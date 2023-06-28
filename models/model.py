import random

import numpy as np
import torch
import torch_scatter
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from torch_cluster import fps

from utils import common_utils
from utils.evaluate_panoptic import eval_one_scan_w_fname, valid_xentropy_ids

from .losses.spg_loss import AffinityLoss
from .pcd_seg_v2 import GASNv2
from .position_embedding import PositionEmbeddingCoordsFourier
from .trasnformer import Transformer, index_points
from .vfe import CFE


def scatter(x, idx, method, dim=0):
    if method == "max":
        return torch_scatter.scatter_max(x, idx, dim=dim)[0]
    elif method == "mean":
        return torch_scatter.scatter_mean(x, idx, dim=dim)
    elif method == "sum":
        return torch_scatter.scatter_add(x, idx, dim=dim)
    else:
        print("unknown method")
        exit(-1)

class Base(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.plus = 1
        sem_cfg = dict(
            lims=cfg.MODEL.LIMS,
            offset=cfg.MODEL.OFFSET,
            target_scale=cfg.MODEL.TARGET_SCALE,
            grid_meters=cfg.MODEL.GRID_METERS,
            scales=cfg.MODEL.SCALES,
            pooling_scale=cfg.MODEL.POOLING_SCALE,
            sizes=cfg.MODEL.SIZES,
            n_class=cfg.MODEL.NCLASS,
            pretrained=cfg.MODEL.get('SEM_PRETRAIN', None)
        )
        self.lims = cfg.MODEL.LIMS
        self.sizes = torch.tensor(cfg.MODEL.SIZES)
        self.sem_backbone = GASNv2(sem_cfg)
        self.merge_func_name = cfg.MODEL.POST_PROCESSING.MERGE_FUNC

    def update_evaluator(self, evaluator, sem_preds, ins_preds, inputs):
        for i in range(len(sem_preds)):
            eval_one_scan_w_fname(evaluator, inputs['points_label'][i].cpu().detach().numpy().reshape(-1),
                inputs['inst_label'][i].cpu().detach().numpy().reshape(-1),
                sem_preds[i], ins_preds[i], inputs['pcd_fname'][i])

    def merge_ins_sem(self, sem_preds, pred_ins_ids, logits=None, inputs=None):
        merged_sem_preds = []
        for i in range(len(sem_preds)):
            if self.merge_func_name == 'merge_ins_sem':
                merged_sem_preds.append(common_utils.merge_ins_sem(sem_preds[i], pred_ins_ids[i]))
            elif self.merge_func_name == 'merge_ins_sem_logits_size_based':
                merged_sem_preds.append(common_utils.merge_ins_sem_logits_size_based(sem_preds[i], pred_ins_ids[i], i, logits, inputs))
            elif self.merge_func_name == 'none':
                merged_sem_preds.append(sem_preds[i])
        return merged_sem_preds

    def calc_sem_label(self, sem_logits, inputs, need_add_one=True):
        pt_pred_labels_cat = torch.argmax(sem_logits, dim=1)
        pt_pred_labels_cat = pt_pred_labels_cat.cpu().detach().numpy()
        if need_add_one:
            pt_pred_labels_cat += self.plus

        pt_pred_labels = []
        start_idx = 0
        end_idx = 0
        for points in inputs['points']:
            end_idx = start_idx + len(points)
            pt_pred_labels.append(pt_pred_labels_cat[start_idx:end_idx])
            start_idx = end_idx
        return pt_pred_labels

    def forward(self, batch, is_test=False, before_merge_evaluator=None, after_merge_evaluator=None, require_merge=True):
        example = {}
        for k, v in batch.items():
            if k not in ['i_iter', 'pcd_fname', 'rank', 'epoch']:
                example[k] = [l.cuda() if l is not None else None for l in v]

        if is_test:
            sem_output_dict = self.sem_backbone(return_loss=False, **example)
            sem_logits = sem_output_dict['semantic_logits']

        out_dict = {}
        if not is_test:
            loss_dict = self.sem_backbone(return_loss=True, **example)
            out_dict['loss'] = sum(list(loss_dict.values()))
            out_dict.update(loss_dict)
        else:
            out_dict['loss'] = torch.zeros(1, requires_grad=True).cuda()
        if is_test:
            pt_sem_preds = self.calc_sem_label(sem_logits, batch, need_add_one=True)
            pt_ins_ids_preds = [np.zeros_like(pt_sem_preds[i]) for i in range(len(pt_sem_preds))]
            merged_sem_preds = pt_sem_preds
            self.update_evaluator(after_merge_evaluator, merged_sem_preds, pt_ins_ids_preds, batch)
            out_dict['sem_preds'] = merged_sem_preds
            out_dict['ins_preds'] = pt_ins_ids_preds

        return out_dict


def find_connected_components(points, batch_idx, dist):
    device = points.device
    bsz = batch_idx.max().item() + 1
    base = 0
    components_inds = torch.zeros_like(batch_idx) - 1

    for i in range(bsz):
        batch_mask = batch_idx == i
        if batch_mask.any():
            this_points = points[batch_mask]
            dist_mat = this_points[:, None, :2] - this_points[None, :, :2] # only care about xy
            dist_mat = (dist_mat ** 2).sum(2) ** 0.5
            adj_mat = dist_mat < dist
            c_inds = connected_components(adj_mat.cpu().numpy(), directed=False)[1]
            c_inds = torch.from_numpy(c_inds).to(device).int() + base
            base = c_inds.max().item() + 1
            components_inds[batch_mask] = c_inds

    assert len(torch.unique(components_inds)) == components_inds.max().item() + 1

    return components_inds


class PANet(Base):
    def __init__(self, cfg) -> None:
        super(PANet, self).__init__(cfg)
        self.cls_voxel_sizes = cfg.MODEL.PROPOSALS.CLS_VOXEL_SIZES
        lims = cfg.MODEL.LIMS
        self.pc_range = [lims[0][0], lims[1][0], lims[2][0], lims[0][1], lims[1][1], lims[2][1]]
        self.cls_bandwidths = cfg.MODEL.PROPOSALS.CLS_BANDWIDTH
        self.cls_connected_radius = cfg.MODEL.PROPOSALS.CLS_RADIUS
        self.stage = cfg.MODEL.STAGE

        dim = cfg.MODEL.PROPOSALS.CHANNELS
        in_channels = 64*6
        self.channel_reduce = torch.nn.Linear(in_channels, dim, bias=False)

        if self.stage > 0:
            self.cfe = CFE(in_channels=dim+3, feat_channels=[64, dim])
            self.transformer = Transformer(dim)
            self.pe = PositionEmbeddingCoordsFourier(dim)
            self.affinity_loss = AffinityLoss()

            self.edge_mlp = torch.nn.Sequential(
                torch.nn.Linear(dim*2+3, dim, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(dim, 1, bias=True),
                torch.nn.Sigmoid()
            )

        self.fix_parameters()

    def fix_parameters(self):
        for k, v in self.named_parameters():
            if 'sem_backbone' in k:
                v.requires_grad = False

    def dynamic_shift(self, points, bandwidth, iters=4):
        X = points
        XT = X.T
        dist =torch.norm(X[:, None]-X, dim=2, p=2)
        K = (dist <= bandwidth).float()
        D = torch.matmul(K, torch.ones([X.shape[0], 1]).to(points)).view(-1)

        for _ in range(iters):
            XT = torch.matmul(XT, K) / D

        X = XT.T

        return X

    def gen_proposals(self, embeddings, sem_logits, ins_fea, batch_idx, gt_centers=None, is_test=True):
        device = embeddings.device
        pc_range = torch.tensor(self.pc_range, device=device)
        valid_classes = torch.tensor(sorted(valid_xentropy_ids), device=device).long()
        sem_preds = sem_logits.argmax(1) + self.plus if is_test else sem_logits
        cluster_inds_list, embeddings_list, feats_list, valid_masks_list = [], [], [], []
        valid = (sem_preds.reshape(-1, 1) == valid_classes).any(dim=-1)
        if valid.sum() == 0:
            return None
        base = 0
        for cls_idx in range(len(valid_classes)):
            fg_mask = sem_preds == valid_classes[cls_idx]
            if not fg_mask.any():
                continue
            cls_embedding = embeddings[fg_mask]
            cls_fea = ins_fea[fg_mask]
            voxel_size = torch.tensor(self.cls_voxel_sizes[cls_idx], device=device)
            coords = torch.div(cls_embedding - pc_range[None, :3], voxel_size[None, :]).floor().int()
            cls_batch_idx = batch_idx[fg_mask]
            coords = torch.cat([cls_batch_idx[:, None], coords], dim=1)

            unq, unq_inv = torch.unique(coords, return_inverse=True, dim=0)
            seed_points = scatter(cls_embedding, unq_inv, 'mean')
            seed_points = self.dynamic_shift(seed_points, self.cls_bandwidths[cls_idx])

            dist = self.cls_connected_radius[cls_idx]
            cls_cluster_inds = find_connected_components(seed_points, unq[:, 0].int(), dist)
            cls_cluster_inds = cls_cluster_inds[unq_inv] + base
            base = cls_cluster_inds.max().item() + 1

            cls_cluster_inds = torch.stack([cls_batch_idx, cls_cluster_inds], dim=1)
            cls_cluster_inds = torch.nn.functional.pad(cls_cluster_inds, (1, 0), 'constant', valid_classes[cls_idx])  # (cls_idx, batch_idx, cluster_idx)

            cluster_inds_list.append(cls_cluster_inds)
            embeddings_list.append(cls_embedding)
            feats_list.append(cls_fea)
            valid_masks_list.append(fg_mask)

        return dict(
            cluster_pt_inds=torch.cat(cluster_inds_list),
            pt_embeddings=torch.cat(embeddings_list),
            pt_feats=torch.cat(feats_list),
            valid_masks_list=valid_masks_list
        )

    def extract_cluster_feats(self, cluster_pt_inds, pt_points, pt_embeddings, pt_feats):
        cluster_inds, unq_inv = torch.unique(cluster_pt_inds, return_inverse=True, dim=0)
        cluster_centers = scatter(pt_embeddings, unq_inv, 'mean')
        f_cluster = pt_points - cluster_centers[unq_inv]

        pt_feats, cluster_feats = self.cfe(cluster_pt_inds, pt_feats, pt_points, f_cluster)

        return dict(
            cluster_inds=cluster_inds,
            cluster_centers=cluster_centers,
            cluster_feats=cluster_feats,
            pt_feats=pt_feats,
            unq_inv=unq_inv
        )

    def interact_neighbor_feats(self, centers, feat, k=6):
        dist = torch.sum((centers[:, None]- centers[None])**2, dim=-1)
        knn_idx = dist.argsort()[:, :k]
        knn_xyz = index_points(centers.unsqueeze(0), knn_idx.unsqueeze(0))[0]
        neighbor_feat = index_points(feat.unsqueeze(0), knn_idx.unsqueeze(0))[0]
        neighbor_pos = knn_xyz - centers[:, None]
        interacted_out_dict = self.transformer(feat.unsqueeze(0), pos_embedding=self.pe, center_pos=centers.unsqueeze(0),
            y=neighbor_feat.unsqueeze(0), neighbor_pos=neighbor_pos.unsqueeze(0))

        discriminative_feats = interacted_out_dict['ct_feat'][0]  # n, c

        return discriminative_feats

    def merge_clusters(self, cluster_inds, cluster_centers, cluster_feats, cluster_affinity_labels=None):
        bsz = cluster_inds[:, 1].max().item()+1
        cls_idxs = torch.unique(cluster_inds[:, 0])
        new_cluster_inds = torch.zeros_like(cluster_inds)-1
        new_cluster_feats = torch.zeros_like(cluster_feats)
        base = 0
        device = cls_idxs.device
        loss = torch.zeros(1, dtype=torch.float, device=device)[0]
        for batch_idx in range(bsz):
            for cls_idx in cls_idxs:
                valid = (cluster_inds[:, 0] == cls_idx) & (cluster_inds[:, 1] == batch_idx)
                if valid.sum() == 0:
                    continue
                valid_centers, valid_feats = cluster_centers[valid], cluster_feats[valid]
                num_c = valid_feats.shape[0]
                discriminative_feats = self.interact_neighbor_feats(valid_centers, valid_feats)
                matrix_feats = discriminative_feats.unsqueeze(1).repeat(1, num_c, 1)
                edges_feats = torch.cat([matrix_feats, matrix_feats.permute(1, 0, 2), torch.abs(valid_centers[None]-valid_centers[:,None])], dim=-1)
                cls_cluster_affinity = self.edge_mlp(edges_feats.flatten(0, 1)).reshape(num_c, num_c)

                new_cluster_feats[valid] = discriminative_feats
                if cluster_affinity_labels is not None:
                    loss += self.affinity_loss(cls_cluster_affinity, cluster_affinity_labels[batch_idx][cls_idx.item()].float())

                cls_cluster_affinity = cls_cluster_affinity.clone().detach()
                with torch.no_grad():
                    cls_cluster_affinity[range(num_c), range(num_c)] += 1
                    if self.training:
                        adj_mat = cls_cluster_affinity > 1
                    else:
                        if cls_idx == 4 and len(valid_xentropy_ids) == 8:
                            adj_mat = cls_cluster_affinity > 0.5
                        else:
                            adj_mat = cls_cluster_affinity > 0.85

                    c_inds = connected_components(adj_mat.cpu().numpy(), directed=False)[1]
                    c_inds = torch.from_numpy(c_inds).to(device).int() + base
                    new_cls_cluster_inds = torch.stack([cluster_inds[valid, 0], cluster_inds[valid, 1], c_inds], dim=1)  # cls, batch_id, c_id
                    new_cluster_inds[valid] = new_cls_cluster_inds
                    base = c_inds.max().item() + 1

        loss /= (cluster_inds.shape[0] + 1e-6)

        new_cluster_inds, unq_inv = torch.unique(new_cluster_inds, return_inverse=True, dim=0)
        new_cluster_feats = scatter(new_cluster_feats, unq_inv, 'mean')

        return dict(
            cluster_inds=new_cluster_inds,
            cluster_feats=new_cluster_feats,
            unq_inv=unq_inv
        ), loss

    def process_proposals(self, cluster_pt_inds, valid_mask_list, batch_idx):
        cls_idxs = torch.unique(cluster_pt_inds[:, 0])
        pt_ins_ids = torch.zeros_like(valid_mask_list[0]).long() - 1
        for i in range(len(cls_idxs)):
            cls_cluster_pt_mask = cluster_pt_inds[:, 0] == cls_idxs[i]
            pt_ins_ids[valid_mask_list[i]] = cluster_pt_inds[cls_cluster_pt_mask, 2]
        pt_ins_ids += 1

        pt_ins_ids_preds = []
        for i in range(self.batch_size):
            mask = batch_idx == i
            pt_ins_ids_preds.append(pt_ins_ids[mask])

        pt_ins_ids_preds = [x.detach().cpu().numpy() for x in pt_ins_ids_preds]

        return pt_ins_ids_preds

    def gen_affinity_labels(self, sem_labels, pt_labels, cluster_pt_inds):
        if pt_labels is None:
            return None
        device = pt_labels.device

        unq, unq_inv = torch.unique(cluster_pt_inds, return_inverse=True, dim=0)
        cluster_idxs = torch.unique(unq_inv)
        cluster_labels = torch.zeros_like(unq[:, 2])
        for idx in cluster_idxs:
            valid = (unq_inv == idx)
            cluster_sem = sem_labels[valid]
            fg_valid = (cluster_sem.reshape(-1, 1) == torch.LongTensor(valid_xentropy_ids).to(device)).any(dim=-1)
            idx_label = pt_labels[valid]
            filter_idx_labels = idx_label[fg_valid]
            if filter_idx_labels.shape[0] == 0:
                cluster_labels[idx] = -1
            else:
                cluster_labels[idx] = torch.mode(filter_idx_labels)[0]

        bsz = unq[:, 1].max().item()+1
        cls_idxs = torch.unique(unq[:, 0])
        cluster_affinity_label = []
        for batch_idx in range(bsz):
            cls_affinity_label = {}
            for cls_idx in cls_idxs:
                valid = (unq[:, 0] == cls_idx) & (unq[:, 1] == batch_idx)
                cls_cluster_labels = cluster_labels[valid]
                invalid_idx = (cls_cluster_labels == -1).nonzero()
                affinity = cls_cluster_labels[None] == cls_cluster_labels[:, None]
                affinity[invalid_idx] = False
                affinity[:, invalid_idx] = False
                affinity[invalid_idx, invalid_idx] = True
                cls_affinity_label[cls_idx.item()] = affinity
            cluster_affinity_label.append(cls_affinity_label)

        return cluster_affinity_label

    def forward(self, batch, is_test=False, before_merge_evaluator=None, after_merge_evaluator=None, require_merge=True):
        example = {}
        for k, v in batch.items():
            if k not in ['i_iter', 'pcd_fname', 'rank', 'epoch']:
                example[k] = [l.cuda() if l is not None else None for l in v]

        with torch.no_grad():
            sem_output_dict = self.sem_backbone(return_loss=False, **example)
            sem_logits = sem_output_dict['semantic_logits']

        onlytest = is_test and (before_merge_evaluator is None and after_merge_evaluator is None)
        batch_size = len(example['points'])
        self.batch_size = batch_size
        with torch.no_grad():
            indicator = [0]
            pc_ibatch = []
            batch_idx = []
            tensor = torch.ones((1,), dtype=torch.long).cuda()
            for i in range(batch_size):
                pc_i = example['points'][i][:, :3]
                pc_ibatch.append(pc_i)
                indicator.append(pc_i.size(0) + indicator[-1])
                batch_idx.append(tensor.new_full((indicator[i+1] - indicator[i],), i))
            points = torch.cat(pc_ibatch, dim=0)
            batch_idx = torch.cat(batch_idx, dim=0)
            inst_label = torch.cat(example['inst_label'], dim=0) if not onlytest else None
            sem_label = torch.cat(example['points_label'], dim=0) if not onlytest else None

        # generate seedpoints for each class
        embedding = points[:, :3]
        sem_logits = sem_logits if is_test else sem_label
        ins_fea = self.channel_reduce(sem_output_dict['points_fea'])
        proposals_pt_info = self.gen_proposals(embedding, sem_logits, ins_fea, batch_idx, is_test=is_test)

        if proposals_pt_info is not None:
            if self.stage == 0:
                cluster_pt_inds = proposals_pt_info['cluster_pt_inds']

            if self.stage > 0:
                pt_points = torch.cat([points[x] for x in proposals_pt_info['valid_masks_list']])
                proposals_info = self.extract_cluster_feats(
                    proposals_pt_info['cluster_pt_inds'],
                    pt_points,
                    proposals_pt_info['pt_embeddings'],
                    proposals_pt_info['pt_feats'])

                pt_labels = torch.cat([inst_label[x] for x in proposals_pt_info['valid_masks_list']]) if not onlytest else None
                sem_labels = torch.cat([sem_label[x] for x in proposals_pt_info['valid_masks_list']]) if not onlytest else None
                proposals_affinity_label = self.gen_affinity_labels(sem_labels, pt_labels, proposals_pt_info['cluster_pt_inds'])
                secondary_proposals_info, loss_affinity = self.merge_clusters(
                    proposals_info['cluster_inds'],
                    proposals_info['cluster_centers'],
                    proposals_info['cluster_feats'],
                    proposals_affinity_label)
                cluster_pt_inds = secondary_proposals_info['cluster_inds'][secondary_proposals_info['unq_inv']][proposals_info['unq_inv']]

        out_dict = {}
        if self.stage > 0 and proposals_pt_info is not None:
            out_dict['loss_affinity'] = loss_affinity
        else:
            out_dict['loss_affinity'] = torch.zeros(1, requires_grad=True)[0].to(points)

        if onlytest or self.stage == 0:
            out_dict['loss'] = torch.zeros(1)[0].to(points)
        else:
            out_dict['loss'] = sum(list(out_dict.values()))

        if is_test:
            pt_sem_preds = self.calc_sem_label(sem_logits, batch, need_add_one=True)
            if proposals_pt_info is not None:
                pt_ins_ids_preds = self.process_proposals(cluster_pt_inds,
                    proposals_pt_info['valid_masks_list'], batch_idx)
            else:
                pt_ins_ids_preds = [np.zeros_like(x) for x in pt_sem_preds]

            if require_merge:
                merged_sem_preds = self.merge_ins_sem(pt_sem_preds, pt_ins_ids_preds)
            else:
                merged_sem_preds = pt_sem_preds
            if before_merge_evaluator is not None:
                self.update_evaluator(before_merge_evaluator, pt_sem_preds, pt_ins_ids_preds, batch)
            if after_merge_evaluator is not None:
                self.update_evaluator(after_merge_evaluator, merged_sem_preds, pt_ins_ids_preds, batch)
            out_dict['sem_preds'] = merged_sem_preds
            out_dict['ins_preds'] = pt_ins_ids_preds

        return out_dict