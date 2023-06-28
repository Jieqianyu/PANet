import numpy as np
import spconv
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch import distributed as dist
from torch.autograd.function import Function

from .losses import lovasz_losses as Lovasz_loss
from .voxel_utils import PcPreprocessor3DSlim, VFELayerMinus


class AllReduce(Function):

    @staticmethod
    def forward(ctx, input):
        input_list = [
            torch.zeros_like(input) for k in range(dist.get_world_size())
        ]
        # Use allgather instead of allreduce in-place operations is unreliable
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


class NaiveSparseSyncBatchNorm1d(nn.BatchNorm1d):
    """Syncronized Batch Normalization for 3D Tensors.
    Note:
        This implementation is modified from
        https://github.com/facebookresearch/detectron2/
        `torch.nn.SyncBatchNorm` has known unknown bugs.
        It produces significantly worse AP (and sometimes goes NaN)
        when the batch size on each worker is quite different
        (e.g., when scale augmentation is used).
        In 3D detection, different workers has points of different shapes,
        whish also cause instability.
        Use this implementation before `nn.SyncBatchNorm` is fixed.
        It is slower than `nn.SyncBatchNorm`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp16_enabled = False

    def forward(self, input):
        assert input.dtype == torch.float32, \
            f'input should be in float32 type, got {input.dtype}'
        if not self.training or dist.get_world_size() == 1:
            return super().forward(input)
        assert input.shape[0] > 0, 'SyncBN does not support empty inputs'
        C = input.shape[1]
        mean = torch.mean(input, dim=[0])
        meansqr = torch.mean(input * input, dim=[0])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (
            mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1)
        bias = bias.reshape(1, -1)
        return input * scale + bias


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, output, proj_labels_copy, ignore_index=255):
        return F.cross_entropy(output, proj_labels_copy, ignore_index=ignore_index)


def voxel_sem_target(point_voxel_coors, sem_label):
    """make sparse voxel tensor of semantic labels
    Args:
        point_voxel_coors(N, bxyz): point-wise voxel coors
        sem_label(N, ): point-wise semantic label
    Return:
        unq_sem(M, ): voxel-wise semantic label
        unq_voxel(M, bxyz): voxel-wise voxel coors
    """
    voxel_sem = torch.cat([point_voxel_coors, sem_label.reshape(-1, 1)], dim=-1)
    unq_voxel_sem, unq_sem_count = torch.unique(voxel_sem, return_counts=True, dim=0)
    unq_voxel, unq_ind = torch.unique(unq_voxel_sem[:, :4], return_inverse=True, dim=0)
    label_max_ind = torch_scatter.scatter_max(unq_sem_count, unq_ind)[1]
    unq_sem = unq_voxel_sem[:, -1][label_max_ind]
    return unq_sem, unq_voxel


class BasicBlock(spconv.SparseModule):
    def __init__(self, C_in, C_out, indice_key):
        super(BasicBlock, self).__init__()
        self.layers_in = spconv.SparseSequential(
            spconv.SubMConv3d(C_in, C_out, 1, indice_key=indice_key, bias=False),
            NaiveSparseSyncBatchNorm1d(C_out, ),
        )
        self.layers = spconv.SparseSequential(
            spconv.SubMConv3d(C_in, C_out, 3, indice_key=indice_key, bias=False),
            NaiveSparseSyncBatchNorm1d(C_out, ),
            nn.LeakyReLU(0.1),
            spconv.SubMConv3d(C_out, C_out, 3, indice_key=indice_key, bias=False),
            NaiveSparseSyncBatchNorm1d(C_out, ),
            # nn.LeakyReLU(0.1)
        )
        self.relu2 = spconv.SparseSequential(
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        identity = self.layers_in(x)
        out = self.layers(x)
        output = spconv.SparseConvTensor(sum([i.features for i in [identity, out]]),
                                         out.indices, out.spatial_shape, out.batch_size)
        output.indice_dict = out.indice_dict
        output.grid = out.grid
        return self.relu2(output)


def make_layers_sp(C_in, C_out, blocks, indice_key):
    layers = []
    layers.append(BasicBlock(C_in, C_out, indice_key))
    for _ in range(1, blocks):
        layers.append(BasicBlock(C_out, C_out, indice_key))
    return spconv.SparseSequential(*layers)


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


def gather(x, idx):
    """
    :param x: voxelwise features
    :param idx:
    :return: pointwise features
    """
    return x[idx]


class SFE(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, layer_name, layer_num=2):
        super(SFE, self).__init__()
        self.spconv_layers = make_layers_sp(in_channels, out_channels, layer_num, layer_name)

    def forward(self, inputs):
        conv_features = self.spconv_layers(inputs)
        return conv_features


class VFELayerMinusSlim(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 normalize=None,
                 last_layer=False,
                 attention=False,
                 name=''):

        super(VFELayerMinusSlim, self).__init__()
        self.name = 'VFELayerMinusSlim' + name
        self.last_vfe = last_layer
        self.normalize = normalize
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels
        # input batch pointnum feature_num
        self.linear = nn.Linear(in_channels, self.units, bias=True)
        self.weight_linear = nn.Linear(6, self.units, bias=True)
        # self.linear.weight.requires_grad = False
        # self.linear.bias.requires_grad = False
        if self.normalize:
            # self.normalize['num_features'] = self.units
            self.norm = nn.BatchNorm1d(self.units)  # , **self.normalize
            # self.norm = build_norm_layer(normalize, self.units)[1]

    def forward(self, inputs, idx_used, sizes, mean=None, activate=False, gs=None):
        x = self.linear(inputs)
        #x = F.relu(x)
        if activate:
            x = F.relu(x)
        if gs is not None:
            x = x * gs
        if mean is not None:
            x_weight = self.weight_linear(mean)
            if activate:
                x_weight = F.relu(x_weight)
            x = x * x_weight
        index, value = torch.unique(idx_used, return_inverse=True, dim=0)
        max_feature, fk = torch_scatter.scatter_max(x, value, dim=0)
        gather_max_feature = max_feature[value, :]
        x_concated = torch.cat((x, gather_max_feature), dim=1)
        # return x_concated, max_feature
        return x_concated


class SGFE(nn.Module):
    def __init__(self, input_channels, output_channels, reduce_channels, name, p_scale=[2, 4, 6, 8]):
        super(SGFE, self).__init__()
        self.inplanes = input_channels
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.name = name

        self.feature_reduce = nn.Linear(input_channels, reduce_channels)
        self.pooling_scale = p_scale
        self.fc_list = nn.ModuleList()
        self.fcs = nn.ModuleList()
        for i, scale in enumerate(self.pooling_scale):
            self.fc_list.append(nn.Sequential(
            nn.Linear(reduce_channels, reduce_channels//2),
            nn.ReLU(),
            ))
            self.fcs.append(nn.Sequential(nn.Linear(reduce_channels//2, reduce_channels//2)))
        self.scale_selection = nn.Sequential(
            nn.Linear(len(self.pooling_scale) * reduce_channels//2,
                                       reduce_channels),nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(reduce_channels//2, reduce_channels//2, bias=False),
                                nn.ReLU(inplace=False))
        self.out_fc = nn.Linear(reduce_channels//2, reduce_channels, bias=False)
        self.linear_output = nn.Sequential(
            nn.Linear(2 * reduce_channels, reduce_channels, bias=False),
            nn.ReLU(),
            nn.Linear(reduce_channels, output_channels),
        )


    def forward(self, coords_info, top_mean_ms,
                input_data, output_scale, method="max",
                with_fm=True, input_coords=None, input_coords_inv=None):

        topoutput_feature_ms = []
        output_feature_pw = []
        reduced_feature = F.relu(self.feature_reduce(input_data))
        # output = fusion_list
        output_list = [reduced_feature]
        for j, ps in enumerate(self.pooling_scale):
            # index = torch.cat([coords_info[ps]['bxyz_indx'][:, 0].unsqueeze(-1),
            #                    torch.flip(coords_info[ps]['bxyz_indx'], dims=[1])[:, :3]], dim=1)
            index = torch.cat([input_coords[:, 0].unsqueeze(-1),
                              (input_coords[:, 1:] // ps).int()], dim=1)
            unq, unq_inv = torch.unique(index, return_inverse=True, dim=0)
            # unq = unq.type(torch.int64)
            fkm = scatter(reduced_feature, unq_inv, method="mean", dim=0)# + torch_scatter.scatter_max(reduced_feature, unq_inv, dim=0)[0]
            att = self.fc_list[j](fkm)[unq_inv]
            out = ( att)
            output_list.append(out)
        scale_features = torch.stack(output_list[1:], dim=1)#.view(-1, len(self.pooling_scale), 64)
        feat_S = scale_features.sum(1)
        feat_Z = self.fc(feat_S)
        attention_vectors = [fc(feat_Z) for fc in self.fcs]
        attention_vectors = torch.sigmoid(torch.stack(attention_vectors, dim=1))
        scale_features = self.out_fc(torch.sum(scale_features * attention_vectors, dim=1))

        output_f = torch.cat([reduced_feature, scale_features], dim=1)
        proj = self.linear_output(output_f)
        proj = proj[input_coords_inv]
        if with_fm:
            index = torch.cat([coords_info[output_scale]['bxyz_indx'][:, 0].unsqueeze(-1),
                               torch.flip(coords_info[output_scale]['bxyz_indx'], dims=[1])[:, :3]], dim=1)

            unq, unq_inv = torch.unique(index, return_inverse=True, dim=0)
            tv_fmap = scatter(proj, unq_inv, method="max", dim=0)
            return proj, tv_fmap, unq, unq_inv
        else:
            return proj, None, None, None


class TinyUnet(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU())
        self.down = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU())
        self.up = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU())
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, 1, stride=1, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer(x)
        x_down = self.down(x)
        x_up = self.up(x_down)
        x = self.fusion(torch.cat([x, x_up], dim=1))

        return x

class GASNv2(nn.Module):
    def __init__(self, train_cfg, **kwargs):
        super(GASNv2, self).__init__()
        params = train_cfg
        self.scales = params['scales']
        self.multi_frame = False
        self.ce_loss = CELoss()
        self.lovasz_loss = Lovasz_loss.Lovasz_loss(ignore=255)

        self.multi_scale_top_layers = nn.ModuleDict()
        self.feature_list = {
            "0.5": [10, 64],
            "1": [10, 64],
        }
        self.target_scale = 1
        for scale in self.scales:
            top_layer = VFELayerMinusSlim(self.feature_list[str(scale)][0],
                                          self.feature_list[str(scale)][1],
                                          "top_layer_" + str(scale))
            if scale == 0.5:
                rescale = int(0.5 * 10)
            else:
                rescale = scale
            self.multi_scale_top_layers[str(rescale)] = top_layer

        self.aggtopmeanproj = nn.Linear(6, 64, bias=True)
        self.aggtopproj = nn.Linear(128, 64, bias=True)
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, ),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        # self.mlp_short = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(96, 64)
        # )

        self.tv_agglayer = VFELayerMinus(64,
                                         128,
                                         "tvagg",
                                         weight_dims=8)

        self.conv1_block = SFE(64, 64, "svpfe_0")
        self.conv2_block = SFE(64, 64, "svpfe_1")
        self.conv3_block = SFE(64, 64, "svpfe_2")
        self.conv4_block = SFE(64, 64, "svpfe_3")

        self.proj1_block = SGFE(input_channels=64, output_channels=64,\
                                reduce_channels=64, name="proj1")

        self.proj2_block = SGFE(input_channels=64, output_channels=64,\
                                reduce_channels=64, name="proj2")
        self.proj3_block = SGFE(input_channels=64, output_channels=64,\
                                reduce_channels=64, name="proj3")
        self.proj4_block = SGFE(input_channels=64, output_channels=64,\
                                reduce_channels=64, name="proj4")

        num_class = params['n_class']  # SemanticKITTI: 19
        self.out_linears = nn.Sequential(
            NaiveSparseSyncBatchNorm1d(64 + 64 + 64  + 64 + 64 + 64, ),
            nn.Linear(64 + 64 + 64 + 64 + 64 + 64, 128, bias=False),
            NaiveSparseSyncBatchNorm1d(128, ),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128, bias=False),
            NaiveSparseSyncBatchNorm1d(128, ),
            nn.LeakyReLU(0.1),

            nn.Linear(128, num_class)
        )

        self.bev_unet = TinyUnet(64, 64)

        self.out2 = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            NaiveSparseSyncBatchNorm1d(64, ),
            nn.LeakyReLU(0.1),
            nn.Linear(64, num_class)
        )
        self.out3 = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            NaiveSparseSyncBatchNorm1d(64, ),
            nn.LeakyReLU(0.1),
            nn.Linear(64, num_class)
        )
        self.out4 = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            NaiveSparseSyncBatchNorm1d(64, ),
            nn.LeakyReLU(0.1),
            nn.Linear(64, num_class)
        )
        # nn.Conv2d(1024, num_class,
        #               kernel_size=1, stride=1, bias=True)
        self.num_class = params['n_class']

        self.reset_params(params)
        self.init_weights(params['pretrained'])

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print('load backbone parameters')
            checkpoint = torch.load(pretrained, map_location='cpu')
            s = self.state_dict()
            for key, val in checkpoint['state_dict'].items():

                # process ckpt from parallel module
                if key[:6] == 'module':
                    key = key[7:]

                if key in s and s[key].shape == val.shape:
                    s[key][...] = val
                elif key not in s:
                    print('ignore weight from not found key {}'.format(key))
                else:                    print('ignore weight of mistached shape in key {}'.format(key))

            self.load_state_dict(s)

    def reset_params(self, params):
        self.x_lims = params['lims'][0]
        self.y_lims = params['lims'][1]
        self.z_lims = params['lims'][2]
        self.offset = params['offset']
        self.target_scale = params['target_scale']
        num_class = 19
        self.grid_meters = params['grid_meters']
        self.sizes = [int(round((self.x_lims[1] - self.x_lims[0]) / self.grid_meters[0])),
                      int(round((self.y_lims[1] - self.y_lims[0]) / self.grid_meters[1])),
                      (int(round((self.z_lims[1] - self.z_lims[0]) / self.grid_meters[2])))]
        self.lims = [self.x_lims, self.y_lims, self.z_lims]

        self.pooling_scale = params['pooling_scale']

        self.preprocess = PcPreprocessor3DSlim(self.lims, self.grid_meters, scales=self.pooling_scale)

    def add_pcmean_and_gridmeanv2(self, pc, idx, idx_used,
                                  xyz_indx, size_x, size_y, lims, m_pergrid, return_mean=False):

        index, value = torch.unique(idx_used, return_inverse=True, dim=0)
        pc_mean = torch_scatter.scatter_mean(pc[:, :3], value, dim=0)[value]

        pc_subtract_mean = pc[:, :3] - pc_mean
        m_pergird = torch.tensor([m_pergrid[0], m_pergrid[1], m_pergrid[2]], dtype=torch.float, device=pc.device)
        xmin_ymin_zmin = torch.tensor([lims[0], lims[1], lims[2]], dtype=torch.float, device=pc.device)

        pc_gridmean = (xyz_indx.type(torch.cuda.FloatTensor) + self.offset) * m_pergird + xmin_ymin_zmin
        grid_center_minus_mean = pc[:, :3] - pc_gridmean
        pc_feature = torch.cat((pc, pc_subtract_mean, grid_center_minus_mean), dim=1)  # same input
        mean = torch.cat((pc_subtract_mean, grid_center_minus_mean), dim=1)  # different input_mean
        # print(pc_feature.size(), mean.size())
        if return_mean:
            return pc_feature, mean
        else:
            return pc_feature

    def extract_geometry_feature(self, pc, out):

        multi_scales_feature = {}
        multi_scales_point_feature = {}
        topoutput_feature_ms = {}
        aggtopoutput_feature_ms = {}
        topoutput_feature_pwms = {}
        topoutput_mean_ms = {}
        for scale in self.scales:
            multi_scales_point_feature[str(scale)] = []
            multi_scales_feature[str(scale)] = []
            topoutput_feature_ms[str(scale)] = []
            aggtopoutput_feature_ms[str(scale)] = []
            topoutput_feature_pwms[str(scale)] = []
            topoutput_mean_ms[str(scale)] = []
        # first stage feature extractor

        for j, scale in enumerate(self.scales):
            size_x = int(round(self.sizes[0] / scale))
            size_y = int(round(self.sizes[1] / scale))
            size_z = int(round(self.sizes[2] / scale))
            # print("size is ", size_x, size_y, size_z, self.sizes)

            idx_i = out[scale]['bxyz_indx']
            idx_l = idx_i.long()
            if scale == 0.5:
                rescale = int(0.5 * 10)
            else:
                rescale = scale
            # print(idx_l.size())
            # print(idx_l[:10, :])
            # exit(0)
            pc_top, topview_mean = self.add_pcmean_and_gridmeanv2(pc, idx_l,
                                                                  idx_l,
                                                                  idx_l[:, 1:], size_x, size_y,
                                                                  [self.lims[0][0], self.lims[1][0], self.lims[2][0]],
                                                                  [self.grid_meters[0] * scale,
                                                                   self.grid_meters[1] * scale,
                                                                   self.grid_meters[2] * scale],
                                                                  return_mean=True)
            # print(torch.max(topview_mean), torch.min(topview_mean))
            topoutput_mean_ms[str(scale)] = topview_mean
            feat = self.multi_scale_top_layers[str(rescale)](pc_top, idx_l,
                                                             size_x * size_y,
                                                             mean=topview_mean)
            topoutput_feature_pwms[str(scale)] = feat

        # feature projection and aggregation
        aggfv_list = []

        tvms_feature = []
        for scale in self.scales:
            tvms_feature.append(topoutput_feature_pwms[str(scale)])
        tvms_feature = torch.cat(tvms_feature, dim=1)
        # size_x = int(self.sizes[0] // self.target_scale)
        # size_y = int(self.sizes[1] // self.target_scale)

        agg_tpfeature = F.relu(self.aggtopmeanproj(topoutput_mean_ms[str(self.target_scale)])) \
                        * F.relu(self.aggtopproj(tvms_feature))

        agg_fusionfeature = agg_tpfeature
        # idx_i = out['view_reprenstation']['top']['idx'][self.target_scale][i]
        # idx_l = idx_i.long()
        # idx_in_used = idx_i.view(-1, 1)
        pidx_i = out[self.target_scale]['bxyz_indx']
        pidx_l = pidx_i.long()
        # pidx_in_used = pidx_i.view(-1, 1)
        index, value = torch.unique(pidx_l, return_inverse=True, dim=0)
        v = self.tv_agglayer.linear(agg_fusionfeature)
        maxf = torch_scatter.scatter_max(v, value, dim=0)[0]

        aggfv_list.append(self.mlp(tvms_feature))
        # out_feature = max_feat.permute(1, 0).contiguous()
        # aggtopoutput_feature_ms[str(self.target_scale)].append(maxf)
        # topip1 = torch.cat(aggtopoutput_feature_ms[str(self.target_scale)], dim=0)
        # aggfv_list = torch.stack(aggfv_list, dim=0)
        return maxf, topoutput_mean_ms, aggfv_list[0], index, value

    def enhance_features(self, vw_fea, vw_coord, vw_size):
        bw_coord = torch.cat([vw_coord[:, 0].reshape(-1, 1), vw_coord[:, -2:]], dim=1)
        unq, unq_inv = torch.unique(bw_coord.int(), return_inverse=True, dim=0)
        bw_fea = scatter(vw_fea, unq_inv, method='max', dim=0)
        bw_fea = spconv.SparseConvTensor(bw_fea, unq.int(), vw_size[-2:], self.batch_size).dense()

        bw_fea = self.bev_unet(bw_fea)

        return bw_fea


    def bev2points(self, bw_fea, bw_size, points, indicator):
        indicator_t = []
        tensor = torch.ones((1,), dtype=torch.long).cuda()
        for i in range(len(indicator) - 1):
            indicator_t.append(tensor.new_full((indicator[i + 1] - indicator[i],), i))
        indicator_t = torch.cat(indicator_t, dim=0)
        x_idx = (points[:, 0] - self.x_lims[0]) / (self.x_lims[1] - self.x_lims[0]) * bw_size[2].float()
        y_idx = (points[:, 1] - self.y_lims[0]) / (self.y_lims[1] - self.y_lims[0]) * bw_size[1].float()
        x_idxlong = x_idx.type(torch.cuda.LongTensor)
        y_idxlong = y_idx.type(torch.cuda.LongTensor)
        return bw_fea[indicator_t, :, y_idxlong, x_idxlong]


    def forward_train(self, pc_tmp, get_ori=False, pw_label=None, grid_label=None):
        batch_size = len(pc_tmp)
        self.batch_size = batch_size
        if pw_label is not None:
            pw_label = torch.cat(pw_label, dim=0)
        with torch.no_grad():
            indicator = [0]
            pc_ibatch = []
            for i in range(batch_size):
                pc_i = pc_tmp[i]
                pc_ibatch.append(pc_i)
                indicator.append(pc_i.size(0) + indicator[-1])
            pc = torch.cat(pc_ibatch, dim=0)
            filter_pc, info = self.preprocess(pc, indicator)

        feature, topoutput_mean_ms, agg_fv1, coord_ind, full_coord = self.extract_geometry_feature(filter_pc, info)
        coord = torch.cat([coord_ind[:, 0].reshape(-1, 1), torch.flip(coord_ind, dims=[1])[:, :3]], dim=1)

        bw_fea = self.enhance_features(feature, coord.int(), np.int32(self.sizes)[::-1].tolist())

        input_tensor = spconv.SparseConvTensor(
            feature, coord.int(), np.int32(self.sizes)[::-1].tolist(), batch_size
        )
        conv1_output = self.conv1_block(input_tensor)

        proj1_pw, proj1_vw, vw1_coord, pw1_coord  = \
            self.proj1_block(info, None, conv1_output.features, output_scale=2, input_coords=coord.int(),
            input_coords_inv=full_coord)

        conv2_input_tensor = spconv.SparseConvTensor(
            proj1_vw, vw1_coord.int(), (np.array(self.sizes, np.int32) // 2)[::-1], batch_size
        )
        conv2_output = self.conv2_block(conv2_input_tensor)

        proj2_pw, proj2_vw, vw2_coord, pw2_coord = \
            self.proj2_block(info, None, conv2_output.features, output_scale=4, input_coords=vw1_coord.int(),
            input_coords_inv=pw1_coord)

        conv3_input_tensor = spconv.SparseConvTensor(
            proj2_vw, vw2_coord.int(), (np.array(self.sizes, np.int32) // 4)[::-1], batch_size
        )
        conv3_output = self.conv3_block(conv3_input_tensor)

        proj3_pw, proj3_vw, vw3_coord, pw3_coord = \
            self.proj3_block(info, None,conv3_output.features, output_scale=4, input_coords=vw2_coord.int(),
            input_coords_inv=pw2_coord)

        conv4_input_tensor = spconv.SparseConvTensor(
            proj3_vw, vw3_coord.int(), (np.array(self.sizes, np.int32) // 4)[::-1], batch_size
        )
        conv4_output = self.conv4_block(conv4_input_tensor)

        proj4_pw, _, _, _ = self.proj4_block(info, None, conv4_output.features, output_scale=4, with_fm=False,
                                             input_coords=vw3_coord.int(),
            input_coords_inv=pw3_coord)

        bw_size = np.array(self.sizes, np.float)[::-1]
        bw_size = torch.from_numpy(bw_size.copy()).cuda()
        b2p = self.bev2points(bw_fea, bw_size, filter_pc, indicator)

        pw_feature = torch.cat([proj1_pw, proj2_pw, proj3_pw, proj4_pw, agg_fv1, b2p], dim=1).contiguous()

        score = self.out_linears(pw_feature)

        if get_ori:
            index_04 = torch.cat([info[2]['bxyz_indx'][:, 0].unsqueeze(-1),
                               torch.flip(info[2]['bxyz_indx'], dims=[1])[:, :3]], dim=1)
            index_08 = torch.cat([info[4]['bxyz_indx'][:, 0].unsqueeze(-1),
                               torch.flip(info[4]['bxyz_indx'], dims=[1])[:, :3]], dim=1)
            vw_label_04 = voxel_sem_target(index_04.int(), pw_label.int())[0]
            vw_label_08 = voxel_sem_target(index_08.int(), pw_label.int())[0]
            return score, [[vw_label_04, self.out2(conv2_output.features)],
            [vw_label_08.clone(), self.out3(conv3_output.features)],
            [vw_label_08, self.out4(conv4_output.features)]]

        return dict(
            semantic_logits = score, # cat
            points_fea = pw_feature # cat
        )

    def forward(self, return_loss=True, **data):
        if return_loss:
            points_label = data["points_label"]
            output_teacher, all_teach_pair = self.forward_train(data['points'], get_ori=True, pw_label=points_label)
            proj_labels_copy = data["points_label"]
            proj_labels_copy = torch.cat(proj_labels_copy, dim=0).long().clone()
            proj_labels_copy[proj_labels_copy == 0] = 256
            proj_labels_copy = proj_labels_copy - 1
            scale_loss = self.lovasz_loss(F.softmax(output_teacher, dim=1), proj_labels_copy)
            focal_loss = self.ce_loss(output_teacher, proj_labels_copy)
            loss_dict = {}
            for i in range(len(all_teach_pair)):
                teach_pair = all_teach_pair[i]
                voxel_labels_copy = teach_pair[0].long().clone()
                voxel_labels_copy[voxel_labels_copy == 0] = 256
                voxel_labels_copy = voxel_labels_copy - 1
                # print(proj_labels_copy.size(), output.size())
                res04_loss = self.lovasz_loss(F.softmax(teach_pair[1], dim=1), voxel_labels_copy)
                res04_loss2 = self.ce_loss(teach_pair[1], voxel_labels_copy)
                loss_dict["voxel_" + str(i) + "lovasz_loss"] = res04_loss
                loss_dict["voxel_" + str(i) + "ce_loss"] = res04_loss2
            loss = {"pw_lovasz_loss": scale_loss,"pw_ce_loss": focal_loss}
            loss.update(loss_dict)
            return loss

        else:
            out_t = self.forward_train(data['points'], get_ori=False)
            return out_t
