import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter


def quantitize(data, lim_min, lim_max, size):
    idx = (data - lim_min) / (lim_max - lim_min) * size
    idx = torch.max(idx, torch.zeros_like(idx))
    idx = torch.min(idx, torch.ones_like(idx) * (size - 1))
    return idx.type(torch.cuda.LongTensor)

def quantitizev2(data, lim_min, lim_max, size, with_res=False):

    #print(size)
    # if size != 12 and size != 6 and size != 24 and size != 2 and size != 48 and size != 4:
    #     print(size)
    #     raise ValueError("fk number")
    idx = (data - lim_min) / (lim_max - lim_min) * size.float()
    idxlong = idx.type(torch.cuda.LongTensor)
    if with_res:
        idx_res = idx - idxlong.float()
        return idxlong, idx_res
    # print(data, lim_min, lim_max, size, idx)
    else:
        return idxlong


def mask_op(data, x_min, x_max):
    mask = (data > x_min) & (data < x_max)
    return mask


class PcPreprocessor3D(nn.Module):
    def __init__(self, lims, grid_meters, scales=[0.5, 1]):
        # todo move to cfg
        super(PcPreprocessor3D, self).__init__()
        self.x_lims = lims[0]
        self.y_lims = lims[1]
        self.z_lims = lims[2]
        self.offset = 0.5
        self.grid_meters = grid_meters

        self.grid_sizes = torch.tensor([int(round((self.x_lims[1] - self.x_lims[0]) / self.grid_meters[0])),
                           int(round((self.y_lims[1] - self.y_lims[0]) / self.grid_meters[1])),
                           int(round((self.z_lims[1] - self.z_lims[0]) / self.grid_meters[2]))])
        self.grid_sizes = self.grid_sizes.cuda().long()
        self.lims = [self.x_lims, self.y_lims, self.z_lims]

        self.scales = scales
        self.view_name = ["top_view", "front_view"]

    def filter_pc(self, pc):
        mask_x = mask_op(pc[:, 0], self.lims[0][0] + 0.0001, self.lims[0][1] - 0.0001)
        mask_y = mask_op(pc[:, 1], self.lims[1][0] + 0.0001, self.lims[1][1] - 0.0001)
        mask_z = mask_op(pc[:, 2], self.lims[2][0] + 0.0001, self.lims[2][1] - 0.0001)
        mask = mask_x & mask_y & mask_z
        filter_pc = pc[mask]
        return filter_pc

    def forward(self, pc, cls_label=None, keep_mask=None):
        if isinstance(pc, list):
            batch = len(pc)
        else:
            batch, _, _ = pc.shape
        pc_list = []
        view_reprenstation = {"top": {"idx": {}, "coord": {}, "res": {}},
                              "front": {"idx":{}, "coord": {}, "res": {}}}
        for i in range(batch):
            pc_i = pc[i][:, :]

            if keep_mask is not None:
                pc_i = pc_i[keep_mask[i]]
            # todo add aug
            filter_pc = pc_i # self.filter_pc(pc_i)
            for scale in self.scales:
                # todo yms the view is a little tricky
                # print(self.grid_sizes[0], scale)
                xidx, xres = quantitizev2(filter_pc[:, 0], self.lims[0][0],
                                  self.lims[0][1], self.grid_sizes[0].float() // scale, with_res=True)
                yidx, yres = quantitizev2(filter_pc[:, 1], self.lims[1][0],
                                  self.lims[1][1], self.grid_sizes[1].float() // scale, with_res=True)
                zidx, zres = quantitizev2(filter_pc[:, 2], self.lims[2][0],
                                  self.lims[2][1], self.grid_sizes[2].float() // scale, with_res=True)
                xy_indx = torch.stack([xidx, yidx], dim=-1)
                xz_indx = torch.stack([xidx, yidx, zidx], dim=-1)
                xz_indx = torch.stack([xidx, yidx, zidx], dim=-1)
                topres = torch.stack([xres, yres], dim=-1)
                frontres = torch.stack([xres, zres], dim=-1)
                size = (self.grid_sizes[0].float() // scale).type(torch.cuda.LongTensor)
                size_z = (self.grid_sizes[2].float() // scale).type(torch.cuda.LongTensor)

                idx = xidx * size + yidx
                idx2 = xidx * size * size_z + yidx * size_z + zidx
                # print(size_z, "scale ", scale)
                # print("info ")
                # print(scale, size, torch.max(idx), self.lims[0][0], self.lims[0][1], torch.max(filter_pc[:, 0]), self.grid_sizes[0])
                # print(scale, size, torch.max(idx), self.lims[1][0], self.lims[1][1], torch.max(filter_pc[:, 1]), self.grid_sizes[0])
                if scale not in view_reprenstation["top"]["idx"]:
                    view_reprenstation["top"]["idx"][scale] = []
                    view_reprenstation["top"]["coord"][scale] = []
                    view_reprenstation["top"]["res"][scale] = []
                    view_reprenstation["front"]["idx"][scale] = []
                    view_reprenstation["front"]["coord"][scale] = []
                    view_reprenstation["front"]["res"][scale] = []
                view_reprenstation["top"]["idx"][scale].append(idx)
                view_reprenstation["top"]["coord"][scale].append(xy_indx)
                view_reprenstation["top"]["res"][scale].append(topres)
                view_reprenstation["front"]["idx"][scale].append(idx2)
                view_reprenstation["front"]["coord"][scale].append(xz_indx)
                view_reprenstation["front"]["res"][scale].append(frontres)
            pc_list.append(filter_pc)
        return {"pc": pc_list, "view_reprenstation": view_reprenstation}


class PolarPreprocessor3D(nn.Module):
    def __init__(self, lims, grid_meters, scales=[0.5, 1]):
        # todo move to cfg
        super(PolarPreprocessor3D, self).__init__()
        self.x_lims = lims[0]
        self.y_lims = lims[1]
        self.z_lims = lims[2]
        self.offset = 0.5
        self.grid_meters = grid_meters

        self.grid_sizes = torch.tensor([int(round((self.x_lims[1] - self.x_lims[0]) / self.grid_meters[0])),
                           int(round((self.y_lims[1] - self.y_lims[0]) / self.grid_meters[1])),
                           int(round((self.z_lims[1] - self.z_lims[0]) / self.grid_meters[2]))])
        self.grid_sizes = self.grid_sizes.cuda().long()
        self.lims = [self.x_lims, self.y_lims, self.z_lims]

        self.scales = scales
        self.view_name = ["top_view", "front_view"]

    def forward(self, pc, cls_label=None, keep_mask=None):
        if isinstance(pc, list):
            batch = len(pc)
        else:
            batch, _, _ = pc.shape
        pc_list = []
        gt_labels_list = []
        idx_scale_list = {}
        xy_idx_scale_list = {}
        view_reprenstation = {"top": {"idx": {}, "coord": {}, "res": {}},
                              "front": {"idx":{}, "coord": {}, "res": {}}}
        for i in range(batch):
            pc_i = pc[i][:, :]

            if keep_mask is not None:
                pc_i = pc_i[keep_mask[i]]
            # todo add aug
            rho = torch.sqrt(pc_i[:, 0] ** 2 + pc_i[:, 1] ** 2)
            phi = torch.atan2(pc_i[:, 1], pc_i[:, 0]) / np.pi * 180.
            filter_pc = torch.stack([rho, phi, pc_i[:, 2], pc_i[:, 3]], dim=-1)
            for scale in self.scales:
                # todo yms the view is a little tricky
                # print(self.grid_sizes[0], scale)
                xidx, xres = quantitizev2(filter_pc[:, 0], self.lims[0][0],
                                  self.lims[0][1], self.grid_sizes[0].float() // scale, with_res=True)
                yidx, yres = quantitizev2(filter_pc[:, 1], self.lims[1][0],
                                  self.lims[1][1], self.grid_sizes[1].float() // scale, with_res=True)
                zidx, zres = quantitizev2(filter_pc[:, 2], self.lims[2][0],
                                  self.lims[2][1], self.grid_sizes[2].float() // scale, with_res=True)
                # if scale == 1:
                #     img = torch.zeros((240, 240), dtype=torch.uint8).cuda()
                #     img[xidx, yidx] = 255
                #     import cv2
                #     cv2.imshow("img", img.detach().cpu().numpy())
                #     cv2.waitKey(0)

                size = (self.grid_sizes[0].float() // scale).type(torch.cuda.LongTensor)
                size_z = (self.grid_sizes[2].float() // scale).type(torch.cuda.LongTensor)
                yidx = torch.clamp(yidx, min=0, max=size-1)
                xy_indx = torch.stack([xidx, yidx], dim=-1)
                # print(torch.min(yidx), torch.max(yidx))
                xz_indx = torch.stack([xidx, yidx, zidx], dim=-1)
                topres = torch.stack([xres, yres], dim=-1)
                frontres = torch.stack([xres, zres], dim=-1)

                idx = xidx * size + yidx
                idx2 = xidx * size * size_z + yidx * size_z + zidx
                if scale not in view_reprenstation["top"]["idx"]:
                    view_reprenstation["top"]["idx"][scale] = []
                    view_reprenstation["top"]["coord"][scale] = []
                    view_reprenstation["top"]["res"][scale] = []
                    view_reprenstation["front"]["idx"][scale] = []
                    view_reprenstation["front"]["coord"][scale] = []
                    view_reprenstation["front"]["res"][scale] = []
                view_reprenstation["top"]["idx"][scale].append(idx)
                view_reprenstation["top"]["coord"][scale].append(xy_indx)
                view_reprenstation["top"]["res"][scale].append(topres)
                view_reprenstation["front"]["idx"][scale].append(idx2)
                view_reprenstation["front"]["coord"][scale].append(xz_indx)
                view_reprenstation["front"]["res"][scale].append(frontres)
            pc_list.append(filter_pc)
        return {"pc": pc_list, "view_reprenstation": view_reprenstation}


class VFELayerMinusPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 normalize=None,
                 last_layer=False,
                 attention=False,
                 weight_dims=7,
                 name=''):

        super(VFELayerMinusPP, self).__init__()
        self.name = 'VFELayerMinusPP' + name
        self.last_vfe = last_layer
        self.normalize = normalize

        self.units = out_channels
        # input batch pointnum feature_num
        self.linear = nn.Linear(in_channels, self.units, bias=True)
        self.weight_linear = nn.Linear(weight_dims, self.units, bias=True)
        # self.linear.weight.requires_grad = False
        # self.linear.bias.requires_grad = False
        if self.normalize:
            # self.normalize['num_features'] = self.units
            self.norm = nn.BatchNorm1d(self.units)  # , **self.normalize
            # self.norm = build_norm_layer(normalize, self.units)[1]

    def forward(self, inputs, idx, idx_used, sizes, mean=None, activate=False, gs=None):
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

        max_feature = x.new_zeros((sizes, x.size(1)))
        idx_used = idx_used.expand_as(x)
        max_feature, fk = torch_scatter.scatter_max(x, idx_used, out=max_feature, dim=0)
        return max_feature


def gather_feature(feature_map, idx_info, mode="nearest"):
    if mode=="nearest":
        return feature_map[idx_info["x"], idx_info["y"], :]
    elif mode == "bilinear":
        xidx = idx_info["x"]
        yidx = idx_info["y"]
        resx = idx_info["resx"]
        resy = idx_info["resy"]
        xsize = idx_info["xsize"]
        ysize = idx_info["ysize"]

        mask = torch.ones_like(xidx).bool()
        mask = (xidx <= (xsize - 2)) & mask
        mask = (yidx <= (ysize - 2)) & mask

        xplusone = torch.clamp(xidx + 1, 0, xsize - 1)
        yplusone = torch.clamp(yidx + 1, 0, ysize - 1)
        feature00 = feature_map[idx_info["x"], idx_info["y"], :]

        feature01 = feature_map[idx_info["x"], yplusone, :]
        feature11 = feature_map[xplusone, yplusone, :]
        feature10 = feature_map[xplusone, idx_info["y"], :]

        feature_bilinear = feature00 * (1 - resx.unsqueeze(dim=-1)) * (1 - resy.unsqueeze(dim=-1)) + \
                           feature01 * (1 - resx.unsqueeze(dim=-1)) * (resy.unsqueeze(dim=-1)) + \
                           feature11 * (resx.unsqueeze(dim=-1)) * (resy.unsqueeze(dim=-1)) + \
                           feature10 * (resx.unsqueeze(dim=-1)) * (1 - resy.unsqueeze(dim=-1))
        output = mask.float().unsqueeze(dim=-1) * feature_bilinear + (1 - mask.float().unsqueeze(dim=-1)) * feature00
        return output
    else:
        raise ValueError("UNKNOWN METHOD FOR MODE")


class VFELayerMinus(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 normalize=None,
                 last_layer=False,
                 attention=False,
                 weight_dims=7,
                 name=''):

        super(VFELayerMinus, self).__init__()
        self.name = 'VFELayerMinus' + name
        self.last_vfe = last_layer
        self.normalize = normalize
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels
        # input batch pointnum feature_num
        self.linear = nn.Linear(in_channels, self.units, bias=True)
        self.weight_linear = nn.Linear(weight_dims, self.units, bias=True)
        # self.linear.weight.requires_grad = False
        # self.linear.bias.requires_grad = False
        if self.normalize:
            # self.normalize['num_features'] = self.units
            self.norm = nn.BatchNorm1d(self.units)  # , **self.normalize
            # self.norm = build_norm_layer(normalize, self.units)[1]

    def forward(self, inputs, idx, idx_used, sizes, mean=None, activate=False, gs=None):
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

        max_feature = x.new_zeros((sizes, x.size(1)))
        idx_used = idx_used.expand_as(x)
        max_feature, fk = torch_scatter.scatter_max(x, idx_used, out=max_feature, dim=0)
        # max_feature = max_feature.cuda()

        if self.last_vfe:
            gather_max_feature = max_feature[idx, :]
            x_concated = torch.cat((x, gather_max_feature), dim=1)
            # return x_concated, max_feature
            return x_concated, max_feature
        else:
            gather_max_feature = max_feature[idx, :]
            x_concated = torch.cat((x, gather_max_feature), dim=1)
            return x_concated, max_feature

class PcPreprocessor3DSlim(nn.Module):
    def __init__(self, lims, grid_meters, scales=[0.5, 1]):
        # todo move to cfg
        super(PcPreprocessor3DSlim, self).__init__()
        self.x_lims = lims[0]
        self.y_lims = lims[1]
        self.z_lims = lims[2]
        self.offset = 0.5
        self.grid_meters = grid_meters

        self.grid_sizes = torch.tensor([int(round((self.x_lims[1] - self.x_lims[0]) / self.grid_meters[0])),
                           int(round((self.y_lims[1] - self.y_lims[0]) / self.grid_meters[1])),
                           int(round((self.z_lims[1] - self.z_lims[0]) / self.grid_meters[2]))])
        self.grid_sizes = self.grid_sizes.cuda().long()
        self.sizes = self.grid_sizes
        self.lims = [self.x_lims, self.y_lims, self.z_lims]

        self.scales = scales
        self.view_name = ["top_view", "front_view"]

    def filter_pc(self, pc):
        mask_x = mask_op(pc[:, 0], self.lims[0][0] + 0.0001, self.lims[0][1] - 0.0001)
        mask_y = mask_op(pc[:, 1], self.lims[1][0] + 0.0001, self.lims[1][1] - 0.0001)
        mask_z = mask_op(pc[:, 2], self.lims[2][0] + 0.0001, self.lims[2][1] - 0.0001)
        mask = mask_x & mask_y & mask_z
        filter_pc = pc[mask]
        return filter_pc

    def forward(self, pc, indicator, keep_mask=None):
        indicator_t = []
        tensor = torch.ones((1,), dtype=torch.long).cuda()
        for i in range(len(indicator) - 1):
            indicator_t.append(tensor.new_full((indicator[i + 1] - indicator[i],), i))
        indicator_t = torch.cat(indicator_t, dim=0)
        # multi-scale perprocess
        # print(indicator)
        # info = {'batch': len(indicator) - 1}
        info = {'batch': len(indicator) - 1}
        for scale in self.scales:
            xidx, xres = quantitizev2(pc[:, 0], self.lims[0][0],
                                              self.lims[0][1], self.sizes[0].float() // scale, with_res=True)
            yidx, yres = quantitizev2(pc[:, 1], self.lims[1][0],
                                              self.lims[1][1], self.sizes[1].float() // scale, with_res=True)
            zidx, zres = quantitizev2(pc[:, 2], self.lims[2][0],
                                              self.lims[2][1], self.sizes[2].float() // scale, with_res=True)
            # print(xidx[:10], self.sizes[0].float() // scale)
            bxyz_indx = torch.stack([indicator_t, xidx, yidx, zidx], dim=-1)
            # idx = xidx * size_y * size_z + yidx
            # print(bxyz_indx.size())
            # exit(0)
            xyz_res = torch.stack([xres, yres, zres], dim=-1)
            # xyz_center = torch.stack([x_center, y_center, z_center], dim=-1)
            info[scale] = {'bxyz_indx': bxyz_indx, 'xyz_res': xyz_res}

        return pc, info