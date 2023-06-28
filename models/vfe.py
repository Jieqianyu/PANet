import torch
import torch.nn as nn
import torch_scatter


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


class DynamicVFELayer(nn.Module):
    """Replace the Voxel Feature Encoder layer in VFE layers.
    This layer has the same utility as VFELayer above
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout=0.0,
                 ):
        super(DynamicVFELayer, self).__init__()
        
        self.norm = nn.BatchNorm1d(out_channels)
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.act = nn.ReLU()
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, inputs):
        """Forward function.
        Args:
            inputs (torch.Tensor): Voxels features of shape (M, C).
                M is the number of points, C is the number of channels of point features.
        Returns:
            torch.Tensor: point features in shape (M, C).
        """
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        if self.dropout is not None:
            inputs = self.dropout(inputs)
        x = self.linear(inputs)
        x = self.norm(x)
        pointwise = self.act(x)
        return pointwise


def build_mlp(in_channel, hidden_dims, is_head=False, bias=False, dropout=0):
    layer_list = []
    last_channel = in_channel
    for i, c in enumerate(hidden_dims):
        act_layer = nn.ReLU()

        norm_layer = nn.BatchNorm1d(c)
        if i == len(hidden_dims) - 1 and is_head:
            layer_list.append(nn.Linear(last_channel, c, bias=True),)
        else:
            sq = [
                nn.Linear(last_channel, c, bias=bias),
                norm_layer,
                act_layer,
            ]
            if dropout > 0:
                sq.append(nn.Dropout(dropout))
            layer_list.append(
                nn.Sequential(
                    *sq
                )
            )

        last_channel = c
    mlp = nn.Sequential(*layer_list)
    return mlp


class CFE(nn.Module):
    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_rel_mlp=True,
                 rel_mlp_hidden_dims=[16,],
                 rel_mlp_in_channel=3,
                 rel_dist_scaler=1.0,
                 with_shortcut=True,
                 xyz_normalizer=[1.0, 1.0, 1.0],
                 dropout=0.0,
                 ):
        super().__init__()
        # overwrite
        self.rel_dist_scaler = rel_dist_scaler
        self.with_shortcut = with_shortcut
        self._with_rel_mlp = with_rel_mlp
        self.xyz_normalizer = xyz_normalizer
        self.in_channels = in_channels
        if with_rel_mlp:
            rel_mlp_hidden_dims.append(in_channels) # not self.in_channels
            self.rel_mlp = build_mlp(rel_mlp_in_channel, rel_mlp_hidden_dims)

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2

            vfe_layers.append(
                DynamicVFELayer(
                    in_filters,
                    out_filters,
                    dropout=dropout,
                )
            )
        self.vfe_layers = nn.ModuleList(vfe_layers)
        self.vfe_fusion = nn.Linear(len(vfe_layers)*feat_channels[-1], feat_channels[-1])
        self.num_vfe = len(vfe_layers)

    def forward(self,
                coors,
                features,
                points,
                f_cluster=None,
        ):
        xyz_normalizer = torch.tensor(self.xyz_normalizer, device=features.device, dtype=features.dtype)
        features_ls = [torch.cat([points / xyz_normalizer[None, :], features], dim=1)]
        
        if self.with_shortcut:
            shortcut = features
        if f_cluster is None:
            # Find distance of x, y, and z from cluster center
            _, unq_inv = torch.unique(coors, return_inverse=True, dim=0)
            voxel_mean = scatter(points, coors, 'mean')
            points_mean = voxel_mean[unq_inv]
            
            f_cluster = (points - points_mean[:, :3]) / self.rel_dist_scaler
        else:
            f_cluster = f_cluster / self.rel_dist_scaler

        if self._with_rel_mlp:
            features_ls[0] = features_ls[0] * self.rel_mlp(f_cluster)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)

        voxel_feats_list = []
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)

            voxel_coors, unq_inv = torch.unique(coors, return_inverse=True, dim=0)
            voxel_feats = scatter(point_feats, unq_inv, 'mean')
            voxel_feats_list.append(voxel_feats)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = voxel_feats[unq_inv]
                features = torch.cat([point_feats, feat_per_point], dim=1)

        voxel_feats = self.vfe_fusion(torch.cat(voxel_feats_list, dim=1))

        if self.with_shortcut and point_feats.shape == shortcut.shape:
            point_feats = point_feats + shortcut
        return point_feats, voxel_feats