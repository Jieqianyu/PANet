"""
Various positional encodings for the transformer.
"""
import numpy as np
import torch
from torch import nn


class PositionEmbeddingCoordsFourier(nn.Module):
    def __init__(
        self,
        d_pos=None,
        d_in=3,
        gauss_scale=1.0,
    ):
        super().__init__()
        assert d_pos is not None
        assert d_pos % 2 == 0
        # define a gaussian matrix input_ch -> output_ch
        B = torch.empty((d_in, d_pos // 2)).normal_()
        B *= gauss_scale
        self.register_buffer("gauss_B", B)
        self.d_pos = d_pos


    def get_fourier_embeddings(self, xyz, num_channels=None):
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        xyz *= 2 * np.pi
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(
            bsize, npoints, d_out
        )
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x npoints d_pos x  embedding
        final_embeds = torch.cat(final_embeds, dim=2)
        return final_embeds

    def forward(self, xyz, num_channels=None):
        assert isinstance(xyz, torch.Tensor)
        assert (xyz.ndim==2 or xyz.ndim == 3 or xyz.ndim==4)

        if xyz.ndim == 2:
            xyz = xyz.unsqueeze(0)
            with torch.no_grad():
                return self.get_fourier_embeddings(xyz, num_channels)[0]

        if xyz.ndim == 4:
            b, x, y, _ = xyz.shape
            xyz = xyz.flatten(0, 1)
            with torch.no_grad():
                return self.get_fourier_embeddings(xyz, num_channels).reshape(b, x, y, -1)

        with torch.no_grad():
            return self.get_fourier_embeddings(xyz, num_channels)


if __name__ == '__main__':
    pe = PositionEmbeddingCoordsFourier(d_pos=64)
    print(pe(torch.rand(5, 3)).shape)
