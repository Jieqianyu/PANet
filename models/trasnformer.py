import math

import torch
from einops import rearrange
from torch import einsum, nn
from torch.nn import functional as F


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# transformer layer
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm_CA(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, y, **kwargs):
        return self.fn(self.norm(x), self.norm(y), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, out_attention=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.out_attention = out_attention

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        if self.out_attention:
            return self.to_out(out), attn
        else:
            return self.to_out(out)


class Cross_attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, out_attention=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.out_attention = out_attention

        self.attend = nn.Softmax(dim=-1)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, y):
        b, n, m, _, h = *y.shape, self.heads
        q = self.to_q(x)
        kv = self.to_kv(y).chunk(2, dim=-1)
        q = rearrange(q, "b n (h d) -> (b n) h 1 d", h=h)
        k, v = map(lambda t: rearrange(t, "b n m (h d) -> (b n) h m d", h=h), kv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "(b n) h 1 d -> b n (h d)", b=b)

        if self.out_attention:
            return self.to_out(out), rearrange(attn, "(b n) h i j -> b n h (i j)", b=b)
        else:
            return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth=1,
        heads=1,
        dim_head=64,
        dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth

        for _ in range(depth):
            self.layers.append(
                PreNorm_CA(
                    dim,
                    Cross_attention(
                        dim,
                        heads=heads,
                        dim_head=dim_head,
                        dropout=dropout,
                        out_attention=False,
                    ),
                )
            )

    def forward(
        self, x, pos_embedding=None, center_pos=None, y=None, neighbor_pos=None
    ):
        if center_pos is not None and pos_embedding is not None:
            center_pos_embedding = pos_embedding(center_pos)
        if neighbor_pos is not None and pos_embedding is not None:
            neighbor_pos_embedding = pos_embedding(neighbor_pos)
        for _, cross_attn in enumerate(self.layers):
            if pos_embedding is not None:
                x_att = cross_attn(
                    x + center_pos_embedding, y + neighbor_pos_embedding
                )
                x = x_att + x
            else:
                x_att = cross_attn(x, y)
                x = x_att + x

        out_dict = {"ct_feat": x}

        return out_dict