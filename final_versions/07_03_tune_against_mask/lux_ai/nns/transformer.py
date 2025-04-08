import torch
import math
import einops
import torch.nn as nn
import numpy as np
from typing import Callable

class ScaledDotProductAttend(nn.ModuleList):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        out = nn.functional.scaled_dot_product_attention(
            q, k, v
        )
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        return out

class AttentionBase(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.attend = ScaledDotProductAttend()
        self.to_out = (
            nn.Linear(self.inner_dim, dim)
            if project_out
            else nn.Identity()
        )

        self.k_norm = (
            nn.LayerNorm(self.inner_dim, elementwise_affine=False) if False else nn.Identity()
        )
        self.q_norm = (
            nn.LayerNorm(self.inner_dim, elementwise_affine=False) if False else nn.Identity()
        )

    def forward_after_projection(
        self,
        qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        q, k, v = qkv

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k, v = map(
            lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v)
        )
        out = self.attend(q, k, v)
        return self.to_out(out)


class SelfAttention(AttentionBase):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        **kwargs
    ):
        super().__init__(
            dim,
            heads,
            dim_head
        )
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)

    def forward(
        self,
        q: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.to_qkv(q).chunk(3, dim=-1)
        return self.forward_after_projection(tuple(qkv))


class CrossAttention(AttentionBase):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        kv_dim: int,
    ):
        super().__init__(
            dim,
            heads,
            dim_head,
        )
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(kv_dim, self.inner_dim * 2, bias=False)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
    ) -> torch.Tensor:
        q = self.to_q(q)
        kv = self.to_kv(kv).chunk(2, dim=-1)
        k, v = kv
        return self.forward_after_projection((q, k, v))



class PreNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        module: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.module = module

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.norm(x)
        return self.module(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        first_layers = [
            nn.Linear(dim, hidden_dim),
            nn.ReLU()
        ]
        self.first_layers = nn.Sequential(*first_layers)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.first_layers(x))

class TransformerBase(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        *,
        self_attention: bool = True,
        kv_dim: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        attention_class = SelfAttention if self_attention else CrossAttention
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            attention_class(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                kv_dim=kv_dim
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(
                                dim,
                                mlp_dim,
                            ),
                        ),
                    ]
                )
            )

class SelfAttentionTransformer(TransformerBase):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
    ):
        super().__init__(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            self_attention = True,
        )

    def forward(
        self,
        q: torch.Tensor,
    ) -> torch.Tensor:
        assert self.dim == q.shape[2]
        for attention_layer, feed_forward_layer in self.layers:
            q = (
                attention_layer(
                    q,
                )
                + q
            )
            q = feed_forward_layer(q) + q
        return q

class CrossAttentionTransformer(TransformerBase):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        kv_dim: int,
    ):
        super().__init__(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            self_attention=False,
            kv_dim=kv_dim,
        )

        self.kv_norm = nn.LayerNorm(kv_dim)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
    ) -> torch.Tensor:
        kv = self.kv_norm(kv)

        for attention_layer, feed_forward_layer in self.layers:
            q = (
                attention_layer(
                    q,
                    kv=kv,
                )
                + q
            )
            q = feed_forward_layer(q) + q
        return q






class PosEmbed2d(nn.Module):
    def __init__(self, h, w, dim):
        super().__init__()

        self.h = h
        self.w = w
        self.dim = dim

        pe = torch.zeros((h, w, dim))

        min_period = 1
        max_period = 30

        assert dim % 4 == 0
        quater = dim // 4

        for x in range(h):
            for y in range(w):
                for i in range(quater):
                    p = (i - 1) / quater
                    period = np.exp((1 - p) * np.log(min_period) + p * np.log(max_period))
                    pe[x, y, i] = np.sin(x / period * 2 * np.pi)
                    pe[x, y, i + quater] = np.cos(x / period * 2 * np.pi)
                    pe[x, y, i + 2 * quater] = np.sin(y / period * 2 * np.pi)
                    pe[x, y, i + 3 * quater] = np.cos(y / period * 2 * np.pi)

        self.register_buffer('pe', pe)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        assert x.shape[1] == self.pe.shape[0]
        assert x.shape[2] == self.pe.shape[1]
        assert x.shape[3] == self.pe.shape[2]
        return x + self.mlp(self.pe.detach())



class TransformerWithPosEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
    ):
        super().__init__()
        self.pos_embed = PosEmbed2d(24, 24, dim)
        self.transformer = SelfAttentionTransformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.pos_embed(x)
        bs, w, h, c = x.shape
        res = self.transformer(x.view(bs, w * h, c))
        return res.view(bs, w, h, c)
