"""Unet
====================
A DiT‑style 3‑D UNet for latent‑diffusion on voxel latents (256 x 32 x 32 x 32)
with optional CLIP text cross‑attention.

Highlights
----------
• Sinusoidal timestep embeddings (1024-dim)
• GroupNorm everywhere (batch-size agnostic)
• Down / Mid / Up blocks each with self + cross-attention
• Residual path keeps channels constant; channel doubling when downsampling
• Built to drop into a Lightning or vanilla PyTorch training loop.

Example
-------
```
model = UNet3D(latent_ch=128,
               base_ch=128,
               text_ctx=77,
               text_dim=768,
               cross_attn=True).cuda()

latent      = torch.randn(4, 128, 8, 8, 8).cuda()
text_embed  = torch.randn(4, 77, 768).cuda()
 timesteps  = torch.randint(0, 1000, (4,), device="cuda")
 
out = model(latent, t=timesteps, text_emb=text_embed)  # same shape as latent
```
"""
from __future__ import annotations

from math import pi
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_timestep_embedding(timesteps: torch.Tensor, dim: int = 1024) -> torch.Tensor:
    """Sinusoidal position embeddings (same as stable‑diff)."""
    half = dim // 2
    freq = torch.exp(
        -torch.arange(half, dtype=torch.float32, device=timesteps.device) * (pi / half)
    )
    args = timesteps.float()[:, None] * freq[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # zero‑pad if dim odd
        emb = F.pad(emb, (0, 1))
    return emb  # shape (B, dim)

class SelfCrossAttn(nn.Module):
    def __init__(self, ch: int, heads: int = 4, text_dim: int = 768, cross: bool = True):
        super().__init__()
        self.q = nn.Conv1d(ch, ch, 1, bias=False)
        self.k = nn.Conv1d(ch, ch, 1, bias=False)
        self.v = nn.Conv1d(ch, ch, 1, bias=False)
        self.heads = heads
        self.scale = (ch // heads) ** -0.5
        self.cross = cross
        if cross:
            self.k_text = nn.Linear(text_dim, ch, bias=False)
            self.v_text = nn.Linear(text_dim, ch, bias=False)
        self.proj = nn.Conv1d(ch, ch, 1)

    def forward(self, x: torch.Tensor, text: Optional[torch.Tensor] = None):
        # flatten spatial dims
        b, c, d, h, w = x.shape
        x_flat = x.view(b, c, -1)  # (B,C,N)

        q = self.q(x_flat)
        k = self.k(x_flat)
        v = self.v(x_flat)

        if self.cross and text is not None:
            k_text = self.k_text(text).transpose(1, 2)  # (B,C,T)
            v_text = self.v_text(text).transpose(1, 2)
            k = torch.cat([k, k_text], dim=-1)
            v = torch.cat([v, v_text], dim=-1)

        # reshape heads
        q = q.view(b, self.heads, -1, q.shape[-1] // self.heads)
        k = k.view(b, self.heads, -1, k.shape[-1] // self.heads)
        v = v.view(b, self.heads, -1, v.shape[-1] // self.heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, Nq, Nk)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v  # (B, H, Nq, C/H)
        out = out.transpose(2, 3).contiguous().view(b, c, -1)
        out = self.proj(out)
        out = out.view(b, c, d, h, w)
        return out + x  # residual

class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k = 3, s = 1, p = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, k, s, p, bias=False)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor):
        return self.act(self.norm(self.conv(x)))

class ResBlock3D(nn.Module):
    def __init__(self, ch: int, emb_dim: int):
        super().__init__()
        self.block1 = ConvGNAct(ch, ch)
        self.block2 = ConvGNAct(ch, ch)
        self.emb_proj = nn.Linear(emb_dim, ch)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        h = self.block1(x)
        h = h + self.emb_proj(emb).view(emb.size(0), -1, 1, 1, 1)
        h = self.block2(h)
        return x + h  # residual

class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, text_dim: int, cross: bool):
        super().__init__()
        self.res1 = ResBlock3D(in_ch, emb_dim)
        self.attn1 = SelfCrossAttn(in_ch, cross=cross, text_dim=text_dim)
        self.down = ConvGNAct(in_ch, out_ch, k=4, s=2, p=1)  # stride‑2 conv

    def forward(self, x, emb, text):
        x = self.res1(x, emb)
        x = self.attn1(x, text)
        x_down = self.down(x)
        return x_down, x  # return skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, emb_dim: int, text_dim: int, cross: bool):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 4, 2, 1)  # nearest neighbour upscale
        self.res1 = ResBlock3D(out_ch + skip_ch, emb_dim)
        self.attn1 = SelfCrossAttn(out_ch + skip_ch, cross=cross, text_dim=text_dim)

    def forward(self, x, skip, emb, text):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, emb)
        x = self.attn1(x, text)
        return x


# ------------------------------------------------------------
#  UNet‑3D main module
# ------------------------------------------------------------

class UNet3D(nn.Module):
    def __init__(
        self,
        latent_ch: int = 128,
        base_ch: int = 128,
        text_dim: int = 768,
        text_ctx: int = 77,
        cross_attn: bool = True,
        time_emb_dim: int = 1024,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.in_conv = ConvGNAct(latent_ch, base_ch)

        # Downsample path
        self.down1 = DownBlock(base_ch, base_ch * 2, time_emb_dim, text_dim, cross_attn)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4, time_emb_dim, text_dim, cross_attn)
        self.down3 = DownBlock(base_ch * 4, base_ch * 8, time_emb_dim, text_dim, cross_attn)

        # Mid (bottleneck)
        self.mid_res = ResBlock3D(base_ch * 8, time_emb_dim)
        self.mid_attn = SelfCrossAttn(base_ch * 8, cross=cross_attn, text_dim=text_dim)

        # Upsample path
        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4, time_emb_dim, text_dim, cross_attn)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2, time_emb_dim, text_dim, cross_attn)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch, time_emb_dim, text_dim, cross_attn)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv3d(base_ch, latent_ch, 3, 1, 1)

    # --------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,  # (B, latent_ch, 8, 8, 8)
        t: torch.Tensor,  # (B,)
        text_emb: Optional[torch.Tensor] = None,  # (B, text_ctx, text_dim)
    ) -> torch.Tensor:
        # timestep embedding
        t_emb = self.time_mlp(get_timestep_embedding(t, self.time_mlp[0].in_features))

        h = self.in_conv(x)

        h, skip1 = self.down1(h, t_emb, text_emb)
        h, skip2 = self.down2(h, t_emb, text_emb)
        h, skip3 = self.down3(h, t_emb, text_emb)

        h = self.mid_res(h, t_emb)
        h = self.mid_attn(h, text_emb)

        h = self.up3(h, skip3, t_emb, text_emb)
        h = self.up2(h, skip2, t_emb, text_emb)
        h = self.up1(h, skip1, t_emb, text_emb)

        h = self.out_conv(self.out_norm(F.silu(h)))
        return h  # same shape as input latent