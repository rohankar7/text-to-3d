"""latent_interpolation.py

Quick script to visualise latent‑space interpolations for the voxel VAE
(checkpoint from epoch 156).

You supply:
    • --ckpt        : path to VAE checkpoint (best_model.pth)
    • --sample_a    : path to first voxel .pt file (shape 1×64×64×64)
    • --sample_b    : path to second voxel .pt file
    • --steps       : how many interpolation frames (default 5)
    • --out_dir     : folder to dump .npy grids and slice PNGs

Run:
    python latent_interpolation.py \
        --ckpt checkpoints/vae/best_model.pth \
        --sample_a data/voxel/airplane_001.pt \
        --sample_b data/voxel/chair_014.pt \
        --steps 7 --out_dir interp_frames

Makes:
    interp_frames/
        alpha_0.00.npy / .png
        alpha_0.17.npy / .png  ... etc.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3‑D voxels)
from vae import VAE  # assumes voxel_vae.py is in PYTHONPATH


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def load_voxel(path: Path) -> torch.Tensor:
    """Load 1×64×64×64 tensor on CPU, values in {0,1}."""
    vox = torch.load(path, map_location="cpu")
    if vox.ndim == 4:
        vox = vox.unsqueeze(0)
    assert vox.shape == (1, 64, 64, 64), f"bad shape {vox.shape}"
    return vox.float()


def viz_slice(voxel: torch.Tensor, save_to: Path, axis: int = 2):
    """Save a mid‑slice PNG for quick eyeballing."""
    vox_np = voxel.squeeze().cpu().numpy()
    mid = vox_np.shape[axis] // 2
    if axis == 0:
        slice_ = vox_np[mid, :, :]
    elif axis == 1:
        slice_ = vox_np[:, mid, :]
    else:
        slice_ = vox_np[:, :, mid]

    plt.figure(figsize=(4, 4))
    plt.imshow(slice_, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load VAE ----
    ckpt = torch.load(cfg.ckpt, map_location=device)
    vae = VAE(latent_dim=128 * 8 * 8 * 8)  # latent_dim arg ignored because we load state_dict
    vae.load_state_dict(ckpt["vae"] if "vae" in ckpt else ckpt)
    vae.eval().to(device)

    # ---- Load two voxel samples ----
    vox_a = load_voxel(cfg.sample_a).to(device)
    vox_b = load_voxel(cfg.sample_b).to(device)

    with torch.no_grad():
        mu_a, _ = vae.encoder_conv(vox_a).view(1, -1), None  # flatten latent grid
        mu_b, _ = vae.encoder_conv(vox_b).view(1, -1), None

    alphas = torch.linspace(0, 1, steps=cfg.steps)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for alpha in alphas:
        z = (1 - alpha) * mu_a + alpha * mu_b
        z = z.view(1, 128, 8, 8, 8).to(device)
        with torch.no_grad():
            recon = vae.decoder_conv(z).cpu()
        alpha_str = f"alpha_{alpha:.2f}"
        np.save(out_dir / f"{alpha_str}.npy", recon.squeeze().numpy())
        viz_slice(recon, out_dir / f"{alpha_str}.png")
        print(f"saved {alpha_str}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--sample_a", type=str, required=True)
    p.add_argument("--sample_b", type=str, required=True)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--out_dir", type=str, default="interp_frames")
    main(p.parse_args())