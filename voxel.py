import trimesh
import torch
import os
import numpy as np
from shapenetcore import get_random_models
from tqdm import tqdm
from config import voxel_res, voxel_dir, directory, suffix_dir
from viz import viz_voxel, viz_mesh


def save_voxels():
    os.makedirs(voxel_dir, exist_ok=True)
    for path in tqdm(sorted(get_random_models())[:50], desc=f"Progress"):
        if f"{"_".join(path.split("/"))}.pt" not in os.listdir(voxel_dir): create_voxels(path)

def analyze_voxel_sparsity(dataloader):
    total_vox = 0
    total_nonzero = 0
    for vox in tqdm(dataloader, desc="Analyzing voxel sparsity"):
        total_vox += vox.numel()
        total_nonzero += (vox > 0).sum().item()
    ratio = total_nonzero / total_vox
    print(f"🔥 Overall Nonzero Voxel Ratio: {ratio:.6f}")

def create_voxels(path):
    try:
        mesh = trimesh.load(f"{directory}/{path}/{suffix_dir}", force="mesh")
        # Normalize: center & scale to fit inside unit cube
        mesh.apply_translation(-mesh.centroid)  # center it
        bbox_min, bbox_max = mesh.bounds
        scale = 1.0 / np.max(bbox_max - bbox_min).max()
        mesh.apply_scale(scale * 0.98)  # shrink a bit to avoid touching edges
        pitch = mesh.extents.max() / (voxel_res - 1) # Dynamic pitch based on desired voxel resolution
        voxelized = mesh.voxelized(pitch=pitch)
        voxelized = voxelized.fill()
        voxel_matrix = voxelized.matrix.astype(np.float32)
        # viz_voxel(voxel_matrix)
        padded = np.zeros((voxel_res, voxel_res, voxel_res), dtype=np.float32) # Centered padding
        min_shape = np.minimum(padded.shape, voxel_matrix.shape)
        offsets = [(voxel_res - s) // 2 for s in min_shape]
        slices = tuple(slice(offset, offset + s) for offset, s in zip(offsets, min_shape))
        padded[slices] = voxel_matrix[:min_shape[0], :min_shape[1], :min_shape[2]]
        voxel_tensor = torch.tensor(padded).unsqueeze(0)  # [1, D, H, W]
        # viz_mesh(voxel_tensor)
        # analyze_voxel_sparsity(voxel_tensor)
        torch.save(voxel_tensor, f"{voxel_dir}/{"_".join(path.split("/"))}.pt")
    except (IndexError, AttributeError, MemoryError) as e:
        print(f"⚠️ Failed on {path}: {e}") # Missing two files from class: 03337140
        return

def main():
    save_voxels()

if __name__ == "__main__":
    main()