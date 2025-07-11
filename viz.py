from skimage import measure
import trimesh
import matplotlib.pyplot as plt
import numpy as np
import torch

def viz_mesh(voxel_pred, voxel_size = 64):
    # Assume voxel_pred shape = [32, 32, 32]
    threshold = 0  # Typical threshold for occupancy
    binary_voxel = (voxel_pred > threshold).detach().cpu().numpy().astype(np.uint8).squeeze(0)
    # print("Min:", binary_voxel.min(), "Max:", binary_voxel.max(), "Unique:", np.unique(binary_voxel))
    verts, faces, normals, values = measure.marching_cubes(binary_voxel, level=threshold)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.show()

def viz_voxel(voxel_data, threshold=0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(np.ndim(voxel_data))
    if np.ndim(voxel_data) == 4:
        if voxel_data.shape[-1] == 4: voxel_data = voxel_data[..., :3]
        if voxel_data.max() > 1: voxel_data  = voxel_data / 255.0 # Normalizing the voxel colors for visualization
        mask = np.any(voxel_data > threshold, axis=-1) # Masking for non-zero voxels with color intensity > 0
        x, y, z = np.indices(voxel_data.shape[:-1])  # Getting the grid coordinates
        ax.scatter(x[mask], y[mask], z[mask], c=voxel_data[mask].reshape(-1, 3), marker='o', s=20)
        ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
    else: ax.voxels(voxel_data, edgecolor='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
# viz_mesh(torch.load("./voxels/02691156_5b86cf0b988987c9fca1f1143bb6bc17.pt", weights_only=False))