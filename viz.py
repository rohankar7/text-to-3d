from skimage import measure
import trimesh
import matplotlib.pyplot as plt
import numpy as np
import torch

def viz_mesh(voxel_pred, voxel_size = 32):
    # Assume voxel_pred shape = [32, 32, 32]
    threshold = 0  # Typical threshold for occupancy
    binary_voxel = (voxel_pred > threshold).detach().cpu().numpy().astype(np.uint8).squeeze(0)
    # print("Min:", binary_voxel.min(), "Max:", binary_voxel.max(), "Unique:", np.unique(binary_voxel))
    # Alpha Transparency view
    # verts, faces, normals, values = measure.marching_cubes(binary_voxel, level=threshold)
    # mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(binary_voxel)
    mesh.show()

def viz_voxel(voxel_data, threshold=0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_data > threshold, edgecolor='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()