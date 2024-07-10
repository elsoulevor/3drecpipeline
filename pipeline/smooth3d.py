import torch
import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.ops import laplacian_smoothing

# Load the STL file
def load_stl(file_path):
    mesh_data = mesh.Mesh.from_file(file_path)
    vertices = torch.tensor(mesh_data.vectors.reshape(-1, 3), dtype=torch.float32)
    faces = torch.arange(vertices.shape[0]).view(-1, 3)
    return Meshes(verts=[vertices], faces=[faces])

# Save the smoothed mesh to an STL file
def save_stl(mesh, file_path):
    vertices = mesh.verts_list()[0].cpu().numpy().reshape(-1, 3, 3)
    smoothed_mesh = mesh.Mesh(np.zeros(vertices.shape[0], dtype=mesh.Mesh.dtype))
    smoothed_mesh.vectors = vertices
    smoothed_mesh.save(file_path)

# Visualize the mesh
def plot_mesh(mesh):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    verts = mesh.verts_list()[0].cpu().numpy()
    faces = mesh.faces_list()[0].cpu().numpy()
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]
    ax.plot_trisurf(x, y, faces, z, linewidth=0.2, antialiased=True)
    plt.show()

# Smooth the mesh using Laplacian smoothing
def smooth_mesh(mesh, iterations=10, lambda_val=0.5):
    for _ in range(iterations):
        mesh = laplacian_smoothing(mesh, lambd=lambda_val)
    return mesh

# Load a mesh from STL
mesh = load_stl("/home/vinhsp/models/srd-depth/test/car4.stl")

# Plot the original mesh
print("Original Mesh")
plot_mesh(mesh)

# Smooth the mesh
smoothed_mesh = smooth_mesh(mesh, iterations=10, lambda_val=0.5)

# Plot the smoothed mesh
print("Smoothed Mesh")
plot_mesh(smoothed_mesh)

# Save the smoothed mesh to STL
save_stl(smoothed_mesh, "path_to_your_smoothed_model.stl")