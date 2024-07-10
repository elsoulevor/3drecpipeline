import numpy as np
from stl import mesh
import PIL.Image as pil


input_image = pil.open('/home/vinhsp/3drecpipeline/SRD-Depth/test/test_disp3.jpeg')
img = np.array(input_image)
(x,y) = img.shape

vertices = np.zeros((x*y, 3))

for i in range (x):
    for j in range (y):
        vertices[(i*y)+j] = [i, j, img[i][j]]


faces = []
for i in range (x-1):
    for j in range (y-1):
            faces.append([j + i*y, j+1 + i*y, j+(i+1)*y])
            faces.append([j+1 + i*y, j + (i+1)*y, j+1 +(i+1)*y])
faces = np.array(faces)

# Create the mesh
cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        cube.vectors[i][j] = vertices[f[j],:]

# Write the mesh to file "cube.stl"
cube.save('/home/vinhsp/3drecpipeline/SRD-Depth/test/test_disp3.stl')