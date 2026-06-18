import numpy as np
from stl import mesh
model = mesh.Mesh.from_file('C:\\Users\\monto\\Downloads\\Xrot-Bone.stl')
supports = mesh.Mesh.from_file('C:\\Users\\monto\\Downloads\\Bone-Sup.stl')

model_z_min = model.z.min()
supports_z_max = supports.z.max()
gap = model_z_min - supports_z_max

if gap > 0:
    supports.vectors[:, :, 2] += gap  # This permanently modifies supports.data

combined = mesh.Mesh(np.concatenate([model.data, supports.data]))
combined.save('C:/Users/monto/Downloads/new3.stl')