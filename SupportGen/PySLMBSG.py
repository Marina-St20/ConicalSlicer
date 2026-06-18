import trimesh
from pyslm.core import Part
from pyslm.support import BlockSupportGenerator, getOverhangMesh

mesh = trimesh.load_mesh("C:\\Users\\monto\\Downloads\\Xrot-Bone-Remesh.stl")

part = Part("part")
part.setGeometry(mesh)
overhead_angle = 10


support_generator = BlockSupportGenerator()
support_structures = support_generator.identifySupportRegions(part, overhangAngle=overhead_angle)

build = [part]

for i,support in enumerate(support_structures):
    support_mesh = support.geometry()

    support_part = Part(f"support_{i}")
    support_part.setGeometry(support_mesh)

    build.append(support_part)

full = trimesh.util.concatenate([spart.geometry for spart in build])

full.export("C:\\Users\\monto\\Downloads\\supported_model.stl")
