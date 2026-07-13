import time

import numpy as np
import trimesh
import FindSupportFaces

def vertical_scan():
    mesh = FindSupportFaces.load_mesh('../Tentacle_Remesh.stl')
    mesh_filter = FindSupportFaces.load_mesh('../Filter_Cone_30.stl')

    extents = mesh.extents
    max_scale = max(extents[0] * np.reciprocal(mesh_filter.extents[0]),
                      extents[1] * np.reciprocal(mesh_filter.extents[1]),
                      extents[2] * np.reciprocal(mesh_filter.extents[2]) * 2)
    max_scale *= 1.01
    scale = np.diag([max_scale,max_scale,max_scale,1])
    center = mesh.center_mass
    center[2] = 0
    scale[:, 3] = np.append(center-mesh_filter.center_mass,1)
    scale[:, 3] += [0,0,-mesh_filter.extents[2]*scale[2,2],0]
    mesh_filter.apply_transform(scale)
    step = .2
    step_num = np.floor(mesh.extents[2] * 2 / step).astype(int)
    manager = trimesh.collision.CollisionManager()
    manager.add_object('mesh', mesh)

    for i in range(step_num):
        transform = trimesh.transformations.translation_matrix([0, 0, i * step])
        
        is_collide, contacts = manager.in_collision_single(
            mesh_filter, 
            transform=transform, 
            return_names=False, 
            return_data=True
        )
        
        if is_collide:
            face_ids = [data._inds['mesh'] for data in contacts]
            unique = np.unique(face_ids)
        else:
            unique = np.array([], dtype=int)

        print(f"{i+1}: Found {len(unique)} intersecting faces.")
        
        # To check end-result alignment
        # if (i == step_num - 1):
        #     mesh_filter.apply_transform(transform)
        #     scene = trimesh.Scene()
        #     scene.add_geometry(mesh)
        #     scene.add_geometry(mesh_filter)
        #     scene.show()

def main():
    start = time.perf_counter()
    vertical_scan()
    end = time.perf_counter()
    print(f"{end-start} seconds")

if __name__ == "__main__":
    main()