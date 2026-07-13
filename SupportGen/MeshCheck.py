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
    component_bases = []
    adjacency = build_adjacency(mesh)

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
        for u in unique:
            faces = adjacency[u]
            stable = any(face[1][0] for face in faces)
            for face in faces:
                face[1][0] = True
            if not stable:
                component_bases.append(u)

        print(f"{i+1}: Found {len(unique)} intersecting faces.")
        
        # To check end-result alignment
        # if (i == step_num - 1):
        #     mesh_filter.apply_transform(transform)
        #     scene = trimesh.Scene()
        #     scene.add_geometry(mesh)
        #     scene.add_geometry(mesh_filter)
        #     scene.show()
    origins = np.array(component_bases, int)
    show_regions(mesh, origins)

    return np.array(component_bases, int)

def build_adjacency(mesh):
    if len(mesh.face_adjacency) == 0:
        return None

    adjacency = [[] for _ in range(len(mesh.faces))]
    for a, b in mesh.face_adjacency:
        a = int(a)
        b = int(b)
        c = [False]

        adjacency[a].append([b, c])
        adjacency[b].append([a, c])
    return adjacency

def show_regions(mesh, face_indices=None, color=[1, 0, 0, 1], colors=None):
    # Visualize the mesh with specified faces highlighted in a different color.
    _mesh = mesh.copy()
    _mesh.unmerge_vertices()
    face_colors = np.tile([0.5, 0.5, 0.5, .1], (len(mesh.faces), 1))  # Default color for all faces
    if face_indices is None:
        face_indices = np.arange(len(mesh.faces))
    if colors is not None:
        face_colors[face_indices] = colors[face_indices]/255
    else:
        face_colors[face_indices] = color
    _mesh.visual.face_colors = face_colors
    _mesh.show()

def main():
    start = time.perf_counter()
    origins = vertical_scan()
    end = time.perf_counter()
    print(f"{origins}")
    print(f"{len(origins)}")
    print(f"{end-start} seconds")

if __name__ == "__main__":
    main()