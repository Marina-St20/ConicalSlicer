from itertools import combinations
import time
import numpy as np
import trimesh

def load_mesh(path):
    mesh = trimesh.load_mesh(path, process=True)
    if mesh.is_empty:
        raise ValueError(f"Loaded mesh is empty: {path}")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh.remove_infinite_values()
    mesh.update_faces(mesh.unique_faces())
    mesh.merge_vertices()
    bad = trimesh.repair.broken_faces(mesh)
    if (len(bad) > 0):
        faces = mesh.faces[bad]
        vertices = np.unique(faces)
        pairs = np.array(list(combinations(vertices, 2)))
        positions = mesh.vertices[pairs]

        v1 = positions[:, 0, :]
        v2 = positions[:, 1, :]

        distances = np.linalg.norm(v1 - v2, axis=1)
        distance_threshold = .1
        mask = distances < distance_threshold

        vertices = pairs[mask]
        for v1, v2 in vertices:
            mesh._data['vertices'][v2] = mesh.vertices[v1]
        mesh._cache.clear()
        mesh.merge_vertices(True, True)
        trimesh.repair.fill_holes(mesh)
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, validate=True)
    print(f"{mesh.bounding_box.extents}")
    return mesh

def vertical_scan():
    mesh = load_mesh('../Earless_Remesh.stl')
    mesh_filter = load_mesh('../Filter_Cone_30.stl')

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
    step = .1
    step_num = np.floor(mesh.extents[2] * 2 / step).astype(int)
    manager = trimesh.collision.CollisionManager()
    manager.add_object('mesh', mesh)
    component_bases = []
    adjacency = build_adjacency(mesh)

    for i in range(step_num):
        for j in range(2):
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
                    if stable or j == 1:
                        face[1][0] = True
                if not stable and j == 1:
                    component_bases.append(u)
        if i % (step_num // 10) == 0:
            percent = np.floor(i / step_num * 100) + 1
            if percent != 0:
                print(f"{percent}%")
        
        # To check end-result alignment
        # if (i == step_num - 1):
        #     mesh_filter.apply_transform(transform)
        #     scene = trimesh.Scene()
        #     scene.add_geometry(mesh)
        #     scene.add_geometry(mesh_filter)
        #     scene.show()

    # Filtering
    origins = np.array(component_bases, int)
    normals = mesh.face_normals[origins]
    mask = normals[:,2] < -.5
    origins = origins[mask]
    pos = mesh.triangles_center[origins]
    mask = pos[:,2] > 1
    origins = origins[mask]
    # Pointed towards center
    # normals = mesh.face_normals[origins]
    # pos = mesh.triangles_center[origins]
    # dots = []
    # for i in range(len(pos)):
    #     loc = pos[i] - center 
    #     loc[2] = 0
    #     dots.append(np.dot(normals[i] / np.linalg.norm(normals[i]), loc / np.linalg.norm(loc)))
    # print(f"{dots}")
    # mask = np.where(np.array(dots) <= 0.2)
    # origins = origins[mask]
    # Grouping
    pos = mesh.triangles_center
    groups = np.array(list(combinations(origins, 2)))
    distances = [[] for _ in range(len(pos))]
    roots = np.array([], int)
    dist_threshold = 5
    if (len(origins) > 1):
        for a, b in groups:
            dist = np.linalg.norm(pos[a] - pos[b])
            distances[a].append([b, dist])
            distances[b].append([a, dist])
        groups = [[] for _ in range(len(pos))]
        for i in range(len(distances)):
            pairs = np.array(distances[i])
            if (len(pairs) > 0):
                mask = []
                for j in range(len(pairs)):
                    pair = pairs[j]
                    if (np.abs(pair[1]) < dist_threshold and 
                        np.linalg.norm(pos[i] - center) - np.linalg.norm(pos[int(pair[0])] - center) < 0):
                        mask.append(j)
                pairs = pairs[mask]
                groups[i] = [i,pairs[:,0].astype(int)]
        filtered = list(filter(None, groups))
        roots = np.array([chain[0] for chain in filtered], dtype=int)
        children = np.concatenate([chain[1] for chain in filtered], dtype=int)
        origins = np.setdiff1d(roots, children, assume_unique=True)

                
    
    show_regions(mesh, origins)
    print(f"{len(origins)} support points found.")
    return np.array(origins, int)

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
    _mesh = mesh.copy()
    _mesh.unmerge_vertices()
    face_colors = np.tile([0.5, 0.5, 0.5, .1], (len(mesh.faces), 1))
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
    print(f"{end-start} seconds")
    return origins

if __name__ == "__main__":
    main()