from collections import deque
import heapq
import sys
import time
import numpy as np
import trimesh
from itertools import combinations

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
        print(f"{bad}")
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
    return mesh

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

def build_adjacency(mesh):
    if len(mesh.face_adjacency) == 0:
        return None

    adjacency = [[] for _ in range(len(mesh.faces))]
    centroids = mesh.triangles_center
    for a, b in mesh.face_adjacency:
        a = int(a)
        b = int(b)

        face_a = centroids[a]
        face_b = centroids[b]

        dist = abs(np.linalg.norm(face_a-face_b))

        adjacency[a].append([b, dist])
        adjacency[b].append([a, dist])
    return adjacency

def build_vectors(mesh):
    if len(mesh.face_adjacency) == 0:
        return None

    adjacency = [[] for _ in range(len(mesh.faces))]
    normals = mesh.face_normals
    for a, b in mesh.face_adjacency:
        a = int(a)
        b = int(b)

        normal_a = normals[a]
        normal_b = normals[b]

        vector = (normal_a + normal_b) / 20
        dot = np.dot(normal_a, normal_b)
        if -.1 < dot and dot < 0:
            dot = -.1
        elif 0 <= dot and dot < .1:
            dot = .1

        adjacency[a].append([b, vector, dot])
        adjacency[b].append([a, vector, dot])
    return adjacency

def build_map(mesh, adjacency, origins=[0]):
    distances = np.full(len(mesh.faces), np.inf, dtype=float)

    # Use to examine full lists
    # np.savetxt("SharedList.txt", mesh.face_adjacency, "%d", ", ")
    # DOESN'T WORK ON BENCHY 
    # txtarray = np.array(adjacency)
    # txtarray = txtarray.reshape(txtarray.shape[0], -1)
    # np.savetxt("AdjacencyTree.txt",txtarray, "%.2f", ", ")


    visited = np.zeros(len(mesh.faces), dtype=bool)
    queue = []
    for i in origins:
        distances[i] = 0.0
        heapq.heappush(queue, (0.0, int(i)))

    while queue:
        face = heapq.heappop(queue)
        current_distance=face[0]
        face_idx=int(face[1])
        if visited[face_idx]:
            continue
        visited[face_idx] = True

        for neighbour_idx, distance in adjacency[face_idx]:
            new_distance = current_distance + distance
            if not visited[neighbour_idx]:
                distances[neighbour_idx] = new_distance
                heapq.heappush(queue, (new_distance, neighbour_idx))

    # For models with multiple components
    while len(visited[False]) > 0:
        remaining = visited[False]
        loop(queue, adjacency, distances, visited, remaining[0])       

    return distances

def build_weights_d(mesh, adjacency, center=None, origins=[0], adjustment = .04):
    weights = np.full(len(mesh.faces), np.inf, dtype=float)
    if center is None:
        center = mesh.bounding_box.centroid.copy()
        center[2] = 0
    centroids = mesh.triangles_center
    visited = np.zeros(len(mesh.faces), dtype=bool)
    queue = []
    for i in origins:
        weights[i] = 0.0
        heapq.heappush(queue, (0.0, int(i)))

    while queue:
        face = heapq.heappop(queue)
        face_idx=int(face[1])
        if visited[face_idx]:
            continue
        visited[face_idx] = True
        current_weight=face[0]
        centroid = centroids[face_idx].copy()
        centroid = centroid - center
        norm = np.linalg.norm(centroid)
        normals = mesh.face_normals

        for neighbour_idx, vector, dot in adjacency[face_idx]:
            # vector = normals[neighbour_idx]
            direction = np.linalg.norm(centroid + vector)
            dot = .1/np.abs(dot)

            weight = (norm - direction - adjustment) * (dot + .1)
            if weight < 0 or dot < .1:
                weight = 0
            new_weight = current_weight + weight
            if not visited[neighbour_idx]:
                weights[neighbour_idx] = new_weight
                heapq.heappush(queue, (new_weight, neighbour_idx))
    return weights

def build_weights_b(mesh, adjacency, center=None, origins=[0], adjustment = .04):
    weights = np.full(len(mesh.faces), np.inf, dtype=float)
    if center is None:
        center = mesh.bounding_box.centroid.copy()
        center[2] = 0
        
    centroids = mesh.triangles_center
    normals = mesh.face_normals # Extract completely outside the loop to fix memory bottleneck
    
    visited = np.zeros(len(mesh.faces), dtype=bool)
    queue = deque()
    
    for i in origins:
        weights[i] = 0.0
        queue.append((0.0, int(i)))
        visited[int(i)] = True # Mark origins visited immediately

    while queue:
        current_weight, face_idx = queue.popleft()
        
        centroid = centroids[face_idx].copy()
        centroid = centroid - (center + [0, 0, centroid[2] * .5])
        norm = np.linalg.norm(centroid)

        for neighbour_idx, vector, dot in adjacency[face_idx]:
            if visited[neighbour_idx]:
                continue
                
            normal_z = normals[face_idx][2]
            if normal_z > -.3:
                normal_z = 0
            direction = np.linalg.norm(centroid + vector)
            
            dot_val = .1 / -dot if dot != 0 else 0 
            
            norm_delta = norm - direction
            weight = (norm_delta - adjustment) * dot_val * normal_z
            
            if -.1001 < dot < .1001:
                weight = 0
                
            new_weight = current_weight + weight
            
            if new_weight < weights[neighbour_idx]:
                weights[neighbour_idx] = new_weight
                
            queue.append((new_weight, neighbour_idx))
            visited[neighbour_idx] = True
            
    return weights


def find_support_faces(mesh, threshold=1.0, max_count=10, center=None, alg="b"):
    if center is None:
        center = mesh.bounding_box.centroid.copy()
        center[2] = 0
    else:
        center = np.asarray(center, dtype=float)

    _, _, origins = mesh.ray.intersects_location(
        ray_origins=[center], ray_directions=[[0, 0, 1]], multiple_hits=True)
    if len(origins) > 0:
        mask = mesh.face_normals[origins][:, 2]
        origins = np.array(origins)[mask < 0]
        mask = mesh.triangles_center[origins][:, 2]
        origins = np.array(origins)[mask > 0.4]

    length = len(origins)
    mask = np.where(mesh.triangles_center[:,2] < .1)
    if len(mask) > 0: 
        origins = np.concat([mask[0], origins])
    
    i=.2
    while (len(origins) == 0):
        mask = np.where(mesh.triangles_center[:,2] < i)
        if len(mask[0]) > 0: 
            origins = np.concat([mask[0], origins])
        i+=.1
    
    if length != 0 or i == .2:
        length = len(mask[0])

    vectors = build_vectors(mesh)
    if vectors is None:
        return origins

    if alg=="b":
        weights = build_weights_b(mesh, vectors, center, origins=origins)
    else:
        weights = build_weights_d(mesh, vectors, center, origins=origins, adjustment=.6)
    current_max = int(np.argmax(weights))
    max_weight = float(weights[current_max])
    mod = max_weight
    i = 0
    while (mod >= max_weight * threshold and i < max_count) or i <= 1:
        origins = np.append(origins, current_max)
        if alg=="b":
            weights = build_weights_b(mesh, vectors, center, origins=origins)
        else:
            weights = build_weights_d(mesh, vectors, center, origins=origins, adjustment=.6)        
        current_max = weights.argmax()
        mod = float(weights[current_max])
        i += 1
    
    origins = origins[length:]

    return np.unique(origins.astype(int))


def loop(queue, adjacency, distances, visited, start):
     if isinstance(queue, deque):
        return
     queue = [(0.0, int(start))]
     while queue:
        face = heapq.heappop(queue)
        face_idx=face[1]
        current_distance=face[0]
        if visited[face_idx]:
            continue
        visited[face_idx] = True

        for neighbour_idx, distance in adjacency[face_idx]:
            new_distance = current_distance + distance
            if not visited[neighbour_idx] and new_distance < distances[neighbour_idx]:
                distances[neighbour_idx] = new_distance
                heapq.heappush(queue, (new_distance, neighbour_idx))

def main():
    if len(sys.argv) < 2:
        print("Usage: python FindSupportFaces.py <path_to_mesh> [optional: threshold_percentage] [optional: string of center]")
        return

    mesh = load_mesh(sys.argv[1])
    print(f"Loaded mesh from {sys.argv[1]} with {len(mesh.faces)} faces.")
    print(f"Mesh bounding box: {mesh.bounding_box.bounds}")

    colors = mesh.visual.face_colors

    if len(sys.argv) > 2:
        threshold = float(sys.argv[2])
    else:
        threshold = 1

    max_count = 10
    if len(sys.argv) > 3:
        max_count = int(sys.argv[3])

    if len(sys.argv) > 4:
        center = np.array([float(x) for x in sys.argv[4].split(',')])
    else:
        center = mesh.bounding_box.centroid.copy()
        center[2] = 0

    print(f"Using center: {center} and threshold: {threshold}")

    alg="b"

    start = time.perf_counter()
    _, _, origins = mesh.ray.intersects_location(
        ray_origins=[center], ray_directions=[[0,0,1]], multiple_hits=True)
    if len(origins) > 0:
        mask = mesh.face_normals[origins][:,2]
        origins = np.array(origins)[mask < 0]
        mask = mesh.triangles_center[origins][:,2]
        origins = np.array(origins)[mask > 0.4]

    length = len(origins)
    mask = np.where(mesh.triangles_center[:,2] < .1)
    print(f"{len(origins)} {len(mask[0])}")
    if len(mask[0]) > 0: 
        origins = np.concat([mask[0], origins])
    
    i = .2
    while (len(origins) == 0):
        mask = np.where(mesh.triangles_center[:,2] < i)
        if len(mask[0]) > 0: 
            origins = np.concat([mask[0], origins])
        i+=.1

    if length != 0 or i==.2:
        length = len(mask[0])
    
    vectors = build_vectors(mesh)
    if alg=="b":
        weights = build_weights_b(mesh, vectors, center, origins=origins)
    else:
        weights = build_weights_d(mesh, vectors, center, origins=origins, adjustment=.04)
    ordered = weights.argsort()
    current_max = weights.argmax()
    mod = weights[current_max]
    max_weight = mod.copy()

    print(f"Max weight: {mod}, threshold: {max_weight * threshold}")

    # Repeat until the maximum weight is below the threshold percentage of the original maximum weight
    i = 0
    while mod >= max_weight * threshold and i < max_count or i <= 1:
        ordered = weights[weights.argsort()]
        origins = np.append(origins, current_max)
        if alg=="b":
            weights = build_weights_b(mesh, vectors, center, origins=origins)
        else:
            weights = build_weights_d(mesh, vectors, center, origins=origins, adjustment=.04)
        ordered = weights.argsort()
        current_max = weights.argmax()
        mod = weights[current_max]
        i += 1

    weights = weights[ordered]
    mesh.update_faces(ordered)
    print(f"{length}")
    print(f"{origins}")
    origins = origins[length:]
    print(f"{origins}")
    origins = np.array([np.where(ordered == i)[0][0] for i in origins])
    end = time.perf_counter()

    print(f"Origin faces: {len(origins)}")
    print(f"Remaining weight: {mod}")
    if mod == 0:
        mod = 1
    for i in range(len(mesh.faces)):
        weight = weights[i]/mod * 255
        red = weight
        colors[i] = [red, 90, 60, 255]
    for i in origins:
        colors[i] = [255, 255, 255, 255]

    print(f"Time to scan: {end-start}")
    show_regions(mesh, colors=colors)

if __name__ == "__main__":
    main()