import heapq
import sys
import time
import numpy as np
import trimesh
from vispy.plot import Fig

def load_mesh(path):
    mesh = trimesh.load_mesh(path)
    if mesh.is_empty:
        raise ValueError(f"Loaded mesh is empty: {path}")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    try:
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
    except Exception:
        pass
    mesh.process(validate=True)
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

        adjacency[a].append([b, vector])
        adjacency[b].append([a, vector])
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

def build_weights(mesh, adjacency, center=None, origins=[0], adjustment = .4):
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

        for neighbour_idx, vector in adjacency[face_idx]:
            vector = normals[neighbour_idx]
            direction = np.linalg.norm(centroid + vector)

            weight = norm - direction - adjustment
            if weight < 0:
                weight = 0
            new_weight = current_weight + weight
            if not visited[neighbour_idx]:
                weights[neighbour_idx] = new_weight
                heapq.heappush(queue, (new_weight, neighbour_idx))
    return weights


def find_support_faces(mesh, threshold=1.0, max_count=10, center=None):
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
        origins = np.array(origins)[mask > 0]
    if len(origins) == 0:
        origins = mesh.nearest.on_surface([center])[2]
    if len(origins) == 0:
        com = mesh.bounding_box.centroid.copy()
        com[2] = 0
        _, _, origins = mesh.ray.intersects_location(
            ray_origins=[com], ray_directions=[[0, 0, 1]], multiple_hits=True)
    if len(origins) == 0:
        return origins

    vectors = build_vectors(mesh)
    if vectors is None:
        return origins

    weights = build_weights(mesh, vectors, center, origins=origins, adjustment=.8)
    current_max = int(np.argmax(weights))
    max_weight = float(weights[current_max])
    mod = max_weight
    i = 0
    while (mod >= max_weight * threshold and i < max_count) or i <= 1:
        origins = np.append(origins, current_max)
        weights = build_weights(mesh, vectors, center, origins=origins, adjustment=0.6)
        current_max = int(np.argmax(weights))
        mod = float(weights[current_max])
        i += 1

    return np.unique(origins.astype(int))


def loop(queue, adjacency, distances, visited, start):
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
    
    start = time.perf_counter()
    _, _, origins = mesh.ray.intersects_location(
        ray_origins=[center], ray_directions=[[0,0,1]], multiple_hits=True)
    if len(origins) > 0:
        mask = mesh.face_normals[origins][:,2]
        origins = np.array(origins)[mask < 0]
        mask = mesh.triangles_center[origins][:,2]
        origins = np.array(origins)[mask > 0]

    if len(origins) == 0:
        origins = mesh.nearest.on_surface([center])[2]

    if len(origins) == 0:
        com = mesh.bounding_box.centroid.copy()
        com[2] = 0
        _,_, origins = mesh.ray.intersects_location(
            ray_origins=[com], ray_directions=[[0,0,1]], multiple_hits=True)
    
    vectors = build_vectors(mesh)
    weights = build_weights(mesh, vectors, center, origins=origins, adjustment=.8)

    ordered = weights.argsort()
    current_max = weights.argmax()
    mod = weights[current_max]
    max_weight = mod.copy()

    print(f"Max weight: {mod}, threshold: {max_weight * threshold}")

    # Repeat until the maximum weight is below the threshold percentage of the original maximum weight; set to greater than 1 to skip
    i = 0
    while mod >= max_weight * threshold and i < max_count or i <= 1:
        ordered = weights[weights.argsort()]
        origins = np.append(origins, current_max)
        weights = build_weights(mesh, vectors, center, origins=origins, adjustment=0.6)
        ordered = weights.argsort()
        current_max = weights.argmax()
        mod = weights[current_max]
        i += 1

    weights = weights[ordered]
    mesh.update_faces(ordered)
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