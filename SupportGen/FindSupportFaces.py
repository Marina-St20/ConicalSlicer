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

def build_map(mesh, origin_face=0):
    distances = np.full(len(mesh.faces), np.inf, dtype=float)

    if len(mesh.face_adjacency) == 0:
        return np.array([origin_face], dtype=int), distances

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

    # Use to examine full lists
    # np.savetxt("SharedList.txt", mesh.face_adjacency, "%d", ", ")
    # DOESN'T WORK ON BENCHY 
    # txtarray = np.array(adjacency)
    # txtarray = txtarray.reshape(txtarray.shape[0], -1)
    # np.savetxt("AdjacencyTree.txt",txtarray, "%.2f", ", ")


    visited = np.zeros(len(mesh.faces), dtype=bool)
    distances[origin_face] = 0.0
    queue = [(0.0, int(origin_face))]

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

def loop(queue, adjacency, ordered_faces, distances, visited, start):
     print(f"Loop")
     queue = [(0.0, int(start))]
     while queue:
        face = heapq.heappop(queue)
        face_idx=face[1]
        current_distance=face[0]
        if visited[face_idx]:
            continue
        visited[face_idx] = True
        ordered_faces.append(face_idx)

        for neighbour_idx, distance in adjacency[face_idx]:
            new_distance = current_distance + distance
            if not visited[neighbour_idx] and new_distance < distances[neighbour_idx]:
                distances[neighbour_idx] = new_distance
                heapq.heappush(queue, (new_distance, neighbour_idx))

def main():
    if len(sys.argv) < 2:
        print("Usage: python FindSupportFaces.py <path_to_mesh> [optional: z_min]")
        return

    mesh_path = sys.argv[1]
    mesh = load_mesh(mesh_path)
    print(f"Loaded mesh from {mesh_path} with {len(mesh.faces)} faces.")

    colors = mesh.visual.face_colors
    center = mesh.bounding_box.centroid.copy()
    if len(sys.argv) > 2:
        center[2] = float(sys.argv[2])
    else:
        center[2] = 0

    start = time.perf_counter()
    distances = build_map(mesh, origin_face=0)
    end = time.perf_counter()
    mod = distances[distances.argmax()]

    for i in range(len(mesh.faces)):
        distance = distances[i]/mod * 255
        red = distance
        colors[i] = [red, 90, 60, 255]

    print(f"Time to scan: {end-start}")
    show_regions(mesh, colors=colors)

if __name__ == "__main__":
    main()