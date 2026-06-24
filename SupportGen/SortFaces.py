import sys
import time
import numpy as np
import trimesh

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

def sort(mesh, target=np.array([0.0, 0.0, 0.0])):
    centroids = mesh.triangles_center
    distances = np.linalg.norm(centroids - target, axis=1)
    sorted_indices = np.argsort(distances)
    return sorted_indices