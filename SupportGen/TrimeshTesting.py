import sys
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


def downward_faces(mesh, z_threshold=-1e-9):
    normals = mesh.face_normals
    if z_threshold > 1:
        z_threshold = z_threshold * (np.pi / 180.0)
    return np.where(normals[:, 2] < z_threshold)[0]


def face_centroids(mesh):
    return mesh.triangles_center


def cast_down(mesh, face_index, max_distance=np.inf, epsilon=1e-8):
    centroids = face_centroids(mesh)
    origin = centroids[face_index].copy()
    origin += np.array([0.0, 0.0, epsilon])
    direction = np.array([0.0, 0.0, -1.0])

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=[origin], ray_directions=[direction], multiple_hits=True)

    if len(locations) == 0:
        return False, None, None

    mask = index_tri != face_index
    if not np.any(mask):
        return False, None, None

    filtered_locations = locations[mask]
    filtered_triangles = index_tri[mask]

    if np.isfinite(max_distance):
        dists = np.linalg.norm(filtered_locations - origin, axis=1)
        within = dists <= max_distance
        if not np.any(within):
            return False, None, None
        filtered_locations = filtered_locations[within]
        filtered_triangles = filtered_triangles[within]

    return True, filtered_locations, filtered_triangles


def main():
    if len(sys.argv) < 2:
        print("Usage: TrimeshTesting.py file.stl [face_index]")
        return
    path = sys.argv[1]
    mesh = load_mesh(path)
    downward = downward_faces(mesh)
    print(f"Total faces: {len(mesh.faces)}")
    print(f"Downward-pointing faces: {len(downward)}")

    if len(sys.argv) == 3:
        fi = int(sys.argv[2])
        hit, locs, tris = cast_down(mesh, fi)
        print(f"Face {fi} downward: {fi in downward}")
        print(f"Intersects below (excluding itself): {hit}")
        if hit:
            for i, (l, t) in enumerate(zip(locs, tris)):
                print(f"  Hit {i}: triangle {t} at {l}")
    else:
        for fi in downward:
            hit, _, _ = cast_down(mesh, int(fi))
            print(f"face {fi}: intersects below = {bool(hit)}")


if __name__ == '__main__':
    main()
