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

def get_size(mesh):
    bounds = mesh.extents
    return bounds[0], bounds[1], bounds[2]

# z_threshold is either a threshold marker between 0-1 or an angle in degrees. 
# Value converted to 0-(-1) and compared with z normal of each face.
# Angle is measured by amount offset from vertical (0 = anything not vertical, 90 = horizontal).
def downward_faces(mesh, z_threshold=-1e-9):
    normals = mesh.face_normals
    if z_threshold > 1:
        z_threshold = np.sin((z_threshold % 180) * (np.pi / 180.0))
    return np.where(normals[:, 2] < -z_threshold)[0]


def face_centroids(mesh):
    return mesh.triangles_center


def cast_down(mesh, face_index, max_distance=np.inf, offset=1e-8, z_threshold=-1e-9):
    lowest_z = mesh.bounds[0,2]
    centroids = face_centroids(mesh)
    origin = centroids[face_index].copy()
    if abs(origin[2] - lowest_z) < z_threshold:
        return False, None, None
    origin += np.array([0.0, 0.0, offset])
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
    dimensions = get_size(mesh)
    print(f"Mesh dimensions: {dimensions[0]:.3f} x {dimensions[1]:.3f} x {dimensions[2]:.3f}")
    downward = downward_faces(mesh, 60)
    print(f"Total faces: {len(mesh.faces)}")
    print(f"Downward-pointing faces: {len(downward)}")

    if len(sys.argv) == 3:
        fi = int(sys.argv[2])
        hit, locs, tris = cast_down(mesh, fi, max_distance=dimensions[2] + 1, z_threshold=1)
        print(f"Face {fi} downward: {fi in downward}")
        print(f"Intersects below (excluding itself): {hit}")
        if hit:
            for i, (l, t) in enumerate(zip(locs, tris)):
                print(f"  Hit {i}: triangle {t} at {l}")
    else:
        intersecting = 0
        intersected_faces = np.ndarray(shape=(1,3))
        for fi in downward:
            hit, locs, tris = cast_down(mesh, int(fi), max_distance=dimensions[2] + 1, z_threshold=1)
            if hit:
                intersecting += 1
            if tris is not None and len(locs) > 0:
                intersected_faces = np.vstack([intersected_faces, locs]) if intersected_faces.size > 0 else locs

        print(f"Total downward faces: {len(downward)}")
        print(f"Faces with intersection: {intersecting}")
        print(f"Total faces intersected below (excluding itself): {len(intersected_faces)}")


if __name__ == '__main__':
    main()
