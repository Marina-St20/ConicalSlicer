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

def get_size(mesh):
    bounds = mesh.extents
    return bounds[0], bounds[1], bounds[2]

# z_threshold is either a threshold marker between 0-1 or an angle in degrees. 
# Value converted to 0-(-1) and compared with z normal of each face.
# Angle is measured by amount offset from vertical (0 = anything not vertical, 90 = horizontal).
def downward_faces(mesh, z_threshold=-1e-9, height_threshold=0):
    normals = mesh.face_normals.copy()
    if height_threshold != 0:
        faces = face_centroids(mesh).copy()
    if z_threshold > 1:
        z_threshold = np.sin((z_threshold % 180) * (np.pi / 180.0))
    down = np.where(normals[:, 2] < -z_threshold)[0]
    above = np.where(faces[:,2] > height_threshold)[0]
    return np.intersect1d(down,above)


def face_centroids(mesh):
    return mesh.triangles_center


def cast_down(mesh, face_index, centroid=None, offset=-1e-8, z_threshold=-1e-9):
    lowest_z = mesh.bounds[0,2]
    if (centroid is None):
        centroids = face_centroids(mesh)
        origin = centroids[face_index].copy()
    else:
        origin = centroid.copy()
    if abs(origin[2] - lowest_z) < z_threshold:
        return False, None, None
    origin += np.array([0.0, 0.0, offset])
    direction = np.array([0.0, 0.0, -1.0])

    locations, _, index_tri = mesh.ray.intersects_location(
        ray_origins=[origin], ray_directions=[direction], multiple_hits=False)

    if len(locations) == 0:
        return False, None, None

    mask = index_tri != face_index
    if not np.any(mask):
        return False, None, None

    filtered_locations = locations[mask]
    filtered_triangles = index_tri[mask]

    return True, filtered_locations, filtered_triangles

def check_clearance(mesh, face_index, centroid=None, max_distance=1, offset=-1e-8, angle_threshold=(.5,.6), z_threshold=1):
    if (centroid is None):
        centroids = face_centroids(mesh)
        origin = centroids[face_index].copy()
    else:
        origin = centroid.copy()
    if origin[2] < z_threshold: return False
    origin += np.array([0.0, 0.0, offset])

    vectors = mesh.vertices - origin
    distances = np.linalg.norm(vectors, axis=1)
    distance_mask = distances <= max_distance

    circle = vectors[distance_mask]
    inner_mask = circle[:,2] > angle_threshold[0]
    cone = circle[inner_mask]

    outer_mask = cone[:,2] < angle_threshold[1]
    cone = cone[outer_mask]
    return len(cone) > 0

def main():
    if len(sys.argv) < 2:
        print("Usage: TrimeshTesting.py file.stl")
        return
    path = sys.argv[1]
    mesh = load_mesh(path)
    print(f"Mesh loaded")
    start_time = time.perf_counter()
    dimensions = get_size(mesh)
    supports = mesh
    z_threshold = 5
    offset = -.2
    downward = downward_faces(mesh, z_threshold=60, height_threshold=z_threshold)
    centroids = face_centroids(mesh)

    for fi in downward:
        face = centroids[int(fi)]
        _face = mesh.faces[int(fi)]
        vertices = mesh.vertices[_face]
        radius = np.linalg.norm(vertices[0]-vertices[1])/3
        if check_clearance(supports,int(fi),face,5,-2,(0,1), z_threshold=z_threshold): continue
        if face[2] < z_threshold: continue

        # locs is the xyz coordinates for each face, tris is the index of each
        _, locs, _ = cast_down(mesh, int(fi), face, offset = offset, z_threshold=z_threshold)
        if locs is not None and len(locs) > 0:
            # Put cylinder with radius of inscribed circle of face down to top intersected face z + offset
            intersection = locs[0]
            print(f"Base: {intersection}")
            support_root = face.copy()
            support_root[2] = intersection[2]
            support = trimesh.creation.cylinder(radius, segment=[face+[0,0,offset],support_root], sections=3)
        else: 
            # Put cylinder w radius of inscribed circle down to z-bounding box
            support_root = face.copy()
            support_root[2] = 0
            support = trimesh.creation.cylinder(radius,segment=[face+[0,0,offset],support_root], sections=3)
        print(f"Root: {support.centroid}")

            # FOR BOTH: Adjust end angle on each side to account for the face angle 

        supports = trimesh.util.concatenate(supports,support)
        print(f"{len(supports.faces)}")

    end_time = time.perf_counter()

    print(f"Mesh dimensions: {dimensions[0]:.3f} x {dimensions[1]:.3f} x {dimensions[2]:.3f}")
    print(f"Downward-pointing faces (ignores base): {len(downward)}")

    print(f"Total faces: {len(mesh.faces)}")
    print(f"Support faces: {len(supports.faces) - len(mesh.faces)}")
    print(f"Time to generate: {end_time - start_time}")

    

    # supports.process(validate=True)
    supports.export("C:/Users/monto/Downloads/benchy_tri_supports.stl", "stl")


if __name__ == '__main__':
    main()
