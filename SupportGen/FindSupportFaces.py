import sys
import time
import numpy as np
import trimesh
import vispy
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

def get_size(mesh):
    bounds = mesh.extents
    return bounds[0], bounds[1], bounds[2]

def cast_ring(mesh, radius=0, origin=[0, 0, 0], division_base=6, angle_offset=0):
    # Cast a ring of rays around the origin point in the XY plane, with a given radius.
    # Returns the locations of intersections and the indices of the intersected faces.
    num_rays = np.floor(radius*division_base)  # Number of rays to cast around the circle
    if num_rays < division_base:
        num_rays = division_base
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False) + angle_offset / division_base
    ray_origins = np.array([[origin[0] + radius * np.cos(angle), origin[1] + radius * np.sin(angle), origin[2]] for angle in angles])
    ray_directions = np.tile(np.array([0.0, 0.0, 1.0]), (num_rays, 1))  # All rays pointing upwards in the Z direction

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=True)

    return locations, index_tri

def cast_box(mesh, divisions=50):
    # Cast rays in a grid pattern over the XY plane of the mesh's bounding box.
    bounds = mesh.bounds
    x_min, y_min, z_min = bounds[0]
    x_max, y_max, z_max = bounds[1]

    x_values = np.linspace(x_min, x_max, divisions)
    y_values = np.linspace(y_min, y_max, divisions)
    ray_origins = np.array([[x, y, z_min - 1] for x in x_values for y in y_values])  
    ray_directions = np.tile(np.array([0.0, 0.0, 1.0]), (len(ray_origins), 1))  

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=True)

    return locations, index_tri

def show_regions(mesh, face_indices, color=[1, 0, 0, 1]):
    # Visualize the mesh with specified faces highlighted in a different color.
    face_colors = np.tile([0.5, 0.5, 0.5, 1], (len(mesh.faces), 1))  # Default color for all faces
    face_colors[face_indices] = color  # Highlight specified faces
    mesh.visual.face_colors = face_colors
    mesh.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python FindSupportFaces.py <path_to_mesh>")
        return

    mesh_path = sys.argv[1]
    mesh = load_mesh(mesh_path)
    print(f"Loaded mesh from {mesh_path} with {len(mesh.faces)} faces.")

    face_indices = np.empty(0, dtype=int)
    bounds = mesh.bounds
    dimensions = mesh.extents

    for i in range(int(max(dimensions[0], dimensions[1]) / 2)):
        radius = i
        faces, fi = cast_ring(mesh, radius, origin=[0, 0, 0], division_base=1, angle_offset=0)
        face_indices = np.append(face_indices, fi)

    # Uniform sampling of the mesh using box casting
    # box_faces, box_fi = cast_box(mesh, divisions=10)
    # print(f"Found {len(box_fi)} faces in box casting.")

    print(f"{face_indices}")
    vertices = np.unique(mesh.faces[face_indices].ravel())
    vertices = mesh.vertices[vertices]

    fig = Fig()
    ax_x = fig[0, 0]

    ax_x.plot((vertices[:, 0], vertices[:,1], vertices[:, 2]), width=.01)
    ax_x.view.camera = 'arcball'
    fig.show(run=True)

    show_regions(mesh, face_indices)


if __name__ == "__main__":
    main()