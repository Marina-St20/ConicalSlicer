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

def cast_ring(mesh, radius=0, origin=[0, 0, 0], division_base=6, angle_offset=0):
    # Cast a ring of rays around the origin point in the XY plane, with a given radius.
    # Returns the locations of intersections and the indices of the intersected faces.
    num_rays = int(np.floor(radius*division_base))  # Number of rays to cast around the circle
    if num_rays < division_base:
        num_rays = division_base
    angles = np.linspace(0, 2 * np.pi, int(num_rays), endpoint=False) + angle_offset / division_base
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

    # Add optional parameters and data for sampling
    face_indices = np.empty(0, dtype=int)
    dimensions = mesh.extents
    normals = mesh.face_normals.copy()

    if len(sys.argv) > 3:
        radius = float(sys.argv[3])
    else:
        radius = int(max(dimensions[0], dimensions[1]))
    if len(sys.argv) > 4:
        density = int(sys.argv[4])
    else:
        density = 1
    print(f"Mesh center: {center}, dimensions: {dimensions}")

    # Sampling using cast_ring
    for i in range(int(radius)):
        ring_radius = i/2
        ring_locs, fi = cast_ring(mesh, ring_radius, origin=center, division_base=density, angle_offset=0)
        face_indices = np.append(face_indices, fi)
    for i in range(len(mesh.faces)):
        red = (-normals[i][2] + 1.01) * 125
        green = (np.max([normals[i][0], normals[i][1]]) + 1.01) * 70
        blue = green/2
        colors[i] = [red, green, blue, 255]



    # Uniform sampling of the mesh using box casting
    box_faces, box_fi = cast_box(mesh, divisions=10)
    print(f"Found {len(box_fi)} faces in box casting.")



    # Point plot of vertices after sampling
    vertices = np.unique(mesh.faces[face_indices].ravel())
    vertices = mesh.vertices[vertices]
    vertex_colors = np.zeros((len(vertices), 4), np.float32)
    vertex_colors = vertex_colors + [0, .4, .3, 1]
    for i in range(len(vertex_colors)):
        vertex_colors[i] = vertex_colors[i] + [i/len(vertex_colors), 0, 0, 0]

    fig = Fig()
    ax_x = fig[0, 0]

    ax_x.plot((vertices[:, 0], vertices[:,1], vertices[:, 2]), symbol='o', marker_size=10, width=.01, face_color=vertex_colors)
    ax_x.view.camera = 'arcball'
    fig.show(run=True)

    show_regions(mesh, face_indices, colors=colors)

if __name__ == "__main__":
    main()