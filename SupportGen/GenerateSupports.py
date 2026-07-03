import argparse
import os
import numpy as np
import trimesh

# Ensure local module import from the SupportGen directory.
import sys
sys.path.insert(0, os.path.dirname(__file__))

from FindSupportFaces import load_mesh, find_support_faces


def parse_center(center_string):
    if center_string is None:
        return None
    values = [float(x.strip()) for x in center_string.split(',') if x.strip()]
    if len(values) != 3:
        raise ValueError('Center must be a comma-separated 3D vertex string like "x,y,z".')
    return np.array(values, dtype=float)


def create_triangular_support(mesh, face_index, bottom_z, radius):
    centroid = mesh.triangles_center[face_index].copy()
    origin = centroid + np.array([0.0, 0.0, -.1], dtype=float)
    direction = np.array([0.0, 0.0, -1.0], dtype=float)

    locations, _, _ = mesh.ray.intersects_location(
        ray_origins=[origin],
        ray_directions=[direction],
        multiple_hits=True,
    )

    support_end_z = bottom_z
    if len(locations) > 0:
        distances = origin[2] - locations[:, 2]
        mask = distances > 1e-6
        if np.any(mask):
            nearest = np.argmin(distances[mask])
            support_end_z = max(bottom_z, locations[mask][nearest, 2] + .1)

    support_height = origin[2] - support_end_z
    if support_height <= 1:
        return None

    support = trimesh.creation.cylinder(radius=radius, height=support_height, sections=16)
    support.visual.face_colors = [180, 120, 40, 255]

    transform = np.eye(4)
    transform[0, 3] = centroid[0]
    transform[1, 3] = centroid[1]
    transform[2, 3] = support_end_z + support_height / 2.0
    transform = check_wall(mesh, face_index, transform, radius=radius, offset=-.2)
    support.apply_transform(transform)

    return support

def check_wall(mesh, face_index, transform, radius=1, offset=.2):
    centroid = mesh.triangles_center[face_index].copy()
    origin = centroid + np.array([0.0, 0.0, offset*2], dtype=float)

    locations = mesh.nearest.on_surface([origin])[0]
    if len(locations) == 0:
        return False
    x = locations[0][0] - centroid[0]
    y = locations[0][1] - centroid[1]
    neg_x = x < 0
    neg_y = y < 0
    radius = radius/2
    x = -radius if neg_x else radius
    y = -radius if neg_y else radius
    transform[0, 3] = transform[0, 3] + x
    transform[1, 3] = transform[1, 3] + y

    return transform


def build_supports(mesh, face_indices, radius=None):
    if len(face_indices) == 0:
        return []

    bottom_z = mesh.bounds[0, 2]
    if radius is None:
        radius = .67
    supports = []

    for face_index in face_indices:
        support = create_triangular_support(mesh, int(face_index), bottom_z, radius)
        if support is not None:
            supports.append(support)

    return supports


def show_mesh_with_supports(mesh, supports):
    if not supports:
        mesh.show()
        return

    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    for support in supports:
        scene.add_geometry(support)
    scene.show()


def main():
    parser = argparse.ArgumentParser(description='Generate triangular support pillars from support face indices.')
    parser.add_argument('mesh_path', help='Path to the mesh file to load.')
    parser.add_argument('--threshold', type=float, default=None, help='Optional threshold multiplier passed to FindSupportFaces.py.')
    parser.add_argument('--max_count', type=int, default=None, help='Optional maximum number of support faces to find passed to FindSupportFaces.py.')
    parser.add_argument('--center', type=str, default=None, help='Optional center point as "x,y,z" passed to FindSupportFaces.py.')
    parser.add_argument('--radius', type=float, default=None, help='Optional fixed support radius for the generated pillars.')
    args = parser.parse_args()

    mesh = load_mesh(args.mesh_path)
    center = parse_center(args.center)

    if args.threshold is None and args.max_count is None:
        support_faces = find_support_faces(mesh, center=center)
    elif args.threshold is None:
        support_faces = find_support_faces(mesh, max_count=args.max_count, center=center)
    elif args.max_count is None:
        support_faces = find_support_faces(mesh, threshold=args.threshold, center=center)
    else:
        support_faces = find_support_faces(mesh, threshold=args.threshold, max_count=args.max_count, center=center)

    support_faces = np.asarray(support_faces, dtype=int)
    print(f'Found {len(support_faces)} support face indices: {support_faces}')

    supports = build_supports(mesh, support_faces, radius=args.radius)
    show_mesh_with_supports(mesh, supports)


if __name__ == '__main__':
    main()
