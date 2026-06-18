import argparse
import os
import numpy as np
from stl import mesh as stl_mesh


def load_stl(file_path):
    return stl_mesh.Mesh.from_file(file_path)


def compute_overhangs(mesh, threshold_degrees):
    normals = mesh.normals.astype(np.float64)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    zero_norm = lengths == 0
    lengths[zero_norm] = 1.0
    normals_u = normals / lengths

    if np.any(zero_norm):
        zero_indices = np.nonzero(zero_norm[:, 0])[0]
        vectors = mesh.vectors[zero_indices]
        v1 = vectors[:, 1] - vectors[:, 0]
        v2 = vectors[:, 2] - vectors[:, 0]
        computed = np.cross(v1, v2)
        comp_len = np.linalg.norm(computed, axis=1, keepdims=True)
        comp_len[comp_len == 0] = 1.0
        normals_u[zero_indices] = computed / comp_len

    downfacing = normals_u[:, 2] < 0.0
    plane_angles = np.degrees(np.arccos(np.clip(np.abs(normals_u[:, 2]), -1.0, 1.0)))
    overhang_mask = downfacing & (plane_angles <= float(threshold_degrees))
    indices = np.nonzero(overhang_mask)[0].tolist()

    overhang_data = []
    for idx in indices:
        overhang_data.append({
            'index': int(idx),
            'angle_from_horizontal': float(plane_angles[idx]),
            'normal': tuple(normals_u[idx]),
            'vertices': [tuple(v) for v in mesh.vectors[idx]],
        })

    return indices, overhang_data


def write_overhang_stl(file_path, mesh, indices):
    overhang_count = len(indices)
    output_mesh = stl_mesh.Mesh(np.zeros(overhang_count, dtype=stl_mesh.Mesh.dtype))
    for i, idx in enumerate(indices):
        output_mesh.vectors[i] = mesh.vectors[idx]
        output_mesh.normals[i] = mesh.normals[idx]
    output_mesh.save(file_path)


def main():
    parser = argparse.ArgumentParser(description='Detect overhangs in an STL mesh using numpy and numpy-stl.')
    parser.add_argument('stl_file', help='Path to the input STL file.')
    parser.add_argument('-a', '--angle', type=float, default=45.0,
                        help='Maximum overhang angle from horizontal in degrees (default: 45).')
    parser.add_argument('-o', '--output', help='Optional output STL file containing only overhang facets.')
    parser.add_argument('-d', '--details', action='store_true',
                        help='Print detailed overhang facet information.')
    args = parser.parse_args()

    if not os.path.isfile(args.stl_file):
        raise FileNotFoundError(f'Input file not found: {args.stl_file}')

    mesh = load_stl(args.stl_file)
    overhang_indices, overhang_data = compute_overhangs(mesh, args.angle)

    print(f'Found {len(overhang_indices)} overhang facets out of {len(mesh.vectors)} total facets.')
    if overhang_indices:
        print('Overhang facet indices:', ', '.join(str(i) for i in overhang_indices))

    if args.details and overhang_data:
        for item in overhang_data:
            print(f"Facet {item['index']}: angle={item['angle_from_horizontal']:.2f} deg, normal={item['normal']}")
            for vertex in item['vertices']:
                print(f'  vertex {vertex}')

    if args.output:
        if overhang_indices:
            write_overhang_stl(args.output, mesh, overhang_indices)
            print(f'Wrote overhang-only STL to: {args.output}')
        else:
            print('No overhang facets to write.')


if __name__ == '__main__':
    main()
