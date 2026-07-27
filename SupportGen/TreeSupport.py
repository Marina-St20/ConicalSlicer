import argparse
import heapq
import numpy as np
import trimesh
import scipy.ndimage as nd
import skimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree

import MeshCheck

def generate(
        mesh=None, 
        origins=None, 
        output_path="../support_demo.stl", 
        step_size=1.0, 
        tip_radius=0.4, 
        base_radius=4, 
):
    if mesh is None or origins is None:
        return

    roots = mesh.triangles_center[origins]
    roots = roots.tolist()
    roots.sort(key=lambda pt: pt[2], reverse=True)
    roots = np.array(roots)
    paths = [[] for _ in range(len(roots))]
    padding = 10
    pitch = .2
    buffer = 1

    # Voxelize for pathfinding algorithm
    print(f"Voxelizing...")
    voxels = mesh.voxelized(pitch)
    # Build clearance
    padded = np.pad(voxels.matrix, ((padding,padding),(padding,padding),(0,0)), "constant", constant_values=0)
    transform = voxels.transform.copy()
    offset = padding * pitch
    transform[0, 3] -= offset
    transform[1, 3] -= offset
    voxels = trimesh.voxel.VoxelGrid(padded, transform)
    # Mask to empty space minus a buffer
    mask = voxels.matrix.astype(bool)
    x,y,z = np.ogrid[-buffer:buffer+1,-buffer:buffer+1,-buffer:buffer+1]
    structure = x**2 + y**2 + z**2 <= buffer**2
    inflated_mask = nd.binary_dilation(mask, structure)
    # Invert to the model, leaving empty space as false
    grid = np.where(inflated_mask == True, 0, 1)
    grid_origin = voxels.translation
    start_nodes = []
    nodes = set()
    
    # Map roots to nodes
    print(f"Building support paths...")
    for root in roots:
        x = int(np.round((root[0] - grid_origin[0]) / step_size)) + padding
        y = int(np.round((root[1] - grid_origin[1]) / step_size)) + padding
        z = int(np.round((root[2] - grid_origin[2]) / step_size))
        if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and 0 <= z < grid.shape[2]):
                print(f"Warning: Point {root} converted to out-of-bounds node {(x, y, z)}. Skipping.")
                continue
        start_nodes.append((x, y, z))

    for root in start_nodes:
        x, y, z = root
        shift = (int(x), int(y), int(z - 1))
        if shift[2] <= 0:
            print(f"Skipped")
        path = build_path(grid, shift, nodes)
        if path is None:
            continue
        paths.append(path)
    print(f"Building the support mesh...")
    supports = wrap(nodes, grid.shape, padding, step_size, tip_radius, base_radius)
    supports.apply_translation((-padding * step_size, -padding * step_size, 0))
    supports.apply_translation(grid_origin)
    supports.export(output_path)
    

def build_path(grid, start, nodes):
    # A* Algorithm
    # Construct possible directions array
    directions = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            directions.append((dx, dy, -1))
    center = (start[0], start[1], 0)
    openings = []
    heapq.heappush(openings, (0, start))
    source = {}
    score = {start: 0}
    while openings:
        _, current = heapq.heappop(openings)
        cx, cy, cz = current

        # End if floored
        if cz == 0 or current in nodes:
            path = []
            while current in source:
                path.append(current)
                current = source[current]
            path.append(start)
            path = path[::-1]
            nodes.update(path)
            return path

        for dx, dy, dz in directions:
            neighbor = (int(cx + dx), int(cy + dy), int(cz + dz))
            nx, ny, nz = neighbor

            # Valid pathing filters
            if not (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and 0 <= nz < grid.shape[2]):
                continue
            if grid[nx, ny, nz] == 0:
                continue

            cost = np.sqrt(dx**2 + dy**2 + dz**2)
            if dx == 0 and dy == 0:
                cost *= .9
            tentative = score[current] + cost
            if neighbor not in score or tentative < score[neighbor]:
                source[neighbor] = current
                score[neighbor] = tentative

                # Check if any nodes are closer than the floor
                min_dist = nz # heuristic_cost(current, center)
                for tx, ty, tz in nodes:
                    if tz == nz: 
                        dist_to_trunk = np.sqrt((nx - tx)**2 + (ny - ty)**2)
                        if dist_to_trunk < min_dist:
                            min_dist = dist_to_trunk
                cost += min_dist
                heapq.heappush(openings, (cost, neighbor))
    print(f"Failed node: {current} at {neighbor}")
    return None

def heuristic_cost(current, goal):
    return np.sqrt((current[0] - goal[0])**2 + (current[1] - goal[1])**2 + (current[2])**2)

def wrap(nodes, grid_shape, padding, voxel_size=1, tradius=.4, bradius=4):
    bradius = 1
    mask = np.zeros(grid_shape, dtype=bool)

    for x,y,z in nodes:
        if 0 <= grid_shape[0] and 0 <= y < grid_shape[1] and 0 <= z < grid_shape[2]:
            mask[x, y, z] = True
    
    mask[:,:,0] = mask[:,:,0] + (mask[:,:,1])
    # Distance Field shaping
    df = nd.distance_transform_edt(~mask)
    max_z = grid_shape[2]
    indices = np.arange(max_z)
    radius_threshold = (bradius - (indices / max_z) * (bradius - tradius)) / voxel_size
    threshold_grid = radius_threshold[np.newaxis, np.newaxis, :]
    df = df - threshold_grid

    verts, faces, normals, _ = skimage.measure.marching_cubes(
        df, 0, spacing=(voxel_size, voxel_size, voxel_size))
    supports = trimesh.Trimesh(verts, faces, vertex_normals=normals)
    supports.fill_holes()

    return supports

def main():
    parser = argparse.ArgumentParser(description='Generate supports for a mesh.')
    parser.add_argument('mesh_path', help='Path to the mesh file to load.')
    parser.add_argument('filter_path', help='Path to filter mesh.')
    parser.add_argument('output_path', help='Path to output.')
    parser.add_argument('--sensitivity', type=float, default=.5, help='Sensitivity, determined by remesh. Low values for high sensitivity, default is .5.')
    parser.add_argument('--center', type=str, default=None, help='Optional center point as "x,y,z" passed to FindSupportFaces.py.')
    parser.add_argument('--bradius', type=float, default=4, help='Support base radius. Defaults to 4.0.')
    parser.add_argument('--tradius', type=float, default=.4, help='Support tip radius. Defaults to 0.4.')
    parser.add_argument('--offset', type=float, default=.2, help='Offset between supports and original mesh.')
    args = parser.parse_args()

    print(f"Scanning...")
    mesh = MeshCheck.load_mesh(args.mesh_path)
    origins = MeshCheck.vertical_scan(mesh, args.filter_path, center=args.center, step=.1, remesh_detail=args.sensitivity)
    generate(
        mesh=mesh, 
        origins=origins, 
        output_path=args.output_path, 
        step_size=1.0, 
        tip_radius=args.tradius, 
        base_radius=args.bradius, 
    )
    print(f"Complete.")

if __name__ == "__main__":
    main()
