import argparse
import heapq
import time
import numpy as np
import trimesh
import scipy.ndimage as nd
import skimage
import pyvista as pv
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
    dilations = int(base_radius / step_size)

    # Voxelize for pathfinding algorithm
    print(f"Voxelizing...")
    voxels = mesh.voxelized(pitch)
    # Build clearance
    padded = np.pad(voxels.matrix, ((padding,padding),(padding,padding),(0,0)), "constant", constant_values=0)
    transform = voxels.transform.copy()
    offset = padding * pitch
    # offset_z = 2 * pitch
    transform[0, 3] -= offset
    transform[1, 3] -= offset
    # transform[2, 3] -= offset_z
    voxels = trimesh.voxel.VoxelGrid(padded, transform)
    # Mask to empty space minus a buffer
    mask = voxels.matrix.astype(bool)
    x,y,z = np.ogrid[-buffer:buffer+1,-buffer:buffer+1,-buffer:buffer+1]
    structure = x**2 + y**2 + z**2 <= buffer**2
    inflated_mask = nd.binary_dilation(mask, structure, dilations)
    grid_origin = voxels.translation
    start_nodes = []
    nodes = set()

    # Draw mesh grid for visualization
    points = np.argwhere(inflated_mask == 1)
    cloud = pv.PolyData(points * pitch + grid_origin)
    cloud["Z Depth"] = points[:, 2]
    surface = pv.wrap(mesh)
    plotter = pv.Plotter()
    plotter.add_mesh(cloud, scalars="Z Depth", cmap = "plasma", render_points_as_spheres=True, point_size = 5)
    plotter.add_mesh(surface, opacity=0.7, style="surface", color="light_gray")
    
    # Preset central pillar in nodes for conic prints
    central_pillar = []
    for i in range(0, int(grid_origin[2]) + 1, 1):
        central_pillar.append((0, 0, i))
    nodes.update(central_pillar)

    # Map roots to nodes
    print(f"Building support paths...")
    for root in roots:
        x = int(np.round((root[0] - grid_origin[0]) / pitch))
        y = int(np.round((root[1] - grid_origin[1]) / pitch))
        z = int(np.round((root[2] - grid_origin[2]) / pitch))
        if not (0 <= x < inflated_mask.shape[0] and 0 <= y < inflated_mask.shape[1] and 0 <= z < inflated_mask.shape[2]):
                print(f"Warning: Point {root} converted to out-of-bounds node {(x, y, z)}. Skipping.")
                continue
        start_nodes.append((x, y, z))

    for root in start_nodes:
        x, y, z = root
        shift = (int(x), int(y), int(z - dilations-1))
        if shift[2] <= 0:
            print(f"Skipped")
        start = time.perf_counter()
        path = build_path(inflated_mask, shift, nodes)
        end = time.perf_counter()
        print(f"Pathfinding for {root} took {end - start:.4f} seconds.")
        if path is None:
            continue
        paths.append(path)
    node_arr = np.array(list(nodes))
    for i, node in enumerate(node_arr):
        scaled = np.asarray(node)*pitch + grid_origin
        x, y, z = scaled
        node_arr[i] = (x, y, z)
        plotter.add_mesh(pv.Sphere(radius=.2, center=(x, y, z)), color="red", opacity=1)
    print(f"Drawing grid")
    plotter.show_grid()
    plotter.show()
    print(f"Building the support mesh...")
    supports = wrap(nodes, inflated_mask.shape, pitch, tip_radius, base_radius)
    supports.apply_translation(grid_origin)
    supports.export(output_path)
    

def build_path(grid, start, nodes):
    # A* Algorithm
    # Construct possible directions array
    directions = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            directions.append((dx, dy, -1))
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
            if grid[nx, ny, nz] == 1:
                continue

            cost = np.sqrt(dx**2 + dy**2 + dz**2)
            tentative = score[current] + cost
            if neighbor not in score or tentative < score[neighbor]:
                source[neighbor] = current
                score[neighbor] = tentative

                # Check if any nodes are closer than the floor
                min_dist = nz 
                for tx, ty, tz in nodes:
                    if tz < nz:
                        continue
                    dist_to_trunk = np.sqrt((nx - tx)**2 + (ny - ty)**2)
                    if dist_to_trunk < min_dist:
                        min_dist = dist_to_trunk
                tentative += min_dist * 1.01
                heapq.heappush(openings, (tentative, neighbor))
    print(f"Failed node: {current} at {neighbor}")
    return None

def wrap(nodes, grid_shape, voxel_size=1, tradius=.4, bradius=4):
    bradius = 4
    tradius = 1
    precision = 1
    adjusted = voxel_size / precision
    grid_shape = np.asarray(grid_shape) / precision
    grid_shape = np.array(grid_shape, dtype=int)
    mask = np.zeros(grid_shape, dtype=bool)
    max_z = 0

    for x,y,z in nodes:
        x = int(x/precision)
        y = int(y/precision)
        z = int(z/precision)
        if z > max_z: max_z = z
        if 0 <= x < grid_shape[0] and 0 <= y < grid_shape[1] and 0 <= z < grid_shape[2]:
            mask[x, y, z] = True
        
    # Distance Field shaping
    mask = ~mask
    np.pad(mask, pad_width=10, mode='constant', constant_values=0)
    df = nd.distance_transform_edt(mask)
    indices = np.zeros(int(grid_shape[2]))
    z_height = np.arange(bradius, tradius, -(bradius - tradius) / max_z)
    radius_threshold = indices + np.pad(z_height, pad_width=(0, indices.shape[0] - z_height.shape[0]), mode='constant')
    threshold_grid = radius_threshold[np.newaxis, np.newaxis, :]
    df = df - threshold_grid
    verts, faces, normals, _ = skimage.measure.marching_cubes(
        df, 0, spacing=(adjusted, adjusted, adjusted))
    supports = trimesh.Trimesh(verts, faces, vertex_normals=normals)

    return supports

def build_raft(mesh: trimesh.Trimesh, z_threshold=.01, extrude_height=2, show_voxels=False):
    vertices = mesh.vertices[:, 2] < z_threshold
    faces = vertices[mesh.faces].all(axis=1) & (mesh.face_normals[:, 2] < 0)
    faces = mesh.faces[faces]
    if len(faces) == 0:
        raise ValueError("No faces found below the threshold with downward normals.")
        
    base = trimesh.Trimesh(vertices=mesh.vertices, faces=faces, process=False)
    raft = trimesh.creation.extrude_triangulation(base.vertices[:, :2], base.faces, height=extrude_height)
    raft = dilate(raft, 6, show_voxels=show_voxels)
    raft.process(validate=True)
    
    return raft

def dilate(mesh: trimesh.Trimesh, units=5, show_voxels=False):
    pitch = .2
    buffer = 1
    dilations = int(units / pitch)
    padding = dilations + 1
    voxels = mesh.voxelized(pitch)

    # Build clearance
    padded = np.pad(voxels.matrix, ((padding,padding),(padding,padding),(padding,padding)), "constant", constant_values=0)
    transform = voxels.transform.copy()
    offset = padding * pitch
    transform[0, 3] -= offset
    transform[1, 3] -= offset
    transform[2, 3] -= offset
    voxels = trimesh.voxel.VoxelGrid(padded, transform)

    # Mask to empty space minus a buffer
    mask = voxels.matrix.astype(bool)
    x,y,z = np.ogrid[-buffer:buffer+1,-buffer:buffer+1,-buffer:buffer+1]
    # structure = x**2 + y**2 <= buffer**2
    structure = nd.generate_binary_structure(2, 1)
    grid = nd.binary_dilation(mask, structure, dilations, axes=(0, 1))
    grid_origin = voxels.translation

    # Distance Field shaping
    np.pad(grid, pad_width=10, mode='constant', constant_values=0)
    df = nd.distance_transform_edt(grid)
    verts, faces, _, _ = skimage.measure.marching_cubes(
        df, 0, spacing=(pitch, pitch, pitch))

    # Draw grid for voxel visualization
    if (show_voxels):
        points = np.argwhere(grid == 1)
        cloud = pv.PolyData(points * pitch + grid_origin)
        cloud["Z Depth"] = points[:, 2]
        surface = pv.wrap(mesh)
        plotter = pv.Plotter()
        plotter.add_mesh(cloud, scalars="Z Depth", cmap = "plasma", render_points_as_spheres=True, point_size = 1)
        plotter.add_mesh(surface, opacity=1, style="surface", color="light_gray")
        plotter.show()

    flipped = faces.copy()
    flipped[:, [0, 1]] = flipped[:, [1, 0]]

    mesh = trimesh.Trimesh(verts, flipped)
    mesh.apply_translation(grid_origin)
    
    return mesh
    
def main():
    options = {
        'mesh_path': {'help': 'Path to the mesh file to load.'},
        'filter_path': {'help': 'Path to filter mesh.'},
        'output_path': {'help': 'Path to output.'},
        '--sensitivity': {'type': float, 'default': .5, 'help': 'Sensitivity, determined by remesh. Low values for high sensitivity, default is .5.'},
        '--center': {'type': str, 'help': 'Optional center point as "x,y,z" passed to FindSupportFaces.py.'},
        '--bradius': {'type': float, 'default': 4.0, 'help': 'Support base radius. Defaults to 4.0.'},
        '--tradius': {'type': float, 'default': 0.4, 'help': 'Support tip radius. Defaults to 0.4.'},
        '--offset': {'type': float, 'default': 0.2, 'help': 'Offset between supports and original mesh.'},
        '--show_raft_cloud': {'type': bool, 'default': False, 'help': 'Show the raft voxel cloud for debugging.'},
        '--show_raft': {'type': bool, 'default': False, 'help': 'Show the raft for debugging.'}
    }
    parser = argparse.ArgumentParser(description='Generate supports for a mesh.')
    for arg, params in options.items():
        parser.add_argument(arg, **params)
    args = parser.parse_args()

    print(f"Scanning...")
    mesh = MeshCheck.load_mesh(args.mesh_path)
    raft = build_raft(mesh, z_threshold=.5, extrude_height=.6, show_voxels=args.show_raft_cloud)

    # Check raft size and positioning
    if (args.show_raft):
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, opacity=0.7, style="surface", color="light_gray")
        plotter.add_mesh(raft, opacity=0.7, style="surface", color="light_gray")
        plotter.show()

    origins = MeshCheck.vertical_scan(mesh, args.filter_path, center=args.center, step=.1, remesh_detail=args.sensitivity)
    generate(
        mesh=mesh, 
        origins=origins, 
        output_path=args.output_path, 
        step_size=1, 
        tip_radius=args.tradius, 
        base_radius=args.bradius, 
    )
    print(f"Complete.")

if __name__ == "__main__":
    main()




    # --- THE SOLID BOTTOM CAP FIX ---
    # To force Marching Cubes to cleanly seal the bottom of the trunks, 
    # we manually clamp the SDF values at Z=0 to positive boundaries, 
    # forcing the mesh generator to wrap a flat, horizontal floor cap.
    # sdf_grid[:, :, 0] = np.maximum(sdf_grid[:, :, 0], 0.0)

    # print("Running Marching Cubes to extract the smooth, tapered shell...")
    # # Tracing level=0.0 extracts the perfect outer skin layer of the tapered tree
    # verts, faces, normals, values = skimage.measure.marching_cubes(
    #     volume=sdf_grid, 
    #     level=0.0, 
    #     spacing=(voxel_size, voxel_size, voxel_size)
    # )
    
    # support_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    # # Strip horizontal padding from vertices, leave Z at 0
    # padding_mm = padding_cells * voxel_size
    # support_mesh.apply_translation((-padding_mm, -padding_mm, 0.0))
    
    # # Snap back to real-world model coordinates
    # support_mesh.apply_translation(grid_origin)