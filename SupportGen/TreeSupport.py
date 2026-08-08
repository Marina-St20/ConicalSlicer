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
    dilations = 2

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
    nodes = np.array(list(nodes))
    for i, node in enumerate(nodes):
        scaled = np.asarray(node)*pitch + grid_origin
        x, y, z = scaled
        nodes[i] = (x, y, z)
        plotter.add_mesh(pv.Sphere(radius=.2, center=(x, y, z)), color="red", opacity=1)
    print(f"Drawing grid")
    plotter.show_grid()
    plotter.show()
    print(f"Building the support mesh...")
    supports = wrap(nodes, inflated_mask.shape, step_size, tip_radius, base_radius)
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
    df = nd.distance_transform_edt(~mask)
    indices = np.zeros(int(grid_shape[2]))
    z_height = np.arange(bradius, tradius, -(bradius - tradius) / max_z)
    radius_threshold = indices + np.pad(z_height, pad_width=(0, indices.shape[0] - z_height.shape[0]), mode='constant')
    threshold_grid = radius_threshold[np.newaxis, np.newaxis, :]
    df = df - threshold_grid
    verts, faces, normals, _ = skimage.measure.marching_cubes(
        df, 0, spacing=(adjusted, adjusted, adjusted))
    supports = trimesh.Trimesh(verts, faces, vertex_normals=normals)

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
        step_size=.4, 
        tip_radius=args.tradius, 
        base_radius=args.bradius, 
    )
    print(f"Complete.")

if __name__ == "__main__":
    main()


## Test
def wrap_paths_to_stl(global_tree_nodes, grid_shape, grid_origin, padding_cells=0, voxel_size=0.2, radius_bottom_mm=4.0, radius_top_mm=1.0, output_filename="tree_supports.stl"):
    """
    Generates perfectly tapered branches with guaranteed flat solid bottom caps
    by building an explicit Signed Distance Field (SDF).
    """
    print(f"Initializing SDF Canvas matching array shape: {grid_shape}")
    
    # We initialize with a high positive value (representing empty space)
    # The surface will be extracted exactly at 0.0
    sdf_grid = np.ones(grid_shape, dtype=float) * 999.0
    
    max_z = grid_shape[2]
    
    print("Populating tapered volumetric fields around tree paths...")
    # Loop over every node in your tree structure
    for pt in global_tree_nodes:
        cx, cy, cz = int(np.round(pt[0])), int(np.round(pt[1])), int(np.round(pt[2]))
        
        if not (0 <= cx < grid_shape[0] and 0 <= cy < grid_shape[1] and 0 <= cz < grid_shape[2]):
            continue
            
        # Calculate the target real-world radius for this specific node's height
        # Linear interpolation from narrow tips at top to thick trunks at the floor (Z=0)
        z_progress = cz / max_z
        current_radius_mm = radius_bottom_mm - (z_progress * (radius_bottom_mm - radius_top_mm))
        
        # Convert radius to voxel units
        voxel_radius = current_radius_mm / voxel_size
        int_rad = int(np.ceil(voxel_radius))
        
        # Define a tight local bounding box around this single node to stay ultra-fast
        x_min, x_max = max(0, cx - int_rad), min(grid_shape[0], cx + int_rad + 1)
        y_min, y_max = max(0, cy - int_rad), min(grid_shape[1], cy + int_rad + 1)
        z_min, z_max = max(0, cz - int_rad), min(grid_shape[2], cz + int_rad + 1)
        
        # Generate local coordinate grids
        x_range = np.arange(x_min, x_max)
        y_range = np.arange(y_min, y_max)
        z_range = np.arange(z_min, z_max)
        
        # Compute 3D distances from the center node
        xx, yy, zz = np.meshgrid(x_range - cx, y_range - cy, z_range - cz, indexing='ij')
        distances = np.sqrt(xx**2 + yy**2 + zz**2)
        
        # --- THE TAPER & HOLLOW/SOLID SIGNED FIELD MATH ---
        # Distance minus radius creates a standard SDF where inside the branch is negative,
        # outside is positive, and the outer skin is EXACTLY 0.0
        local_sdf = distances - voxel_radius
        
        # Merge this node's sphere into the global canvas using a boolean minimum operator
        sdf_grid[x_min:x_max, y_min:y_max, z_min:z_max] = np.minimum(
            sdf_grid[x_min:x_max, y_min:y_max, z_min:z_max], 
            local_sdf
        )

    # --- THE SOLID BOTTOM CAP FIX ---
    # To force Marching Cubes to cleanly seal the bottom of the trunks, 
    # we manually clamp the SDF values at Z=0 to positive boundaries, 
    # forcing the mesh generator to wrap a flat, horizontal floor cap.
    sdf_grid[:, :, 0] = np.maximum(sdf_grid[:, :, 0], 0.0)

    print("Running Marching Cubes to extract the smooth, tapered shell...")
    # Tracing level=0.0 extracts the perfect outer skin layer of the tapered tree
    verts, faces, normals, values = skimage.measure.marching_cubes(
        volume=sdf_grid, 
        level=0.0, 
        spacing=(voxel_size, voxel_size, voxel_size)
    )
    
    support_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    # Strip horizontal padding from vertices, leave Z at 0
    padding_mm = padding_cells * voxel_size
    support_mesh.apply_translation((-padding_mm, -padding_mm, 0.0))
    
    # Snap back to real-world model coordinates
    support_mesh.apply_translation(grid_origin)