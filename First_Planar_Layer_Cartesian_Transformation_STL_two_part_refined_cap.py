import numpy as np
from stl import mesh
import time
import os
import trimesh

def print_xy_size_check(label, points):
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)

    x_size = max_xyz[0] - min_xyz[0]
    y_size = max_xyz[1] - min_xyz[1]
    z_size = max_xyz[2] - min_xyz[2]

    print(f"{label}:")
    print(f"  X size: {x_size:.5f} mm")
    print(f"  Y size: {y_size:.5f} mm")
    print(f"  Z size: {z_size:.5f} mm")

def refinement_one_triangle(triangle):
    point1 = triangle[0]
    point2 = triangle[1]
    point3 = triangle[2]
    midpoint12 = (point1 + point2) / 2
    midpoint23 = (point2 + point3) / 2
    midpoint31 = (point3 + point1) / 2
    triangle1 = np.array([point1, midpoint12, midpoint31])
    triangle2 = np.array([point2, midpoint23, midpoint12])
    triangle3 = np.array([point3, midpoint31, midpoint23])
    triangle4 = np.array([midpoint12, midpoint23, midpoint31])
    return np.array([triangle1, triangle2, triangle3, triangle4])


def refinement_triangulation(triangle_array, num_iterations):
    refined_array = triangle_array
    for i in range(num_iterations):
        n_triangles = refined_array.shape[0] * 4
        refined_array = np.array(list(map(refinement_one_triangle, refined_array)))
        refined_array = np.reshape(refined_array, (n_triangles, 3, 3))
    return refined_array

def adaptive_refinement_triangulation(triangle_array, cone_type, cone_angle_deg,
                                      max_z_delta=0.10, max_edge_length=1.5,
                                      max_iterations=3):
    """
    Adaptively refine only triangles that are likely to cause visible artifacts
    after conical transformation.

    max_z_delta:
        Maximum allowed conical Z offset variation across one triangle.

    max_edge_length:
        Maximum allowed triangle edge length.

    max_iterations:
        Safety cap to prevent runaway file size.
    """
    cone_angle_rad = np.radians(cone_angle_deg)
    tan_a = np.tan(cone_angle_rad)

    if cone_type == 'outward':
        c = 1
    elif cone_type == 'inward':
        c = -1
    else:
        raise ValueError(f"{cone_type} is not an admissible type")

    refined = triangle_array

    for iteration in range(max_iterations):
        keep = []
        split = []

        for tri in refined:
            xy = tri[:, :2]
            r = np.sqrt(xy[:, 0]**2 + xy[:, 1]**2)
            cone_z_offset = c * tan_a * r

            z_delta = np.max(cone_z_offset) - np.min(cone_z_offset)

            e01 = np.linalg.norm(tri[1] - tri[0])
            e12 = np.linalg.norm(tri[2] - tri[1])
            e20 = np.linalg.norm(tri[0] - tri[2])
            max_edge = max(e01, e12, e20)

            if z_delta > max_z_delta or max_edge > max_edge_length:
                split.append(tri)
            else:
                keep.append(tri)

        print(
            f"Adaptive refinement pass {iteration + 1}: "
            f"keeping {len(keep)}, splitting {len(split)}"
        )

        if not split:
            break

        split_refined = np.array(list(map(refinement_one_triangle, split)))
        split_refined = np.reshape(split_refined, (-1, 3, 3))

        if keep:
            refined = np.concatenate([np.array(keep), split_refined], axis=0)
        else:
            refined = split_refined

    return refined


def transformation_cone(points, cone_type, cone_angle_deg):
    """
    Cartesian-printer-friendly conical transformation.
    Uses a shallow cone angle (recommended: 5-20 degrees).

    The scaling factor is derived from the cone angle:
        scale = 1 / cos(cone_angle)   (XY scale to preserve footprint)
        z_offset = tan(cone_angle) * r  (Z lift per unit radius)

    Forward transform:
        x' = scale * x
        y' = scale * y
        z' = z + c * tan(cone_angle) * sqrt(x^2 + y^2)

    where c = -1 for inward, +1 for outward.
    """
    cone_angle_rad = np.radians(cone_angle_deg)
    scale = 1.0 / np.cos(cone_angle_rad)
    tan_a = np.tan(cone_angle_rad)

    if cone_type == 'outward':
        c = 1
    elif cone_type == 'inward':
        c = -1
    else:
        raise ValueError('{} is not an admissible type for the transformation'.format(cone_type))

    def T(x, y, z):
        r = np.sqrt(x**2 + y**2)
        return np.array([scale * x, scale * y, z + c * tan_a * r])

    points_transformed = list(map(T, points[:, 0], points[:, 1], points[:, 2]))
    return np.array(points_transformed)

def center_model(vectors_refined):
    vectors_refined = vectors_refined.copy()

    min_xyz = np.min(vectors_refined, axis=0)
    max_xyz = np.max(vectors_refined, axis=0)

    center_x = (min_xyz[0] + max_xyz[0]) / 2
    center_y = (min_xyz[1] + max_xyz[1]) / 2
    min_z = min_xyz[2]

    print("Original STL bounds:")
    print(f"  X: {min_xyz[0]:.3f} to {max_xyz[0]:.3f}, center {center_x:.3f}")
    print(f"  Y: {min_xyz[1]:.3f} to {max_xyz[1]:.3f}, center {center_y:.3f}")
    print(f"  Z: {min_xyz[2]:.3f} to {max_xyz[2]:.3f}")

    vectors_refined[:, 0] -= center_x
    vectors_refined[:, 1] -= center_y
    vectors_refined[:, 2] -= min_z

    return vectors_refined

def sit_model_on_build_plate(vectors_transformed):
    vectors_transformed = vectors_transformed.copy()

    min_z_after = np.min(vectors_transformed[:, 2])
    vectors_transformed[:, 2] -= min_z_after

    min_after = np.min(vectors_transformed, axis=0)
    max_after = np.max(vectors_transformed, axis=0)

    print("Transformed STL bounds:")
    print(f"  X: {min_after[0]:.3f} to {max_after[0]:.3f}, center {(min_after[0] + max_after[0]) / 2:.3f}")
    print(f"  Y: {min_after[1]:.3f} to {max_after[1]:.3f}, center {(min_after[1] + max_after[1]) / 2:.3f}")
    print(f"  Z: {min_after[2]:.3f} to {max_after[2]:.3f}")

    return vectors_transformed


def save_trimesh_as_stl(tri_mesh, output_path):
    """Save a trimesh.Trimesh as STL after basic validation."""
    if tri_mesh is None or tri_mesh.is_empty:
        raise ValueError(f"Cannot save empty mesh: {output_path}")

    tri_mesh.remove_unreferenced_vertices()
    tri_mesh.process(validate=True)
    tri_mesh.export(output_path)



def refine_mesh_to_max_edge(tri_mesh, max_edge_length=0.50, max_iterations=8):
    """
    Subdivide a trimesh until every triangle edge is no longer than
    max_edge_length. This is especially important for the newly capped
    cut face, because the conical transform only moves mesh vertices.
    """
    if max_edge_length is None or max_edge_length <= 0:
        return tri_mesh.copy()

    vertices, faces = trimesh.remesh.subdivide_to_size(
        vertices=np.asarray(tri_mesh.vertices),
        faces=np.asarray(tri_mesh.faces),
        max_edge=max_edge_length,
        max_iter=max_iterations,
    )

    refined = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )
    refined.merge_vertices()
    refined.remove_unreferenced_vertices()
    refined.fix_normals()
    return refined

def split_original_mesh_at_z(original_mesh, split_z):
    """
    Split a watertight model into two capped solids at original-model Z=split_z.

    Returns:
        planar_slab: geometry from Z=0 through split_z
        upper_body:  geometry above split_z

    The input mesh must already be centered in X/Y and placed with min Z=0.
    """
    if split_z <= 0:
        raise ValueError("split_z must be greater than zero.")

    # slice_plane keeps the positive side of the plane normal.
    upper_body = original_mesh.slice_plane(
        plane_origin=[0.0, 0.0, split_z],
        plane_normal=[0.0, 0.0, 1.0],
        cap=True,
    )

    planar_slab = original_mesh.slice_plane(
        plane_origin=[0.0, 0.0, split_z],
        plane_normal=[0.0, 0.0, -1.0],
        cap=True,
    )

    if planar_slab is None or planar_slab.is_empty:
        raise ValueError("Planar slab is empty. Increase planar_slab_height.")

    if upper_body is None or upper_body.is_empty:
        raise ValueError("Upper body is empty. Decrease planar_slab_height.")

    # Put the cut face of the upper part at Z=0 before conical transformation.
    upper_body = upper_body.copy()
    upper_body.apply_translation([0.0, 0.0, -split_z])

    return planar_slab, upper_body


def transform_trimesh_conically(tri_mesh, cone_type, cone_angle_deg):
    """Apply the existing conical point transform to a trimesh mesh."""
    transformed = tri_mesh.copy()
    transformed.vertices = transformation_cone(
        np.asarray(transformed.vertices),
        cone_type=cone_type,
        cone_angle_deg=cone_angle_deg,
    )

    # Put the transformed upper STL on Bambu Studio's build plate.
    transformed.apply_translation([0.0, 0.0, -transformed.bounds[0, 2]])
    transformed.remove_unreferenced_vertices()
    return transformed


def generate_two_part_stls(
    path,
    output_dir,
    cone_type,
    cone_angle_deg,
    planar_slab_height,
    upper_max_edge_length=0.50,
):
    """
    Generate two independently sliceable STL files:

      1. PLANAR_<name>_slab_<height>mm.stl
         Original unwarped bottom slab.

      2. CONICAL_<name>_upper_<angle>deg.stl
         Original model above the slab, moved down to Z=0 and transformed.
    """
    start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    original = trimesh.load_mesh(path, force='mesh', process=True)
    if original.is_empty:
        raise ValueError("Loaded STL is empty.")

    original = original.copy()

    # Match center_model(): center X/Y and place original minimum Z at zero.
    bounds = original.bounds
    center_x = (bounds[0, 0] + bounds[1, 0]) / 2.0
    center_y = (bounds[0, 1] + bounds[1, 1]) / 2.0
    min_z = bounds[0, 2]
    original.apply_translation([-center_x, -center_y, -min_z])

    print("Centered original STL bounds:")
    print(original.bounds)
    print(f"Planar slab cutoff: Z=0 through Z={planar_slab_height:.5f} mm")

    planar_slab, upper_body = split_original_mesh_at_z(
        original,
        split_z=planar_slab_height,
    )

    print(
        f"Refining capped upper mesh to maximum edge length "
        f"{upper_max_edge_length:.3f} mm before conical transformation..."
    )
    upper_body = refine_mesh_to_max_edge(
        upper_body,
        max_edge_length=upper_max_edge_length,
    )

    conical_upper = transform_trimesh_conically(
        upper_body,
        cone_type=cone_type,
        cone_angle_deg=cone_angle_deg,
    )

    base = os.path.basename(path)
    name, _ = os.path.splitext(base)

    planar_path = os.path.join(
        output_dir,
        f"PLANAR_{name}_slab_{planar_slab_height:.3f}mm.stl",
    )
    upper_path = os.path.join(
        output_dir,
        f"CONICAL_{name}_upper_{cone_angle_deg}deg.stl",
    )

    save_trimesh_as_stl(planar_slab, planar_path)
    save_trimesh_as_stl(conical_upper, upper_path)

    print(f"Planar slab saved to:\n  {planar_path}")
    print(f"Conical upper body saved to:\n  {upper_path}")
    print(f"Two-part STL generation completed in {time.time() - start:.1f}s")

    return planar_path, upper_path


def transformation_STL_file(path, output_dir, cone_type, nb_iterations, cone_angle_deg):
    start = time.time()
    my_mesh = mesh.Mesh.from_file(path)
    vectors = my_mesh.vectors
    #vectors_refined = refinement_triangulation(vectors, nb_iterations)

    if nb_iterations == 0:
        vectors_refined = vectors
    else:
        vectors_refined = adaptive_refinement_triangulation(
            vectors,
            cone_type=cone_type,
            cone_angle_deg=cone_angle_deg,
            max_z_delta=0.10,
            max_edge_length=1.5,
            max_iterations=nb_iterations
        )

    vectors_refined = np.reshape(vectors_refined, (-1, 3))

    vectors_refined = center_model(vectors_refined)

    print_xy_size_check("Centered original STL before cone transform", vectors_refined)

    cone_angle_rad = np.radians(cone_angle_deg)
    expected_xy_scale = 1.0 / np.cos(cone_angle_rad)

    print(f"Expected STL XY expansion for {cone_angle_deg} deg:")
    print(f"  scale = 1/cos(angle) = {expected_xy_scale:.6f}")
    print(f"  percent = {expected_xy_scale * 100.0:.3f}%")

    vectors_transformed = transformation_cone(vectors_refined, cone_type, cone_angle_deg)

    print_xy_size_check("Transformed STL after cone transform", vectors_transformed)

    vectors_transformed = sit_model_on_build_plate(vectors_transformed)

    vectors_transformed = np.reshape(vectors_transformed, (-1, 3, 3))
    my_mesh_transformed = np.zeros(vectors_transformed.shape[0], dtype=mesh.Mesh.dtype)
    my_mesh_transformed['vectors'] = vectors_transformed
    my_mesh_transformed = mesh.Mesh(my_mesh_transformed)

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    file_name = f"E_Safe_Polar_{name}_{cone_angle_deg}deg_transformed{ext}"
    output_path = os.path.join(output_dir, file_name)
    my_mesh_transformed.save(output_path)

    end = time.time()
    print('STL file generated in {:.1f}s, saved in {}'.format(end - start, output_path))
    return None


# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------

file_path = r"C:\Users\canca\Documents\Conical Slicer Repo\ConicalSlicer\Dragon.stl"
dir_transformed = r"C:\Users\canca\Documents\Conical Slicer Repo\ConicalSlicer\TransformedFiles"

transformation_type = "outward"       # "inward" or "outward"
number_iterations = 0                  # used only in single-STL mode
cone_angle_degrees = 30

# MASTER SWITCH
# True  -> output a planar bottom slab plus a separate conical upper STL.
# False -> preserve the original single fully conical STL workflow.
USE_PLANAR_FOUNDATION = True

# Must match the planar first-layer height selected in Bambu Studio.
# Example: Bambu initial layer height 0.20 mm -> use 0.20 here.
PLANAR_SLAB_HEIGHT = 0.20
UPPER_MAX_EDGE_LENGTH = 0.50  # mm; densifies the new capped cut face


# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------

if USE_PLANAR_FOUNDATION:
    generate_two_part_stls(
        path=file_path,
        output_dir=dir_transformed,
        cone_type=transformation_type,
        cone_angle_deg=cone_angle_degrees,
        planar_slab_height=PLANAR_SLAB_HEIGHT,
        upper_max_edge_length=UPPER_MAX_EDGE_LENGTH,
    )
else:
    transformation_STL_file(
        path=file_path,
        output_dir=dir_transformed,
        cone_type=transformation_type,
        nb_iterations=number_iterations,
        cone_angle_deg=cone_angle_degrees,
    )
