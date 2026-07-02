import numpy as np
from stl import mesh
import time
import os

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
    file_name = f"Safe_Polar_{name}_{cone_angle_deg}deg_transformed{ext}"
    output_path = os.path.join(output_dir, file_name)
    my_mesh_transformed.save(output_path)

    end = time.time()
    print('STL file generated in {:.1f}s, saved in {}'.format(end - start, output_path))
    return None


# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------

#file_path = r"C:\Professional\3D4E\5AxisPrinter\ConicalSlicing\ASTM_Dogbone.stl"
file_path = r"C:\Users\canca\Documents\Conical Slicer Repo\ConicalSlicer\d20_medium.stl"
dir_transformed = r"C:\Users\canca\Documents\Conical Slicer Repo\ConicalSlicer\TransformedFiles"
transformation_type = 'outward'       # 'inward' or 'outward'
number_iterations = 0                # mesh refinement iterations
cone_angle_degrees = 0   

transformation_STL_file(
    path=file_path,
    output_dir=dir_transformed,
    cone_type=transformation_type,
    nb_iterations=number_iterations,
    cone_angle_deg=cone_angle_degrees,
)
