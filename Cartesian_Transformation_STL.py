import numpy as np
from stl import mesh
import time
import os


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
    vectors_refined = refinement_triangulation(vectors, nb_iterations)
    vectors_refined = np.reshape(vectors_refined, (-1, 3))

    vectors_refined = center_model(vectors_refined)

    vectors_transformed = transformation_cone(vectors_refined, cone_type, cone_angle_deg)

    vectors_transformed = sit_model_on_build_plate(vectors_transformed)

    vectors_transformed = np.reshape(vectors_transformed, (-1, 3, 3))
    my_mesh_transformed = np.zeros(vectors_transformed.shape[0], dtype=mesh.Mesh.dtype)
    my_mesh_transformed['vectors'] = vectors_transformed
    my_mesh_transformed = mesh.Mesh(my_mesh_transformed)

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    file_name = f"Polar_{name}_{cone_angle_deg}deg_transformed{ext}"
    output_path = os.path.join(output_dir, file_name)
    my_mesh_transformed.save(output_path)

    end = time.time()
    print('STL file generated in {:.1f}s, saved in {}'.format(end - start, output_path))
    return None


# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------

#file_path = r"C:\Professional\3D4E\5AxisPrinter\ConicalSlicing\ASTM_Dogbone.stl"
file_path = r"C:\Users\canca\OneDrive\Documents\Conical Slicer Repo\ConicalSlicer\Flat Normal Dogbone.stl"
dir_transformed = r"C:\Users\canca\OneDrive\Documents\Conical Slicer Repo\ConicalSlicer\TransformedFiles"
transformation_type = 'outward'       # 'inward' or 'outward'
number_iterations = 3                # mesh refinement iterations
cone_angle_degrees = 10            # recommended: 5-20 deg for cartesian printers

transformation_STL_file(
    path=file_path,
    output_dir=dir_transformed,
    cone_type=transformation_type,
    nb_iterations=number_iterations,
    cone_angle_deg=cone_angle_degrees,
)
