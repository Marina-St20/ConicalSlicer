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

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def cone_angle_from_z_rad(z, z_min, z_max, max_cone_angle_deg, angle_ramp_power=1.0):
    """
    Height-dependent cone angle.

    z_min -> 0 degrees
    z_max -> max_cone_angle_deg

    angle_ramp_power:
        1.0 = linear
        2.0 = gentler near bottom, stronger near top
        0.5 = stronger near bottom
    """
    height = z_max - z_min

    if height <= 1e-9:
        return 0.0

    frac = (z - z_min) / height
    frac = clamp(frac, 0.0, 1.0)

    frac = frac ** angle_ramp_power

    return np.radians(max_cone_angle_deg * frac)

def transformation_cone(points, cone_type, max_cone_angle_deg, angle_ramp_power=1.0):
    """
    Height-dependent conical transformation.

    Instead of using one constant cone angle everywhere, the cone angle
    increases with original model Z height:

        bottom of model -> 0 deg
        top of model    -> max_cone_angle_deg

    Forward transform:
        angle(z) = max_angle * normalized_z^power

        x' = x / cos(angle(z))
        y' = y / cos(angle(z))
        z' = z + c * tan(angle(z)) * r

    where c = +1 for outward, -1 for inward.
    """
    max_cone_angle_deg = clamp(max_cone_angle_deg, 0.0, 89.0)

    if cone_type == 'outward':
        c = 1
    elif cone_type == 'inward':
        c = -1
    else:
        raise ValueError('{} is not an admissible type for the transformation'.format(cone_type))

    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    model_height = z_max - z_min

    print("Height-dependent cone transform:")
    print(f"  Model Z range after centering: {z_min:.3f} to {z_max:.3f} mm")
    print(f"  Original model height: {model_height:.3f} mm")
    print(f"  Max cone angle at top: {max_cone_angle_deg:.3f} deg")
    print(f"  Angle ramp power: {angle_ramp_power:.3f}")
    print("")
    print("COPY THIS VALUE INTO THE POLAR BACKTRANSFORM FILE:")
    print(f"  original_model_height_mm = {model_height:.5f}")
    print("")

    def T(x, y, z):
        r = np.sqrt(x**2 + y**2)

        angle_rad = cone_angle_from_z_rad(
            z=z,
            z_min=z_min,
            z_max=z_max,
            max_cone_angle_deg=max_cone_angle_deg,
            angle_ramp_power=angle_ramp_power,
        )

        scale = 1.0 / np.cos(angle_rad)
        tan_a = np.tan(angle_rad)

        return np.array([
            scale * x,
            scale * y,
            z + c * tan_a * r
        ])

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

def transformation_STL_file(path, output_dir, cone_type, nb_iterations, cone_angle_deg, angle_ramp_power=1.0):
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

    vectors_transformed = transformation_cone(
        vectors_refined,
        cone_type,
        max_cone_angle_deg=cone_angle_deg,
        angle_ramp_power=angle_ramp_power,
    )

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
file_path = r"C:\Users\canca\Documents\Conical Slicer Repo\ConicalSlicer\ISO REAL Cone Angle Fix 2.5 Benchy.stl"
dir_transformed = r"C:\Users\canca\Documents\Conical Slicer Repo\ConicalSlicer\TransformedFiles"
transformation_type = 'outward'       # 'inward' or 'outward'
number_iterations = 0                # mesh refinement iterations
cone_angle_degrees = 60            # max cone angle at the TOP of the model
angle_ramp_power = 1.0             # 1.0 linear, 2.0 gentler bottom, 0.5 stronger bottom

transformation_STL_file(
    path=file_path,
    output_dir=dir_transformed,
    cone_type=transformation_type,
    nb_iterations=number_iterations,
    cone_angle_deg=cone_angle_degrees,
    angle_ramp_power=angle_ramp_power,
)
