import re
import numpy as np
import os
import time
import zipfile
import io


def read_gcode_from_file(path):
    """Read gcode from either a plain .gcode file or a .ufp package."""
    if path.endswith('.ufp'):
        with zipfile.ZipFile(path, 'r') as zf:
            with zf.open('3D/model.gcode') as f:
                return io.TextIOWrapper(f, encoding='utf-8').readlines()
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return f.readlines()


def detect_bed_center(data):
    """
    Auto-detect the center of the print from the G-code bounding box.
    Only considers extrusion moves (lines with both E and X/Y).
    """
    pattern_X = r'X([-0-9]+[.]?[0-9]*)'
    pattern_Y = r'Y([-0-9]+[.]?[0-9]*)'
    pattern_E = r'E[-0-9]+[.]?[0-9]*'
    pattern_G = r'\AG[01] '

    x_coords, y_coords = [], []
    for row in data:
        if re.search(pattern_G, row) and re.search(pattern_E, row):
            mx = re.search(pattern_X, row)
            my = re.search(pattern_Y, row)
            if mx:
                x_coords.append(float(mx.group(1)))
            if my:
                y_coords.append(float(my.group(1)))

    if not x_coords or not y_coords:
        raise ValueError("Could not detect print bounding box from G-code.")

    cx = (min(x_coords) + max(x_coords)) / 2
    cy = (min(y_coords) + max(y_coords)) / 2
    print(f"  X range: {min(x_coords):.1f} to {max(x_coords):.1f}, center: {cx:.1f}")
    print(f"  Y range: {min(y_coords):.1f} to {max(y_coords):.1f}, center: {cy:.1f}")
    return cx, cy


def insert_Z(row, z_value):
    pattern_X = r'X[-0-9]+[.]?[0-9]*'
    pattern_Y = r'Y[-0-9]+[.]?[0-9]*'
    pattern_Z = r'Z[-0-9]+[.]?[0-9]*'
    match_x = re.search(pattern_X, row)
    match_y = re.search(pattern_Y, row)
    match_z = re.search(pattern_Z, row)

    if match_z is not None:
        row_new = re.sub(pattern_Z, 'Z' + str(round(z_value, 3)), row)
    else:
        if match_y is not None:
            row_new = row[:match_y.end()] + ' Z' + str(round(z_value, 3)) + row[match_y.end():]
        elif match_x is not None:
            row_new = row[:match_x.end()] + ' Z' + str(round(z_value, 3)) + row[match_x.end():]
        else:
            row_new = 'Z' + str(round(z_value, 3)) + ' ' + row
    return row_new


def replace_E(row, dist_old, dist_new):
    """
    Scale the E value by the ratio of new 3D path length to old 2D path length.
    Since BambuStudio uses relative E (M83), each move's E is independent.
    """
    pattern_E = r'E[-0-9]+[.]?[0-9]*'
    match_e = re.search(pattern_E, row)
    if match_e is None:
        return row
    e_val_old = float(match_e.group(0).replace('E', ''))
    if dist_old == 0:
        e_val_new = 0.0
    else:
        e_val_new = round(e_val_old * dist_new / dist_old, 6)
    row_new = row[:match_e.start()] + 'E' + str(e_val_new) + row[match_e.end():]
    return row_new


def backtransform_data(data, cone_type, cone_angle_deg, maximal_length, bed_center_x, bed_center_y):
    """
    Backtransform G-Code for a Bambu Lab Cartesian printer (A1 mini).

    The forward transform (applied to the STL, centered at origin, then placed
    at bed center by the slicer) was:
        x' = x / cos(a)          (XY scaled from origin)
        y' = y / cos(a)
        z' = z + c * tan(a) * sqrt(x^2 + y^2)

    After slicing, the slicer placed the part at (bed_center_x, bed_center_y),
    so in GCode coordinates:
        x_gc = x' + bed_center_x
        y_gc = y' + bed_center_y
        z_gc = z'   (Z is unchanged by placement)

    The inverse (what we apply here) only needs to fix XY â€” Z is already
    correct because the slicer sliced the transformed shape and its layer
    Z values already encode the conical surface height:

        x_bt = (x_gc - bed_center_x) * cos(a) + bed_center_x
        y_bt = (y_gc - bed_center_y) * cos(a) + bed_center_y
        z_bt = z_gc   (NO cone offset added â€” it's already in z_gc)

    Segmentation is still applied so that long XY moves get intermediate Z
    values interpolated from the layer Z (which varies smoothly along the
    conical surface due to the slicer's layer heights).
    """
    cone_angle_rad = np.radians(cone_angle_deg)

    pattern_X = r'X[-0-9]+[.]?[0-9]*'
    pattern_Y = r'Y[-0-9]+[.]?[0-9]*'
    pattern_Z = r'Z[-0-9]+[.]?[0-9]*'
    pattern_E = r'E[-0-9]+[.]?[0-9]*'
    pattern_G = r'\AG[01] '

    new_data = []
    x_old, y_old = bed_center_x, bed_center_y
    x_new, y_new = bed_center_x, bed_center_y
    z_layer = 0.0
    update_x, update_y = False, False

    for row in data:
        g_match = re.search(pattern_G, row)
        if g_match is None:
            new_data.append(row)
            continue

        x_match = re.search(pattern_X, row)
        y_match = re.search(pattern_Y, row)
        z_match = re.search(pattern_Z, row)

        if x_match is None and y_match is None and z_match is None:
            new_data.append(row)
            continue

        # Update tracked layer Z from the raw GCode value
        if z_match is not None:
            z_layer = float(z_match.group(0).replace('Z', ''))
        if x_match is not None:
            x_new = float(x_match.group(0).replace('X', ''))
            update_x = True
        if y_match is not None:
            y_new = float(y_match.group(0).replace('Y', ''))
            update_y = True

        e_match = re.search(pattern_E, row)

        # Backtransform XY only: undo the radial scaling relative to bed center
        x_old_bt = (x_old - bed_center_x) * np.cos(cone_angle_rad) + bed_center_x
        y_old_bt = (y_old - bed_center_y) * np.cos(cone_angle_rad) + bed_center_y
        x_new_bt = (x_new - bed_center_x) * np.cos(cone_angle_rad) + bed_center_x
        y_new_bt = (y_new - bed_center_y) * np.cos(cone_angle_rad) + bed_center_y

        # Segment long moves so intermediate Z values are interpolated correctly
        dist_transformed = np.linalg.norm([x_new - x_old, y_new - y_old])
        num_segm = max(1, int(dist_transformed // maximal_length + 1))

        x_vals = np.linspace(x_old_bt, x_new_bt, num_segm + 1)
        y_vals = np.linspace(y_old_bt, y_new_bt, num_segm + 1)

        # Z comes directly from the slicer â€” no cone offset added
        # For segmented moves, z_layer is constant across the segment
        # (the slicer already encoded varying Z per layer)
        z_vals = np.full(num_segm + 1, z_layer)

        # 2D distance per segment in transformed space (denominator for E scaling)
        dist_transformed_per_seg = dist_transformed / num_segm

        # 3D distances in backtransformed space (numerator for E scaling)
        distances_bt = np.array([
            np.linalg.norm([
                x_vals[i] - x_vals[i-1],
                y_vals[i] - y_vals[i-1],
                z_vals[i] - z_vals[i-1]
            ])
            for i in range(1, num_segm + 1)
        ])

        # Build replacement rows
        replacement_rows = ''
        for j in range(num_segm):
            single_row = row
            single_row = re.sub(pattern_X, 'X' + str(round(x_vals[j+1], 3)), single_row)
            single_row = re.sub(pattern_Y, 'Y' + str(round(y_vals[j+1], 3)), single_row)
            single_row = insert_Z(single_row, z_vals[j+1])
            if e_match is not None:
                single_row = replace_E(single_row, dist_transformed_per_seg, distances_bt[j])
            replacement_rows += single_row

        if update_x:
            x_old = x_new
            update_x = False
        if update_y:
            y_old = y_new
            update_y = False

        new_data.append(replacement_rows)

    return new_data


def translate_data(data, translate_x, translate_y, z_desired):
    """
    Optionally shift the G-code in X/Y and ensure minimum extrusion Z
    is at z_desired (first layer height).
    """
    pattern_X = r'X[-0-9]+[.]?[0-9]*'
    pattern_Y = r'Y[-0-9]+[.]?[0-9]*'
    pattern_Z = r'Z[-0-9]+[.]?[0-9]*'
    pattern_E = r'E[-0-9]+[.]?[0-9]*'
    pattern_G = r'\AG[01] '

    # First pass: find minimum Z among extrusion moves
    z_min = None
    for row in data:
        g_match = re.search(pattern_G, row)
        z_match = re.search(pattern_Z, row)
        e_match = re.search(pattern_E, row)
        if g_match and z_match and e_match:
            z_val = float(z_match.group(0).replace('Z', ''))
            if z_min is None or z_val < z_min:
                z_min = z_val
    z_translate = (z_desired - z_min) if z_min is not None else 0.0

    if z_translate != 0:
        print(f"  Applying Z translation of {z_translate:.3f} mm to set minimum Z to {z_desired} mm")

    # Second pass: apply translations
    new_data = []
    for row in data:
        g_match = re.search(pattern_G, row)
        if g_match is None:
            new_data.append(row)
            continue
        x_match = re.search(pattern_X, row)
        y_match = re.search(pattern_Y, row)
        z_match = re.search(pattern_Z, row)
        if x_match and translate_x != 0:
            x_val = round(float(x_match.group(0).replace('X', '')) + translate_x, 3)
            row = re.sub(pattern_X, 'X' + str(x_val), row)
        if y_match and translate_y != 0:
            y_val = round(float(y_match.group(0).replace('Y', '')) + translate_y, 3)
            row = re.sub(pattern_Y, 'Y' + str(y_val), row)
        if z_match:
            z_val = max(round(float(z_match.group(0).replace('Z', '')) + z_translate, 3), z_desired)
            row = re.sub(pattern_Z, 'Z' + str(z_val), row)
        new_data.append(row)

    return new_data


def backtransform_file(path, output_dir, cone_type, cone_angle_deg, maximal_length,
                       x_shift, y_shift, z_desired, bed_center_x=None, bed_center_y=None):
    """
    Full pipeline: read -> detect center -> backtransform XY -> translate -> save.

    bed_center_x / bed_center_y: the XY position in GCode space where the slicer
    placed the center of your model. For the Bambu A1 mini, this is (90, 90).
    Leave as None to auto-detect from the GCode bounding box.
    """
    start = time.time()

    print("Reading G-code...")
    data = read_gcode_from_file(path)

    if bed_center_x is None or bed_center_y is None:
        print("Auto-detecting bed center from G-code bounding box...")
        bed_center_x, bed_center_y = detect_bed_center(data)
    else:
        print(f"Using manually specified bed center: ({bed_center_x}, {bed_center_y})")

    print(f"Backtransforming with cone_type='{cone_type}', angle={cone_angle_deg}deg...")
    data_bt = backtransform_data(data, cone_type, cone_angle_deg, maximal_length,
                                  bed_center_x, bed_center_y)

    data_bt_string = ''.join(data_bt)
    data_bt = [row + ' \n' for row in data_bt_string.split('\n')]

    print("Applying Z/XY translation...")
    data_bt = translate_data(data_bt, x_shift, y_shift, z_desired)
    data_bt_string = ''.join(data_bt)

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    file_name = f"{name}_bt_{cone_type}_{cone_angle_deg}deg.gcode"
    output_path = os.path.join(output_dir, file_name)

    with open(output_path, 'w+', encoding='utf-8', newline='\n') as f_out:
        f_out.write(data_bt_string)

    end = time.time()
    print(f"Done! GCode generated in {end - start:.1f}s, saved to:\n  {output_path}")


# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------

file_path = r"C:\Professional\3D4E\5AxisPrinter\ConicalSlicing\SlicedTransformedGcode\ASTM_Dogbone_outward_10deg_transformed.gcode"
dir_backtransformed = r"C:\Professional\3D4E\5AxisPrinter\ConicalSlicing\DeformedGcode"

transformation_type = 'outward'   # must match Cartesian_Transformation_STL.py
cone_angle_degrees  = 10.0        # must match Cartesian_Transformation_STL.py exactly

max_length = 2.0   # max segment length in mm (smaller = smoother curves)

# Bambu A1 mini bed center. Set to None to auto-detect from G-code bounding box.
# Manual override recommended for reliability: A1 mini bed is 180x180, center = (90, 90)
override_bed_center_x = 90.0
override_bed_center_y = 90.0

delta_x    = 0.0   # XY shift after backtransform (leave 0 for Bambu)
delta_y    = 0.0
z_height   = 0.2   # desired minimum Z = first layer height

# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------

backtransform_file(
    path           = file_path,
    output_dir     = dir_backtransformed,
    cone_type      = transformation_type,
    cone_angle_deg = cone_angle_degrees,
    maximal_length = max_length,
    x_shift        = delta_x,
    y_shift        = delta_y,
    z_desired      = z_height,
    bed_center_x   = override_bed_center_x,
    bed_center_y   = override_bed_center_y,
)