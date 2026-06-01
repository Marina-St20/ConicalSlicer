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
    Backtransform G-Code for a standard Cartesian printer.

    The forward transform was:
        x' = (x - cx) / cos(a) + cx
        y' = (y - cy) / cos(a) + cy
        z' = z - c * tan(a) * sqrt((x-cx)^2 + (y-cy)^2)   [inward: c=-1, outward: c=+1]

    The inverse (what we apply here):
        x_bt = (x' - cx) * cos(a) + cx
        y_bt = (y' - cy) * cos(a) + cy
        z_bt = z_layer + c * tan(a) * sqrt((x_bt-cx)^2 + (y_bt-cy)^2)

    where c = +1 for inward (inverts the forward c=-1), -1 for outward.
    All coordinates stay in bed space throughout.
    """
    cone_angle_rad = np.radians(cone_angle_deg)
    scale = 1.0 / np.cos(cone_angle_rad)   # forward scale factor
    tan_a = np.tan(cone_angle_rad)

    if cone_type == 'inward':
        c = 1    # inverse of forward c=-1
    elif cone_type == 'outward':
        c = -1   # inverse of forward c=+1
    else:
        raise ValueError('{} is not an admissible cone type'.format(cone_type))

    pattern_X = r'X[-0-9]+[.]?[0-9]*'
    pattern_Y = r'Y[-0-9]+[.]?[0-9]*'
    pattern_Z = r'Z[-0-9]+[.]?[0-9]*'
    pattern_E = r'E[-0-9]+[.]?[0-9]*'
    pattern_G = r'\AG[01] '

    new_data = []
    x_old, y_old = 0.0, 0.0
    x_new, y_new = 0.0, 0.0
    z_layer = 0.0
    z_max = 0.0
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

        # Update tracked positions
        if z_match is not None:
            z_layer = float(z_match.group(0).replace('Z', ''))
        if x_match is not None:
            x_new = float(x_match.group(0).replace('X', ''))
            update_x = True
        if y_match is not None:
            y_new = float(y_match.group(0).replace('Y', ''))
            update_y = True

        e_match = re.search(pattern_E, row)

        # Backtransform XY: undo the forward scaling (relative to cone axis / bed center)
        x_old_bt = (x_old - bed_center_x) * np.cos(cone_angle_rad) + bed_center_x
        y_old_bt = (y_old - bed_center_y) * np.cos(cone_angle_rad) + bed_center_y
        x_new_bt = (x_new - bed_center_x) * np.cos(cone_angle_rad) + bed_center_x
        y_new_bt = (y_new - bed_center_y) * np.cos(cone_angle_rad) + bed_center_y

        # Segment long moves for accurate Z interpolation
        dist_transformed = np.linalg.norm([x_new - x_old, y_new - y_old])
        num_segm = max(1, int(dist_transformed // maximal_length + 1))

        x_vals = np.linspace(x_old_bt, x_new_bt, num_segm + 1)
        y_vals = np.linspace(y_old_bt, y_new_bt, num_segm + 1)

        # Compute Z along the cone surface for each sub-point
        # Radius is measured from the cone axis (bed center)
        z_vals = np.array([
            z_layer + c * tan_a * np.sqrt((x - bed_center_x)**2 + (y - bed_center_y)**2)
            for x, y in zip(x_vals, y_vals)
        ])

        # Track max printed Z; cap travel moves to avoid nozzle crashing
        if e_match is not None:
            if np.max(z_vals) > z_max or z_max == 0:
                z_max = np.max(z_vals)
        else:
            z_vals = np.minimum(z_vals, z_max + 1.0)

        # 2D distance per segment in transformed (Cura) space — used for E scaling denominator
        dist_transformed_per_seg = dist_transformed / num_segm

        # 3D distances in backtransformed space — used for E scaling numerator
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
    Optionally shift the G-code in X/Y and ensure minimum Z is at z_desired.
    For the Ultimaker S3, these should typically both be 0.
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
    Full pipeline: read -> detect center -> backtransform -> translate -> save.

    If bed_center_x / bed_center_y are None, they are auto-detected from the
    G-code bounding box. For the Ultimaker S3, this is the recommended mode.
    """
    start = time.time()

    print("Reading G-code...")
    data = read_gcode_from_file(path)

    if bed_center_x is None or bed_center_y is None:
        print("Auto-detecting bed center from G-code bounding box...")
        bed_center_x, bed_center_y = detect_bed_center(data)
    else:
        print(f"Using manually specified bed center: ({bed_center_x}, {bed_center_y})")

    print(f"Backtransforming with cone_type='{cone_type}', angle={cone_angle_deg}°...")
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
    # Always save as .gcode regardless of input format
    file_name = f"{name}_bt_{cone_type}_{cone_angle_deg}deg.gcode"
    output_path = os.path.join(output_dir, file_name)

    with open(output_path, 'w+', encoding='utf-8') as f_out:
        f_out.write(data_bt_string)

    end = time.time()
    print(f"Done! GCode generated in {end - start:.1f}s, saved to:\n  {output_path}")


# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------

file_path = r"C:\Professional\3D4E\5AxisPrinter\ConicalSlicing\TransformedFiles\3DBenchy_inward_45.0deg_transformed.stl"   # or .ufp
dir_backtransformed = r"C:\Professional\3D4E\5AxisPrinter\ConicalSlicing\DeformedGcode"

transformation_type = 'inward'    # must match Transformation_STL.py
cone_angle_degrees = 45.0         # must match Transformation_STL.py exactly

max_length = 2.0                  # max segment length in mm (smaller = smoother Z curves)

# For Ultimaker S3: leave as None to auto-detect from G-code bounding box.
# Or override manually, e.g.: bed_center_x = 132.0, bed_center_y = 130.0
override_bed_center_x = None
override_bed_center_y = None

# For Ultimaker S3 with Griffin G-code, leave these as 0 —
# coordinates stay in the printer's own bed space.
delta_x = 0.0
delta_y = 0.0
z_height = 0.2                    # desired minimum Z (should match your first layer height)

# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------

backtransform_file(
    path=file_path,
    output_dir=dir_backtransformed,
    cone_type=transformation_type,
    cone_angle_deg=cone_angle_degrees,
    maximal_length=max_length,
    x_shift=delta_x,
    y_shift=delta_y,
    z_desired=z_height,
    bed_center_x=override_bed_center_x,
    bed_center_y=override_bed_center_y,
)
