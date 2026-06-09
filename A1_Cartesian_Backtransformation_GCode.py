import re
import numpy as np
import os
import time
import zipfile
import io

FIXED_HEADER_PATH = r"C:\Users\canca\OneDrive\Documents\Conical Slicer Repo\ConicalSlicer\A1_HEADERBLOCKSTART.txt"
#FIXED_HEADER_PATH = r"C:\Users\canca\OneDrive\Documents\Conical Slicer Repo\ConicalSlicer\A1_SLOW_HEADERBLOCKSTART.txt"

NUM = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)'

def square_pyramid_rho(dx, dy, rotation_deg=0.0):
    """
    Square pyramid height-field distance.

    dx, dy are coordinates relative to the pyramid center.
    rotation_deg rotates the square/pyramid orientation in XY.
    """
    theta = np.deg2rad(rotation_deg)

    # Rotate coordinates
    x_rot = dx * np.cos(theta) + dy * np.sin(theta)
    y_rot = -dx * np.sin(theta) + dy * np.cos(theta)

    # Square pyramid "radius"
    rho = np.maximum(np.abs(x_rot), np.abs(y_rot))

    return rho

def read_gcode_from_file(path):
    """Read gcode from either a plain .gcode file or a .ufp package."""
    if path.endswith('.ufp'):
        with zipfile.ZipFile(path, 'r') as zf:
            with zf.open('3D/model.gcode') as f:
                return io.TextIOWrapper(f, encoding='utf-8').readlines()
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return f.readlines()


def read_fixed_header(path):
    """
    Read the fixed header block from a text file.
    Returns the content as a single string (with trailing newline).
    """
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def strip_original_header(data):
    """
    Remove everything from the start of the G-code up to and including
    '; MACHINE_START_GCODE_END' (the last line of the header block).
    Returns the remaining lines as a list.
    """
    end_marker = '; MACHINE_START_GCODE_END'
    for i, row in enumerate(data):
        if end_marker in row:
            return data[i + 1:]  # everything after the marker
    # Fallback: if marker not found, return data unchanged
    print("  WARNING: '; MACHINE_START_GCODE_END' not found in G-code. Header not replaced.")
    return data


def detect_bed_center(data):
    """
    Auto-detect the center of the print from the G-code bounding box.
    Only considers extrusion moves (lines with both E and X/Y).
    ^Because travel moves may go outside of the print area
    """
    #Find X, Y, E, G patterns via regex matching
    #pattern_X = r'X([-0-9]+[.]?[0-9]*)'
    #pattern_Y = r'Y([-0-9]+[.]?[0-9]*)'
    #pattern_E = r'E[-0-9]+[.]?[0-9]*'
    #pattern_E = r'E-?\d*\.?\d+'
    pattern_G = r'\AG[01] '
    pattern_X = rf'X{NUM}'
    pattern_Y = rf'Y{NUM}'
    pattern_Z = rf'Z{NUM}'
    pattern_E = rf'E{NUM}'

    x_coords, y_coords = [], [] #store x,y coordinates
    for row in data: #loop through g-code lines, if line contains G0/G1 and E, extract X/Y and add to list
        if re.search(pattern_G, row) and re.search(pattern_E, row): #ex. G1 X10 Y20 E0.5 (ONLY extrusion/printing moves)
            mx = re.search(pattern_X, row) #extract X
            my = re.search(pattern_Y, row) #extract Y
            if mx:
                x_coords.append(float(mx.group(1))) #gets string and converts to float number
            if my:
                y_coords.append(float(my.group(1))) #same thing for y

    if not x_coords or not y_coords: #there's no x or y coords at all
        raise ValueError("Could not detect print bounding box from G-code.")

    # get the min and max coords of x and y and calculate the center of x and y
    cx = (min(x_coords) + max(x_coords)) / 2
    cy = (min(y_coords) + max(y_coords)) / 2
    print(f"  X range: {min(x_coords):.1f} to {max(x_coords):.1f}, center: {cx:.1f}")
    print(f"  Y range: {min(y_coords):.1f} to {max(y_coords):.1f}, center: {cy:.1f}")
    return cx, cy


def insert_Z(row, z_value):
    #pattern_X = r'X[-0-9]+[.]?[0-9]*'
    #pattern_Y = r'Y[-0-9]+[.]?[0-9]*'
    #pattern_Z = r'Z[-0-9]+[.]?[0-9]*'
    pattern_X = rf'X{NUM}'
    pattern_Y = rf'Y{NUM}'
    pattern_Z = rf'Z{NUM}'
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


def backtransform_data(data, cone_type, cone_angle_deg, maximal_length, bed_center_x, bed_center_y,
                       fixed_e=0.0275, use_fixed_e=False):
    """
    Backtransform G-Code for a Bambu Lab Cartesian printer (A1).

    Forward transform (applied to STL centered at origin, then placed at bed center):
        x' = x / cos(a)
        y' = y / cos(a)
        z' = z + c * tan(a) * sqrt(x^2 + y^2)

    In G-code space after slicer placement at (bed_center_x, bed_center_y):
        x_gc = x' + bed_center_x
        y_gc = y' + bed_center_y
        z_gc = z'

    Full inverse (XY undo + Z cone offset removal):
        x_bt = (x_gc - bed_center_x) * cos(a) + bed_center_x
        y_bt = (y_gc - bed_center_y) * cos(a) + bed_center_y
        r_bt = sqrt((x_bt - bed_center_x)^2 + (y_bt - bed_center_y)^2)
        z_bt = z_gc - c * tan(a) * r_bt      <-- critical fix

    All extrusion (E) values are replaced with a fixed constant (fixed_e).
    """
    cone_angle_rad = np.radians(cone_angle_deg)
    tan_a = np.tan(cone_angle_rad)
    c = 1 if cone_type == 'outward' else -1

    # pattern_X = r'X[-0-9]+[.]?[0-9]*'
    # pattern_Y = r'Y[-0-9]+[.]?[0-9]*'
    # pattern_Z = r'Z[-0-9]+[.]?[0-9]*'
    # pattern_E = r'E[-0-9]+[.]?[0-9]*'
    pattern_G = r'\AG[01] '
    pattern_X = rf'X{NUM}'
    pattern_Y = rf'Y{NUM}'
    pattern_Z = rf'Z{NUM}'
    pattern_E = rf'E{NUM}'

    e_replacement = f'E{fixed_e}'

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
        e_match = re.search(pattern_E, row)

        if x_match is None and y_match is None and z_match is None:
            # G-code move with no XYZ — only replace E if present
            # if e_match is not None:
            #     new_data.append(re.sub(pattern_E, e_replacement, row))
            # else:
            #     new_data.append(row)
            # continue
            # E-only / retract / prime moves.
            # Usually leave these unchanged unless you specifically want fixed E everywhere.
            if e_match is not None and use_fixed_e:
                new_data.append(re.sub(pattern_E, e_replacement, row, count=1))
            else:
                new_data.append(row)
            continue

        if z_match is not None:
            z_layer = float(z_match.group(0).replace('Z', ''))
        if x_match is not None:
            x_new = float(x_match.group(0).replace('X', ''))
            update_x = True
        if y_match is not None:
            y_new = float(y_match.group(0).replace('Y', ''))
            update_y = True

        # Undo XY radial scaling relative to bed center
        x_old_bt = (x_old - bed_center_x) * np.cos(cone_angle_rad) + bed_center_x
        y_old_bt = (y_old - bed_center_y) * np.cos(cone_angle_rad) + bed_center_y
        x_new_bt = (x_new - bed_center_x) * np.cos(cone_angle_rad) + bed_center_x
        y_new_bt = (y_new - bed_center_y) * np.cos(cone_angle_rad) + bed_center_y

        # Segment long moves for smooth Z interpolation
        dist_transformed = np.linalg.norm([x_new - x_old, y_new - y_old])
        #num_segm = max(1, int(dist_transformed // maximal_length + 1)) # old
        num_segm = max(1, int(np.ceil(dist_transformed / maximal_length)))  # fixed

        x_vals = np.linspace(x_old_bt, x_new_bt, num_segm + 1)
        y_vals = np.linspace(y_old_bt, y_new_bt, num_segm + 1)

        # Compute backtransformed Z by removing the cone offset at each segment point
        # r_vals = np.sqrt((x_vals - bed_center_x)**2 + (y_vals - bed_center_y)**2)
        # z_vals = z_layer - c * tan_a * r_vals

        # Compute backtransformed Z by removing the square-pyramid offset at each segment point
        dx_vals = x_vals - bed_center_x
        dy_vals = y_vals - bed_center_y

        rho_vals = square_pyramid_rho(dx_vals, dy_vals, rotation_deg=0.0)

        z_vals = z_layer - c * tan_a * rho_vals

        replacement_rows = ''
        for j in range(num_segm):
            single_row = row
            single_row = re.sub(pattern_X, 'X' + str(round(x_vals[j+1], 3)), single_row)
            single_row = re.sub(pattern_Y, 'Y' + str(round(y_vals[j+1], 3)), single_row)
            single_row = insert_Z(single_row, z_vals[j+1])
            if e_match is not None:
                # print("Replacing:", e_match.group(0))
                # single_row = re.sub(pattern_E, e_replacement, single_row)
                if use_fixed_e:
                    # Old behavior: force every segment to the same fixed E.
                    single_row = re.sub(pattern_E, e_replacement, single_row, count=1)
                else:
                    # New behavior: preserve original relative extrusion total.
                    e_original = float(e_match.group(0).replace('E', ''))
                    e_segment = e_original / num_segm
                    single_row = re.sub(pattern_E, f'E{e_segment:.5f}', single_row, count=1)
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

    Only extrusion moves AFTER the first '; CHANGE_LAYER' or '; Z_HEIGHT:'
    comment are considered when finding z_min, to exclude startup/purge moves.
    """
    # pattern_X = r'X[-0-9]+[.]?[0-9]*'
    # pattern_Y = r'Y[-0-9]+[.]?[0-9]*'
    # pattern_Z = r'Z[-0-9]+[.]?[0-9]*'
    # pattern_E = r'E[-0-9]+[.]?[0-9]*'
    pattern_G = r'\AG[01] '
    pattern_X = rf'X{NUM}'
    pattern_Y = rf'Y{NUM}'
    pattern_Z = rf'Z{NUM}'
    pattern_E = rf'E{NUM}'

    # First pass: find minimum Z among extrusion moves AFTER layer 1 begins
    z_min = None
    in_print = False
    for row in data:
        if '; CHANGE_LAYER' in row or '; Z_HEIGHT:' in row:
            in_print = True
        if not in_print:
            continue
        g_match = re.search(pattern_G, row)
        z_match = re.search(pattern_Z, row)
        e_match = re.search(pattern_E, row)
        if g_match and z_match and e_match:
            z_val = float(z_match.group(0).replace('Z', ''))
            if z_min is None or z_val < z_min:
                z_min = z_val

    # Fallback: if no layer comments found, scan all extrusion moves
    if z_min is None:
        print("  WARNING: No '; CHANGE_LAYER' or '; Z_HEIGHT:' comments found.")
        print("  Falling back to searching ALL extrusion moves for z_min.")
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
        print(f"  Applying Z translation of {z_translate:.3f} mm to set minimum print Z to {z_desired} mm")

    # Second pass: apply X/Y/Z translations to all G0/G1 moves
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
            z_val = max(round(float(z_match.group(0).replace('Z', '')) + z_translate, 3), z_desired) #clamp to z_desired minimum
            #z_val = round(float(z_match.group(0).replace('Z', '')) + z_translate, 3) #no clamp
            row = re.sub(pattern_Z, 'Z' + str(z_val), row)
            if z_val < z_desired:
                print(f"WARNING: Z below desired minimum after translation: {z_val}")
        new_data.append(row)

    return new_data

def remove_unwanted_blocks(data):
    """
    Remove non-model utility blocks from Bambu/Orca G-code.

    Currently removes:
      - ; SKIPPABLE_START ... ; SKIPPABLE_END blocks
      - especially timelapse/safe-position chunks

    Keeps:
      - real print moves
      - temperatures
      - fan commands
      - homing/start/end G-code
      - layer comments
    """
    cleaned = []
    in_skippable = False
    removed_lines = 0
    removed_blocks = 0
    current_skiptype = None

    for row in data:
        if '; SKIPPABLE_START' in row:
            in_skippable = True
            removed_blocks += 1
            removed_lines += 1
            current_skiptype = None
            continue

        if in_skippable:
            removed_lines += 1

            if '; SKIPTYPE:' in row:
                current_skiptype = row.strip()

            if '; SKIPPABLE_END' in row:
                in_skippable = False
                current_skiptype = None

            continue

        cleaned.append(row)

    print(f"  Removed {removed_blocks} skippable block(s), {removed_lines} total line(s).")
    return cleaned

def backtransform_file(path, output_dir, cone_type, cone_angle_deg, maximal_length,
                       x_shift, y_shift, z_desired, fixed_header_path,
                       bed_center_x=None, bed_center_y=None, fixed_e=0.0275, use_fixed_e=False):
    """
    Full pipeline:
      read -> strip original header -> prepend fixed header
      -> detect center -> backtransform XY+Z (fixed E) -> translate -> save.
    """
    start = time.time()

    print("Reading G-code...")
    data = read_gcode_from_file(path)

    print("Replacing header block with fixed header...")
    body = strip_original_header(data)

    print("Removing timelapse/skippable utility blocks...")
    body = remove_unwanted_blocks(body)

    fixed_header = read_fixed_header(fixed_header_path)

    if bed_center_x is None or bed_center_y is None:
        print("Auto-detecting bed center from G-code bounding box...")
        bed_center_x, bed_center_y = detect_bed_center(body)
    else:
        print(f"Using manually specified bed center: ({bed_center_x}, {bed_center_y})")

    #print(f"Backtransforming with cone_type='{cone_type}', angle={cone_angle_deg}deg, fixed E={fixed_e}...")
    #data_bt = backtransform_data(body, cone_type, cone_angle_deg, maximal_length,
    #                             bed_center_x, bed_center_y, fixed_e=fixed_e)

    if use_fixed_e:
        print(f"Backtransforming with cone_type='{cone_type}', angle={cone_angle_deg}deg, fixed E={fixed_e}...")
    else:
        print(f"Backtransforming with cone_type='{cone_type}', angle={cone_angle_deg}deg, preserving original relative E...")

    data_bt = backtransform_data(
        body,
        cone_type,
        cone_angle_deg,
        maximal_length,
        bed_center_x,
        bed_center_y,
        fixed_e=fixed_e,
        use_fixed_e=use_fixed_e
    )

    data_bt_string = ''.join(data_bt)
    data_bt = [row + ' \n' for row in data_bt_string.split('\n')]

    print("Applying Z/XY translation...")
    data_bt = translate_data(data_bt, x_shift, y_shift, z_desired)
    data_bt_string = ''.join(data_bt)

    # Prepend fixed header (ensure it ends with a newline before body)
    if not fixed_header.endswith('\n'):
        fixed_header += '\n'
    final_output = fixed_header + data_bt_string

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    file_name = f"{name}_bt_square_pyramid_{cone_type}_{cone_angle_deg}deg.gcode"
    output_path = os.path.join(output_dir, file_name)

    with open(output_path, 'w+', encoding='utf-8', newline='\n') as f_out:
        f_out.write(final_output)

    end = time.time()
    print(f"Done! GCode generated in {end - start:.1f}s, saved to:\n  {output_path}")


# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------

file_path           = r"C:\Users\canca\OneDrive\Documents\Conical Slicer Repo\ConicalSlicer\SlicedTransformedGcode\XYZ_quality_cube_correct_orientation_square_pyramid_outward_20deg_transformed_PLA_25m25s.gcode"
dir_backtransformed = r"C:\Users\canca\OneDrive\Documents\Conical Slicer Repo\ConicalSlicer\DeformedGcode"
fixed_header_path   = FIXED_HEADER_PATH   # path to HEADERBLOCKSTART.txt

transformation_type = 'outward'   # must match Cartesian_Transformation_STL.py
cone_angle_degrees  =  20         # must match Cartesian_Transformation_STL.py exactly

max_length = 2.0   # max segment length in mm (smaller = smoother curves)

# Bambu A1 bed center. Set to None to auto-detect from G-code bounding box.
override_bed_center_x = 128.0
override_bed_center_y = 128.0

delta_x  = 0.0   # XY shift after backtransform (leave 0 for Bambu)
delta_y  = 0.0
z_height = 0.2   # desired minimum Z = first layer height

fixed_extrusion = 0.0275  # constant E value applied to every extrusion move
use_fixed_extrusion = False  # False = preserve slicer E values, True = force fixed E

# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------

backtransform_file(
    path              = file_path,
    output_dir        = dir_backtransformed,
    cone_type         = transformation_type,
    cone_angle_deg    = cone_angle_degrees,
    maximal_length    = max_length,
    x_shift           = delta_x,
    y_shift           = delta_y,
    z_desired         = z_height,
    fixed_header_path = fixed_header_path,
    bed_center_x      = override_bed_center_x,
    bed_center_y      = override_bed_center_y,
    fixed_e           = fixed_extrusion,
    use_fixed_e       = use_fixed_extrusion,
)