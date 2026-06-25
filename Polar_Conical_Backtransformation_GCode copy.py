import re
import numpy as np
import os
import time
import zipfile
import io

FIXED_HEADER_PATH = r"C:\Users\canca\Documents\Conical Slicer Repo\ConicalSlicer\POLAR_HEADERBLOCKSTART.txt"
#FIXED_HEADER_PATH = r"C:\Users\canca\Documents\Conical Slicer Repo\ConicalSlicer\A1_SLOW_HEADERBLOCKSTART.txt"

NUM = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)'

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

def make_simple_start_gcode(
    nozzle_temp=240,
    bed_temp=60,
):
    """
    Simple starter G-code for first 4-axis printer tests.

    Assumptions:
        G90 = absolute machine axes
        M83 = relative extrusion
        G28 = home machine
        M190 = set bed temp and wait
        M109 = set nozzle temp and wait
    """
    return f"""\
; ------------------------------------------------------------
; SIMPLE 4-AXIS START G-CODE
; ------------------------------------------------------------
G90 ; absolute positioning for machine axes
M82 ; absolute extrusion mode
M140 S{bed_temp} ; start heating bed
M104 S{nozzle_temp} ; start heating nozzle
M190 S{bed_temp} ; wait for bed temperature
M109 S{nozzle_temp} ; wait for nozzle temperature
G28 ; home all axes
G90 ; absolute positioning after homing
M83 ; relative extrusion mode after homing
; ------------------------------------------------------------
; BEGIN GENERATED 4-AXIS TOOLPATH
; ------------------------------------------------------------
"""


def make_simple_end_gcode(
    end_z_lift=10.0,
):
    """
    Simple end G-code for first 4-axis printer tests.
    """
    return f"""\
; ------------------------------------------------------------
; END GENERATED 4-AXIS TOOLPATH
; SIMPLE 4-AXIS END G-CODE
; ------------------------------------------------------------
G91 ; relative positioning for lift
G1 Z{end_z_lift:.3f} F2000 ; lift Z out of the way
G90 ; back to absolute positioning
M104 S0 ; turn off nozzle heater
M140 S0 ; turn off bed heater
M84 ; disable motors
; ------------------------------------------------------------
; END FILE
; ------------------------------------------------------------
"""

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

def remove_everything_before_first_layer_marker(data):
    """
    Remove all slicer/body content before the first real layer marker.

    This is better than starting at the first positive E move, because
    Bambu/Orca can have purge/wipe extrusion before the actual model layer.
    """
    cleaned = []
    found_layer = False
    removed_lines = 0

    layer_markers = (
        '; CHANGE_LAYER',
        '; Z_HEIGHT:',
        ';LAYER_CHANGE',
        '; layer num/',
    )

    for row in data:
        if not found_layer:
            if any(marker in row for marker in layer_markers):
                found_layer = True
                cleaned.append(row)
            else:
                removed_lines += 1
            continue

        cleaned.append(row)

    print(f"  Removed {removed_lines} line(s) before first layer marker.")

    if not found_layer:
        raise ValueError("Could not find first layer marker.")

    return cleaned


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

def cartesian_to_cxzb(
    x_cart,
    y_cart,
    z_cart,
    bed_center_x,
    bed_center_y,
    head_tilt_rad,
    nozzle_offset,
    prev_theta,
    theta_accum,
    c_sign=1.0,
    b_sign=-1.0,
):
    """
    Convert backtransformed Cartesian XYZ into machine-native CXZB coordinates.

    C = accumulated bed rotation angle, degrees
    X = radial distance from bed center, mm
    Z = vertical height, mm
    B = head tilt angle, degrees
    """

    dx = x_cart - bed_center_x
    dy = y_cart - bed_center_y

    radius = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    # Prevent C from jumping at +/-180 degrees.
    delta_theta = theta - prev_theta
    if delta_theta > np.pi:
        delta_theta -= 2 * np.pi
    if delta_theta < -np.pi:
        delta_theta += 2 * np.pi

    theta_accum += delta_theta

    # Nozzle pivot compensation.
    radius_comp = radius + np.sin(head_tilt_rad) * nozzle_offset
    z_comp = z_cart + (np.cos(head_tilt_rad) - 1.0) * nozzle_offset

    c_axis = c_sign * np.rad2deg(theta_accum)
    x_axis = radius_comp
    z_axis = z_comp
    b_axis = b_sign * np.rad2deg(head_tilt_rad)

    return c_axis, x_axis, z_axis, b_axis, theta, theta_accum

def auto_shift_cxzb_z(data_bt_string, desired_min_z):
    """
    Shift all generated CXZB Z values upward so the minimum machine Z
    is desired_min_z.

    This runs after CXZB conversion, so it shifts machine Z directly.
    """
    pattern_Z = rf'Z{NUM}'

    z_values = []

    for row in data_bt_string.splitlines():
        if row.startswith("G0") or row.startswith("G1"):
            z_match = re.search(pattern_Z, row)
            if z_match is not None:
                z_values.append(float(z_match.group(0).replace("Z", "")))

    if not z_values:
        print("  WARNING: No Z values found in generated CXZB G-code. No Z shift applied.")
        return data_bt_string

    min_generated_z = min(z_values)
    z_shift = desired_min_z - min_generated_z

    print(f"  Minimum generated machine Z before auto-shift: {min_generated_z:.3f} mm")

    if z_shift <= 0:
        print("  No auto Z shift needed.")
        return data_bt_string

    print(f"  Applying automatic machine Z shift of {z_shift:.3f} mm")

    def replace_z(match):
        old_z = float(match.group(0).replace("Z", ""))
        new_z = old_z + z_shift
        return f"Z{new_z:.5f}"

    shifted_lines = []
    for row in data_bt_string.splitlines():
        if row.startswith("G0") or row.startswith("G1"):
            row = re.sub(pattern_Z, replace_z, row, count=1)
        shifted_lines.append(row)

    return "\n".join(shifted_lines) + "\n"

def auto_shift_first_layer_extrusion_min_z(data_bt_string, desired_first_layer_z):
    """
    Shift all generated CXZB Z values so the LOWEST positive-extrusion
    move on the FIRST LAYER is at desired_first_layer_z.

    This is the correct shift for true conical backtransform:
        - ignores travel moves
        - ignores E-only moves
        - only looks at positive extrusion
        - only looks at the first layer
    """
    pattern_Z = rf'Z{NUM}'
    pattern_E = rf'E{NUM}'

    in_first_layer = False
    first_layer_done = False

    first_layer_extrusion_zs = []

    for row in data_bt_string.splitlines():
        stripped = row.strip()

        # Start first layer at first layer marker.
        if stripped.startswith("; CHANGE_LAYER") and not in_first_layer and not first_layer_done:
            in_first_layer = True
            continue

        # Stop when second layer starts.
        if stripped.startswith("; CHANGE_LAYER") and in_first_layer:
            first_layer_done = True
            break

        if not in_first_layer:
            continue

        if not row.startswith(("G0", "G1")):
            continue

        z_match = re.search(pattern_Z, row)
        e_match = re.search(pattern_E, row)

        if z_match is None or e_match is None:
            continue

        e_val = float(e_match.group(0).replace("E", ""))

        if e_val <= 0:
            continue

        z_val = float(z_match.group(0).replace("Z", ""))
        first_layer_extrusion_zs.append(z_val)

    if not first_layer_extrusion_zs:
        raise ValueError("Could not find any positive-extrusion Z moves on the first layer.")

    min_first_layer_extrusion_z = min(first_layer_extrusion_zs)
    max_first_layer_extrusion_z = max(first_layer_extrusion_zs)

    z_shift = desired_first_layer_z - min_first_layer_extrusion_z

    print(f"  First-layer positive-extrusion Z before shift:")
    print(f"    min: {min_first_layer_extrusion_z:.5f} mm")
    print(f"    max: {max_first_layer_extrusion_z:.5f} mm")
    print(f"  Applying Z shift of {z_shift:.5f} mm so lowest first-layer extrusion is {desired_first_layer_z:.5f} mm")

    def replace_z(match):
        old_z = float(match.group(0).replace("Z", ""))
        new_z = old_z + z_shift
        return f"Z{new_z:.5f}"

    shifted_lines = []

    for row in data_bt_string.splitlines():
        if row.startswith(("G0", "G1")):
            row = re.sub(pattern_Z, replace_z, row, count=1)
        shifted_lines.append(row)

    return "\n".join(shifted_lines) + "\n"

def lift_nonextrusion_moves_below_min_z(
    data_bt_string,
    min_allowed_z=0.0,
    travel_safe_z=0.2,
):
    """
    Lift non-extruding travel moves that dip below the bed.

    Important:
        - Positive-extrusion moves define the printed geometry.
        - We do NOT lift the whole file.
        - We only modify G0/G1 moves with no positive extrusion.
        - If a positive-extrusion move is below min_allowed_z, raise an error.
    """
    pattern_Z = rf'Z{NUM}'
    pattern_E = rf'E{NUM}'

    cleaned_lines = []
    lifted_count = 0
    lowest_lifted_z = None

    for line_number, row in enumerate(data_bt_string.splitlines(), start=1):
        if not row.startswith(("G0", "G1")):
            cleaned_lines.append(row)
            continue

        z_match = re.search(pattern_Z, row)

        if z_match is None:
            cleaned_lines.append(row)
            continue

        z_val = float(z_match.group(0).replace("Z", ""))

        if z_val >= min_allowed_z:
            cleaned_lines.append(row)
            continue

        e_match = re.search(pattern_E, row)

        is_positive_extrusion = False
        if e_match is not None:
            e_val = float(e_match.group(0).replace("E", ""))
            is_positive_extrusion = e_val > 0

        if is_positive_extrusion:
            raise ValueError(
                f"Positive-extrusion move below minimum Z after first-layer shift. "
                f"Line {line_number}: Z={z_val:.5f}, min={min_allowed_z:.5f}. "
                f"Row: {row}"
            )

        # Non-extruding travel/retract/wipe move below bed:
        # lift just this move to travel_safe_z.
        new_z = max(travel_safe_z, min_allowed_z)

        def replace_z(match):
            return f"Z{new_z:.5f}"

        row = re.sub(pattern_Z, replace_z, row, count=1)

        lifted_count += 1
        if lowest_lifted_z is None or z_val < lowest_lifted_z:
            lowest_lifted_z = z_val

        cleaned_lines.append(row)

    if lifted_count > 0:
        print(
            f"  Lifted {lifted_count} non-extrusion move(s) below Z{min_allowed_z:.5f} "
            f"to Z{travel_safe_z:.5f}."
        )
        print(f"  Lowest lifted non-extrusion Z was {lowest_lifted_z:.5f} mm.")
    else:
        print("  No below-bed non-extrusion moves needed lifting.")

    return "\n".join(cleaned_lines) + "\n"

def make_safety_limits():
    """
    Safety limits for the 4-axis printer.

    Machine axes:
        X = radial machine axis, mm
        Z = vertical machine axis, mm
        C = bed rotation, degrees
        B = head/nozzle rotation, degrees

    Current known limits:
        -150 <= X <= 150 mm
           0 <= Z <= 280 mm
        C is continuous / unlimited
        -120 <= B <= 120 degrees

    Shadow-box values are placeholders for now.
    Fill them in later when the real head/bed geometry is known.
    """
    return {
        # -----------------------------------------------------------
        # Real machine axis limits
        # -----------------------------------------------------------

        "min_x": -150.0,
        # Real printer limit: minimum X/radial position in mm.

        "max_x": 150.0,
        # Real printer limit: maximum X/radial position in mm.

        "min_z": 0.0,
        # Real printer limit: minimum Z in mm.

        "max_z": 280.0,
        # Real printer limit: maximum Z in mm.

        "min_b": -120.0,
        # Real printer limit: minimum B angle in degrees.

        "max_b": 120.0,
        # Real printer limit: maximum B angle in degrees.

        "min_c": None,
        # C axis is currently treated as unlimited.
        # TODO LATER:
        # If C is not actually continuous, replace None with the real minimum C angle.

        "max_c": None,
        # C axis is currently treated as unlimited.
        # TODO LATER:
        # If C is not actually continuous, replace None with the real maximum C angle.

        # -----------------------------------------------------------
        # Shadow-box placeholders
        # -----------------------------------------------------------
        # These are intentionally disabled for now.
        # Turn on later after measuring the print head / nozzle assembly.

        "enable_shadow_box_check": False,

        "bed_radius_mm": 150.0,
        # TODO REPLACE LATER:
        # Set this to the real usable rotating-bed radius.
        # For now this matches the +/-150 mm X limit.

        "head_shadow_radius_mm": 0.0,
        # TODO REPLACE LATER:
        # Approximate horizontal clearance radius of the head/nozzle assembly.
        # Example later: 25.0, 35.0, 50.0, etc.

        "head_shadow_z_below_nozzle_mm": 0.0,
        # TODO REPLACE LATER:
        # Distance below the nozzle tip occupied by anything on the head.
        # Usually this should stay 0.0 unless hardware protrudes below the nozzle.

        "head_shadow_z_above_nozzle_mm": 0.0,
        # TODO REPLACE LATER:
        # Vertical clearance needed above the nozzle for head/gantry geometry.

        "safety_margin_mm": 0.0,
        # TODO LATER:
        # Optional extra margin to shrink the safe X/Z envelope.
        # Example: 2.0 means stay 2 mm away from X/Z limits.
    }


def check_axis_limit(axis_name, value, min_value, max_value, tolerance=0.0, line_context=""):
    """
    Check one machine axis.

    If min_value or max_value is None, that side is treated as unlimited.
    """
    if min_value is not None and value < min_value - tolerance:
        raise ValueError(
            f"Safety limit error: {axis_name} below minimum. "
            f"{axis_name}={value:.3f}, min={min_value:.3f}, tolerance={tolerance:.3f}. "
            f"{line_context}"
        )

    if max_value is not None and value > max_value + tolerance:
        raise ValueError(
            f"Safety limit error: {axis_name} above maximum. "
            f"{axis_name}={value:.3f}, max={max_value:.3f}, tolerance={tolerance:.3f}. "
            f"{line_context}"
        )


def check_safety_limits(
    c_axis,
    x_axis,
    z_axis,
    b_axis,
    safety_limits,
    line_context="",
    check_min_z=True,
):
    """
    Check generated 4-axis machine coordinates against safety limits.

    This should be called after Cartesian XYZ has been converted into
    machine-native C/X/Z/B coordinates.
    """
    if safety_limits is None:
        return

    margin = safety_limits.get("safety_margin_mm", 0.0)

    min_x = safety_limits["min_x"] + margin if safety_limits["min_x"] is not None else None
    max_x = safety_limits["max_x"] - margin if safety_limits["max_x"] is not None else None
    min_z = safety_limits["min_z"] + margin if safety_limits["min_z"] is not None else None
    max_z = safety_limits["max_z"] - margin if safety_limits["max_z"] is not None else None

    tolerance_mm = safety_limits.get("limit_tolerance_mm", 0.0)
    tolerance_deg = safety_limits.get("limit_tolerance_deg", 0.0)

    # check_axis_limit("X", x_axis, min_x, max_x)
    # check_axis_limit("Z", z_axis, min_z, max_z)
    # check_axis_limit("B", b_axis, safety_limits["min_b"], safety_limits["max_b"])
    # check_axis_limit("C", c_axis, safety_limits["min_c"], safety_limits["max_c"])

    check_axis_limit("X", x_axis, min_x, max_x, tolerance=tolerance_mm, line_context=line_context)

    if check_min_z:
        check_axis_limit("Z", z_axis, min_z, max_z, tolerance=tolerance_mm, line_context=line_context)
    else:
        check_axis_limit("Z", z_axis, None, max_z, tolerance=tolerance_mm, line_context=line_context)

    check_axis_limit(
        "B",
        b_axis,
        safety_limits["min_b"],
        safety_limits["max_b"],
        tolerance=tolerance_deg,
        line_context=line_context,
    )

    check_axis_limit(
        "C",
        c_axis,
        safety_limits["min_c"],
        safety_limits["max_c"],
        tolerance=tolerance_deg,
        line_context=line_context,
    )

    # Optional rough shadow-box check.
    # This is a placeholder collision envelope, not a full CAD collision model.
    if safety_limits.get("enable_shadow_box_check", False):
        bed_radius = safety_limits["bed_radius_mm"]
        head_shadow_radius = safety_limits["head_shadow_radius_mm"]
        z_below = safety_limits["head_shadow_z_below_nozzle_mm"]
        z_above = safety_limits["head_shadow_z_above_nozzle_mm"]

        shadow_x_min = x_axis - head_shadow_radius
        shadow_x_max = x_axis + head_shadow_radius

        if shadow_x_min < safety_limits["min_x"] or shadow_x_max > safety_limits["max_x"]:
            raise ValueError(
                f"Shadow-box error: head shadow exceeds X travel. "
                f"shadow X range={shadow_x_min:.3f} to {shadow_x_max:.3f}. "
                f"{line_context}"
            )

        if abs(x_axis) + head_shadow_radius > bed_radius:
            raise ValueError(
                f"Shadow-box error: head shadow exceeds bed radius. "
                f"abs(X)+shadow={abs(x_axis) + head_shadow_radius:.3f}, "
                f"bed_radius={bed_radius:.3f}. "
                f"{line_context}"
            )

        shadow_z_min = z_axis - z_below
        shadow_z_max = z_axis + z_above

        if shadow_z_min < safety_limits["min_z"]:
            raise ValueError(
                f"Shadow-box error: head shadow below min Z. "
                f"shadow_z_min={shadow_z_min:.3f}. "
                f"{line_context}"
            )

        if shadow_z_max > safety_limits["max_z"]:
            raise ValueError(
                f"Shadow-box error: head shadow above max Z. "
                f"shadow_z_max={shadow_z_max:.3f}. "
                f"{line_context}"
            )


def print_safety_limits_summary(safety_limits):
    """
    Print active safety settings before conversion.
    """
    if safety_limits is None:
        print("  Safety limits: DISABLED")
        return

    print("  Safety limits: ENABLED")
    print(f"    X: {safety_limits['min_x']} to {safety_limits['max_x']} mm")
    print(f"    Z: {safety_limits['min_z']} to {safety_limits['max_z']} mm")
    print(f"    B: {safety_limits['min_b']} to {safety_limits['max_b']} deg")

    if safety_limits["min_c"] is None and safety_limits["max_c"] is None:
        print("    C: unlimited / continuous")
    else:
        print(f"    C: {safety_limits['min_c']} to {safety_limits['max_c']} deg")

    if safety_limits.get("enable_shadow_box_check", False):
        print("    Shadow box: ENABLED")
        print(f"      bed_radius_mm:                 {safety_limits['bed_radius_mm']}")
        print(f"      head_shadow_radius_mm:         {safety_limits['head_shadow_radius_mm']}")
        print(f"      head_shadow_z_below_nozzle_mm: {safety_limits['head_shadow_z_below_nozzle_mm']}")
        print(f"      head_shadow_z_above_nozzle_mm: {safety_limits['head_shadow_z_above_nozzle_mm']}")
    else:
        print("    Shadow box: DISABLED")

def validate_final_cxzb_gcode(data_bt_string, safety_limits):
    """
    Validate the final generated C/X/Z/B G-code AFTER auto Z shifting.

    This is where we should enforce the true Z minimum, because before
    auto_shift_cxzb_z(...), temporary negative Z values may exist.
    """
    if safety_limits is None:
        return

    pattern_C = rf'C{NUM}'
    pattern_X = rf'X{NUM}'
    pattern_Z = rf'Z{NUM}'
    pattern_B = rf'B{NUM}'

    checked_moves = 0

    for line_number, row in enumerate(data_bt_string.splitlines(), start=1):
        if not row.startswith(("G0", "G1")):
            continue

        c_match = re.search(pattern_C, row)
        x_match = re.search(pattern_X, row)
        z_match = re.search(pattern_Z, row)
        b_match = re.search(pattern_B, row)

        if x_match is None or z_match is None or b_match is None:
            continue

        c_axis = float(c_match.group(0).replace("C", "")) if c_match is not None else 0.0
        x_axis = float(x_match.group(0).replace("X", ""))
        z_axis = float(z_match.group(0).replace("Z", ""))
        b_axis = float(b_match.group(0).replace("B", ""))

        check_safety_limits(
            c_axis=c_axis,
            x_axis=x_axis,
            z_axis=z_axis,
            b_axis=b_axis,
            safety_limits=safety_limits,
            check_min_z=True,
            line_context=f"Final G-code line {line_number}: {row}",
        )

        checked_moves += 1

    print(f"  Final safety validation passed on {checked_moves} G0/G1 move(s).")

def backtransform_data(
    data,
    cone_type,
    cone_angle_deg,
    maximal_length,
    bed_center_x,
    bed_center_y,
    nozzle_offset,
    fixed_e=0.0275,
    use_fixed_e=False,
    c_sign=1.0,
    b_sign=-1.0,
    safety_limits=None,
    machine_z_lift=0.0,
    use_conical_z_backtransform=True,
):
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

    head_tilt_rad = c * cone_angle_rad

    # pattern_X = r'X[-0-9]+[.]?[0-9]*'
    # pattern_Y = r'Y[-0-9]+[.]?[0-9]*'
    # pattern_Z = r'Z[-0-9]+[.]?[0-9]*'
    # pattern_E = r'E[-0-9]+[.]?[0-9]*'
    pattern_G = r'\AG[01] '
    pattern_X = rf'X{NUM}'
    pattern_Y = rf'Y{NUM}'
    pattern_Z = rf'Z{NUM}'
    pattern_E = rf'E{NUM}'
    pattern_F = rf'F{NUM}'

    e_replacement = f'E{fixed_e}'

    new_data = []
    x_old, y_old = bed_center_x, bed_center_y
    x_new, y_new = bed_center_x, bed_center_y
    z_layer = 0.0
    update_x, update_y = False, False

    current_feedrate = None

    prev_theta = 0.0
    theta_accum = 0.0

    for row in data:
        g_match = re.search(pattern_G, row)
        if g_match is None:
            new_data.append(row)
            continue

        x_match = re.search(pattern_X, row)
        y_match = re.search(pattern_Y, row)
        z_match = re.search(pattern_Z, row)
        e_match = re.search(pattern_E, row)
        f_match = re.search(pattern_F, row)

        if f_match is not None:
            current_feedrate = float(f_match.group(0).replace('F', ''))

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

        # Compute backtransformed Z.
        r_vals = np.sqrt((x_vals - bed_center_x)**2 + (y_vals - bed_center_y)**2)

        if use_conical_z_backtransform:
            # Full conical/non-planar mode.
            z_vals = z_layer - c * tan_a * r_vals
        else:
            # Flat-bed first hardware test mode.
            # Keeps the slicer's layer Z flat so the first layer can actually print on the bed.
            z_vals = np.full_like(r_vals, z_layer)

        replacement_rows = ''

        command = g_match.group(0).strip()

        for j in range(num_segm):
            x_cart = x_vals[j + 1]
            y_cart = y_vals[j + 1]
            #z_cart = z_vals[j + 1]
            z_cart = z_vals[j + 1] + machine_z_lift

            c_axis, x_axis, z_axis, b_axis, prev_theta, theta_accum = cartesian_to_cxzb(
                x_cart=x_cart,
                y_cart=y_cart,
                z_cart=z_cart,
                bed_center_x=bed_center_x,
                bed_center_y=bed_center_y,
                head_tilt_rad=head_tilt_rad,
                nozzle_offset=nozzle_offset,
                prev_theta=prev_theta,
                theta_accum=theta_accum,
                c_sign=c_sign,
                b_sign=b_sign,
            )

            check_safety_limits(
                c_axis=c_axis,
                x_axis=x_axis,
                z_axis=z_axis,
                b_axis=b_axis,
                safety_limits=safety_limits,
                check_min_z=False,
                line_context=(
                    f"Pre-shift generated move: "
                    f"C{c_axis:.3f} X{x_axis:.3f} Z{z_axis:.3f} B{b_axis:.3f} | "
                    f"Original row: {row.strip()}"
                ),
            )

            new_line = (
                f"{command} "
                f"C{c_axis:.5f} "
                f"X{x_axis:.5f} "
                f"Z{z_axis:.5f} "
                f"B{b_axis:.5f}"
            )

            if e_match is not None:
                if use_fixed_e:
                    e_out = fixed_e
                else:
                    e_original = float(e_match.group(0).replace('E', ''))
                    e_out = e_original / num_segm
                new_line += f" E{e_out:.5f}"

            if current_feedrate is not None:
                new_line += f" F{current_feedrate:.1f}"

            replacement_rows += new_line + "\n"

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

def remove_preprint_motion_moves(data):
    """
    Remove G0/G1 motion moves before the actual model print starts.

    Why:
        The fixed header should handle startup/homing/purge behavior.
        The slicer body can still contain pre-print travel/purge moves like:
            G1 X-28.5 F18000

        Those are not model geometry and should not be converted into C/X/Z/B.

    Start of real print is detected using common slicer layer markers:
        ; CHANGE_LAYER
        ; Z_HEIGHT:
        ;LAYER_CHANGE
        ; layer num/
    """
    cleaned = []
    in_print = False
    removed_motion_lines = 0

    pattern_G = r'\AG[01] '

    for row in data:
        stripped = row.strip()

        if (
            '; CHANGE_LAYER' in row
            or '; Z_HEIGHT:' in row
            or ';LAYER_CHANGE' in row
            or '; layer num/' in row
        ):
            in_print = True
            cleaned.append(row)
            continue

        g_match = re.search(pattern_G, row)

        # Before the model starts, remove only motion lines.
        # Keep comments, temperature commands, fan commands, etc.
        if not in_print and g_match is not None:
            removed_motion_lines += 1
            continue

        cleaned.append(row)

    print(f"  Removed {removed_motion_lines} pre-print G0/G1 motion line(s).")
    return cleaned

def remove_out_of_bounds_nonprint_moves(
    data,
    cartesian_min_x=0.0,
    cartesian_max_x=256.0,
    cartesian_min_y=0.0,
    cartesian_max_y=256.0,
):
    """
    Remove non-extrusion travel/utility G0/G1 moves that go outside
    the normal Cartesian slicer bed area.

    Why:
        Some slicers include machine utility moves in the body, for example:
            G1 X-28.5 F18000

        That is not model geometry. If we backtransform it, it can create
        invalid 4-axis X/C/Z/B moves.

    This only removes moves with no positive extrusion.
    Real print moves with E > 0 are kept.
    """
    cleaned = []
    removed_lines = 0

    pattern_G = r'\AG[01] '
    pattern_X = rf'X{NUM}'
    pattern_Y = rf'Y{NUM}'
    pattern_E = rf'E{NUM}'

    current_x = None
    current_y = None

    for row in data:
        g_match = re.search(pattern_G, row)

        if g_match is None:
            cleaned.append(row)
            continue

        x_match = re.search(pattern_X, row)
        y_match = re.search(pattern_Y, row)
        e_match = re.search(pattern_E, row)

        # Determine whether this is a positive-extrusion print move.
        is_positive_extrusion = False
        if e_match is not None:
            e_val = float(e_match.group(0).replace("E", ""))
            is_positive_extrusion = e_val > 0

        # Candidate modal position after this row.
        candidate_x = current_x
        candidate_y = current_y

        if x_match is not None:
            candidate_x = float(x_match.group(0).replace("X", ""))

        if y_match is not None:
            candidate_y = float(y_match.group(0).replace("Y", ""))

        # Only remove non-print moves.
        if not is_positive_extrusion:
            out_of_bounds = False

            if candidate_x is not None:
                if candidate_x < cartesian_min_x or candidate_x > cartesian_max_x:
                    out_of_bounds = True

            if candidate_y is not None:
                if candidate_y < cartesian_min_y or candidate_y > cartesian_max_y:
                    out_of_bounds = True

            # should just remove machine/slicer utility moves - not benchy geometry :3
            if out_of_bounds:
                removed_lines += 1
                print(f"  Commented out out-of-bounds non-print move: {row.strip()}")

                cleaned.append(
                    f"; REMOVED_OUT_OF_BOUNDS_NONPRINT_MOVE: {row.strip()}\n"
                )
                continue

        # Keep row and update modal XY state.
        if x_match is not None:
            current_x = candidate_x

        if y_match is not None:
            current_y = candidate_y

        cleaned.append(row)

    print(f"  Commented out {removed_lines} out-of-bounds non-print move(s).")
    return cleaned

def clean_final_gcode_for_4axis_first_print(
    gcode_string,
    max_feedrate_print=1200.0,
    max_feedrate_travel=3000.0,
    comment_removed_lines=True,
):
    """
    Final cleanup for first 4-axis hardware tests.

    Does three things:
        1. Comments out unsupported arc/plane commands: G2, G3, G17
        2. Comments out Bambu/Orca-specific commands
        3. Clamps high feedrates to safer first-test values

    max_feedrate_print:
        Max feedrate for extrusion moves, mm/min.
        Example: 1200 = 20 mm/s

    max_feedrate_travel:
        Max feedrate for non-extrusion travel moves, mm/min.
        Example: 3000 = 50 mm/s
    """
    pattern_F = rf'F{NUM}'
    pattern_E = rf'E{NUM}'

    # Commands from the uploaded file that are Bambu/Orca-specific or suspicious
    # for a custom 4-axis controller.
    blocked_commands = {
        # Arc / plane commands that your current converter does not backtransform.
        "G2",
        "G3",
        "G17",

        # Bambu / Orca / printer-specific commands found in the generated file.
        "G392",
        "M73",
        "M73.2",
        "M201.2",
        "M620",
        "M621",
        "M622",
        "M623",
        "M971",
        "M981",
        "M991",
        "M1002",
        "M1003",
        "M1006",
        "T255",

        # Extra cleanup for custom 4-axis first print.
        "M17",
        "M18",
        "M220",

        # accelration changes - let printer/controller use its own conservative acceleration settings
        "M17",
        "M18",
        "M220",
        "M204", #acceleration
        "M106", #fan command?
    }

    cleaned_lines = []

    removed_counts = {}
    clamped_feedrate_count = 0

    for row in gcode_string.splitlines():
        stripped = row.strip()

        if stripped == "" or stripped.startswith(";"):
            cleaned_lines.append(row)
            continue

        command = stripped.split()[0]

        if command in blocked_commands:
            removed_counts[command] = removed_counts.get(command, 0) + 1

            if comment_removed_lines:
                cleaned_lines.append(f"; REMOVED_FOR_4AXIS_FIRST_PRINT: {row}")
            continue

        # Clamp feedrates on G0/G1 moves.
        if command in {"G0", "G1"}:
            f_match = re.search(pattern_F, row)
            e_match = re.search(pattern_E, row)

            if f_match is not None:
                old_f = float(f_match.group(0).replace("F", ""))

                is_print_move = False
                if e_match is not None:
                    e_val = float(e_match.group(0).replace("E", ""))
                    is_print_move = e_val > 0

                if is_print_move:
                    new_f = min(old_f, max_feedrate_print)
                else:
                    new_f = min(old_f, max_feedrate_travel)

                if new_f < old_f:
                    clamped_feedrate_count += 1
                    row = re.sub(pattern_F, f"F{new_f:.1f}", row, count=1)

        cleaned_lines.append(row)

    print("  Final 4-axis cleanup summary:")

    if removed_counts:
        for command in sorted(removed_counts.keys()):
            print(f"    Commented out {removed_counts[command]} {command} command(s).")
    else:
        print("    No blocked Bambu/arc commands found.")

    print(f"    Clamped feedrate on {clamped_feedrate_count} G0/G1 move(s).")
    print(f"    Max print feedrate:  {max_feedrate_print:.1f} mm/min")
    print(f"    Max travel feedrate: {max_feedrate_travel:.1f} mm/min")

    return "\n".join(cleaned_lines) + "\n"

def auto_shift_cxzb_z_to_first_moving_extrusion(data_bt_string, desired_first_print_z):
    """
    Shift all generated CXZB Z values so the first real moving positive-extrusion
    G0/G1 move starts at desired_first_print_z.

    Ignores E-only prime/unretract moves like:
        G1 E.8 F1200

    Finds the first line like:
        G1 C... X... Z... B... Epositive ... F...
    """
    pattern_C = rf'C{NUM}'
    pattern_X = rf'X{NUM}'
    pattern_Z = rf'Z{NUM}'
    pattern_B = rf'B{NUM}'
    pattern_E = rf'E{NUM}'

    current_z = None
    first_print_z = None
    first_print_row = None

    for row in data_bt_string.splitlines():
        if not row.startswith(("G0", "G1")):
            continue

        z_match = re.search(pattern_Z, row)
        e_match = re.search(pattern_E, row)

        if z_match is not None:
            current_z = float(z_match.group(0).replace("Z", ""))

        if e_match is None:
            continue

        e_val = float(e_match.group(0).replace("E", ""))

        if e_val <= 0:
            continue

        has_motion_axis = (
            re.search(pattern_C, row) is not None
            or re.search(pattern_X, row) is not None
            or re.search(pattern_Z, row) is not None
            or re.search(pattern_B, row) is not None
        )

        if not has_motion_axis:
            continue

        if current_z is None:
            continue

        first_print_z = current_z
        first_print_row = row
        break

    if first_print_z is None:
        raise ValueError("Could not find first moving positive-extrusion G0/G1 move with known Z.")

    z_shift = desired_first_print_z - first_print_z

    print(f"  First moving positive-extrusion machine Z before shift: {first_print_z:.5f} mm")
    print(f"  First moving positive-extrusion row: {first_print_row.strip()}")
    print(f"  Applying Z shift of {z_shift:.5f} mm so first moving extrusion starts at {desired_first_print_z:.5f} mm")

    def replace_z(match):
        old_z = float(match.group(0).replace("Z", ""))
        new_z = old_z + z_shift
        return f"Z{new_z:.5f}"

    shifted_lines = []

    for row in data_bt_string.splitlines():
        if row.startswith(("G0", "G1")):
            row = re.sub(pattern_Z, replace_z, row, count=1)
        shifted_lines.append(row)

    return "\n".join(shifted_lines) + "\n"


def keep_only_simple_4axis_print_gcode(gcode_string, keep_layer_comments=True):
    """
    Keep only simple custom 4-axis print G-code.

    Keeps:
        - simple start/end commands
        - real G0/G1 C/X/Z/B machine moves
        - optional layer comments

    Removes:
        - Bambu comments
        - REMOVED_FOR_4AXIS_FIRST_PRINT audit comments
        - E-only moves like G1 E.8 or G1 E-.8
        - feedrate-only moves like G1 F1200
        - WIPE_START / WIPE_END comments
        - OBJECT_ID / FEATURE / LINE_WIDTH comments
    """
    simple_allowed_commands = {
        "G90",
        "G91",
        "G92",
        "G28",
        "M83",
        "M104",
        "M109",
        "M140",
        "M190",
        "M84",
    }

    cleaned_lines = []
    removed_count = 0

    for row in gcode_string.splitlines():
        stripped = row.strip()

        if stripped == "":
            continue

        # Keep only useful comments.
        if stripped.startswith(";"):
            if (
                "SIMPLE 4-AXIS" in stripped
                or "BEGIN GENERATED 4-AXIS TOOLPATH" in stripped
                or "END GENERATED 4-AXIS TOOLPATH" in stripped
                or "END FILE" in stripped
            ):
                cleaned_lines.append(row)
                continue

            if keep_layer_comments and (
                stripped.startswith("; CHANGE_LAYER")
                or stripped.startswith("; Z_HEIGHT:")
                or stripped.startswith("; LAYER_HEIGHT:")
            ):
                cleaned_lines.append(row)
                continue

            removed_count += 1
            continue

        command = stripped.split()[0]

        # Keep simple start/end commands.
        # if command in simple_allowed_commands:
        #     cleaned_lines.append(row)
        #     continue

        # Only G0/G1 model motion from here.
        if command not in {"G0", "G1"}:
            removed_count += 1
            continue

        has_c = re.search(rf'(^|\s)C{NUM}', row) is not None
        has_x = re.search(rf'(^|\s)X{NUM}', row) is not None
        has_z = re.search(rf'(^|\s)Z{NUM}', row) is not None
        has_b = re.search(rf'(^|\s)B{NUM}', row) is not None

        has_machine_motion_axis = has_c or has_x or has_z or has_b

        if has_machine_motion_axis:
            cleaned_lines.append(row)
            continue

        # Remove E-only moves and feedrate-only moves.
        removed_count += 1

    print(f"  Model-only final cleanup removed {removed_count} line(s).")

    return "\n".join(cleaned_lines) + "\n"

def backtransform_file(
    path,
    output_dir,
    cone_type,
    cone_angle_deg,
    maximal_length,
    x_shift,
    y_shift,
    z_desired,
    fixed_header_path,
    bed_center_x=None,
    bed_center_y=None,
    nozzle_offset=0.0,
    fixed_e=0.0275,
    use_fixed_e=False,
    c_sign=1.0,
    b_sign=-1.0,
    safety_limits=None,
    machine_z_lift=0.0,
    use_conical_z_backtransform=True,
):
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

    print("Removing pre-print motion moves...")
    body = remove_preprint_motion_moves(body)

    print("Removing everything before first layer marker...")
    body = remove_everything_before_first_layer_marker(body)

    print("Removing out-of-bounds non-print moves...")
    body = remove_out_of_bounds_nonprint_moves(
        body,
        cartesian_min_x=0.0,
        cartesian_max_x=256.0,
        cartesian_min_y=0.0,
        cartesian_max_y=256.0,
    )

    # fixed_header = read_fixed_header(fixed_header_path)

    fixed_header = make_simple_start_gcode(
        nozzle_temp=240,
        bed_temp=60,
    )

    fixed_footer = make_simple_end_gcode(
        end_z_lift=10.0,
    )

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

    print_safety_limits_summary(safety_limits)

    data_bt = backtransform_data(
        body,
        cone_type,
        cone_angle_deg,
        maximal_length,
        bed_center_x,
        bed_center_y,
        nozzle_offset=nozzle_offset,
        fixed_e=fixed_e,
        use_fixed_e=use_fixed_e,
        c_sign=c_sign,
        b_sign=b_sign,
        safety_limits=safety_limits,
        machine_z_lift=machine_z_lift,
        use_conical_z_backtransform=use_conical_z_backtransform,
    )

    data_bt_string = ''.join(data_bt)

    print("Cleaning final G-code for first 4-axis hardware test...")
    data_bt_string = clean_final_gcode_for_4axis_first_print(
        data_bt_string,
        max_feedrate_print=1200.0,
        max_feedrate_travel=3000.0,
        comment_removed_lines=False,
    )

    print("Keeping only simple model-print G-code...")
    data_bt_string = keep_only_simple_4axis_print_gcode(
        data_bt_string,
        keep_layer_comments=True,
    )

    if use_conical_z_backtransform:
        print("Conical Z mode: shifting so lowest FIRST-LAYER EXTRUSION equals desired first-layer height...")
        data_bt_string = auto_shift_first_layer_extrusion_min_z(
            data_bt_string,
            desired_first_layer_z=z_desired,
        )
    else:
        print("Flat test mode: shifting so first moving extrusion starts at desired first-layer height...")
        data_bt_string = auto_shift_cxzb_z_to_first_moving_extrusion(
            data_bt_string,
            desired_first_print_z=z_desired,
        )
    
    # print("Lifting any below-bed non-extrusion travel moves...")
    # data_bt_string = lift_nonextrusion_moves_below_min_z(
    #     data_bt_string,
    #     min_allowed_z=safety_limits["min_z"],
    #     travel_safe_z=z_desired,
    # )

    validate_final_cxzb_gcode(data_bt_string, safety_limits)

    #data_bt = [row + ' \n' for row in data_bt_string.split('\n')]

    #print("Applying Z/XY translation...")
    #data_bt = translate_data(data_bt, x_shift, y_shift, z_desired)
    #data_bt_string = ''.join(data_bt)

    # Prepend fixed header (ensure it ends with a newline before body)
    # if not fixed_header.endswith('\n'):
    #     fixed_header += '\n'
    # final_output = fixed_header + data_bt_string

    # Add simple start/end G-code around generated body.
    if not fixed_header.endswith('\n'):
        fixed_header += '\n'

    if not fixed_footer.endswith('\n'):
        fixed_footer += '\n'

    final_output = fixed_header + data_bt_string + fixed_footer

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    file_name = f"{name}_bt_final.gcode"
    output_path = os.path.join(output_dir, file_name)

    with open(output_path, 'w+', encoding='utf-8', newline='\n') as f_out:
        f_out.write(final_output)

    end = time.time()
    print(f"Done! GCode generated in {end - start:.1f}s, saved to:\n  {output_path}")


# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------

file_path           = r"C:\Users\canca\Documents\Conical Slicer Repo\ConicalSlicer\SlicedTransformedGcode\Safe_Polar_Side Dogbone_30deg_transformed_PLA_49m30s.gcode"
dir_backtransformed = r"C:\Users\canca\Documents\Conical Slicer Repo\ConicalSlicer\DeformedGcode"
fixed_header_path   = FIXED_HEADER_PATH   # path to HEADERBLOCKSTART.txt

transformation_type = 'outward'   # must match Cartesian_Transformation_STL.py
cone_angle_degrees  =  30         # must match Cartesian_Transformation_STL.py exactly

max_length = 2.0   # max segment length in mm (smaller = smoother curves)

# Bambu A1 bed center. Set to None to auto-detect from G-code bounding box.
override_bed_center_x = 128.0
override_bed_center_y = 128.0

delta_x  = 0.0   # XY shift after backtransform (leave 0 for Bambu)
delta_y  = 0.0
z_height = 0.2   # desired minimum Z = first layer height
machine_z_lift = 0.0   # lift compensated CXZB Z so it stays above min_z

fixed_extrusion = 0.0275  # constant E value applied to every extrusion move
use_fixed_extrusion = False  # False = preserve slicer E values, True = force fixed E

nozzle_offset = 43.5  # mm, replace with real value
b_sign = -1.0         # flip to +1 if B tilts wrong way
c_sign = 1.0          # flip to -1 if bed rotates opposite direction

# min_radius = 0.0
# max_radius = 150.0    # replace with your machine limit
# min_z = 0.0
# max_z = 250.0         # replace with your machine limit
# min_b = -45.0
# max_b = 45.0

# ---------------------------------------------------------------
# Safety limits / shadow box
# ---------------------------------------------------------------

safety_limits = make_safety_limits()

# Real known 4-axis printer limits:
safety_limits["min_x"] = -150.0
safety_limits["max_x"] = 150.0

# Machine Z can be negative because B-axis/nozzle-angle compensation
# can place the pivot/axis below Z0 while the nozzle tip is still correct.
# During conical debugging, do not enforce a lower machine-Z clamp.
safety_limits["min_z"] = None
safety_limits["max_z"] = 280.0

safety_limits["min_b"] = -120.0
safety_limits["max_b"] = 120.0

# C is continuous / unlimited.
safety_limits["min_c"] = None
safety_limits["max_c"] = None

# Optional extra safety margin.
# Keep 0.0 while debugging. Later, use something like 1.0 or 2.0 mm.
safety_limits["safety_margin_mm"] = 0.0

safety_limits["limit_tolerance_mm"] = 0.25
safety_limits["limit_tolerance_deg"] = 0.001

# Shadow box is OFF until you measure the head/nozzle assembly.
safety_limits["enable_shadow_box_check"] = False

use_conical_z_backtransform = True  # False = flat-bed first hardware print test #should be True for real conical backtransform

# TODO REPLACE LATER:
# Fill these in when you know the real print head and bed geometry.
safety_limits["bed_radius_mm"] = 150.0
safety_limits["head_shadow_radius_mm"] = 0.0
safety_limits["head_shadow_z_below_nozzle_mm"] = 0.0
safety_limits["head_shadow_z_above_nozzle_mm"] = 0.0

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
    nozzle_offset     = nozzle_offset,
    fixed_e           = fixed_extrusion,
    use_fixed_e       = use_fixed_extrusion,
    c_sign            = c_sign,
    b_sign            = b_sign,
    safety_limits     = safety_limits,
    machine_z_lift    = machine_z_lift,
    use_conical_z_backtransform = use_conical_z_backtransform,
)