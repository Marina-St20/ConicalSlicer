import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DIAGNOSTIC_FILE = Path("extrusion_diagnostics.txt")

# Only affects the overview plots.
# The detailed cross-section keeps more points.
OVERVIEW_SAMPLE_STEP = 20

# Width of the angular slice used for the radius-versus-Z cross-section.
# Smaller values produce a cleaner cross-section but include fewer points.
CROSS_SECTION_HALF_WIDTH_DEG = 2.0

# Select the direction of the cross-section:
#   0°   = positive X direction
#   90°  = positive Y direction
#   180° = negative X direction
#   270° = negative Y direction
CROSS_SECTION_ANGLE_DEG = 0.0

# Approximate layer grouping tolerance in millimeters.
# Increase slightly if one layer gets split into multiple groups.
LAYER_Z_TOLERANCE = 0.03

FRONT_SLICE_HALF_WIDTH_MM = 0.25


NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"

BED_CENTER_X = 128.0
BED_CENTER_Y = 128.0


def extract_value(line: str, name: str) -> float | None:
    """Extract a numeric value written as name=number."""
    pattern = rf"\b{re.escape(name)}=({NUMBER_PATTERN})"
    match = re.search(pattern, line)

    if match is None:
        return None

    return float(match.group(1))


def load_diagnostics(path: Path):
    x_values = []
    y_values = []
    z_values = []
    e_values = []
    radius_values = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            x = extract_value(line, "x")
            y = extract_value(line, "y")
            z = extract_value(line, "z")
            radius = extract_value(line, "radius")
            e_per_mm = extract_value(line, "move_e_per_mm")

            if x is None or y is None or z is None:
                continue

            # Recalculate radius if it is missing.
            if radius is None:
                radius = np.hypot(x, y)

            x_values.append(x)
            y_values.append(y)
            z_values.append(z)
            radius_values.append(radius)
            e_values.append(
                np.nan if e_per_mm is None else e_per_mm
            )

    if not x_values:
        raise ValueError(
            "No XYZ points were found. Each diagnostic line must contain "
            "x=..., y=..., and z=...."
        )

    return (
        np.asarray(x_values, dtype=float),
        np.asarray(y_values, dtype=float),
        np.asarray(z_values, dtype=float),
        np.asarray(radius_values, dtype=float),
        np.asarray(e_values, dtype=float),
    )


def wrapped_angle_difference_deg(angle, target):
    """
    Return the smallest signed angular difference in degrees.

    Result is in the range [-180, 180).
    """
    return (angle - target + 180.0) % 360.0 - 180.0


def select_radial_cross_section(
    x,
    y,
    z,
    radius,
    e_per_mm,
    target_angle_deg,
    half_width_deg,
):
    """
    Select points near one radial direction relative to the cone center.
    """

    # Coordinates relative to the actual cone/bed center.
    relative_x = x - BED_CENTER_X
    relative_y = y - BED_CENTER_Y

    theta_deg = np.degrees(
        np.arctan2(relative_y, relative_x)
    )

    angle_difference = wrapped_angle_difference_deg(
        theta_deg,
        target_angle_deg,
    )

    mask = np.abs(angle_difference) <= half_width_deg

    if not np.any(mask):
        available_min = np.min(theta_deg)
        available_max = np.max(theta_deg)

        raise ValueError(
            f"No points found near {target_angle_deg:.1f}° "
            f"± {half_width_deg:.1f}°. "
            f"Available point angles range from "
            f"{available_min:.1f}° to {available_max:.1f}°."
        )

    # Recalculate radius from the correct center rather than trusting
    # an older diagnostic value.
    corrected_radius = np.hypot(
        relative_x,
        relative_y,
    )

    return (
        relative_x[mask],
        relative_y[mask],
        z[mask],
        corrected_radius[mask],
        e_per_mm[mask],
        theta_deg[mask],
    )


def estimate_layer_ids(z, tolerance):
    """
    Group nearby Z values into approximate layers.

    This is only a diagnostic grouping. It does not assume that every
    nonplanar layer has one perfectly constant Z value.
    """
    order = np.argsort(z)
    sorted_z = z[order]

    sorted_layer_ids = np.zeros(len(z), dtype=int)

    current_layer = 0
    running_layer_z = sorted_z[0]
    points_in_layer = 1

    for index in range(1, len(sorted_z)):
        z_value = sorted_z[index]

        if abs(z_value - running_layer_z) > tolerance:
            current_layer += 1
            running_layer_z = z_value
            points_in_layer = 1
        else:
            points_in_layer += 1
            running_layer_z += (
                z_value - running_layer_z
            ) / points_in_layer

        sorted_layer_ids[index] = current_layer

    layer_ids = np.empty_like(sorted_layer_ids)
    layer_ids[order] = sorted_layer_ids

    return layer_ids


def plot_cross_section(radius, z, e_per_mm):
    """
    Main diagnostic plot.

    Radius is horizontal and Cartesian Z is vertical, making adjacent
    conical layers much easier to inspect.
    """
    figure, axes = plt.subplots(figsize=(13, 8))

    valid_e = np.isfinite(e_per_mm)

    if np.any(valid_e):
        points = axes.scatter(
            radius[valid_e],
            z[valid_e],
            c=e_per_mm[valid_e],
            s=8,
            alpha=0.8,
        )

        colorbar = figure.colorbar(points, ax=axes)
        colorbar.set_label("Extrusion per millimeter")
    else:
        axes.scatter(
            radius,
            z,
            s=8,
            alpha=0.8,
        )

    axes.set_xlabel("Radius from cone center (mm)")
    axes.set_ylabel("Cartesian Z (mm)")
    axes.set_title(
        "Radial Cross-Section of Backtransformed Toolpath\n"
        f"Direction: {CROSS_SECTION_ANGLE_DEG:.1f}° "
        f"± {CROSS_SECTION_HALF_WIDTH_DEG:.1f}°"
    )

    axes.grid(True)
    axes.set_aspect("auto")

    figure.tight_layout()
    plt.show()


def plot_cross_section_by_layer(radius, z):
    """
    Plot approximate layers as individual curves.

    This makes radial increases in layer spacing easier to identify.
    """
    layer_ids = estimate_layer_ids(
        z,
        tolerance=LAYER_Z_TOLERANCE,
    )

    figure, axes = plt.subplots(figsize=(13, 8))

    unique_layers = np.unique(layer_ids)

    for layer_id in unique_layers:
        mask = layer_ids == layer_id

        if np.count_nonzero(mask) < 2:
            continue

        layer_radius = radius[mask]
        layer_z = z[mask]

        order = np.argsort(layer_radius)

        axes.plot(
            layer_radius[order],
            layer_z[order],
            linewidth=0.8,
            alpha=0.7,
        )

    axes.set_xlabel("Radius from cone center (mm)")
    axes.set_ylabel("Cartesian Z (mm)")
    axes.set_title(
        "Approximate Adjacent Layers in Radial Cross-Section"
    )
    axes.grid(True)

    figure.tight_layout()
    plt.show()


def plot_local_layer_spacing(radius, z):
    """
    Estimate vertical separation between neighboring Z samples at similar radii.

    This is not the true cone-normal spacing, but it is useful for detecting
    whether visible vertical separation grows with radius.
    """
    radial_bin_width = 0.25

    min_radius = np.min(radius)
    max_radius = np.max(radius)

    bin_edges = np.arange(
        min_radius,
        max_radius + radial_bin_width,
        radial_bin_width,
    )

    spacing_radius = []
    median_spacing = []
    maximum_spacing = []

    for left_edge, right_edge in zip(
        bin_edges[:-1],
        bin_edges[1:],
    ):
        mask = (
            (radius >= left_edge)
            & (radius < right_edge)
        )

        z_bin = np.sort(np.unique(np.round(z[mask], decimals=4)))

        if len(z_bin) < 2:
            continue

        differences = np.diff(z_bin)

        # Remove tiny differences that are likely points from the same path.
        differences = differences[
            differences > LAYER_Z_TOLERANCE
        ]

        if len(differences) == 0:
            continue

        spacing_radius.append(
            (left_edge + right_edge) / 2.0
        )
        median_spacing.append(
            np.median(differences)
        )
        maximum_spacing.append(
            np.max(differences)
        )

    if not spacing_radius:
        print(
            "Could not estimate layer spacing. Try increasing "
            "CROSS_SECTION_HALF_WIDTH_DEG or decreasing "
            "LAYER_Z_TOLERANCE."
        )
        return

    figure, axes = plt.subplots(figsize=(12, 7))

    axes.plot(
        spacing_radius,
        median_spacing,
        marker=".",
        linewidth=1,
        label="Median apparent spacing",
    )

    axes.plot(
        spacing_radius,
        maximum_spacing,
        linewidth=0.8,
        alpha=0.6,
        label="Maximum apparent spacing",
    )

    axes.set_xlabel("Radius from cone center (mm)")
    axes.set_ylabel("Apparent vertical spacing (mm)")
    axes.set_title(
        "Apparent Layer Spacing Versus Radius"
    )
    axes.grid(True)
    axes.legend()

    figure.tight_layout()
    plt.show()


def plot_xy_overview(x, y):
    step = max(1, OVERVIEW_SAMPLE_STEP)

    relative_x = x - BED_CENTER_X
    relative_y = y - BED_CENTER_Y

    figure, axes = plt.subplots(figsize=(8, 8))

    axes.scatter(
        relative_x[::step],
        relative_y[::step],
        s=2,
        alpha=0.5,
    )

    max_extent = max(
        np.max(np.abs(relative_x)),
        np.max(np.abs(relative_y)),
        1.0,
    )

    angle_rad = np.radians(CROSS_SECTION_ANGLE_DEG)

    direction_x = max_extent * np.cos(angle_rad)
    direction_y = max_extent * np.sin(angle_rad)

    axes.plot(
        [0.0, direction_x],
        [0.0, direction_y],
        linewidth=2,
        label="Selected cross-section",
    )

    axes.set_xlabel("X relative to cone center (mm)")
    axes.set_ylabel("Y relative to cone center (mm)")
    axes.set_title("Top-Down Toolpath Overview")
    axes.set_aspect("equal", adjustable="box")
    axes.grid(True)
    axes.legend()

    figure.tight_layout()
    plt.show()

def plot_front_view(x, y, z):
    """
    Forward-facing toolpath view.

    Horizontal axis:
        X relative to the cone center

    Vertical axis:
        Z height
    """
    relative_x = x - BED_CENTER_X

    figure, axes = plt.subplots(figsize=(14, 9))

    axes.scatter(
        relative_x,
        z,
        s=1,
        alpha=0.35,
        rasterized=True,
    )

    axes.set_xlabel("X relative to cone center (mm)")
    axes.set_ylabel("Z height (mm)")
    axes.set_title("Front View of Backtransformed Toolpath")
    axes.set_aspect("equal", adjustable="box")
    axes.grid(True)

    figure.tight_layout()
    plt.show()

def plot_front_slice(x, y, z):
    """
    Forward-facing X-Z cross-section through the cone center.

    Keeps points whose Y coordinate is close to the cone center.
    """
    relative_x = x - BED_CENTER_X
    relative_y = y - BED_CENTER_Y

    mask = np.abs(relative_y) <= FRONT_SLICE_HALF_WIDTH_MM

    if not np.any(mask):
        raise ValueError(
            "No points found in the front slice. Increase "
            "FRONT_SLICE_HALF_WIDTH_MM."
        )

    slice_x = relative_x[mask]
    slice_z = z[mask]

    print(
        f"Front slice contains {len(slice_x):,} points "
        f"within ±{FRONT_SLICE_HALF_WIDTH_MM:.3f} mm of center Y."
    )

    figure, axes = plt.subplots(figsize=(14, 9))

    axes.scatter(
        slice_x,
        slice_z,
        s=4,
        alpha=0.7,
        rasterized=True,
    )

    axes.set_xlabel("X relative to cone center (mm)")
    axes.set_ylabel("Z height (mm)")
    axes.set_title(
        "Front-Facing X-Z Toolpath Cross-Section\n"
        f"Y slice: ±{FRONT_SLICE_HALF_WIDTH_MM:.3f} mm"
    )

    axes.set_aspect("equal", adjustable="box")
    axes.grid(True)

    figure.tight_layout()
    plt.show()

def main():
    if not DIAGNOSTIC_FILE.exists():
        raise FileNotFoundError(
            f"Could not find: {DIAGNOSTIC_FILE.resolve()}"
        )

    x, y, z, radius, e_per_mm = load_diagnostics(
        DIAGNOSTIC_FILE
    )

    print(f"Loaded {len(x):,} total points.")
    print(
        f"Full radius range: "
        f"{radius.min():.3f} to {radius.max():.3f} mm"
    )
    print(
        f"Full Z range: "
        f"{z.min():.3f} to {z.max():.3f} mm"
    )

    (
        section_x,
        section_y,
        section_z,
        section_radius,
        section_e,
        section_theta,
    ) = select_radial_cross_section(
        x=x,
        y=y,
        z=z,
        radius=radius,
        e_per_mm=e_per_mm,
        target_angle_deg=CROSS_SECTION_ANGLE_DEG,
        half_width_deg=CROSS_SECTION_HALF_WIDTH_DEG,
    )

    print(
        f"Selected {len(section_x):,} cross-section points "
        f"near {CROSS_SECTION_ANGLE_DEG:.1f}°."
    )

    plot_xy_overview(x, y)

    plot_front_view(x, y, z)

    plot_front_slice(x, y, z)

    plot_cross_section(
        section_radius,
        section_z,
        section_e,
    )

    plot_cross_section_by_layer(
        section_radius,
        section_z,
    )

    plot_local_layer_spacing(
        section_radius,
        section_z,
    )


if __name__ == "__main__":
    main()