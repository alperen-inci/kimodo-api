# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coordinate system conversions between Z-up (lzyx) and Kimodo Y-up.

Kimodo generates in Y-up (right-handed):
    Right   = +X
    Up      = +Y
    Forward = +Z

DART / AMASS / Blender convention (lzyx, left-handed, Z-up):
    Right   = +X
    Forward = +Y
    Up      = +Z

The AMASS export transform (from kimodo.exports.smplx) is:

    M = [[-1, 0, 0],
         [ 0, 0, 1],
         [ 0, 1, 0]]

    v_lzyx = M @ v_yup       (positions)
    R_lzyx = M @ R_yup @ M^T (rotations)

This module provides the inverse (lzyx -> Y-up) for input conversion.
Since M is an involution (M @ M = I), the inverse is M itself.
"""

import numpy as np

# Combined transform: Y-up -> Z-up with 180° rotation around Z.
# This matches kimodo.exports.smplx.get_amass_parameters z_up=True.
M = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

# Inverse is the same matrix (involution).
M_INV = M.copy()


def lzyx_pos_to_yup(pos: np.ndarray) -> np.ndarray:
    """Convert position(s) from lzyx (Z-up) to Kimodo Y-up.

    Args:
        pos: (..., 3) array in lzyx coordinates.

    Returns:
        (..., 3) array in Y-up coordinates.
    """
    return pos @ M_INV.T


def yup_pos_to_lzyx(pos: np.ndarray) -> np.ndarray:
    """Convert position(s) from Kimodo Y-up to lzyx (Z-up).

    Args:
        pos: (..., 3) array in Y-up coordinates.

    Returns:
        (..., 3) array in lzyx coordinates.
    """
    return pos @ M.T


def lzyx_root2d(x_lzyx: float, y_lzyx: float) -> tuple[float, float]:
    """Convert a 2D ground-plane point from lzyx to Kimodo's root2d [x, z].

    In lzyx: X=right, Y=forward (ground plane is XY).
    In Kimodo Y-up: X=right, Z=forward (ground plane is XZ).

    The M transform negates X, so:
        root2d_x = -x_lzyx
        root2d_z =  y_lzyx

    Returns:
        (root2d_x, root2d_z) in Kimodo's Y-up XZ plane.
    """
    return -x_lzyx, y_lzyx


def lzyx_heading_to_model_angle(heading_deg: float) -> float:
    """Convert an lzyx facing direction to Kimodo's root heading angle (rad).

    Input ``heading_deg``: 0 = +Y (lzyx forward), positive rotates toward
    +X (lzyx right, clockwise viewed top-down). The facing unit vector in
    lzyx is therefore ``(dir_x, dir_y) = (sin θ, cos θ)``.

    Kimodo's ``compute_heading_angle`` (feature_utils.py) returns
    ``atan2(diff_z, -diff_x)`` on the hip vector in its Y-up frame, which
    works out (after the lzyx→Y-up X-negation in ``lzyx_root2d``) to:

        model_angle = atan2(dir_x_lzyx, -dir_y_lzyx)

    Sanity: facing +Y (forward) → atan2(0, -1) = π; facing -Y (backward)
    → atan2(0, 1) = 0. global_root_heading = [cos(model_angle),
    sin(model_angle)].

    NOTE: sign/offset of this convention is to be confirmed empirically
    via the bench heading sweep; flip here if the character faces the
    opposite way.
    """
    import math
    theta = math.radians(heading_deg)
    dir_x, dir_y = math.sin(theta), math.cos(theta)
    return math.atan2(dir_x, -dir_y)
