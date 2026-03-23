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
