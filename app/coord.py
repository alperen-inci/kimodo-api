# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coordinate system conversions between DART/AMASS lzyx (Z-up) and Kimodo Y-up.

Kimodo generates in Y-up (right-handed):
    Right   = +X
    Up      = +Y
    Forward = +Z

DART / AMASS convention (lzyx, left-handed, Z-up):
    Right   = +X
    Forward = +Y
    Up      = +Z

DART-compatible transform (pure Y<->Z swap, NO X negation):
    M = [[1, 0, 0],
         [0, 0, 1],
         [0, 1, 0]]

    v_lzyx = M @ v_yup       (positions)
    R_lzyx = M @ R_yup @ M^T (rotations)

Note: Kimodo's built-in get_amass_parameters uses rot_z_180 which negates X.
We do NOT use that here — DART/UnrealSMPLXImporter expects no X negation.
"""

import numpy as np

# Pure Y<->Z swap — DART/AMASS compatible (no X negation).
M = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

# M is an involution (M @ M = I), so inverse = M.
M_INV = M.copy()


def lzyx_pos_to_yup(pos: np.ndarray) -> np.ndarray:
    """Convert position(s) from lzyx (Z-up) to Kimodo Y-up."""
    return pos @ M_INV.T


def yup_pos_to_lzyx(pos: np.ndarray) -> np.ndarray:
    """Convert position(s) from Kimodo Y-up to lzyx (Z-up)."""
    return pos @ M.T


def lzyx_root2d(x_lzyx: float, y_lzyx: float) -> tuple[float, float]:
    """Convert a 2D ground-plane point from lzyx to Kimodo's root2d [x, z].

    In lzyx: X=right, Y=forward (ground plane is XY).
    In Kimodo Y-up: X=right, Z=forward (ground plane is XZ).

    No X negation (DART-compatible):
        root2d_x = x_lzyx
        root2d_z = y_lzyx

    Returns:
        (root2d_x, root2d_z) in Kimodo's Y-up XZ plane.
    """
    return x_lzyx, y_lzyx
