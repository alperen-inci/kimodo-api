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
    ``atan2(diff_z, -diff_x)`` on the hip vector in its Y-up frame.
    Empirically (bench heading sweep 2026-06-15), the naive derivation
    ``atan2(dir_x, -dir_y)`` was 180° inverted — heading_deg=0 produced
    a backward-facing character (run flipped 10/10 fwd → 0/10). The
    correct mapping is the 180° rotation:

        model_angle = atan2(-dir_x_lzyx, dir_y_lzyx)

    Sanity: facing +Y (forward) → atan2(0, 1) = 0; facing -Y (backward)
    → atan2(0, -1) = π. global_root_heading = [cos(model_angle),
    sin(model_angle)].

    Fully calibrated against the bench (2026-06-15):
      0°  → +Y (forward)   [roundtrip walk 5/10 → 10/10]
      90° → +X (right)     [perpendicular test: +turn delta]
      180°→ -Y (backward)
      270°→ -X (left)      [perpendicular test: -turn delta]
    """
    import math
    theta = math.radians(heading_deg)
    dir_x, dir_y = math.sin(theta), math.cos(theta)
    return math.atan2(-dir_x, dir_y)


def lzyx_params_to_yup_fk_inputs(
    all_aa: np.ndarray, trans: np.ndarray, pelvis_offset: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Shared lzyx ingest decode: SMPL-X-style params -> Kimodo FK inputs.

    THE single implementation of the Z-up -> Y-up ingest conversion. It was
    copy-pasted in four places in service.py (history load x2 formats,
    inbetween keyframes, single-keyframe constraint) before 2026-09; keep it
    here only.

    Args:
        all_aa: (T, J, 3) local axis-angle, joint 0 = root orient, in lzyx.
        trans: (T, 3) lzyx translation as exported by get_amass_parameters
            z_up=True (pelvis-offset frame: the offset is folded in before the
            rotation and removed after — see kimodo.exports.smplx).
        pelvis_offset: (3,) the skeleton's neutral root joint position (Y-up).

    Returns:
        (root_positions (T, 3) float32 Y-up, local_rot_mats (T, J, 3, 3)
        float32 Y-up) — ready for kimodo.skeleton.fk.
    """
    import torch
    from kimodo.geometry import axis_angle_to_matrix

    trans_yup = np.matmul(trans + pelvis_offset, M_INV.T) - pelvis_offset
    root_positions = (trans_yup + pelvis_offset).astype(np.float32)

    root_rots_mat = axis_angle_to_matrix(
        torch.tensor(all_aa[:, 0], dtype=torch.float32)
    ).numpy()
    root_rots_yup = np.matmul(M_INV.T, root_rots_mat)  # undo M @ R

    body_rots = axis_angle_to_matrix(
        torch.tensor(all_aa[:, 1:], dtype=torch.float32)
    ).numpy()
    local_rot_mats = np.concatenate(
        [root_rots_yup[:, np.newaxis, :, :], body_rots], axis=1
    ).astype(np.float32)
    return root_positions, local_rot_mats


# ---------------------------------------------------------------------------
# Self-describing input handling (canonical-axis migration, Phase 2.4)
# ---------------------------------------------------------------------------

# Wire spellings accepted per frame. "kimodo_zup" is the UE SDK's name for the
# exact frame this service exports as lzyx (its NPZ writer stamps it on every
# history/pose file it uploads).
_COORD_ALIASES = {
    "lzyx": "lzyx",
    "kimodo_zup": "lzyx",
    "smplx_yup": "smplx_yup",
}


def normalize_coord(value: str) -> str:
    """Canonical spelling of a coord tag ('lzyx' or 'smplx_yup'); raises on unknown."""
    key = str(value).strip().lower()
    if key not in _COORD_ALIASES:
        raise ValueError(
            f"unknown coord {value!r}; known: {sorted(set(_COORD_ALIASES))}"
        )
    return _COORD_ALIASES[key]


def npz_coord(data, default: str = "lzyx") -> str:
    """The frame an uploaded NPZ declares for itself, else ``default``.

    Files are SELF-DESCRIBING (COORDINATE_CONTRACT.md §2): a `coord` field in
    the file OVERRIDES any request-level coord_in, so a canonical file can
    never be mis-read as lzyx by a stale caller (or vice versa).
    """
    keys = set(getattr(data, "files", None) or data.keys())
    if "coord" in keys:
        return normalize_coord(np.asarray(data["coord"]).item())
    return normalize_coord(default)


def params_to_yup_fk_inputs(
    all_aa: np.ndarray, trans: np.ndarray, pelvis_offset: np.ndarray,
    coord: str = "lzyx",
) -> tuple[np.ndarray, np.ndarray]:
    """Coord-aware ingest decode: SMPL-X-style params -> Kimodo FK inputs.

    "lzyx" is the packed export frame (see lzyx_params_to_yup_fk_inputs).
    "smplx_yup" is canonical SMPL-X: axes already match Kimodo's Y-up and the
    translation is the clean AMASS parameter (root - pelvis_offset), so the
    decode is just the offset add + axis-angle -> matrices.
    """
    if normalize_coord(coord) == "lzyx":
        return lzyx_params_to_yup_fk_inputs(all_aa, trans, pelvis_offset)

    import torch
    from kimodo.geometry import axis_angle_to_matrix

    root_positions = (trans + pelvis_offset).astype(np.float32)
    local_rot_mats = (
        axis_angle_to_matrix(torch.tensor(all_aa, dtype=torch.float32))
        .numpy()
        .astype(np.float32)
    )
    return root_positions, local_rot_mats


def root2d_from_pos(pos, coord: str = "lzyx") -> tuple[float, float]:
    """Ground-plane root2d [x, z] (Kimodo Y-up) from a wire position.

    lzyx ground plane is XY -> lzyx_root2d; canonical ground plane is XZ and
    the axes already match Kimodo's, so it reads straight off. Both share the
    same deliberate approximation as the existing lzyx path: the ~1 cm
    pelvis-offset XZ term is ignored.
    """
    if normalize_coord(coord) == "lzyx":
        return lzyx_root2d(float(pos[0]), float(pos[1]))
    return float(pos[0]), float(pos[2])


def heading_to_model_angle(heading_deg: float, coord: str = "lzyx") -> float:
    """Coord-aware facing-direction -> model heading angle (rad).

    The canonical (smplx_yup) zero-direction is CALIBRATED, not derived on
    paper (the lzyx map needed two bench fixes — see
    lzyx_heading_to_model_angle); until the Phase 2.5 bench sweep runs,
    canonical requests must not silently guess.
    """
    if normalize_coord(coord) == "lzyx":
        return lzyx_heading_to_model_angle(heading_deg)
    raise NotImplementedError(
        "heading for coord_in='smplx_yup' is not calibrated yet "
        "(canonical-axis migration Phase 2.5: 0/90/180/270 bench sweep)"
    )
