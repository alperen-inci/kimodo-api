# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""lzyx_params_to_yup_fk_inputs must be BIT-IDENTICAL to the four inline
blocks it replaced (service.py history-load x2, inbetween keyframes,
single-keyframe constraint — all copies of the same decode).

The reference below is the pre-refactor block, verbatim. No model, no GPU:
only kimodo.geometry.axis_angle_to_matrix (pure torch) is needed.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KIMODO_ROOT = os.environ.get("KIMODO_ROOT", "/mnt/hdd/repos/unreal_body/kimodo")
if os.path.isdir(KIMODO_ROOT):
    sys.path.insert(0, KIMODO_ROOT)

torch = pytest.importorskip("torch")

# kimodo/__init__ pulls the whole model stack (peft etc.); geometry.py itself is
# pure torch. Load it file-direct so the test runs on a torch-only machine, while
# registering it under its real name so app.coord's `from kimodo.geometry import
# axis_angle_to_matrix` resolves to the same module.
import importlib.util
import types

_GEOM = os.path.join(KIMODO_ROOT, "kimodo", "geometry.py")
if not os.path.isfile(_GEOM):
    pytest.skip(f"kimodo geometry not found at {_GEOM} (set KIMODO_ROOT)", allow_module_level=True)
if "kimodo.geometry" not in sys.modules:
    _pkg = types.ModuleType("kimodo")
    _pkg.__path__ = [os.path.join(KIMODO_ROOT, "kimodo")]
    _spec = importlib.util.spec_from_file_location("kimodo.geometry", _GEOM)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["kimodo"] = _pkg
    sys.modules["kimodo.geometry"] = _mod
    _spec.loader.exec_module(_mod)

from app.coord import M_INV, lzyx_params_to_yup_fk_inputs  # noqa: E402


def _reference_inline_decode(all_aa, trans, pelvis_offset):
    """The pre-2026-09 service.py block, verbatim (history DART/API variant)."""
    from kimodo.geometry import axis_angle_to_matrix

    trans_yup = np.matmul(trans + pelvis_offset, M_INV.T) - pelvis_offset
    root_positions = (trans_yup + pelvis_offset).astype(np.float32)

    root_rots_mat = axis_angle_to_matrix(
        torch.tensor(all_aa[:, 0], dtype=torch.float32)
    ).numpy()
    root_rots_yup = np.matmul(M_INV.T, root_rots_mat)

    body_rots = axis_angle_to_matrix(
        torch.tensor(all_aa[:, 1:], dtype=torch.float32)
    ).numpy()
    local_rot_mats = np.concatenate(
        [root_rots_yup[:, np.newaxis, :, :], body_rots], axis=1
    ).astype(np.float32)
    return root_positions, local_rot_mats


def test_helper_matches_the_inline_blocks_bit_for_bit():
    rng = np.random.default_rng(7)
    T, J = 24, 22
    all_aa = rng.normal(scale=0.8, size=(T, J, 3)).astype(np.float64)
    trans = rng.normal(scale=2.0, size=(T, 3)).astype(np.float64)
    pelvis_offset = np.array([0.002, -0.35, 0.01], dtype=np.float32)

    ref_pos, ref_rots = _reference_inline_decode(all_aa, trans, pelvis_offset)
    got_pos, got_rots = lzyx_params_to_yup_fk_inputs(all_aa, trans, pelvis_offset)

    assert got_pos.dtype == ref_pos.dtype and got_rots.dtype == ref_rots.dtype
    assert np.array_equal(got_pos, ref_pos), "root_positions diverged"
    assert np.array_equal(got_rots, ref_rots), "local_rot_mats diverged"


def test_round_trip_through_the_export_transform():
    """The decode must invert the export packing: applying the forward
    z_up transform (M with the pelvis-offset frame) to the decoded Y-up
    values reproduces the lzyx inputs."""
    rng = np.random.default_rng(11)
    T, J = 8, 22
    all_aa = rng.normal(scale=0.5, size=(T, J, 3))
    trans = rng.normal(scale=1.5, size=(T, 3))
    pelvis_offset = np.array([0.0, -0.3494, 0.0], dtype=np.float32)

    root_positions, local_rot_mats = lzyx_params_to_yup_fk_inputs(all_aa, trans, pelvis_offset)

    # forward: trans_lzyx = M @ (trans_yup + off) - off, with trans_yup = root_pos - off
    trans_yup = root_positions - pelvis_offset
    re_lzyx = np.matmul(trans_yup + pelvis_offset, np.asarray(M_INV, dtype=np.float64).T) - pelvis_offset
    assert np.allclose(re_lzyx, trans, atol=1e-6)

    # forward rotation: R_lzyx = M @ R_yup
    from kimodo.geometry import axis_angle_to_matrix
    ref_root = axis_angle_to_matrix(torch.tensor(all_aa[:, 0], dtype=torch.float32)).numpy()
    re_root = np.matmul(np.asarray(M_INV, dtype=np.float32), local_rot_mats[:, 0])
    assert np.allclose(re_root, ref_root, atol=1e-6)
