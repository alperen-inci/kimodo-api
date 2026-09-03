# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""coord_out export gates (canonical-axis migration Phase 2).

1. lzyx regression: coord_out="lzyx" output is BYTE-identical to the
   pre-change export (z_up=True, no `coord` field) — proof the default wire
   did not move.
2. Mathematical equivalence: the canonical output IS the lzyx output with the
   z_up packing removed — applying the packing (M with the pelvis-offset
   frame on trans, M @ R on the root) to the canonical NPZ reproduces the
   lzyx NPZ.
3. Self-description: the canonical NPZ carries coord='smplx_yup'; lzyx none.

No GPU, no model: a mock skeleton drives MotionService._export_npz directly;
kimodo's pure-math submodules are loaded file-direct (kimodo/__init__ pulls
the model stack).
"""
import importlib.util
import io
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KIMODO_ROOT = os.environ.get("KIMODO_ROOT", "/mnt/hdd/repos/unreal_body/kimodo")

torch = pytest.importorskip("torch")
if not os.path.isdir(KIMODO_ROOT):
    pytest.skip(f"kimodo not found at {KIMODO_ROOT} (set KIMODO_ROOT)", allow_module_level=True)

if "kimodo.exports.smplx" not in sys.modules:
    _pkg = types.ModuleType("kimodo")
    _pkg.__path__ = [os.path.join(KIMODO_ROOT, "kimodo")]
    sys.modules.setdefault("kimodo", _pkg)
    _exp = types.ModuleType("kimodo.exports")
    _exp.__path__ = [os.path.join(KIMODO_ROOT, "kimodo", "exports")]
    sys.modules.setdefault("kimodo.exports", _exp)
    for _name, _rel in [
        ("kimodo.tools", "kimodo/tools.py"),
        ("kimodo.geometry", "kimodo/geometry.py"),
        ("kimodo.assets", "kimodo/assets.py"),
        ("kimodo.exports.smplx", "kimodo/exports/smplx.py"),
    ]:
        if _name in sys.modules:
            continue
        _spec = importlib.util.spec_from_file_location(_name, os.path.join(KIMODO_ROOT, _rel))
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_name] = _mod
        _spec.loader.exec_module(_mod)

from app.service import KimodoService as MotionService  # noqa: E402
from app.coord import M  # noqa: E402


class _MockSkeleton:
    def __init__(self):
        self.root_idx = 0
        # SMPL-X-ish neutral pelvis (metres, Y-up)
        self.neutral_joints = torch.tensor(
            [[0.002, -0.35, 0.01]] + [[0.0, 0.0, 0.0]] * 21, dtype=torch.float32
        )


def _make_svc():
    """A real KimodoService shell without __init__ (no model, no GPU): only the
    attributes _export_npz/_pack_dart_npz actually touch are provided."""
    svc = MotionService.__new__(MotionService)
    svc.skeleton = _MockSkeleton()
    svc.amass_converter = None
    return svc


def _make_output(seed=3, T=16, J=22):
    rng = np.random.default_rng(seed)
    aa = torch.tensor(rng.normal(scale=0.6, size=(T, J, 3)), dtype=torch.float32)
    from kimodo.geometry import axis_angle_to_matrix
    rots = axis_angle_to_matrix(aa).numpy()  # (T, J, 3, 3)
    return {
        "local_rot_mats": rots.astype(np.float32),
        "root_positions": rng.normal(scale=1.0, size=(T, 3)).astype(np.float32) + np.array([0, 0.9, 0], dtype=np.float32),
    }


def _legacy_reference_export(svc, output):
    """The pre-change export, verbatim: z_up=True, no coord field."""
    from kimodo.exports.smplx import get_amass_parameters
    local_rot_mats = output["local_rot_mats"][np.newaxis]
    root_positions = output["root_positions"][np.newaxis]
    trans, root_orient, pose_body = get_amass_parameters(
        local_rot_mats, root_positions, svc.skeleton, z_up=True
    )
    trans, root_orient, pose_body = trans[0], root_orient[0], pose_body[0]
    T = trans.shape[0]
    parts = [root_orient, pose_body,
             np.zeros((T, 3), dtype=np.float32),
             np.zeros((T, 6), dtype=np.float32),
             np.zeros((T, 90), dtype=np.float32)]
    poses = np.concatenate(parts, axis=-1).astype(np.float32)
    buf = io.BytesIO()
    np.savez(buf, poses=poses, trans=trans.astype(np.float32),
             betas=np.zeros(16, dtype=np.float32), gender="neutral",
             mocap_framerate=np.int64(30),
             n_body_joints=np.int64(pose_body.shape[-1] // 3))
    return buf.getvalue()


@pytest.fixture()
def svc():
    return _make_svc()


def test_lzyx_output_is_byte_identical_to_the_legacy_export(svc):
    output = _make_output()
    got = svc._export_npz(output, return_format="npz", coord_out="lzyx")
    ref = _legacy_reference_export(svc, output)
    ga = dict(np.load(io.BytesIO(got), allow_pickle=True))
    ra = dict(np.load(io.BytesIO(ref), allow_pickle=True))
    assert sorted(ga.keys()) == sorted(ra.keys()), "lzyx key set moved"
    assert "coord" not in ga, "lzyx must NOT carry coord (legacy loaders key on its absence)"
    for k in ra:
        assert np.array_equal(np.asarray(ga[k]), np.asarray(ra[k])), f"field {k} diverged"


def test_canonical_output_is_the_unpacked_lzyx(svc):
    output = _make_output(seed=9)
    lz = dict(np.load(io.BytesIO(svc._export_npz(
        output, return_format="npz", coord_out="lzyx")), allow_pickle=True))
    cn = dict(np.load(io.BytesIO(svc._export_npz(
        output, return_format="npz", coord_out="smplx_yup")), allow_pickle=True))

    assert str(cn["coord"]) == "smplx_yup"

    off = svc.skeleton.neutral_joints[0].numpy()
    # trans: re-apply the z_up packing to the canonical trans -> lzyx trans
    repacked = (cn["trans"] + off) @ M.T - off
    assert np.allclose(repacked, lz["trans"], atol=1e-6)
    # canonical trans is CLEAN: root_positions - pelvis_offset, no packing bias
    assert np.allclose(cn["trans"], output["root_positions"] - off, atol=1e-6)

    # root orient: R_lzyx == M @ R_yup
    from kimodo.geometry import axis_angle_to_matrix
    r_yup = axis_angle_to_matrix(torch.tensor(cn["poses"][:, :3])).numpy()
    r_lz = axis_angle_to_matrix(torch.tensor(lz["poses"][:, :3])).numpy()
    assert np.allclose(np.matmul(M, r_yup), r_lz, atol=1e-5)

    # body joints identical in both frames
    assert np.allclose(cn["poses"][:, 3:], lz["poses"][:, 3:], atol=0.0)
