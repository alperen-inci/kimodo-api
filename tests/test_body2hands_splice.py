# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the body2hands splice helper.

The splice must:
  1. Replace ONLY poses[:, 75:165] (hand channels).
  2. Leave every other field byte-identical to the original NPZ.
  3. Return None on shape/format mismatch.

Run with:
    cd kimodo-api/
    python -m pytest tests/test_body2hands_splice.py -v
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from app.body2hands import HAND_SLICE, splice_hands_only


def _make_npz(T: int = 60, pose_dim: int = 165, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    np.savez(
        buf,
        poses=rng.standard_normal((T, pose_dim)).astype(np.float32),
        trans=rng.standard_normal((T, 3)).astype(np.float32),
        betas=rng.standard_normal(10).astype(np.float32),
        gender=np.array("neutral"),
        mocap_framerate=np.array(30.0, dtype=np.float32),
    )
    return buf.getvalue()


def _load(npz_bytes: bytes) -> dict:
    d = np.load(io.BytesIO(npz_bytes), allow_pickle=True)
    return {k: d[k] for k in d.files}


def test_splice_replaces_only_hand_channels():
    original = _make_npz(seed=1)
    b2h = _make_npz(seed=2)  # fully different content
    result = splice_hands_only(original, b2h)
    assert result is not None

    o = _load(original)
    b = _load(b2h)
    r = _load(result)

    # Hand channels must come from b2h.
    np.testing.assert_array_equal(r["poses"][:, HAND_SLICE], b["poses"][:, HAND_SLICE])
    # Non-hand pose channels must match original.
    np.testing.assert_array_equal(r["poses"][:, :75], o["poses"][:, :75])
    # All other fields must match original.
    for field in ("trans", "betas", "gender", "mocap_framerate"):
        np.testing.assert_array_equal(r[field], o[field])


def test_splice_preserves_dtype():
    original = _make_npz(seed=3)
    b2h = _make_npz(seed=4)
    result = splice_hands_only(original, b2h)
    assert result is not None
    r = _load(result)
    assert r["poses"].dtype == np.float32


def test_splice_rejects_shape_mismatch_frames():
    original = _make_npz(T=60, seed=5)
    b2h = _make_npz(T=90, seed=6)
    assert splice_hands_only(original, b2h) is None


def test_splice_rejects_shape_mismatch_dims():
    original = _make_npz(T=60, pose_dim=165, seed=7)
    b2h_buf = io.BytesIO()
    rng = np.random.default_rng(8)
    np.savez(
        b2h_buf,
        poses=rng.standard_normal((60, 120)).astype(np.float32),
        trans=rng.standard_normal((60, 3)).astype(np.float32),
    )
    assert splice_hands_only(original, b2h_buf.getvalue()) is None


def test_splice_rejects_missing_poses():
    original = _make_npz(seed=9)
    bad_buf = io.BytesIO()
    np.savez(bad_buf, trans=np.zeros((60, 3), dtype=np.float32))
    assert splice_hands_only(original, bad_buf.getvalue()) is None


def test_splice_rejects_invalid_npz_bytes():
    original = _make_npz(seed=10)
    assert splice_hands_only(original, b"not-an-npz-file") is None


def test_splice_preserves_exact_bytes_outside_hands():
    """Stricter check: all non-hand pose columns match bit-for-bit."""
    original = _make_npz(seed=11)
    b2h = _make_npz(seed=12)
    result = splice_hands_only(original, b2h)
    assert result is not None

    o = _load(original)
    r = _load(result)

    for col in range(165):
        if 75 <= col < 165:
            continue
        np.testing.assert_array_equal(r["poses"][:, col], o["poses"][:, col])
