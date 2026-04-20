# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Client for the Body2Hands API — adds generated hand motion to an SMPL-X NPZ.

Design:
    * Post-processing step applied after kimodo generates a body motion.
    * Never raises: on any failure (unreachable, timeout, bad response,
      shape mismatch) returns ``None`` so the caller can fall back to the
      unenriched kimodo output.
    * Only sends the NPZ payload; all body2hands parameters use their
      server-side defaults.
    * Defensively re-splices: we take *only* the hand channel slice
      ``poses[:, 75:165]`` from the body2hands response and overwrite that
      slice on our original NPZ. Every other field — root orientation, body
      pose, jaw/eyes, translation, betas, gender, framerate — is preserved
      byte-for-byte from the kimodo output regardless of what body2hands
      sends back.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import httpx
import numpy as np

log = logging.getLogger("kimodo_api.body2hands")

# Hand channels in SMPL-X 165-dim pose layout.
HAND_SLICE = slice(75, 165)
MIN_POSE_DIMS = 165


async def health(base_url: str, timeout: float = 3.0) -> Optional[dict]:
    """Probe the body2hands /health endpoint. Returns body on success, None on failure."""
    url = base_url.rstrip("/") + "/health"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        log.warning("Body2Hands health check failed (%s): %s", url, e)
        return None


async def enrich(base_url: str, npz_bytes: bytes, timeout: float = 120.0) -> Optional[bytes]:
    """POST the NPZ to body2hands and return the enriched NPZ bytes.

    Sends only the required ``motion`` field so body2hands applies its own
    defaults for ddim_steps, seed, smoothing, etc.

    Returns the raw response body on 200, or None on any failure.
    """
    url = base_url.rstrip("/") + "/generate/body2hands"
    files = {"motion": ("motion.npz", npz_bytes, "application/octet-stream")}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, files=files)
            resp.raise_for_status()
            return resp.content
    except httpx.HTTPStatusError as e:
        body = e.response.text[:300] if e.response is not None else ""
        log.warning("Body2Hands returned HTTP %s: %s", e.response.status_code, body)
        return None
    except Exception as e:
        log.warning("Body2Hands request failed (%s): %s", url, e)
        return None


def splice_hands_only(original_npz: bytes, b2h_npz: bytes) -> Optional[bytes]:
    """Return a new NPZ that is the original with only hand channels replaced.

    Preserves every field of the original NPZ. Overwrites ``poses[:, 75:165]``
    with the hand slice from the body2hands NPZ. If shapes disagree or either
    NPZ is malformed, returns None (caller should fall back to original).
    """
    try:
        orig = np.load(io.BytesIO(original_npz), allow_pickle=True)
        b2h = np.load(io.BytesIO(b2h_npz), allow_pickle=True)
    except Exception as e:
        log.warning("Body2Hands splice: failed to load NPZ (%s)", e)
        return None

    if "poses" not in orig.files or "poses" not in b2h.files:
        log.warning("Body2Hands splice: 'poses' missing in one of the NPZs")
        return None

    orig_poses = orig["poses"]
    b2h_poses = b2h["poses"]

    if orig_poses.shape != b2h_poses.shape:
        log.warning(
            "Body2Hands splice: shape mismatch orig=%s b2h=%s",
            orig_poses.shape, b2h_poses.shape,
        )
        return None

    if orig_poses.ndim != 2 or orig_poses.shape[-1] < MIN_POSE_DIMS:
        log.warning(
            "Body2Hands splice: unexpected poses shape %s (need (T,>=%d))",
            orig_poses.shape, MIN_POSE_DIMS,
        )
        return None

    new_poses = orig_poses.copy()
    new_poses[:, HAND_SLICE] = b2h_poses[:, HAND_SLICE].astype(new_poses.dtype)

    out = {name: orig[name] for name in orig.files}
    out["poses"] = new_poses

    buf = io.BytesIO()
    np.savez(buf, **out)
    return buf.getvalue()
