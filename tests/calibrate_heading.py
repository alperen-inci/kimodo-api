#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 2.5 heading calibration sweep (needs the live service + GPU).

For each coord ('lzyx' control, 'smplx_yup' candidate) and each heading
H in {0, 90, 180, 270}: generate a walk toward a 3.5 m target placed in
direction H, with a waypoint heading_deg=H, then measure the OUTPUT's actual
ground-plane travel direction and report the error vs H.

Direction conventions under test (same semantics, per-frame axes):
    lzyx:      0 = +Y (forward), 90 = +X (subject's right)
    smplx_yup: 0 = +Z (forward), 90 = -X (subject's right)

Usage:  python3 tests/calibrate_heading.py [http://localhost:8020]
"""
import io
import json
import math
import sys
import urllib.request

import numpy as np

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8020"
DIST = 3.5
SEED = 11


def gen(spec: dict) -> dict:
    body, boundary = [], "XBOUNDX"
    spec_json = json.dumps(spec)
    payload = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"spec_json\"\r\n\r\n"
        f"{spec_json}\r\n--{boundary}--\r\n"
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/generate/kimodo", data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return dict(np.load(io.BytesIO(r.read()), allow_pickle=True))


def direction(coord: str, deg: float):
    t = math.radians(deg)
    if coord == "lzyx":
        return math.sin(t), math.cos(t)      # ground plane (X, Y)
    return -math.sin(t), math.cos(t)         # ground plane (X, Z)


def travel_deg(coord: str, trans: np.ndarray) -> float:
    d = trans[-1] - trans[0]
    if coord == "lzyx":
        return math.degrees(math.atan2(d[0], d[1])) % 360.0
    return math.degrees(math.atan2(-d[0], d[2])) % 360.0


def main():
    print(f"service: {BASE}  dist={DIST}m seed={SEED}")
    worst = 0.0
    for coord in ("lzyx", "smplx_yup"):
        for H in (0.0, 90.0, 180.0, 270.0):
            dx, dfwd = direction(coord, H)
            pos = [DIST * dx, DIST * dfwd, 0.0] if coord == "lzyx" else [DIST * dx, 0.0, DIST * dfwd]
            spec = {
                "fps": 30, "seed": SEED, "diffusion_steps": 50,
                "coord_in": coord, "coord_out": coord,
                "segments": [{
                    "type": "trajectory", "text": "a person walks forward",
                    "start_frame": 0, "end_frame": 120,
                    "points": [{"frame": 119, "pos": pos, "heading_deg": H}],
                }],
            }
            out = gen(spec)
            got = travel_deg(coord, np.asarray(out["trans"], float))
            err = min(abs(got - H), 360.0 - abs(got - H))
            worst = max(worst, err)
            dist = float(np.linalg.norm(out["trans"][-1] - out["trans"][0]))
            print(f"  {coord:10s} H={H:5.0f}  travel={got:6.1f} deg  err={err:5.1f} deg  dist={dist:.2f} m")
    print(f"worst error: {worst:.1f} deg  -> {'PASS' if worst < 25 else 'FAIL'} (gate < 25 deg)")
    return 0 if worst < 25 else 1


if __name__ == "__main__":
    sys.exit(main())
