#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare Kimodo CLI (kimodo_gen) vs Kimodo API export paths.

Strategy
--------
For each test case the script runs model inference **once** and then exports
the *same* raw output dict through two code paths:

  A) **CLI path** — the same export logic used by ``kimodo_gen``
     (raw NPZ + AMASSConverter.convert_save_npz)
  B) **API path** — the same export logic used by ``kimodo-api/app/service.py``
     (get_amass_parameters → poses(T,165) DART-compatible NPZ)

Because both paths share the same raw inference result, any numerical
difference is **purely** an export / coordinate-conversion / packing bug —
not a stochastic model difference.

A second mode ("dual-inference") re-runs inference with the same seed to
verify that the seeding is deterministic. This is opt-in with --dual.

Usage (inside Docker container or venv with kimodo installed)
-------------------------------------------------------------
    cd kimodo/kimodo-api/
    python tests/compare_outputs.py                  # shared inference
    python tests/compare_outputs.py --quick           # fewer steps
    python tests/compare_outputs.py --dual            # run inference twice
    python tests/compare_outputs.py --output-dir /tmp/kimodo_compare
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import textwrap
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
API_DIR = SCRIPT_DIR.parent
REPO_ROOT = API_DIR.parent
sys.path.insert(0, str(API_DIR))
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------
@dataclass
class TestCase:
    name: str
    description: str
    texts: list[str]
    num_frames: list[int]
    seed: int = 0
    diffusion_steps: int = 100
    num_samples: int = 1
    post_processing: bool = True
    num_transition_frames: int = 5
    cfg_weight: list[float] = field(default_factory=lambda: [2.0, 2.0])
    # Per-segment trajectory: None = no trajectory for that segment
    trajectory_points: Optional[list[Optional[list[dict]]]] = None


def get_test_cases(quick: bool = False) -> list[TestCase]:
    steps = 30 if quick else 100
    return [
        TestCase(
            name="text_walk_forward",
            description="Simple text: walk forward (3s)",
            texts=["A person walks forward."],
            num_frames=[90],
            seed=42,
            diffusion_steps=steps,
        ),
        TestCase(
            name="text_wave_hello",
            description="Simple text: wave hello (3s)",
            texts=["Wave hello with right hand."],
            num_frames=[90],
            seed=0,
            diffusion_steps=steps,
        ),
        TestCase(
            name="text_dance",
            description="Simple text: dance (5s)",
            texts=["A person dances energetically."],
            num_frames=[150],
            seed=123,
            diffusion_steps=steps,
        ),
        TestCase(
            name="text_multi_prompt",
            description="Multi-prompt: wave then walk (6s)",
            texts=["Wave hello with right hand.", "Walk forward."],
            num_frames=[90, 90],
            seed=42,
            diffusion_steps=steps,
        ),
        TestCase(
            name="trajectory_walk_right",
            description="Trajectory: walk right (5s, 1 waypoint)",
            texts=["Walk to the right."],
            num_frames=[150],
            seed=0,
            diffusion_steps=steps,
            trajectory_points=[[{"frame": 149, "pos": [2.0, 0.0, 0.96]}]],
        ),
        TestCase(
            name="trajectory_walk_diagonal",
            description="Trajectory: walk diagonal (5s, 3 waypoints)",
            texts=["Walk diagonally forward and to the right."],
            num_frames=[150],
            seed=0,
            diffusion_steps=steps,
            trajectory_points=[
                [
                    {"frame": 50, "pos": [1.0, 1.0, 0.96]},
                    {"frame": 100, "pos": [2.0, 2.0, 0.96]},
                    {"frame": 149, "pos": [3.0, 3.0, 0.96]},
                ]
            ],
        ),
        TestCase(
            name="trajectory_multi_segment",
            description="Multi-trajectory: right then forward (8s)",
            texts=["Walk to the right.", "Walk forward."],
            num_frames=[120, 120],
            seed=0,
            diffusion_steps=steps,
            trajectory_points=[
                [
                    {"frame": 60, "pos": [1.0, 0.0, 0.96]},
                    {"frame": 119, "pos": [2.0, 0.0, 0.96]},
                ],
                [
                    {"frame": 60, "pos": [2.0, 1.5, 0.96]},
                    {"frame": 119, "pos": [2.0, 3.0, 0.96]},
                ],
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def build_constraints(model, test: TestCase) -> list:
    """Build Root2DConstraintSet list from test trajectory points (lzyx input)."""
    from kimodo.constraints import Root2DConstraintSet

    constraints = []
    if not test.trajectory_points:
        return constraints
    for seg_idx, seg_pts in enumerate(test.trajectory_points):
        if not seg_pts:
            continue
        abs_offset = sum(test.num_frames[:seg_idx])
        frames, pos2d = [], []
        for pt in seg_pts:
            frames.append(abs_offset + pt["frame"])
            # lzyx → Kimodo root2d: negate X, use Y as forward
            pos2d.append([-pt["pos"][0], pt["pos"][1]])

        device = model.skeleton.device if hasattr(model.skeleton, "device") else "cpu"
        constraints.append(
            Root2DConstraintSet(
                model.skeleton,
                frame_indices=torch.tensor(frames, dtype=torch.long),
                smooth_root_2d=torch.tensor(pos2d, dtype=torch.float32, device=device),
            )
        )
    return constraints


def run_inference(model, test: TestCase) -> dict:
    """Run inference once; returns the raw model output dict (numpy)."""
    from kimodo.tools import seed_everything

    seed_everything(test.seed)
    constraint_lst = build_constraints(model, test)

    output = model(
        test.texts,
        test.num_frames,
        num_denoising_steps=test.diffusion_steps,
        multi_prompt=True,
        constraint_lst=constraint_lst,
        cfg_weight=test.cfg_weight,
        num_samples=test.num_samples,
        num_transition_frames=test.num_transition_frames,
        post_processing=test.post_processing,
        return_numpy=True,
    )
    return output


# ---------------------------------------------------------------------------
# Export: CLI path (same as kimodo_gen)
# ---------------------------------------------------------------------------
def export_cli(model, output: dict, test: TestCase, out_dir: Path) -> dict[str, Path]:
    """Export using the same logic as kimodo_gen."""
    from kimodo.exports.smplx import AMASSConverter

    paths = {}
    n_samples = int(output["posed_joints"].shape[0])

    # 1. Raw NPZ (every key from model output)
    raw_path = out_dir / f"{test.name}_cli_raw.npz"
    if n_samples == 1:
        single = {
            k: (v[0] if hasattr(v, "shape") and v.ndim > 0 and v.shape[0] == n_samples else v)
            for k, v in output.items()
        }
        np.savez(str(raw_path), **single)
    else:
        np.savez(str(raw_path), **output)
    paths["raw"] = raw_path

    # 2. AMASS NPZ (same as kimodo_gen for smplx)
    converter = AMASSConverter(skeleton=model.skeleton, fps=model.fps)
    amass_path = out_dir / f"{test.name}_cli_amass.npz"
    converter.convert_save_npz(output, str(amass_path), z_up=True)
    paths["amass"] = amass_path

    return paths


# ---------------------------------------------------------------------------
# Export: API path (same as kimodo-api service._export_npz)
# ---------------------------------------------------------------------------
def export_api(model, output: dict, test: TestCase, out_dir: Path) -> dict[str, Path]:
    """Export using the same logic as kimodo-api/app/service.py."""
    from kimodo.exports.smplx import AMASSConverter, get_amass_parameters

    paths = {}

    local_rot_mats = output["local_rot_mats"]
    root_positions = output["root_positions"]

    # Ensure batch dim (mirrors service._export_npz)
    if local_rot_mats.ndim == 4:
        local_rot_mats = local_rot_mats[np.newaxis]
        root_positions = root_positions[np.newaxis]

    trans, root_orient, pose_body = get_amass_parameters(
        local_rot_mats, root_positions, model.skeleton, z_up=True
    )

    # Squeeze single sample
    if trans.shape[0] == 1:
        trans = trans[0]
        root_orient = root_orient[0]
        pose_body = pose_body[0]

    T = trans.shape[0]

    # DART-compatible poses (mirrors service._pack_dart_npz — skeleton-aware)
    converter = AMASSConverter(skeleton=model.skeleton, fps=model.fps)
    n_body_dims = pose_body.shape[-1]

    parts = [root_orient, pose_body]
    for key, default_shape in [("pose_jaw", 3), ("pose_eye", 6), ("pose_hand", 90)]:
        param = converter.default_frame_params.get(key, np.zeros(default_shape))
        if param.ndim == 1:
            parts.append(np.tile(param, (T, 1)).astype(np.float32))
        else:
            parts.append(param[:T].astype(np.float32))

    poses = np.concatenate(parts, axis=-1).astype(np.float32)
    betas = converter.output_dict_base.get("betas", np.zeros(16, dtype=np.float32))
    n_body_joints = n_body_dims // 3

    dart_path = out_dir / f"{test.name}_api_dart.npz"
    np.savez(
        str(dart_path),
        poses=poses,
        trans=trans.astype(np.float32),
        betas=betas.astype(np.float32),
        gender="neutral",
        mocap_framerate=np.int64(30),
        n_body_joints=np.int64(n_body_joints),
        skeleton=model.skeleton.name,
    )
    paths["dart"] = dart_path

    # Also save AMASS NPZ for direct comparison
    amass_path = out_dir / f"{test.name}_api_amass.npz"
    converter.convert_save_npz(output, str(amass_path), z_up=True)
    paths["amass"] = amass_path

    return paths


# ---------------------------------------------------------------------------
# Numerical comparison
# ---------------------------------------------------------------------------
@dataclass
class FieldDiff:
    name: str
    shape_a: tuple
    shape_b: tuple
    dtype_a: str
    dtype_b: str
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float
    num_mismatched: int
    total_elements: int
    atol: float
    match: bool
    note: str = ""
    worst_indices: str = ""  # human-readable worst-diff locations


def compare_arrays(name: str, a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> FieldDiff:
    sa, sb = tuple(a.shape), tuple(b.shape)
    da, db = str(a.dtype), str(b.dtype)

    if sa != sb:
        return FieldDiff(name, sa, sb, da, db, float("inf"), float("inf"), float("inf"),
                         -1, max(a.size, b.size), atol, False,
                         f"SHAPE MISMATCH {sa} vs {sb}")

    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    absd = np.abs(af - bf)
    maxabs = float(np.max(absd))
    meanabs = float(np.mean(absd))
    denom = np.maximum(np.abs(af), np.abs(bf))
    denom = np.where(denom < 1e-12, 1.0, denom)
    maxrel = float(np.max(absd / denom))
    nbad = int(np.sum(absd > atol))

    # Find worst indices for debugging
    worst = ""
    if maxabs > atol and a.ndim <= 3:
        flat_idx = int(np.argmax(absd))
        multi_idx = np.unravel_index(flat_idx, a.shape)
        worst = f"worst at {multi_idx}: CLI={a[multi_idx]:.8f} API={b[multi_idx]:.8f} diff={absd[flat_idx]:.2e}"

    return FieldDiff(name, sa, sb, da, db, maxabs, meanabs, maxrel,
                     nbad, af.size, atol, maxabs <= atol, worst_indices=worst)


def compare_npz(path_a: Path, path_b: Path, atol: float) -> list[FieldDiff]:
    da = np.load(str(path_a), allow_pickle=True)
    db = np.load(str(path_b), allow_pickle=True)
    ka, kb = set(da.keys()), set(db.keys())
    common = sorted(ka & kb)
    diffs = []

    for key in common:
        a, b = da[key], db[key]
        if a.dtype.kind in ("U", "S", "O") or b.dtype.kind in ("U", "S", "O"):
            eq = (str(a) == str(b))
            diffs.append(FieldDiff(
                key, a.shape, b.shape, str(a.dtype), str(b.dtype),
                0 if eq else float("inf"), 0, 0,
                0 if eq else 1, 1, 0, eq,
                note=f"string: '{a}'" if eq else f"string: '{a}' vs '{b}'"))
            continue
        diffs.append(compare_arrays(key, a, b, atol))

    for key in sorted(ka - kb):
        a = da[key]
        diffs.append(FieldDiff(key, tuple(a.shape), (), str(a.dtype), "N/A",
                               float("inf"), float("inf"), float("inf"),
                               -1, a.size, atol, False, note="ONLY IN CLI"))
    for key in sorted(kb - ka):
        b = db[key]
        diffs.append(FieldDiff(key, (), tuple(b.shape), "N/A", str(b.dtype),
                               float("inf"), float("inf"), float("inf"),
                               -1, b.size, atol, False, note="ONLY IN API"))
    return diffs


def cross_compare_amass_vs_dart(amass_path: Path, dart_path: Path, atol: float) -> list[FieldDiff]:
    """Compare CLI's AMASS (root_orient, pose_body, trans) vs API's DART (poses, trans)."""
    cli = np.load(str(amass_path), allow_pickle=True)
    api = np.load(str(dart_path), allow_pickle=True)
    diffs = []

    api_poses = api["poses"]
    T = api_poses.shape[0]

    cli_root = cli["root_orient"]
    cli_body = cli["pose_body"]
    cli_trans = cli["trans"]

    if cli_root.shape[0] != T:
        diffs.append(FieldDiff("frame_count", (cli_root.shape[0],), (T,), "", "",
                               abs(cli_root.shape[0] - T), 0, 0, 1, 1, 0, False,
                               note=f"CLI={cli_root.shape[0]} vs API={T}"))
        return diffs

    # Decompose API poses — skeleton-aware slicing
    n_body_dims = cli_body.shape[-1]  # actual body pose dims from CLI AMASS
    body_end = 3 + n_body_dims
    jaw_end = body_end + 3
    eye_end = jaw_end + 6
    hand_end = eye_end + 90

    api_root = api_poses[:, :3]
    api_body = api_poses[:, 3:body_end]
    api_jaw = api_poses[:, body_end:jaw_end]
    api_eye = api_poses[:, jaw_end:eye_end]
    api_hand = api_poses[:, eye_end:hand_end]

    diffs.append(compare_arrays(
        f"root_orient (amass vs poses[:,:3])", cli_root, api_root, atol))
    diffs.append(compare_arrays(
        f"pose_body (amass vs poses[:,3:{body_end}])", cli_body, api_body, atol))
    diffs.append(compare_arrays("trans", cli_trans, api["trans"], atol))

    # jaw/eye should be zeros
    diffs.append(FieldDiff(
        "pose_jaw (should be zeros)", (), api_jaw.shape, "N/A", str(api_jaw.dtype),
        float(np.max(np.abs(api_jaw))), float(np.mean(np.abs(api_jaw))), 0,
        int(np.sum(np.abs(api_jaw) > atol)), api_jaw.size, atol,
        float(np.max(np.abs(api_jaw))) <= atol))

    diffs.append(FieldDiff(
        "pose_eye (should be zeros)", (), api_eye.shape, "N/A", str(api_eye.dtype),
        float(np.max(np.abs(api_eye))), float(np.mean(np.abs(api_eye))), 0,
        int(np.sum(np.abs(api_eye) > atol)), api_eye.size, atol,
        float(np.max(np.abs(api_eye))) <= atol))

    # hand should match mean_hands from CLI
    if "pose_hand" in cli:
        cli_hand = cli["pose_hand"]
        diffs.append(compare_arrays(
            f"pose_hand (amass vs poses[:,{eye_end}:{hand_end}])", cli_hand, api_hand, atol))

    if "betas" in cli and "betas" in api:
        diffs.append(compare_arrays("betas", cli["betas"], api["betas"], atol))

    return diffs


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(diffs: list[FieldDiff], label: str) -> bool:
    all_ok = all(d.match for d in diffs)
    tag = "PASS" if all_ok else "FAIL"
    print(f"\n  [{tag}] {label}")
    for d in diffs:
        icon = "  OK" if d.match else "FAIL"
        line = f"    [{icon}] {d.name}"
        if d.note:
            line += f"  — {d.note}"
        if d.match:
            if not d.note:
                line += f"  shape={d.shape_a}  max_diff={d.max_abs_diff:.2e}"
        else:
            if d.shape_a and d.shape_b:
                line += (f"  max_abs={d.max_abs_diff:.2e}  mean_abs={d.mean_abs_diff:.2e}"
                         f"  bad={d.num_mismatched}/{d.total_elements}")
            if d.worst_indices:
                print(line)
                line = f"           {d.worst_indices}"
        print(line)
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare Kimodo CLI vs API export paths.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Both export paths receive the SAME raw model output, so any
            difference is purely a bug in the export/packing/coordinate logic.

            With --dual, inference runs twice (same seed) to also verify
            seeding determinism.

            Output files are saved to --output-dir for manual inspection.
            Load any NPZ in Python:
                d = np.load("file.npz", allow_pickle=True)
                print(list(d.keys()), d["poses"].shape)

            Common diff root causes:
              1. Batch dim: @ensure_batched may add/remove a leading dim
              2. Coordinate: check coord.py signs match smplx.py transform
              3. Pose assembly: poses column order must be
                 [root(3) body(63) jaw(3) eye(6) hand(90)] = 165
        """),
    )
    parser.add_argument("--output-dir", default="/tmp/kimodo_compare")
    parser.add_argument("--quick", action="store_true", help="30 diffusion steps")
    parser.add_argument("--dual", action="store_true", help="Run inference twice to test seed determinism")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model", type=str, default="smplx")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(" Kimodo CLI vs API — Export Path Comparison")
    print("=" * 70)
    print(f"  Device:     {device}")
    print(f"  Model:      {args.model}")
    print(f"  Output:     {out_dir}")
    print(f"  Quick:      {args.quick}")
    print(f"  Dual infer: {args.dual}")
    print(f"  Atol:       {args.atol}")
    print()

    # ---- Load model ----
    print("Loading model ...")
    from kimodo import load_model
    t0 = time.time()
    model = load_model(args.model, device=device, default_family="Kimodo")
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s — {model.skeleton.name}, {model.skeleton.dim} joints\n")

    cases = get_test_cases(quick=args.quick)
    total = len(cases)
    passed = 0
    failed_names = []

    for i, tc in enumerate(cases):
        hdr = f"Test {i+1}/{total}: {tc.name}"
        print("=" * 70)
        print(hdr)
        print(f"  {tc.description}")
        print(f"  texts={tc.texts}  frames={tc.num_frames}  seed={tc.seed}")

        # ---- Inference ----
        print("  Running inference ...")
        t0 = time.time()
        output = run_inference(model, tc)
        print(f"  Inference done in {time.time()-t0:.1f}s  "
              f"— output shapes: joints={output['posed_joints'].shape}")

        # ---- Export both paths from same output ----
        cli_paths = export_cli(model, output, tc, out_dir)
        api_paths = export_api(model, output, tc, out_dir)

        # ---- Compare 1: AMASS NPZ (apples-to-apples) ----
        d1 = compare_npz(cli_paths["amass"], api_paths["amass"], args.atol)
        ok1 = print_report(d1, "AMASS NPZ (CLI vs API) — same export function")

        # ---- Compare 2: CLI AMASS vs API DART (cross-format) ----
        d2 = cross_compare_amass_vs_dart(cli_paths["amass"], api_paths["dart"], args.atol)
        ok2 = print_report(d2, "Cross-format (CLI AMASS → API DART poses)")

        # ---- Optional: dual-inference determinism ----
        ok3 = True
        if args.dual:
            print("\n  Running inference again (same seed) for determinism check ...")
            t0 = time.time()
            output2 = run_inference(model, tc)
            print(f"  Second inference done in {time.time()-t0:.1f}s")
            api2_paths = export_api(model, output2, tc, out_dir)
            # rename to avoid overwrite
            dart2 = out_dir / f"{tc.name}_api_dart_run2.npz"
            os.rename(str(api2_paths["dart"]), str(dart2))
            d3 = compare_npz(api_paths["dart"], dart2, args.atol)
            ok3 = print_report(d3, "Determinism: API run1 vs API run2 (same seed)")

        test_ok = ok1 and ok2 and ok3
        if test_ok:
            passed += 1
        else:
            failed_names.append(tc.name)

    # ---- Summary ----
    print()
    print("=" * 70)
    print(" Summary")
    print("=" * 70)
    print(f"  Total:  {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {total - passed}")
    print()
    for tc in cases:
        tag = "PASS" if tc.name not in failed_names else "FAIL"
        print(f"  [{tag}] {tc.name}")
    print()
    print(f"  Output dir: {out_dir}")

    # Save machine-readable results
    results_path = out_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "total": total, "passed": passed, "failed": total - passed,
            "failed_names": failed_names,
            "cases": [tc.name for tc in cases],
        }, f, indent=2)
    print(f"  Results:    {results_path}")

    if failed_names:
        print(f"\n  *** {len(failed_names)} TEST(S) FAILED ***")
        print("  Check the diff details above.")
        print("  Compare NPZ files manually:")
        print(f"    python3 -c \"import numpy as np; a=np.load('{out_dir}/<name>_cli_amass.npz'); ...")
        sys.exit(1)
    else:
        print("\n  ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
