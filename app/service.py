# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core inference service wrapping kimodo model loading, constraint building, and NPZ export."""

from __future__ import annotations

import io
import json
import logging
import math
import os
import threading
import time
from typing import Optional

import numpy as np
import torch

log = logging.getLogger("kimodo_api.service")

# ---------------------------------------------------------------------------
# Auto-chunking: split segments longer than MAX_CHUNK_FRAMES so each piece
# stays within the model's effective quality window.  The model's internal
# _multiprompt handles 5-frame history + 10% over-generate between chunks.
# ---------------------------------------------------------------------------
MAX_CHUNK_FRAMES = int(os.environ.get("KIMODO_MAX_CHUNK_FRAMES", "300"))   # 10s @ 30fps
CHUNK_SIZE = int(os.environ.get("KIMODO_CHUNK_SIZE", "300"))               # 10s @ 30fps
MIN_CHUNK_FRAMES = 60  # avoid tiny tail chunks (< 2s)

# Dense-path mode (matches demo's "Make Smooth Path").
# Margin is the soft-constraint slack used by the ADMM smoother — same default
# as kimodo.motion_rep.smooth_root.get_smooth_root_pos. Conflicts between a
# trajectory waypoint and a Full-Body root anchor at the same frame are
# resolved by FullBody winning; a warning is logged if they disagree beyond
# DENSE_PATH_CONFLICT_WARN_M.
DENSE_PATH_MARGIN_M = 0.06
DENSE_PATH_MIN_ANCHORS_FOR_SMOOTH = 3  # 2 anchors → linear only; 3+ → ADMM
DENSE_PATH_CONFLICT_WARN_M = 0.10


def _split_long_segments(
    texts: list[str], num_frames: list[int],
) -> tuple[list[str], list[int], bool]:
    """Split any segment exceeding *MAX_CHUNK_FRAMES* into balanced pieces.

    Instead of greedy chunking (300, 300, ..., tiny-tail), this divides
    frames evenly across ``ceil(nf / MAX_CHUNK_FRAMES)`` chunks so every
    chunk stays within the model's quality sweet-spot (~5-10 s).

    Returns (texts_out, num_frames_out, did_chunk).
    """
    out_texts: list[str] = []
    out_frames: list[int] = []
    did_chunk = False

    for text, nf in zip(texts, num_frames):
        if nf <= MAX_CHUNK_FRAMES:
            out_texts.append(text)
            out_frames.append(nf)
            continue

        did_chunk = True
        n_chunks = math.ceil(nf / MAX_CHUNK_FRAMES)
        base = nf // n_chunks
        extra = nf % n_chunks
        chunks: list[int] = []
        for i in range(n_chunks):
            # Distribute leftover frames across the first 'extra' chunks
            chunks.append(base + (1 if i < extra else 0))

        for c in chunks:
            out_texts.append(text)
            out_frames.append(c)

        log.info(
            "  Balanced-chunked '%s' (%d frames / %.1fs) → %d chunks %s",
            text[:50], nf, nf / 30.0, len(chunks), chunks,
        )

    return out_texts, out_frames, did_chunk


# Lazy imports — heavy deps loaded at model-load time, not at import time.
_kimodo_loaded = False


def _ensure_kimodo_imports():
    global _kimodo_loaded
    if _kimodo_loaded:
        return
    # These are heavy (torch, transformers, etc.), so import lazily.
    import kimodo  # noqa: F401
    from kimodo.constraints import Root2DConstraintSet  # noqa: F401
    from kimodo.exports.smplx import AMASSConverter, get_amass_parameters  # noqa: F401
    from kimodo.tools import seed_everything  # noqa: F401

    _kimodo_loaded = True


class KimodoService:
    """Wraps Kimodo model lifecycle: load, generate, export."""

    def __init__(self, model_name: str = "smplx", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.skeleton = None
        self.amass_converter = None
        self._lock = threading.Lock()
        self._loaded = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load the Kimodo model. Safe to call multiple times (idempotent)."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            _ensure_kimodo_imports()
            from kimodo import load_model
            from kimodo.exports.smplx import AMASSConverter

            log.info("Loading Kimodo model '%s' on device '%s' ...", self.model_name, self.device)
            t0 = time.time()
            self.model = load_model(self.model_name, device=self.device, default_family="Kimodo")
            self.skeleton = self.model.skeleton
            self.amass_converter = AMASSConverter(skeleton=self.skeleton, fps=self.model.fps)
            self._loaded = True
            log.info(
                "Model loaded in %.1fs — skeleton=%s, fps=%d, joints=%d",
                time.time() - t0,
                self.skeleton.name,
                self.model.fps,
                self.skeleton.dim,
            )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        *,
        texts: list[str],
        num_frames: list[int],
        constraint_lst: list,
        seed: int = 0,
        diffusion_steps: int = 100,
        cfg_weight: list[float] | None = None,
        num_samples: int = 1,
        post_processing: bool = True,
        num_transition_frames: int = 5,
        return_format: str = "npz",
        history_info: dict | None = None,
        first_heading_angle_override: float | None = None,
        coord_out: str = "lzyx",
    ) -> dict:
        """Run inference and return output dict.

        Args:
            history_info: If provided, dict with keys:
                - "num_over_generate": int — extra frames to prepend for history overlap
                - "heading_angle": float — initial heading from history's last pose
                Constraints are already in constraint_lst.

        Returns:
            dict with keys: npz_bytes, meta
        """
        _ensure_kimodo_imports()
        from kimodo.tools import seed_everything

        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        cfg_weight = cfg_weight or [2.0, 2.0]
        seed_everything(seed)

        # --- Over-generate for history continuation ---
        num_over = 0
        first_heading_angle = first_heading_angle_override  # from spec, None if not set
        if history_info:
            num_over = history_info["num_over_generate"]
            first_heading_angle = history_info.get("heading_angle")  # history overrides spec
            # Add overlap frames to the first segment
            num_frames = num_frames.copy()
            num_frames[0] += num_over
            log.info("  History: over-generating %d extra frames on segment 0, heading=%.3f",
                      num_over, first_heading_angle if first_heading_angle is not None else 0.0)

        # --- Auto-chunk long segments ---
        texts, num_frames, did_chunk = _split_long_segments(texts, num_frames)

        total_frames = sum(num_frames)
        log.info(
            "Generating: %d segment(s)%s, %d total frames (%.1fs), "
            "seed=%d, steps=%d, samples=%d, post_process=%s",
            len(texts),
            " (auto-chunked)" if did_chunk else "",
            total_frames,
            total_frames / 30.0,
            seed,
            diffusion_steps,
            num_samples,
            post_processing,
        )
        for i, (t, nf) in enumerate(zip(texts, num_frames)):
            log.info("  segment %d: '%s' — %d frames (%.1fs)", i, t, nf, nf / 30.0)
        if constraint_lst:
            log.info("  constraints: %d constraint set(s)", len(constraint_lst))

        # --- Build heading angle tensor ---
        heading_tensor = None
        if first_heading_angle is not None:
            heading_tensor = torch.tensor([first_heading_angle], dtype=torch.float32,
                                          device=self.device)

        t0 = time.time()
        with self._lock:
            output = self.model(
                texts,
                num_frames,
                num_denoising_steps=diffusion_steps,
                multi_prompt=True,
                constraint_lst=constraint_lst,
                cfg_weight=cfg_weight,
                num_samples=num_samples,
                post_processing=post_processing,
                num_transition_frames=num_transition_frames,
                first_heading_angle=heading_tensor,
                return_numpy=True,
            )
        elapsed = time.time() - t0
        log.info("Generation done in %.1fs", elapsed)

        # --- Translate output back from origin + trim history overlap ---
        if num_over > 0:
            # Translate back: model generated at origin, shift to A's end position
            root_origin = history_info.get("root_origin_2d_yup")
            if root_origin:
                ox, oz = root_origin  # Y-up: [X, Z]
                log.info("  Translating output back from origin: offset=[%.3f, %.3f]", ox, oz)
                for key in ["posed_joints"]:
                    if key in output:
                        val = output[key]
                        if val.ndim >= 3:
                            val[..., 0] += ox  # X
                            val[..., 2] += oz  # Z (in Y-up)
                        else:
                            val[..., 0] += ox
                            val[..., 2] += oz
                for key in ["root_positions", "smooth_root_pos"]:
                    if key in output:
                        val = output[key]
                        if val.ndim >= 2:
                            val[..., 0] += ox
                            val[..., 2] += oz

            # Trim the over-generated history frames
            total_out = output["posed_joints"].shape[-3]
            log.info("  Trimming first %d history frames from %d total → %d",
                      num_over, total_out, total_out - num_over)
            for key in output:
                val = output[key]
                if hasattr(val, "shape") and val.ndim >= 2:
                    if val.ndim >= 3:
                        output[key] = val[:, num_over:]
                    else:
                        output[key] = val[num_over:]

            # Prepend history's last frame so Unreal bake has no jump.
            # Only prepend to the keys used by _export_npz: local_rot_mats, root_positions
            history_last_frame = history_info.get("last_frame")
            if history_last_frame:
                log.info("  Prepending history last frame to output")
                for key in ("local_rot_mats", "root_positions"):
                    if key in output and key in history_last_frame:
                        val = output[key]
                        hf = history_last_frame[key]  # (1, ...) — single frame, no batch
                        log.info("    key=%s val.shape=%s hf.shape=%s", key, val.shape, hf.shape)
                        # val: (B, T, ...) or (T, ...) after trim
                        # hf: (1, ...) — single frame without batch
                        # We need to concat along the time axis (axis 1 for batched, axis 0 for unbatched)
                        if val.ndim == hf.ndim:
                            # Both same ndim: concat on axis 0 (time)
                            output[key] = np.concatenate([hf, val], axis=0)
                        elif val.ndim == hf.ndim + 1:
                            # val has batch dim, hf doesn't: add batch dim to hf
                            output[key] = np.concatenate([hf[np.newaxis], val], axis=1)
                        log.info("    result shape=%s", output[key].shape)

        # ---- Export to NPZ ----
        # Use history betas if available so the output character matches the input.
        override_betas = None
        if history_info and history_info.get("betas") is not None:
            override_betas = history_info["betas"]
        npz_bytes = self._export_npz(output, return_format=return_format,
                                     override_betas=override_betas,
                                     coord_out=coord_out)

        # Count frames on an array the export actually packs: root_positions
        # carries the prepended history frame, posed_joints does NOT — counting
        # the latter made X-Kimodo-Meta total_frames off by one whenever
        # history was used.
        actual_frames = int(output["root_positions"].shape[-2])
        meta = {
            "texts": texts,
            "num_frames": [nf - num_over if i == 0 else nf for i, nf in enumerate(num_frames)],
            "seed": seed,
            "diffusion_steps": diffusion_steps,
            "num_samples": num_samples,
            "elapsed_sec": round(elapsed, 2),
            "fps": int(self.model.fps),
            "total_frames": actual_frames,
            "return_format": return_format,
            "history_frames_trimmed": num_over,
        }

        return {"npz_bytes": npz_bytes, "meta": meta}

    # ------------------------------------------------------------------
    # NPZ export
    # ------------------------------------------------------------------
    def _export_npz(self, output: dict, return_format: str = "npz",
                    override_betas: np.ndarray | None = None,
                    coord_out: str = "lzyx") -> bytes:
        """Convert model output to NPZ bytes.

        Two formats:
          - 'npz': DART-compatible (poses/trans/betas/gender/mocap_framerate)
          - 'amass_npz': AMASS-style (root_orient/pose_body/pose_hand/...)

        ``coord_out``: "lzyx" keeps today's wire byte-for-byte (z_up export,
        NO ``coord`` field — downstream loaders key their legacy up-axis fix
        on its absence). "smplx_yup" skips the z_up transform entirely, which
        both yields canonical axes AND kills the (M - I) @ pelvis_offset
        translation bias the z_up packing folds in; the NPZ then carries
        ``coord='smplx_yup'`` (self-describing, COORDINATE_CONTRACT.md §2).
        """
        from kimodo.exports.smplx import get_amass_parameters

        # get_amass_parameters handles batched input; squeeze if needed
        local_rot_mats = output["local_rot_mats"]
        root_positions = output["root_positions"]

        # Ensure batch dim
        if local_rot_mats.ndim == 4:
            local_rot_mats = local_rot_mats[np.newaxis]
            root_positions = root_positions[np.newaxis]

        trans, root_orient, pose_body = get_amass_parameters(
            local_rot_mats, root_positions, self.skeleton,
            z_up=(coord_out != "smplx_yup"),
        )

        # Squeeze batch dim for single sample
        if trans.shape[0] == 1:
            trans = trans[0]
            root_orient = root_orient[0]
            pose_body = pose_body[0]

        T = trans.shape[-2] if trans.ndim >= 2 else trans.shape[0]

        if return_format == "amass_npz":
            return self._pack_amass_npz(trans, root_orient, pose_body, T,
                                       override_betas=override_betas,
                                       coord_out=coord_out)
        else:
            return self._pack_dart_npz(trans, root_orient, pose_body, T,
                                       override_betas=override_betas,
                                       coord_out=coord_out)

    def _pack_dart_npz(
        self,
        trans: np.ndarray,
        root_orient: np.ndarray,
        pose_body: np.ndarray,
        T: int,
        override_betas: np.ndarray | None = None,
        coord_out: str = "lzyx",
    ) -> bytes:
        """Pack into DART-compatible NPZ.

        For SMPLX (22 joints): poses = (T, 165)
            [root_orient(3) | body_pose(63) | jaw(3) | eye(6) | hand(90)]
        For other skeletons (e.g. SOMA 30 joints): poses = (T, 3 + n_body*3 + 99)
            [root_orient(3) | body_pose(n_body*3) | jaw(3) | eye(6) | hand(90)]
        """
        # Skeleton-aware assembly
        n_body_dims = pose_body.shape[-1]  # 63 for SMPLX, 87 for SOMA30, etc.

        parts = [root_orient, pose_body]

        # Add jaw/eye/hand from AMASS converter defaults
        if self.amass_converter and hasattr(self.amass_converter, "default_frame_params"):
            for key, default_shape in [("pose_jaw", 3), ("pose_eye", 6), ("pose_hand", 90)]:
                param = self.amass_converter.default_frame_params.get(
                    key, np.zeros(default_shape)
                )
                if param.ndim == 1:
                    parts.append(np.tile(param, (T, 1)).astype(np.float32))
                else:
                    parts.append(param[:T].astype(np.float32))
        else:
            parts.append(np.zeros((T, 3), dtype=np.float32))   # jaw
            parts.append(np.zeros((T, 6), dtype=np.float32))   # eye
            parts.append(np.zeros((T, 90), dtype=np.float32))  # hand

        poses = np.concatenate(parts, axis=-1).astype(np.float32)
        total_dims = 3 + n_body_dims + 3 + 6 + 90  # root + body + jaw + eye + hand
        assert poses.shape[-1] == total_dims, (
            f"Expected {total_dims} pose dims, got {poses.shape[-1]}"
        )

        # Toggle: True = zero betas (neutral template), False = model default betas
        use_zero_betas = True

        if override_betas is not None:
            betas = override_betas
            log.info("  Using history betas for export: %s", betas[:6])
        elif use_zero_betas:
            betas = np.zeros(16, dtype=np.float32)
        else:
            betas = self.amass_converter.output_dict_base.get(
                "betas", np.zeros(16, dtype=np.float32)
            ) if self.amass_converter else np.zeros(16, dtype=np.float32)

        # Store joint count so consumer knows the layout
        n_body_joints = n_body_dims // 3

        buf = io.BytesIO()
        fields = dict(
            poses=poses,
            trans=trans.astype(np.float32),
            betas=betas.astype(np.float32),
            gender="neutral",
            mocap_framerate=np.int64(30),
            n_body_joints=np.int64(n_body_joints),
        )
        # Self-describing frame tag, CANONICAL OUTPUT ONLY: legacy lzyx stays
        # byte-identical because downstream (UE loader) keys its up-axis fix on
        # the ABSENCE of `coord` in coord-less legacy files.
        if coord_out == "smplx_yup":
            fields["coord"] = "smplx_yup"
        np.savez(buf, **fields)
        return buf.getvalue()

    def _pack_amass_npz(
        self,
        trans: np.ndarray,
        root_orient: np.ndarray,
        pose_body: np.ndarray,
        T: int,
        override_betas: np.ndarray | None = None,
        coord_out: str = "lzyx",
    ) -> bytes:
        """Pack into AMASS-style NPZ."""
        base = dict(self.amass_converter.output_dict_base)
        if override_betas is not None:
            base["betas"] = override_betas
            log.info("  Using history betas for AMASS export: %s", override_betas[:6])
        for key, val in self.amass_converter.default_frame_params.items():
            import einops

            base[key] = einops.repeat(val, "d -> t d", t=T)
        base["mocap_time_length"] = T / 30.0
        base["trans"] = trans.astype(np.float32)
        base["root_orient"] = root_orient.astype(np.float32)
        base["pose_body"] = pose_body.astype(np.float32)
        if coord_out == "smplx_yup":  # see _pack_dart_npz
            base["coord"] = "smplx_yup"

        buf = io.BytesIO()
        np.savez(buf, **base)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # History / continuation
    # ------------------------------------------------------------------
    def build_history_constraints(
        self, npz_path: str, num_history_frames: int = 5, coord_in: str = "lzyx"
    ) -> dict:
        """Build FullBodyConstraintSet from the last N frames of a previous motion NPZ.

        This replicates what Kimodo's _multiprompt does internally between
        segments: extract the last N frames and lock the first N frames of
        the new generation to match.  The model over-generates by
        num_history_frames, then the caller trims those frames from output.

        The NPZ can be either:
          - API output (DART format): keys 'poses', 'trans'
          - Raw kimodo output: keys 'local_rot_mats', 'root_positions', 'posed_joints'
          - AMASS format: keys 'root_orient', 'pose_body', 'trans'

        Args:
            npz_path: Path to the previous motion NPZ file.
            num_history_frames: Number of frames from the END to use as constraints.

        Returns:
            Dict with keys:
              - "constraints": list with one FullBodyConstraintSet
              - "heading_angle": float (radians) from history's last pose
              - "num_over_generate": int (= num_history_frames)
        """
        _ensure_kimodo_imports()
        from kimodo.constraints import FullBodyConstraintSet
        from kimodo.skeleton import fk

        data = np.load(npz_path, allow_pickle=True)
        keys = set(data.keys())

        # Self-describing frame: the file's own `coord` overrides coord_in.
        from .coord import npz_coord
        coord = npz_coord(data, default=coord_in)

        log.info("Loading history NPZ: keys=%s coord=%s", sorted(keys), coord)

        # --- Detect format and extract local_rot_mats + root_positions (Y-up) ---
        if "local_rot_mats" in keys and "root_positions" in keys:
            # Raw kimodo output (already Y-up, no coord conversion needed)
            local_rot_mats = data["local_rot_mats"]  # (T, J, 3, 3)
            root_positions = data["root_positions"]   # (T, 3)
            log.info("  Format: raw kimodo (local_rot_mats + root_positions)")

        elif "poses" in keys and "trans" in keys:
            # DART/API format — need to convert from Z-up back to Y-up
            poses = data["poses"]   # (T, 165 or similar)
            trans = data["trans"]   # (T, 3)

            # Determine body joints from poses layout
            n_body_joints = int(data["n_body_joints"]) if "n_body_joints" in keys else 21
            body_end = 3 + n_body_joints * 3

            root_orient_aa = poses[:, :3]           # (T, 3)
            body_pose_aa = poses[:, 3:body_end]     # (T, n_body*3)

            # Combine into full local rotation axis-angle: (T, J, 3)
            body_reshaped = body_pose_aa.reshape(-1, n_body_joints, 3)
            all_aa = np.concatenate(
                [root_orient_aa[:, np.newaxis, :], body_reshaped], axis=1
            )  # (T, J, 3)

            # Coord-aware decode to Y-up FK inputs (shared)
            from .coord import params_to_yup_fk_inputs
            pelvis_offset = self.skeleton.neutral_joints[self.skeleton.root_idx].cpu().numpy()
            root_positions, local_rot_mats = params_to_yup_fk_inputs(
                all_aa, trans, pelvis_offset, coord=coord
            )

            log.info("  Format: DART/API (poses+trans, %d body joints, coord=%s)", n_body_joints, coord)

        elif "root_orient" in keys and "pose_body" in keys and "trans" in keys:
            # AMASS format — same conversion as DART but fields are separate
            root_orient_aa = data["root_orient"]  # (T, 3)
            body_pose_aa = data["pose_body"]      # (T, 63)
            trans = data["trans"]                 # (T, 3)

            n_body_joints = body_pose_aa.shape[-1] // 3
            body_reshaped = body_pose_aa.reshape(-1, n_body_joints, 3)
            all_aa = np.concatenate(
                [root_orient_aa[:, np.newaxis, :], body_reshaped], axis=1
            )

            from .coord import params_to_yup_fk_inputs
            pelvis_offset = self.skeleton.neutral_joints[self.skeleton.root_idx].cpu().numpy()
            root_positions, local_rot_mats = params_to_yup_fk_inputs(
                all_aa, trans, pelvis_offset, coord=coord
            )

            log.info("  Format: AMASS (root_orient+pose_body+trans, %d body joints, coord=%s)", n_body_joints, coord)

        else:
            raise ValueError(
                f"Unrecognized NPZ format. Keys: {sorted(keys)}. "
                "Expected 'poses'+'trans' (DART), 'root_orient'+'pose_body'+'trans' (AMASS), "
                "or 'local_rot_mats'+'root_positions' (raw kimodo)."
            )

        # --- Take last N frames ---
        T = local_rot_mats.shape[0]
        if num_history_frames > T:
            log.warning(
                "  Requested %d history frames but NPZ only has %d — using all",
                num_history_frames, T,
            )
            num_history_frames = T

        hist_local_rots = local_rot_mats[-num_history_frames:]   # (N, J, 3, 3)
        hist_root_pos = root_positions[-num_history_frames:]      # (N, 3)

        # --- Run FK to get global positions and rotations ---
        device = self.skeleton.device if hasattr(self.skeleton, "device") else "cpu"
        hist_local_rots_t = torch.tensor(hist_local_rots, dtype=torch.float32, device=device)
        hist_root_pos_t = torch.tensor(hist_root_pos, dtype=torch.float32, device=device)

        global_rots, posed_joints, _ = fk(hist_local_rots_t, hist_root_pos_t, self.skeleton)

        # --- Compute smooth root 2D (XZ in Y-up) ---
        smooth_root_2d = posed_joints[:, self.skeleton.root_idx, [0, 2]]  # (N, 2)

        # --- Translate constraints to origin (critical!) ---
        # _multiprompt does this: translate_2d(observed_motion, -last_smooth_root_2d)
        # The model generates motion starting from origin. Without this,
        # the model sees the constraint at position (4m, 0) and interprets
        # the existing momentum as "keep moving forward".
        # We translate to origin, generate, then translate output back.
        root_origin_2d = smooth_root_2d[0].clone()  # first constraint frame's XZ
        log.info("  Translating constraints to origin: offset=[%.3f, %.3f]",
                  root_origin_2d[0], root_origin_2d[1])

        # Shift joint positions to origin (XZ only, Y=height unchanged)
        posed_joints_centered = posed_joints.clone()
        posed_joints_centered[:, :, 0] -= root_origin_2d[0]  # X
        posed_joints_centered[:, :, 2] -= root_origin_2d[1]  # Z (in Y-up)
        smooth_root_2d_centered = smooth_root_2d - root_origin_2d

        # --- Build constraint on frames [0, N) with centered positions ---
        constraint = FullBodyConstraintSet(
            self.skeleton,
            frame_indices=torch.arange(num_history_frames, device=device),
            global_joints_positions=posed_joints_centered,
            global_joints_rots=global_rots,
            smooth_root_2d=smooth_root_2d_centered,
        )

        # --- Compute heading angle from history's last pose ---
        from kimodo.motion_rep.feature_utils import compute_heading_angle
        heading = compute_heading_angle(posed_joints.unsqueeze(0), self.skeleton)  # (1, N)
        heading_angle = float(heading[0, -1].cpu())  # last frame heading

        log.info(
            "  History constraint: %d frames, heading=%.3f rad, root end=[%.3f, %.3f, %.3f]",
            num_history_frames,
            heading_angle,
            hist_root_pos[-1, 0], hist_root_pos[-1, 1], hist_root_pos[-1, 2],
        )

        # --- Save history's last frame (Y-up) for prepending to output ---
        # We store local_rot_mats and root_positions of the LAST frame.
        # After generation, these will go through the same export pipeline
        # (get_amass_parameters) so the output NPZ starts with history's
        # exact last frame — no jump when Unreal bakes.
        last_frame_data = {}
        for key, arr in [
            ("local_rot_mats", hist_local_rots_t.cpu().numpy()),
            ("root_positions", hist_root_pos),
            ("posed_joints", posed_joints.cpu().numpy()),
            ("global_rot_mats", global_rots.cpu().numpy()),
        ]:
            # Take last frame, keep batch-compatible shape
            last = arr[-1:]  # (1, ...) — single frame
            last_frame_data[key] = last

        # Also need foot_contacts and other keys the model outputs
        # We'll fill those with zeros for the single prepended frame
        # (they get populated during generate() trimming step)

        # --- Preserve history betas so export uses the same body shape ---
        # Normalized to the export's 16-slot layout: uploads have carried 10-
        # and 300-coefficient betas historically, and echoing an odd length
        # produced NPZs whose betas field disagreed with our own exports.
        history_betas = None
        if "betas" in keys:
            history_betas = np.array(data["betas"], dtype=np.float32).flatten()
            if history_betas.shape[0] != 16:
                log.warning(
                    "  History betas length %d != 16 — normalizing (pad/truncate)",
                    history_betas.shape[0],
                )
                fixed = np.zeros(16, dtype=np.float32)
                n = min(16, history_betas.shape[0])
                fixed[:n] = history_betas[:n]
                history_betas = fixed
            log.info("  History betas: %s (shape=%s)", history_betas[:6], history_betas.shape)

        return {
            "constraints": [constraint],
            "heading_angle": heading_angle,
            "num_over_generate": num_history_frames,
            "root_origin_2d_yup": [float(root_origin_2d[0]), float(root_origin_2d[1])],
            "last_frame": last_frame_data,
            "betas": history_betas,
        }

    # ------------------------------------------------------------------
    # Constraint building
    # ------------------------------------------------------------------
    def build_constraints(
        self, segments: list, coord_in: str = "lzyx", staged_files: dict | None = None,
        origin_offset_2d: "torch.Tensor | None" = None,
        pose_anchors: list | None = None,
        dense_paths: "dict[int, np.ndarray] | None" = None,
    ) -> list:
        """Build kimodo constraint objects from parsed segment specs.

        Handles trajectory (Root2DConstraintSet) and inbetween (FullBodyConstraintSet).
        When history is used, the model generates at origin. All constraints must be
        translated by origin_offset_2d (the history's root position in Y-up XZ plane)
        so they're in the same origin-centered frame.

        ``pose_anchors`` is a list of ``(abs_frame, x_yup, z_yup)`` tuples extracted
        from external pose NPZs.  These are merged into the trajectory waypoint
        anchors so that intermediate waypoints follow the real path through both
        targets and pose positions.
        """
        _ensure_kimodo_imports()
        from kimodo.constraints import FullBodyConstraintSet, Root2DConstraintSet
        from kimodo.skeleton import fk

        from .coord import root2d_from_pos

        # Spec-level positions/headings arrive in coord_in (uploaded NPZs
        # override per file via their own `coord`).
        def root2d_pos_fn(pos):
            return root2d_from_pos(pos, coord=coord_in)

        staged_files = staged_files or {}
        dense_paths = dense_paths or {}
        constraints = []

        for i, seg in enumerate(segments):
            dense_path = dense_paths.get(i)
            if seg.type.value == "trajectory":
                constraints.extend(self._build_trajectory_constraint(
                    seg, root2d_pos_fn, origin_offset_2d, pose_anchors=pose_anchors,
                    dense_path=dense_path, coord_in=coord_in))

            elif seg.type.value == "text":
                if dense_path is not None:
                    # Dense mode covers text segments uniformly via per-frame
                    # Root2D — no separate pose-anchor trajectory needed.
                    constraints.extend(self._build_dense_root2d_constraint(
                        seg, dense_path, origin_offset_2d))
                elif pose_anchors:
                    constraints.extend(self._build_trajectory_from_pose_anchors(
                        seg, pose_anchors, origin_offset_2d))

            elif seg.type.value == "inbetween":
                constraints.extend(
                    self._build_inbetween_constraint(
                        seg, staged_files, origin_offset_2d,
                        dense_path=dense_path, coord_in=coord_in,
                    )
                )
                if dense_path is not None:
                    # Inbetween already adds a FullBody constraint at its
                    # destination frames; also lay down the per-frame
                    # Root2D so the path is followed between them.
                    constraints.extend(self._build_dense_root2d_constraint(
                        seg, dense_path, origin_offset_2d))

        return constraints

    def _build_trajectory_constraint(self, seg, root2d_pos_fn, origin_offset_2d=None,
                                     pose_anchors: list | None = None,
                                     dense_path: "np.ndarray | None" = None,
                                     coord_in: str = "lzyx") -> list:
        from kimodo.constraints import Root2DConstraintSet

        # Dense mode: one Root2DConstraintSet covering every frame of the
        # segment with the pre-computed smoothed path. Skip the per-chunk
        # interp shim entirely — every chunk already has per-frame guidance.
        if dense_path is not None:
            return self._build_dense_root2d_constraint(seg, dense_path, origin_offset_2d)

        abs_offset = seg.start_frame
        frame_indices = []
        root2d_positions = []
        # abs_frame -> model heading angle (rad), only for waypoints that
        # carry an explicit heading_deg. Used to additionally constrain the
        # root facing direction at those frames (e.g. to force a turn). The
        # heading frames are split into a separate Root2DConstraintSet below
        # because Root2DConstraintSet applies global_root_heading to ALL of
        # its frames or none.
        from .coord import heading_to_model_angle
        heading_by_frame: dict[int, float] = {}

        for pt in seg.points:
            abs_frame = abs_offset + pt.frame
            frame_indices.append(abs_frame)
            rx, rz = root2d_pos_fn(pt.pos)
            root2d_positions.append([rx, rz])
            if getattr(pt, "heading_deg", None) is not None:
                heading_by_frame[abs_frame] = heading_to_model_angle(
                    pt.heading_deg, coord=coord_in)

        if not frame_indices:
            return []

        # Merge pose anchor positions into the waypoint list so that
        # intermediate waypoints follow the real path through both
        # targets AND pose positions (poses may be off the straight line).
        # Only for multi-chunk segments — single chunk doesn't need extra waypoints.
        total_seg_frames = seg.end_frame - seg.start_frame
        if pose_anchors and total_seg_frames > MAX_CHUNK_FRAMES:
            seg_end = seg.end_frame
            for pa_frame, pa_x, pa_z in pose_anchors:
                if abs_offset <= pa_frame < seg_end and pa_frame not in frame_indices:
                    frame_indices.append(pa_frame)
                    root2d_positions.append([pa_x, pa_z])
                    log.info("  Pose anchor merged into trajectory: frame %d pos=[%.3f, %.3f]",
                             pa_frame, pa_x, pa_z)
        elif pose_anchors:
            log.info("  Skipping pose-anchor merge: segment %d frames ≤ %d (single chunk)",
                     total_seg_frames, MAX_CHUNK_FRAMES)

        # --- Intermediate target interpolation ---
        # Insert linearly interpolated waypoints at chunk boundaries so every
        # chunk gets trajectory guidance.  Anchors now include both user targets
        # and pose root positions.
        total_seg_frames = seg.end_frame - seg.start_frame
        frame_indices, root2d_positions = self._insert_intermediate_waypoints(
            frame_indices, root2d_positions, abs_offset, total_seg_frames)

        device = self.skeleton.device if hasattr(self.skeleton, "device") else "cpu"
        root2d_t = torch.tensor(root2d_positions, dtype=torch.float32, device=device)

        # Translate to origin (same as history constraints) so model sees
        # trajectory waypoints relative to origin, not absolute position.
        if origin_offset_2d is not None:
            offset = origin_offset_2d.to(device=device, dtype=torch.float32)
            root2d_t = root2d_t - offset.unsqueeze(0)
            log.info("  Trajectory translated to origin by offset=[%.3f, %.3f]",
                      offset[0].item(), offset[1].item())

        # Partition frames into heading-bearing and position-only. A single
        # Root2DConstraintSet applies global_root_heading to all its frames,
        # so heading frames go in their own set (still carrying position,
        # equal to the trajectory's natural value here → no position
        # tug-of-war). Position-only frames stay in the plain set.
        #
        # NOTE: _insert_intermediate_waypoints shifts every original
        # waypoint by -1 (final_frame = target_frame - 1), so the heading
        # frame's abs index no longer exact-matches heading_by_frame. Map
        # each headed input to its NEAREST output row instead.
        row_angle: dict[int, float] = {}
        for hf, ang in heading_by_frame.items():
            best_i = min(range(len(frame_indices)),
                         key=lambda i: abs(frame_indices[i] - hf))
            row_angle[best_i] = ang
        headed_rows = sorted(row_angle)
        plain_rows = [i for i in range(len(frame_indices)) if i not in row_angle]

        constraints = []
        if plain_rows:
            plain_frames = torch.tensor([frame_indices[i] for i in plain_rows],
                                        dtype=torch.long, device=device)
            constraints.append(Root2DConstraintSet(
                self.skeleton,
                frame_indices=plain_frames,
                smooth_root_2d=root2d_t[plain_rows],
            ))
        if headed_rows:
            headed_frames = torch.tensor([frame_indices[i] for i in headed_rows],
                                         dtype=torch.long, device=device)
            angles = torch.tensor([row_angle[i] for i in headed_rows],
                                  dtype=torch.float32, device=device)
            grh = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
            constraints.append(Root2DConstraintSet(
                self.skeleton,
                frame_indices=headed_frames,
                smooth_root_2d=root2d_t[headed_rows],
                global_root_heading=grh,
            ))
            log.info("  Heading-constrained %d frame(s): %s (model angle rad: %s)",
                     len(headed_rows),
                     [frame_indices[i] for i in headed_rows],
                     [round(float(a), 3) for a in angles])

        log.info("  Built Root2D constraint: %d waypoints (%d heading), frames %s",
                 len(frame_indices), len(headed_rows), frame_indices)
        return constraints

    def _build_dense_root2d_constraint(
        self, seg, dense_path: "np.ndarray", origin_offset_2d=None,
    ) -> list:
        """Build a single Root2DConstraintSet with per-frame XZ guidance.

        ``dense_path`` is segment-local ``(seg_len, 2)`` in Y-up. Frame indices
        in the resulting constraint are absolute (``[seg.start_frame,
        seg.end_frame)``); ``_multiprompt.crop_move`` slices them per chunk.
        """
        from kimodo.constraints import Root2DConstraintSet

        seg_len = seg.end_frame - seg.start_frame
        if dense_path.shape[0] != seg_len:
            raise ValueError(
                f"dense_path length {dense_path.shape[0]} does not match "
                f"segment length {seg_len}"
            )

        device = self.skeleton.device if hasattr(self.skeleton, "device") else "cpu"
        xz = torch.from_numpy(np.ascontiguousarray(dense_path)).to(
            device=device, dtype=torch.float32
        )
        if origin_offset_2d is not None:
            offset = origin_offset_2d.to(device=device, dtype=torch.float32)
            xz = xz - offset.unsqueeze(0)
            log.info(
                "  Dense Root2D[seg @ frame %d..%d]: translated by offset=[%.3f, %.3f]",
                seg.start_frame, seg.end_frame,
                float(offset[0].item()), float(offset[1].item()),
            )
        frame_indices = torch.arange(
            seg.start_frame, seg.end_frame, dtype=torch.long, device=device,
        )
        constraint = Root2DConstraintSet(
            self.skeleton,
            frame_indices=frame_indices,
            smooth_root_2d=xz,
        )
        log.info(
            "  Built dense Root2D constraint: %d frames [%d, %d)",
            seg_len, seg.start_frame, seg.end_frame,
        )
        return [constraint]

    def _build_trajectory_from_pose_anchors(self, seg, pose_anchors, origin_offset_2d=None) -> list:
        """Build Root2DConstraintSet from pose anchor positions for a text segment.

        When only poses are provided (no explicit targets), pose root positions
        serve as trajectory waypoints so the character moves through them.

        Only applies for multi-chunk segments (> MAX_CHUNK_FRAMES).  For single-chunk
        segments (e.g. bridge inbetween), the FullBody constraint alone handles spatial
        positioning — adding Root2D waypoints would conflict.
        """
        from kimodo.constraints import Root2DConstraintSet

        abs_offset = seg.start_frame
        seg_end = seg.end_frame
        total_seg_frames = seg_end - abs_offset

        # Single chunk: FullBody constraint is sufficient, skip trajectory
        if total_seg_frames <= MAX_CHUNK_FRAMES:
            log.info("  Skipping pose-anchor trajectory: segment %d frames ≤ %d (single chunk)",
                     total_seg_frames, MAX_CHUNK_FRAMES)
            return []

        frame_indices = []
        root2d_positions = []

        for pa_frame, pa_x, pa_z in pose_anchors:
            if abs_offset <= pa_frame < seg_end:
                frame_indices.append(pa_frame)
                root2d_positions.append([pa_x, pa_z])

        if not frame_indices:
            return []

        log.info("  Building trajectory from %d pose anchor(s)", len(frame_indices))

        total_seg_frames = seg_end - abs_offset
        frame_indices, root2d_positions = self._insert_intermediate_waypoints(
            frame_indices, root2d_positions, abs_offset, total_seg_frames)

        device = self.skeleton.device if hasattr(self.skeleton, "device") else "cpu"
        root2d_t = torch.tensor(root2d_positions, dtype=torch.float32, device=device)

        if origin_offset_2d is not None:
            offset = origin_offset_2d.to(device=device, dtype=torch.float32)
            root2d_t = root2d_t - offset.unsqueeze(0)

        constraint = Root2DConstraintSet(
            self.skeleton,
            frame_indices=torch.tensor(frame_indices, dtype=torch.long, device=device),
            smooth_root_2d=root2d_t,
        )
        log.info("  Built Root2D from pose anchors: %d waypoints, frames %s",
                 len(frame_indices), frame_indices)
        return [constraint]

    @staticmethod
    def _compute_chunk_boundaries(total_frames: int) -> list[int]:
        """Return absolute frame indices where balanced chunks start.

        Mirrors the logic in ``_split_long_segments`` so that intermediate
        waypoints land inside the correct chunks.

        Example: 750 frames → chunks [250, 250, 250] → boundaries [0, 250, 500]
        """
        if total_frames <= MAX_CHUNK_FRAMES:
            return [0]
        n_chunks = math.ceil(total_frames / MAX_CHUNK_FRAMES)
        base = total_frames // n_chunks
        extra = total_frames % n_chunks
        boundaries = []
        cursor = 0
        for i in range(n_chunks):
            boundaries.append(cursor)
            cursor += base + (1 if i < extra else 0)
        return boundaries

    @staticmethod
    def _insert_intermediate_waypoints(
        frame_indices: list[int],
        root2d_positions: list[list[float]],
        abs_offset: int,
        total_segment_frames: int = 0,
    ) -> tuple[list[int], list[list[float]]]:
        """Insert linearly interpolated waypoints at chunk boundaries.

        Computes the balanced chunk boundaries for the segment's total frame
        count, then ensures every chunk that falls between two user waypoints
        gets an interpolated target at the boundary so the model has guidance
        in every chunk.

        The first "prev" is the segment start (abs_offset) at position (0, 0).
        """
        if not frame_indices:
            return frame_indices, root2d_positions

        # Compute where chunks will actually split
        if total_segment_frames > MAX_CHUNK_FRAMES:
            chunk_bounds = KimodoService._compute_chunk_boundaries(total_segment_frames)
            # Convert to absolute frames
            chunk_bounds = [abs_offset + b for b in chunk_bounds]
        else:
            chunk_bounds = []

        anchors = list(zip(frame_indices, root2d_positions))
        anchors.sort(key=lambda a: a[0])

        prev_frame = abs_offset
        prev_pos = [0.0, 0.0]

        out_frames: list[int] = []
        out_positions: list[list[float]] = []
        num_inserted = 0

        for target_frame, target_pos in anchors:
            gap = target_frame - prev_frame
            if gap > 0 and chunk_bounds:
                # Insert at every chunk boundary that falls between prev and target
                for cb in chunk_bounds:
                    # Place 1 frame before boundary so it lands inside the chunk
                    inter_frame = cb - 1
                    if inter_frame <= prev_frame or inter_frame >= target_frame:
                        continue
                    t = (inter_frame - prev_frame) / gap
                    inter_pos = [
                        prev_pos[0] + t * (target_pos[0] - prev_pos[0]),
                        prev_pos[1] + t * (target_pos[1] - prev_pos[1]),
                    ]
                    out_frames.append(inter_frame)
                    out_positions.append(inter_pos)
                    num_inserted += 1

            # Keep the original waypoint
            final_frame = max(target_frame - 1, prev_frame)
            if not out_frames or out_frames[-1] != final_frame:
                out_frames.append(final_frame)
                out_positions.append(target_pos)
            prev_frame = target_frame
            prev_pos = target_pos

        if num_inserted > 0:
            log.info(
                "  Interpolated %d intermediate waypoints at chunk boundaries, total %d waypoints",
                num_inserted, len(out_frames),
            )

        return out_frames, out_positions

    # ------------------------------------------------------------------
    # Dense-path computation (mirrors demo's "Make Smooth Path")
    # ------------------------------------------------------------------
    def _compute_dense_path(
        self,
        anchor_local_frames: list[int],
        anchor_xz_yup: list[list[float]],
        seg_len: int,
        smooth: bool = True,
    ) -> "np.ndarray | None":
        """Build a per-frame dense XZ trajectory for one segment.

        Args:
            anchor_local_frames: segment-local frame indices in ``[0, seg_len)``.
            anchor_xz_yup: matching ``[x, z]`` positions in Kimodo Y-up.
            seg_len: total frames in the segment.
            smooth: if True and there are >= 3 anchors, apply the ADMM smoother
                (``kimodo.motion_rep.smooth_root.smooth_signal`` with margin
                ``DENSE_PATH_MARGIN_M``).

        Returns:
            ``(seg_len, 2)`` float32 array of XZ values, or ``None`` if fewer
            than 2 unique anchors (sparse mode falls back).

        Auto-bypass: if anchors already cover every frame in ``[0, seg_len)``
        the supplied values are returned verbatim — no interpolation, no
        smoothing — so callers that pre-computed a path get exactly what they
        sent.
        """
        if not anchor_local_frames or len(anchor_local_frames) < 2:
            return None

        # Sort + dedup by frame (keep last write — caller controls ordering)
        by_frame: dict[int, list[float]] = {}
        for f, pos in zip(anchor_local_frames, anchor_xz_yup):
            if 0 <= f < seg_len:
                by_frame[int(f)] = [float(pos[0]), float(pos[1])]
        if len(by_frame) < 2:
            return None

        sorted_frames = np.array(sorted(by_frame.keys()), dtype=np.int64)
        sorted_xz = np.array([by_frame[f] for f in sorted_frames], dtype=np.float32)

        # Auto-bypass: every frame already provided → trust the caller
        if (
            sorted_frames.shape[0] == seg_len
            and sorted_frames[0] == 0
            and sorted_frames[-1] == seg_len - 1
            and (sorted_frames == np.arange(seg_len)).all()
        ):
            log.info(
                "  Dense-path: caller provided %d/%d frames → bypass smoothing",
                seg_len, seg_len,
            )
            return sorted_xz.astype(np.float32)

        # Extend anchors to cover [0, seg_len-1] for interp1d (clamp by repeat)
        if sorted_frames[0] > 0:
            sorted_frames = np.concatenate(([0], sorted_frames))
            sorted_xz = np.concatenate((sorted_xz[:1], sorted_xz), axis=0)
        if sorted_frames[-1] < seg_len - 1:
            sorted_frames = np.concatenate((sorted_frames, [seg_len - 1]))
            sorted_xz = np.concatenate((sorted_xz, sorted_xz[-1:]), axis=0)

        from scipy.interpolate import interp1d
        t = np.arange(seg_len, dtype=np.int64)
        dense_x = interp1d(sorted_frames, sorted_xz[:, 0], kind="linear")(t)
        dense_z = interp1d(sorted_frames, sorted_xz[:, 1], kind="linear")(t)
        dense = np.stack([dense_x, dense_z], axis=1).astype(np.float32)

        if smooth and len(by_frame) >= DENSE_PATH_MIN_ANCHORS_FOR_SMOOTH:
            try:
                from kimodo.motion_rep.smooth_root import smooth_signal
                margins = np.full(seg_len, DENSE_PATH_MARGIN_M, dtype=np.float32)
                dense = smooth_signal(dense, margins).astype(np.float32)
                log.info(
                    "  Dense-path: %d anchors → %d frames smoothed (ADMM, margin=%.2fm)",
                    len(by_frame), seg_len, DENSE_PATH_MARGIN_M,
                )
            except Exception as e:
                # If smoothing fails for any reason (e.g., scipy issue),
                # fall back to the linearly interpolated path rather than
                # aborting the whole request.
                log.warning("  Dense-path: ADMM smoothing failed (%s); using linear interp only", e)
        else:
            log.info(
                "  Dense-path: %d anchors → %d frames (linear only, smooth=%s)",
                len(by_frame), seg_len, smooth,
            )
        return dense

    def _extract_inbetween_anchors(
        self, seg, staged_files: dict, coord_in: str = "lzyx"
    ) -> "list[tuple[int, float, float]]":
        """Pull Full-Body root XZ (Y-up Kimodo) at this inbetween's destination
        frames from the referenced NPZ — lightweight, no FK.

        Returns ``[(seg_local_frame, x_yup, z_yup), ...]`` matching what
        ``_build_inbetween_constraint`` would assign to ``smooth_root_2d`` at
        each constrained frame.

        For standard SMPL-X/SOMA/G1 skeletons the pelvis neutral has zero XZ
        offset, so ``root2d_from_pos(trans[i], coord)`` equals
        ``posed_joints[i, root_idx, [0, 2]]`` to numerical precision.
        """
        from .coord import npz_coord, root2d_from_pos

        ref_spec = seg.ref_smplx
        if ref_spec is None or ref_spec.file_name not in staged_files:
            return []
        try:
            ref_data = np.load(staged_files[ref_spec.file_name], allow_pickle=True)
            ref_trans = ref_data["trans"]
            ref_T = ref_trans.shape[0]
            ref_coord = npz_coord(ref_data, default=coord_in)
        except Exception as e:
            log.warning("  Could not read inbetween ref '%s' for dense-path anchors: %s",
                        ref_spec.file_name, e)
            return []

        n_frames = seg.end_frame - seg.start_frame
        src_start = ref_spec.smplx_src_start_frame
        mask_mode = seg.mask_mode or "endpoints"

        if mask_mode == "none":
            return []
        elif mask_mode == "endpoints":
            dest_frames = [0, n_frames - 1]
            src_frames = [src_start, min(src_start + ref_T - 1, ref_T - 1)]
        elif mask_mode == "all":
            actual_len = min(n_frames, ref_T - src_start)
            dest_frames = list(range(actual_len))
            src_frames = [src_start + i for i in range(actual_len)]
        elif mask_mode == "keyframes":
            dest_frames = list(seg.keyframes or [])
            src_frames = list(seg.keyframes_src_frames) if seg.keyframes_src_frames \
                else [src_start + kf for kf in dest_frames]
        else:
            return []

        anchors: list[tuple[int, float, float]] = []
        for df, sf in zip(dest_frames, src_frames):
            if sf < 0 or sf >= ref_T:
                continue
            if df < 0 or df >= n_frames:
                continue
            rx, rz = root2d_from_pos(ref_trans[sf], coord=ref_coord)
            anchors.append((int(df), rx, rz))
        return anchors

    def precompute_dense_paths(
        self,
        segments: list,
        pose_anchors: "list[tuple[int, float, float]]",
        staged_files: dict,
        enabled: bool = True,
        coord_in: str = "lzyx",
    ) -> "dict[int, np.ndarray]":
        """Compute one dense per-frame XZ trajectory per segment that has
        enough anchors.

        Anchors per segment come from:
          • trajectory points (``seg.points`` for trajectory segments),
          • Full-Body root XZ from the inbetween's reference NPZ at the
            destination frames (for inbetween segments),
          • external pose-constraint root XZ (``pose_anchors``) that fall
            inside the segment range.

        Full-Body anchors WIN over trajectory anchors at the same segment-local
        frame; a warning is logged if they disagree beyond
        ``DENSE_PATH_CONFLICT_WARN_M`` metres.

        Returns ``{segment_index → (seg_len, 2) np.ndarray (Y-up)}`` for
        every segment with >= 2 unique anchors. Other segments are absent
        from the returned dict and callers fall back to sparse mode for them.
        """
        if not enabled:
            return {}

        from .coord import root2d_from_pos

        dense_paths: dict[int, np.ndarray] = {}
        for i, seg in enumerate(segments):
            seg_start = seg.start_frame
            seg_end = seg.end_frame
            seg_len = seg_end - seg_start

            # Step 1: trajectory waypoints (segment-local frames)
            traj_by_local: dict[int, list[float]] = {}
            if seg.type.value == "trajectory" and seg.points:
                for pt in seg.points:
                    if 0 <= pt.frame < seg_len:
                        rx, rz = root2d_from_pos(pt.pos, coord=coord_in)
                        traj_by_local[int(pt.frame)] = [rx, rz]

            # Step 2: Full-Body root anchors from inbetween reference NPZ
            fb_by_local: dict[int, list[float]] = {}
            if seg.type.value == "inbetween":
                for local_f, x, z in self._extract_inbetween_anchors(seg, staged_files, coord_in=coord_in):
                    fb_by_local[int(local_f)] = [x, z]

            # Step 3: external pose anchors that fall inside this segment
            for abs_f, x, z in pose_anchors:
                if seg_start <= abs_f < seg_end:
                    local_f = int(abs_f - seg_start)
                    fb_by_local[local_f] = [x, z]

            # Step 4: merge — Full-Body wins over trajectory on conflict
            merged: dict[int, list[float]] = dict(traj_by_local)
            for local_f, fb_xz in fb_by_local.items():
                if local_f in merged:
                    dx = fb_xz[0] - merged[local_f][0]
                    dz = fb_xz[1] - merged[local_f][1]
                    dist = float(np.hypot(dx, dz))
                    if dist > DENSE_PATH_CONFLICT_WARN_M:
                        log.warning(
                            "  Dense-path: trajectory waypoint at seg=%d local-frame=%d "
                            "overridden by Full-Body anchor (Δ=%.3fm > %.2fm threshold)",
                            i, local_f, dist, DENSE_PATH_CONFLICT_WARN_M,
                        )
                merged[local_f] = fb_xz

            if len(merged) < 2:
                if merged:
                    log.info(
                        "  Dense-path: segment %d has only %d anchor(s) → sparse mode",
                        i, len(merged),
                    )
                continue

            frames = sorted(merged.keys())
            xzs = [merged[f] for f in frames]
            dense = self._compute_dense_path(frames, xzs, seg_len, smooth=True)
            if dense is not None:
                dense_paths[i] = dense
                log.info(
                    "  Dense-path[seg=%d]: %d anchors → %d-frame dense path "
                    "(traj=%d, fb=%d)",
                    i, len(merged), seg_len, len(traj_by_local), len(fb_by_local),
                )
        return dense_paths

    def _build_inbetween_constraint(self, seg, staged_files: dict, origin_offset_2d=None,
                                    coord_in: str = "lzyx",
                                    dense_path: "np.ndarray | None" = None) -> list:
        """Build FullBodyConstraintSet from an inbetween segment.

        Accepts the same request format as DART API:
          ref_smplx: {file_name, smplx_src_start_frame}
          mask_mode: "endpoints" | "keyframes" | "all" | "none"
          keyframes: [int, ...]              (segment-local destination frames)
          keyframes_src_frames: [int, ...]   (source frames in ref NPZ)
        """
        from kimodo.constraints import FullBodyConstraintSet
        from kimodo.skeleton import fk

        ref_spec = seg.ref_smplx
        if ref_spec.file_name not in staged_files:
            raise ValueError(f"ref_smplx references '{ref_spec.file_name}' but it was not uploaded")

        # Load reference NPZ (DART format: poses, trans); frame per file
        from .coord import npz_coord
        ref_data = np.load(staged_files[ref_spec.file_name], allow_pickle=True)
        ref_poses = ref_data["poses"]    # (T, 165)
        ref_trans = ref_data["trans"]    # (T, 3)
        ref_T = ref_poses.shape[0]
        ref_coord = npz_coord(ref_data, default=coord_in)
        src_start = ref_spec.smplx_src_start_frame

        log.info("  Inbetween ref NPZ: %d frames, src_start=%d", ref_T, src_start)

        # Determine which frames to constrain
        n_frames = seg.end_frame - seg.start_frame
        mask_mode = seg.mask_mode or "endpoints"

        if mask_mode == "none":
            log.info("  mask_mode=none → no constraints")
            return []
        elif mask_mode == "endpoints":
            dest_frames = [0, n_frames - 1]
            # Use first and last frame of the ref NPZ (not offset by n_frames)
            src_frames = [src_start, min(src_start + ref_T - 1, ref_T - 1)]
        elif mask_mode == "all":
            # Constrain every frame — need enough ref frames
            actual_len = min(n_frames, ref_T - src_start)
            dest_frames = list(range(actual_len))
            src_frames = [src_start + i for i in range(actual_len)]
        elif mask_mode == "keyframes":
            dest_frames = list(seg.keyframes)
            if seg.keyframes_src_frames:
                src_frames = list(seg.keyframes_src_frames)
            else:
                src_frames = [src_start + kf for kf in dest_frames]
        else:
            raise ValueError(f"Unknown mask_mode: {mask_mode}")

        # Validate source frames
        for sf in src_frames:
            if sf < 0 or sf >= ref_T:
                raise ValueError(f"Source frame {sf} out of range [0, {ref_T})")

        log.info("  mask_mode=%s: %d keyframes, dest=%s, src=%s",
                  mask_mode, len(dest_frames), dest_frames, src_frames)

        # Extract poses at keyframe source frames
        n_body_joints = (ref_poses.shape[-1] - 3 - 99) // 3  # typically 21
        kf_poses = ref_poses[src_frames]     # (K, 165)
        kf_trans = ref_trans[src_frames]     # (K, 3)

        # Parse into root_orient + body_pose (axis-angle)
        kf_root_aa = kf_poses[:, :3]                    # (K, 3)
        kf_body_aa = kf_poses[:, 3:3 + n_body_joints * 3]  # (K, n_body*3)
        kf_body_aa = kf_body_aa.reshape(-1, n_body_joints, 3)  # (K, n_body, 3)

        # Combine into full local rotation: (K, J, 3)
        kf_all_aa = np.concatenate(
            [kf_root_aa[:, np.newaxis, :], kf_body_aa], axis=1
        )  # (K, J, 3)

        # Coord-aware decode to Y-up FK inputs (shared)
        from .coord import params_to_yup_fk_inputs
        pelvis_offset = self.skeleton.neutral_joints[self.skeleton.root_idx].cpu().numpy()
        root_positions, local_rot_mats = params_to_yup_fk_inputs(
            kf_all_aa, kf_trans, pelvis_offset, coord=ref_coord
        )

        # FK to get global positions and rotations
        device = self.skeleton.device if hasattr(self.skeleton, "device") else "cpu"
        local_rots_t = torch.tensor(local_rot_mats, dtype=torch.float32, device=device)
        root_pos_t = torch.tensor(root_positions, dtype=torch.float32, device=device)
        global_rots, posed_joints, _ = fk(local_rots_t, root_pos_t, self.skeleton)

        # Build constraint with absolute frame indices
        abs_offset = seg.start_frame
        abs_frames = [abs_offset + df for df in dest_frames]

        smooth_root_2d = posed_joints[:, self.skeleton.root_idx, [0, 2]]

        # Dense-path override: when a per-frame smooth path is supplied, force
        # this FullBody's smooth_root_2d to the dense path's value at each
        # destination frame so the Root2D dense constraint and FullBody agree
        # bit-for-bit. The pose geometry (joint rotations + global_joints_positions)
        # is unchanged; only the per-frame root XZ reference shifts.
        if dense_path is not None:
            try:
                local_dest = np.asarray(dest_frames, dtype=np.int64)
                dense_local = dense_path[local_dest]  # (K, 2) in Y-up
                smooth_root_2d = torch.from_numpy(
                    np.ascontiguousarray(dense_local)
                ).to(device=device, dtype=torch.float32)
                log.info("  Inbetween smooth_root_2d overridden from dense path "
                         "at %d frames", len(dest_frames))
            except Exception as e:
                log.warning("  Inbetween dense_path override failed (%s); using FK root XZ", e)

        # Translate to origin (same as history constraints)
        if origin_offset_2d is not None:
            offset = origin_offset_2d.to(device=device, dtype=torch.float32)
            posed_joints[:, :, 0] -= offset[0]  # X in Y-up
            posed_joints[:, :, 2] -= offset[1]  # Z in Y-up
            smooth_root_2d = smooth_root_2d - offset.unsqueeze(0)
            log.info("  Inbetween translated to origin by offset=[%.3f, %.3f]",
                      offset[0].item(), offset[1].item())

        constraint = FullBodyConstraintSet(
            self.skeleton,
            frame_indices=torch.tensor(abs_frames, dtype=torch.long, device=device),
            global_joints_positions=posed_joints,
            global_joints_rots=global_rots,
            smooth_root_2d=smooth_root_2d,
        )

        log.info("  Built FullBody constraint: %d keyframes at abs frames %s", len(abs_frames), abs_frames)
        return [constraint]

    def extract_pose_root2d(
        self,
        pose_constraints: list,
        staged_files: dict,
        frame_offset: int = 0,
        coord_in: str = "lzyx",
    ) -> list[tuple[int, float, float]]:
        """Extract root 2D positions from pose NPZs (lightweight, no FK).

        Returns list of ``(abs_frame, x_yup, z_yup)`` tuples that can be used
        as additional anchors for trajectory waypoint interpolation.
        """
        from .coord import npz_coord, root2d_from_pos

        results = []
        for pc in pose_constraints:
            if pc.file_name not in staged_files:
                continue
            ref_data = np.load(staged_files[pc.file_name], allow_pickle=True)
            ref_trans = ref_data["trans"]  # (T, 3) in the file's own frame
            src_frame = pc.smplx_src_frame
            if src_frame >= ref_trans.shape[0]:
                continue

            # Root 2D in Kimodo Y-up, per the file's own declared frame
            ref_coord = npz_coord(ref_data, default=coord_in)
            rx, rz = root2d_from_pos(ref_trans[src_frame], coord=ref_coord)
            abs_frame = pc.frame + frame_offset
            results.append((abs_frame, rx, rz))

        return results

    def build_pose_constraints(
        self,
        pose_constraints: list,
        staged_files: dict,
        frame_offset: int = 0,
        origin_offset_2d: "torch.Tensor | None" = None,
        segments: list | None = None,
        dense_paths: "dict[int, np.ndarray] | None" = None,
        coord_in: str = "lzyx",
    ) -> list:
        """Build FullBodyConstraintSet(s) from ExternalPoseConstraint entries.

        Each entry specifies a single frame + an uploaded NPZ.  The constraints
        are returned as a flat list so they can be appended to ``constraint_lst``
        and applied on top of trajectory or text segments.

        When ``segments`` and ``dense_paths`` are supplied, each pose's
        ``smooth_root_2d`` is overridden to the dense path's value at the
        absolute frame (so it matches the per-frame Root2D constraint of the
        enclosing segment, exactly like the demo's "Make Smooth Path" wiring).
        """
        _ensure_kimodo_imports()
        from kimodo.constraints import FullBodyConstraintSet
        from kimodo.skeleton import fk

        constraints = []
        for pc in pose_constraints:
            if pc.file_name not in staged_files:
                raise ValueError(
                    f"pose_constraint references '{pc.file_name}' but it was not uploaded"
                )

            from .coord import npz_coord
            ref_data = np.load(staged_files[pc.file_name], allow_pickle=True)
            ref_poses = ref_data["poses"]   # (T, 165)
            ref_trans = ref_data["trans"]    # (T, 3)
            ref_coord = npz_coord(ref_data, default=coord_in)

            src_frame = pc.smplx_src_frame
            if src_frame >= ref_poses.shape[0]:
                raise ValueError(
                    f"smplx_src_frame {src_frame} out of range [0, {ref_poses.shape[0]})"
                )

            # Extract single frame
            kf_poses = ref_poses[src_frame:src_frame + 1]   # (1, 165)
            kf_trans = ref_trans[src_frame:src_frame + 1]    # (1, 3)

            n_body_joints = (
                int(ref_data["n_body_joints"]) if "n_body_joints" in ref_data
                else (ref_poses.shape[-1] - 3 - 99) // 3  # same heuristic as _build_inbetween_constraint
            )
            kf_root_aa = kf_poses[:, :3]
            kf_body_aa = kf_poses[:, 3:3 + n_body_joints * 3].reshape(1, n_body_joints, 3)
            kf_all_aa = np.concatenate([kf_root_aa[:, np.newaxis, :], kf_body_aa], axis=1)

            # Coord-aware decode to Y-up FK inputs (shared)
            from .coord import params_to_yup_fk_inputs
            pelvis_offset = self.skeleton.neutral_joints[self.skeleton.root_idx].cpu().numpy()
            root_positions, local_rot_mats = params_to_yup_fk_inputs(
                kf_all_aa, kf_trans, pelvis_offset, coord=ref_coord
            )

            device = self.skeleton.device if hasattr(self.skeleton, "device") else "cpu"
            local_rots_t = torch.tensor(local_rot_mats, dtype=torch.float32, device=device)
            root_pos_t = torch.tensor(root_positions, dtype=torch.float32, device=device)
            global_rots, posed_joints, _ = fk(local_rots_t, root_pos_t, self.skeleton)

            smooth_root_2d = posed_joints[:, self.skeleton.root_idx, [0, 2]]

            abs_frame = pc.frame + frame_offset

            # Dense-path override: look up which segment contains abs_frame and
            # fetch its dense path's XZ at the segment-local index. Falls back
            # silently to the FK root XZ above if no match.
            if segments and dense_paths:
                for seg_idx, seg in enumerate(segments):
                    if seg.start_frame <= abs_frame < seg.end_frame:
                        dp = dense_paths.get(seg_idx)
                        if dp is not None:
                            local_f = abs_frame - seg.start_frame
                            dense_xz = dp[local_f]  # (2,) Y-up
                            smooth_root_2d = torch.tensor(
                                [[float(dense_xz[0]), float(dense_xz[1])]],
                                dtype=torch.float32, device=device,
                            )
                            log.info(
                                "  Pose at abs frame %d: smooth_root_2d "
                                "overridden from dense path of seg %d",
                                abs_frame, seg_idx,
                            )
                        break

            # Translate to origin
            if origin_offset_2d is not None:
                offset = origin_offset_2d.to(device=device, dtype=torch.float32)
                posed_joints[:, :, 0] -= offset[0]
                posed_joints[:, :, 2] -= offset[1]
                smooth_root_2d = smooth_root_2d - offset.unsqueeze(0)

            constraint = FullBodyConstraintSet(
                self.skeleton,
                frame_indices=torch.tensor([abs_frame], dtype=torch.long, device=device),
                global_joints_positions=posed_joints,
                global_joints_rots=global_rots,
                smooth_root_2d=smooth_root_2d,
            )
            constraints.append(constraint)
            log.info("  Built external pose constraint at abs frame %d (from %s frame %d)",
                     abs_frame, pc.file_name, src_frame)

        return constraints
