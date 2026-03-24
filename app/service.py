# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core inference service wrapping kimodo model loading, constraint building, and NPZ export."""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
from typing import Optional

import numpy as np
import torch

log = logging.getLogger("kimodo_api.service")

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
        first_heading_angle = None
        if history_info:
            num_over = history_info["num_over_generate"]
            first_heading_angle = history_info.get("heading_angle")
            # Add overlap frames to the first segment
            num_frames = num_frames.copy()
            num_frames[0] += num_over
            log.info("  History: over-generating %d extra frames on segment 0, heading=%.3f",
                      num_over, first_heading_angle if first_heading_angle is not None else 0.0)

        total_frames = sum(num_frames)
        log.info(
            "Generating: %d segment(s), %d total frames (%.1fs), "
            "seed=%d, steps=%d, samples=%d, post_process=%s",
            len(texts),
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
        npz_bytes = self._export_npz(output, return_format=return_format)

        actual_frames = int(output["posed_joints"].shape[-3])
        meta = {
            "texts": texts,
            "num_frames": [nf - num_over if i == 0 else nf for i, nf in enumerate(num_frames)],
            "seed": seed,
            "diffusion_steps": diffusion_steps,
            "num_samples": num_samples,
            "elapsed_sec": round(elapsed, 2),
            "model": self.model_name,
            "skeleton": self.skeleton.name,
            "fps": int(self.model.fps),
            "total_frames": actual_frames,
            "return_format": return_format,
            "history_frames_trimmed": num_over,
        }

        return {"npz_bytes": npz_bytes, "meta": meta}

    # ------------------------------------------------------------------
    # NPZ export
    # ------------------------------------------------------------------
    def _export_npz(self, output: dict, return_format: str = "npz") -> bytes:
        """Convert model output to NPZ bytes.

        Two formats:
          - 'npz': DART-compatible (poses/trans/betas/gender/mocap_framerate)
          - 'amass_npz': AMASS-style (root_orient/pose_body/pose_hand/...)
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
            local_rot_mats, root_positions, self.skeleton, z_up=True
        )

        # Kimodo's get_amass_parameters includes rot_z_180 which puts the character
        # facing -Y (backward in DART/UE convention). DART expects +Y = forward.
        # Undo rot_z_180 on BOTH translation and root orient to match DART convention.
        # rot_z_180 negates X and Y in translation, and rotates root 180° around Z.
        import torch as _torch
        from kimodo.geometry import axis_angle_to_matrix, matrix_to_axis_angle

        # Undo translation: negate X and Y (rot_z_180 negated them)
        if trans.ndim == 3:
            trans[:, :, 0] *= -1.0
            trans[:, :, 1] *= -1.0
        else:
            trans[:, 0] *= -1.0
            trans[:, 1] *= -1.0

        # Undo root orient: apply inverse rot_z_180 (= rot_z_180 itself, it's an involution)
        _rot_z_180 = _torch.tensor([
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=_torch.float32)

        def _undo_rot_z_180(ro_aa):
            ro_t = _torch.tensor(ro_aa, dtype=_torch.float32)
            ro_mat = axis_angle_to_matrix(ro_t)
            ro_fixed = _torch.einsum('ij,...jk->...ik', _rot_z_180, ro_mat)
            return matrix_to_axis_angle(ro_fixed).numpy()

        if root_orient.ndim == 3:
            for b in range(root_orient.shape[0]):
                root_orient[b] = _undo_rot_z_180(root_orient[b])
        else:
            root_orient = _undo_rot_z_180(root_orient)

        # Squeeze batch dim for single sample
        if trans.shape[0] == 1:
            trans = trans[0]
            root_orient = root_orient[0]
            pose_body = pose_body[0]

        T = trans.shape[-2] if trans.ndim >= 2 else trans.shape[0]

        if return_format == "amass_npz":
            return self._pack_amass_npz(trans, root_orient, pose_body, T)
        else:
            return self._pack_dart_npz(trans, root_orient, pose_body, T)

    def _pack_dart_npz(
        self,
        trans: np.ndarray,
        root_orient: np.ndarray,
        pose_body: np.ndarray,
        T: int,
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

        betas = np.zeros(16, dtype=np.float32)
        if self.amass_converter:
            betas = self.amass_converter.output_dict_base.get(
                "betas", np.zeros(16, dtype=np.float32)
            )

        # Store skeleton info so consumer knows the layout
        n_body_joints = n_body_dims // 3
        skeleton_name = self.skeleton.name if self.skeleton else "unknown"

        buf = io.BytesIO()
        np.savez(
            buf,
            poses=poses,
            trans=trans.astype(np.float32),
            betas=betas.astype(np.float32),
            gender="neutral",
            mocap_framerate=np.int64(30),
            n_body_joints=np.int64(n_body_joints),
            skeleton=skeleton_name,
        )
        return buf.getvalue()

    def _pack_amass_npz(
        self,
        trans: np.ndarray,
        root_orient: np.ndarray,
        pose_body: np.ndarray,
        T: int,
    ) -> bytes:
        """Pack into AMASS-style NPZ."""
        base = dict(self.amass_converter.output_dict_base)
        for key, val in self.amass_converter.default_frame_params.items():
            import einops

            base[key] = einops.repeat(val, "d -> t d", t=T)
        base["mocap_time_length"] = T / 30.0
        base["trans"] = trans.astype(np.float32)
        base["root_orient"] = root_orient.astype(np.float32)
        base["pose_body"] = pose_body.astype(np.float32)

        buf = io.BytesIO()
        np.savez(buf, **base)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # History / continuation
    # ------------------------------------------------------------------
    def build_history_constraints(
        self, npz_path: str, num_history_frames: int = 5
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
        from kimodo.geometry import axis_angle_to_matrix
        from kimodo.skeleton import fk

        data = np.load(npz_path, allow_pickle=True)
        keys = set(data.keys())

        log.info("Loading history NPZ: keys=%s", sorted(keys))

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

            # Convert Z-up (lzyx) trans back to Y-up for FK
            # Inverse of get_amass_parameters z_up transform
            from .coord import M_INV
            pelvis_offset = self.skeleton.neutral_joints[self.skeleton.root_idx].cpu().numpy()
            trans_yup = np.matmul(trans + pelvis_offset, M_INV.T) - pelvis_offset
            root_positions = (trans_yup + pelvis_offset).astype(np.float32)

            # Undo the root orient Z-up rotation
            root_rots_aa = all_aa[:, 0]  # (T, 3)
            root_rots_mat = axis_angle_to_matrix(
                torch.tensor(root_rots_aa, dtype=torch.float32)
            ).numpy()  # (T, 3, 3)
            M_inv_t = torch.tensor(M_INV, dtype=torch.float32)
            root_rots_yup = np.matmul(M_INV.T, root_rots_mat)  # undo M @ R

            # Build local_rot_mats: (T, J, 3, 3)
            body_rots = axis_angle_to_matrix(
                torch.tensor(all_aa[:, 1:], dtype=torch.float32)
            ).numpy()
            local_rot_mats = np.concatenate(
                [root_rots_yup[:, np.newaxis, :, :], body_rots], axis=1
            )

            log.info("  Format: DART/API (poses+trans, %d body joints, converted Z-up→Y-up)", n_body_joints)

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

            from .coord import M_INV
            pelvis_offset = self.skeleton.neutral_joints[self.skeleton.root_idx].cpu().numpy()
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
            )

            log.info("  Format: AMASS (root_orient+pose_body+trans, %d body joints)", n_body_joints)

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

        return {
            "constraints": [constraint],
            "heading_angle": heading_angle,
            "num_over_generate": num_history_frames,
            "root_origin_2d_yup": [float(root_origin_2d[0]), float(root_origin_2d[1])],
            "last_frame": last_frame_data,
        }

    # ------------------------------------------------------------------
    # Constraint building
    # ------------------------------------------------------------------
    def build_constraints(
        self, segments: list, coord_in: str = "lzyx", staged_files: dict | None = None
    ) -> list:
        """Build kimodo constraint objects from parsed segment specs.

        Handles trajectory (Root2DConstraintSet) and inbetween (FullBodyConstraintSet).
        """
        _ensure_kimodo_imports()
        from kimodo.constraints import FullBodyConstraintSet, Root2DConstraintSet
        from kimodo.geometry import axis_angle_to_matrix
        from kimodo.skeleton import fk

        from .coord import lzyx_root2d

        staged_files = staged_files or {}
        constraints = []

        for seg in segments:
            if seg.type.value == "trajectory":
                constraints.extend(self._build_trajectory_constraint(seg, lzyx_root2d))

            elif seg.type.value == "inbetween":
                constraints.extend(
                    self._build_inbetween_constraint(seg, staged_files)
                )

        return constraints

    def _build_trajectory_constraint(self, seg, lzyx_root2d) -> list:
        from kimodo.constraints import Root2DConstraintSet

        abs_offset = seg.start_frame
        frame_indices = []
        root2d_positions = []

        for pt in seg.points:
            abs_frame = abs_offset + pt.frame
            frame_indices.append(abs_frame)
            rx, rz = lzyx_root2d(pt.pos[0], pt.pos[1])
            root2d_positions.append([rx, rz])

        if not frame_indices:
            return []

        device = self.skeleton.device if hasattr(self.skeleton, "device") else "cpu"
        constraint = Root2DConstraintSet(
            self.skeleton,
            frame_indices=torch.tensor(frame_indices, dtype=torch.long, device=device),
            smooth_root_2d=torch.tensor(root2d_positions, dtype=torch.float32, device=device),
        )
        log.info("  Built Root2D constraint: %d waypoints, frames %s", len(frame_indices), frame_indices)
        return [constraint]

    def _build_inbetween_constraint(self, seg, staged_files: dict) -> list:
        """Build FullBodyConstraintSet from an inbetween segment.

        Accepts the same request format as DART API:
          ref_smplx: {file_name, smplx_src_start_frame}
          mask_mode: "endpoints" | "keyframes" | "all" | "none"
          keyframes: [int, ...]              (segment-local destination frames)
          keyframes_src_frames: [int, ...]   (source frames in ref NPZ)
        """
        from kimodo.constraints import FullBodyConstraintSet
        from kimodo.geometry import axis_angle_to_matrix
        from kimodo.skeleton import fk

        from .coord import M_INV

        ref_spec = seg.ref_smplx
        if ref_spec.file_name not in staged_files:
            raise ValueError(f"ref_smplx references '{ref_spec.file_name}' but it was not uploaded")

        # Load reference NPZ (DART format: poses, trans)
        ref_data = np.load(staged_files[ref_spec.file_name], allow_pickle=True)
        ref_poses = ref_data["poses"]    # (T, 165)
        ref_trans = ref_data["trans"]    # (T, 3)
        ref_T = ref_poses.shape[0]
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

        # Convert Z-up (lzyx) to Y-up for Kimodo FK
        pelvis_offset = self.skeleton.neutral_joints[self.skeleton.root_idx].cpu().numpy()
        trans_yup = np.matmul(kf_trans + pelvis_offset, M_INV.T) - pelvis_offset
        root_positions = (trans_yup + pelvis_offset).astype(np.float32)

        # Undo Z-up rotation on root orient
        root_rots_mat = axis_angle_to_matrix(
            torch.tensor(kf_all_aa[:, 0], dtype=torch.float32)
        ).numpy()
        root_rots_yup = np.matmul(M_INV.T, root_rots_mat)

        # Body local rotations (no coord conversion needed)
        body_rots = axis_angle_to_matrix(
            torch.tensor(kf_all_aa[:, 1:], dtype=torch.float32)
        ).numpy()

        local_rot_mats = np.concatenate(
            [root_rots_yup[:, np.newaxis, :, :], body_rots], axis=1
        ).astype(np.float32)  # (K, J, 3, 3)

        # FK to get global positions and rotations
        device = self.skeleton.device if hasattr(self.skeleton, "device") else "cpu"
        local_rots_t = torch.tensor(local_rot_mats, dtype=torch.float32, device=device)
        root_pos_t = torch.tensor(root_positions, dtype=torch.float32, device=device)
        global_rots, posed_joints, _ = fk(local_rots_t, root_pos_t, self.skeleton)

        # Build constraint with absolute frame indices
        abs_offset = seg.start_frame
        abs_frames = [abs_offset + df for df in dest_frames]

        smooth_root_2d = posed_joints[:, self.skeleton.root_idx, [0, 2]]

        constraint = FullBodyConstraintSet(
            self.skeleton,
            frame_indices=torch.tensor(abs_frames, dtype=torch.long, device=device),
            global_joints_positions=posed_joints,
            global_joints_rots=global_rots,
            smooth_root_2d=smooth_root_2d,
        )

        log.info("  Built FullBody constraint: %d keyframes at abs frames %s", len(abs_frames), abs_frames)
        return [constraint]
