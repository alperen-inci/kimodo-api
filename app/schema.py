# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for the Kimodo API request/response schema.

The request format intentionally mirrors dart-api's /generate/timeline
so that Unreal Engine client code can target both backends with minimal changes.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class SegmentType(str, Enum):
    text = "text"
    trajectory = "trajectory"
    inbetween = "inbetween"


class TrajectoryPoint(BaseModel):
    """A waypoint on the root trajectory.

    ``frame`` is segment-local (0 = first frame of this segment).
    ``pos`` is [x, y, z] in the coordinate system specified by ``coord_in``
    (default lzyx: X=right, Y=forward, Z=up).
    """

    frame: int = Field(..., ge=0, description="Segment-local frame index")
    pos: list[float] = Field(
        ..., min_length=3, max_length=3, description="[x, y, z] position"
    )


class RefSmplxSpec(BaseModel):
    """Reference SMPL-X NPZ for inbetween constraints."""

    file_name: str = Field(..., description="Uploaded NPZ filename (must match multipart upload)")
    smplx_src_start_frame: int = Field(0, ge=0, description="Start frame in the reference NPZ")


class SegmentSpec(BaseModel):
    """A single timeline segment."""

    type: SegmentType

    # --- timing (exactly one pair required) ---
    start_frame: Optional[int] = Field(None, ge=0)
    end_frame: Optional[int] = Field(None, ge=1)
    start_sec: Optional[float] = Field(None, ge=0.0)
    end_sec: Optional[float] = Field(None, gt=0.0)

    # --- common ---
    text: str = Field(
        ..., min_length=1, description="Motion description for text conditioning"
    )

    # --- trajectory-specific ---
    points: Optional[list[TrajectoryPoint]] = Field(
        None, description="Root trajectory waypoints (segment-local frames)"
    )
    joints: Optional[list[int]] = Field(
        None,
        description=(
            "Joint indices to constrain. Currently only [0] (pelvis/root) is "
            "supported. Defaults to [0]."
        ),
    )

    # --- inbetween-specific (same format as DART API) ---
    ref_smplx: Optional[RefSmplxSpec] = Field(
        None, description="Reference SMPL-X NPZ for keyframe constraints"
    )
    mask_mode: Optional[str] = Field(
        "endpoints",
        description="Constraint mode: 'endpoints' | 'keyframes' | 'all' | 'none'",
    )
    keyframes: Optional[list[int]] = Field(
        None, description="Segment-local frame indices to constrain (for mask_mode='keyframes')"
    )
    keyframes_src_frames: Optional[list[int]] = Field(
        None, description="Corresponding source frame indices in ref_smplx NPZ"
    )

    @model_validator(mode="after")
    def validate_timing_and_type(self) -> "SegmentSpec":
        # resolve timing
        has_frames = self.start_frame is not None and self.end_frame is not None
        has_secs = self.start_sec is not None and self.end_sec is not None
        if not has_frames and not has_secs:
            raise ValueError(
                "Provide either (start_frame, end_frame) or (start_sec, end_sec)"
            )
        if has_secs and not has_frames:
            fps = 30
            self.start_frame = int(math.floor(self.start_sec * fps + 1e-6))
            self.end_frame = int(math.floor(self.end_sec * fps + 1e-6))
        if self.end_frame <= self.start_frame:
            raise ValueError(
                f"end_frame ({self.end_frame}) must be > start_frame ({self.start_frame})"
            )
        n_frames = self.end_frame - self.start_frame
        # trajectory must have points
        if self.type == SegmentType.trajectory:
            if not self.points:
                raise ValueError("Trajectory segment requires at least one point")
            for pt in self.points:
                if pt.frame >= n_frames:
                    raise ValueError(
                        f"Point frame {pt.frame} out of range for segment "
                        f"with {n_frames} frames [0, {n_frames})"
                    )
        # inbetween must have ref_smplx
        if self.type == SegmentType.inbetween:
            if not self.ref_smplx:
                raise ValueError("Inbetween segment requires ref_smplx")
            if self.mask_mode == "keyframes" and not self.keyframes:
                raise ValueError("mask_mode='keyframes' requires keyframes list")
            if self.keyframes:
                for kf in self.keyframes:
                    if kf < 0 or kf >= n_frames:
                        raise ValueError(
                            f"Keyframe {kf} out of range [0, {n_frames})"
                        )
        return self


class HistorySpec(BaseModel):
    """History motion for continuation.

    Upload a previous motion NPZ (via multipart ``files`` field) and reference
    it here by ``file_name``.  The last ``num_frames`` of the history motion
    will be used as FullBody constraints on the first frames of the new
    generation, creating a smooth continuation.
    """

    file_name: str = Field(..., description="Uploaded NPZ filename (must match multipart upload)")
    num_frames: int = Field(
        5, ge=1, le=60,
        description="Number of frames from the END of history to use as transition constraint",
    )


class TimelineSpec(BaseModel):
    """Top-level request specification for /generate/timeline.

    Designed to be compatible with dart-api's spec format.
    """

    # --- model ---
    model: str = Field(
        "smplx",
        description="Model variant.",
    )

    # --- history / continuation ---
    history_smplx: Optional[HistorySpec] = Field(
        None,
        description="Previous motion NPZ for continuation. Upload the file via multipart 'files' field.",
    )

    # --- timing ---
    fps: int = Field(30, description="Frames per second. Must be 30.")

    # --- coordinate system ---
    coord_in: Literal["lzyx"] = Field(
        "lzyx",
        description="Input coordinate system. lzyx = left-handed, Z-up, Y-forward, X-right.",
    )
    coord_out: Literal["lzyx"] = Field(
        "lzyx",
        description="Output coordinate system. lzyx = left-handed, Z-up, Y-forward, X-right.",
    )

    # --- generation params ---
    seed: int = Field(0, ge=0, description="Random seed for reproducibility")
    diffusion_steps: int = Field(
        100, ge=1, le=1000, description="Number of DDIM denoising steps"
    )
    cfg_weight: list[float] = Field(
        [2.0, 2.0],
        description="Classifier-free guidance weights [text_cfg, constraint_cfg]",
    )
    num_samples: int = Field(
        1, ge=1, le=8, description="Number of motion samples to generate"
    )

    # --- post-processing ---
    post_processing: bool = Field(
        True, description="Apply foot-skating cleanup and constraint enforcement"
    )

    # --- heading ---
    first_heading_angle: Optional[float] = Field(
        None, description="Initial body heading in radians. None = auto (0.0 if no history). Used for first generation without history."
    )

    # --- transition ---
    num_transition_frames: int = Field(
        5, ge=0, le=30, description="Transition blend frames between segments"
    )

    # --- segments ---
    segments: list[SegmentSpec] = Field(
        ..., min_length=1, description="Ordered list of timeline segments"
    )

    # --- output format ---
    return_format: Literal["npz", "amass_npz"] = Field(
        "npz",
        description=(
            "'npz' returns DART-compatible format (poses/trans). "
            "'amass_npz' returns AMASS-style (root_orient/pose_body/...)."
        ),
    )

    @model_validator(mode="after")
    def validate_spec(self) -> "TimelineSpec":
        if self.fps != 30:
            raise ValueError("fps must be 30")
        # validate contiguous segments
        for i, seg in enumerate(self.segments):
            if i == 0 and seg.start_frame != 0:
                raise ValueError("First segment must start at frame 0")
            if i > 0:
                prev = self.segments[i - 1]
                if seg.start_frame != prev.end_frame:
                    raise ValueError(
                        f"Segment {i} start_frame ({seg.start_frame}) must equal "
                        f"previous segment end_frame ({prev.end_frame})"
                    )
        return self


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    detail: Optional[str] = None
