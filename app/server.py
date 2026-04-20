# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimodo API — FastAPI server for text-driven and trajectory-constrained motion generation.

Endpoints:
    GET  /health              — Service health and model status
    POST /generate/kimodo   — Generate motion from a timeline specification
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid

import tempfile

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from . import body2hands as body2hands_client
from .schema import HealthResponse, SegmentType, TimelineSpec
from .service import KimodoService

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("KIMODO_API_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("kimodo_api")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
DEVICE = os.environ.get("KIMODO_DEVICE", "cuda")
MODEL_NAME = os.environ.get("KIMODO_MODEL", "smplx")

BODY2HANDS_URL = os.environ.get("KIMODO_BODY2HANDS_URL", "http://localhost:8021")
BODY2HANDS_ENABLED = os.environ.get("KIMODO_BODY2HANDS_ENABLED", "true").lower() == "true"
BODY2HANDS_TIMEOUT_SEC = float(os.environ.get("KIMODO_BODY2HANDS_TIMEOUT_SEC", "120"))

app = FastAPI(
    title="Kimodo Motion Generation API",
    version="1.0.0",
    description=(
        "REST API for Kimodo text-driven and trajectory-constrained "
        "body motion generation. Outputs SMPL-X NPZ (Z-up)."
    ),
)
service = KimodoService(model_name=MODEL_NAME, device=DEVICE)


@app.on_event("startup")
async def startup():
    log.info("Starting Kimodo API — model=%s, device=%s", MODEL_NAME, DEVICE)
    log.info(
        "Body2Hands post-processing: enabled=%s url=%s timeout=%.0fs",
        BODY2HANDS_ENABLED, BODY2HANDS_URL, BODY2HANDS_TIMEOUT_SEC,
    )
    try:
        service.load()
    except Exception:
        log.exception("Failed to load model at startup. /health will report not ready.")


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    """Check service health and model readiness."""
    b2h_info: dict = {"enabled": BODY2HANDS_ENABLED, "url": BODY2HANDS_URL}
    if BODY2HANDS_ENABLED:
        b2h_health = await body2hands_client.health(BODY2HANDS_URL, timeout=3.0)
        if b2h_health is None:
            b2h_info.update(reachable=False, model_loaded=False)
        else:
            b2h_info.update(
                reachable=True,
                model_loaded=bool(b2h_health.get("model_loaded")),
                device=b2h_health.get("device"),
                checkpoint=b2h_health.get("checkpoint"),
            )

    if service.is_loaded:
        return HealthResponse(status="ok", model_loaded=True, body2hands=b2h_info)
    else:
        return HealthResponse(
            status="not_ready", model_loaded=False,
            detail="Model not loaded", body2hands=b2h_info,
        )


# ---------------------------------------------------------------------------
# /generate/kimodo
# ---------------------------------------------------------------------------
@app.post("/generate/kimodo")
async def generate_timeline(
    request: Request,
    spec_json: str = Form(..., description="JSON timeline specification"),
    files: list[UploadFile] = File(default=[], description="Optional file uploads (e.g. history NPZ)"),
):
    """Generate motion from a timeline specification.

    Accepts a multipart form with:
      - ``spec_json``: JSON string matching ``TimelineSpec``
      - ``files`` (optional, repeatable): file uploads referenced by ``history_smplx.file_name``

    Returns a raw ``.npz`` file as ``application/octet-stream``.
    """
    req_id = uuid.uuid4().hex[:8]
    log.info("[%s] /generate/kimodo — received request (%d file(s))", req_id, len(files))

    # ---- Parse spec ----
    try:
        spec_dict = json.loads(spec_json)
        spec = TimelineSpec(**spec_dict)
    except json.JSONDecodeError as e:
        log.error("[%s] Invalid JSON in spec_json: %s", req_id, e)
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    except Exception as e:
        log.error("[%s] Spec validation failed: %s", req_id, e)
        raise HTTPException(status_code=400, detail="Invalid request specification")

    log.info(
        "[%s] Parsed spec: %d segment(s), seed=%d, steps=%d, samples=%d, format=%s, history=%s",
        req_id,
        len(spec.segments),
        spec.seed,
        spec.diffusion_steps,
        spec.num_samples,
        spec.return_format,
        spec.history_smplx.file_name if spec.history_smplx else "none",
    )

    # ---- Check model ----
    if not service.is_loaded:
        log.error("[%s] Model not loaded", req_id)
        raise HTTPException(status_code=503, detail="Model not loaded")

    # ---- Stage uploaded files to temp dir ----
    staged_files: dict[str, str] = {}
    tmp_dir = None
    try:
        if files:
            tmp_dir = tempfile.mkdtemp(prefix="kimodo_req_")
            for f in files:
                if f.filename:
                    dst = os.path.join(tmp_dir, f.filename)
                    content = await f.read()
                    with open(dst, "wb") as fh:
                        fh.write(content)
                    staged_files[f.filename] = dst
                    log.info("[%s] Staged file: %s (%d bytes)", req_id, f.filename, len(content))

        # ---- Build history constraints ----
        history_info = None
        history_constraints = []
        if spec.history_smplx:
            fname = spec.history_smplx.file_name
            if fname not in staged_files:
                raise HTTPException(
                    status_code=400,
                    detail=f"history_smplx references '{fname}' but it was not uploaded",
                )
            try:
                history_result = service.build_history_constraints(
                    npz_path=staged_files[fname],
                    num_history_frames=spec.history_smplx.num_frames,
                )
                history_constraints = history_result["constraints"]
                history_info = {
                    "num_over_generate": history_result["num_over_generate"],
                    "heading_angle": history_result["heading_angle"],
                    "root_origin_2d_yup": history_result["root_origin_2d_yup"],
                    "last_frame": history_result["last_frame"],
                }
                log.info("[%s] History: %d constraint frames, heading=%.3f, over-gen=%d",
                         req_id, spec.history_smplx.num_frames,
                         history_result["heading_angle"],
                         history_result["num_over_generate"])
            except Exception as e:
                log.error("[%s] Failed to build history constraints: %s", req_id, e)
                raise HTTPException(status_code=400, detail="History constraint error")

        # ---- Build texts and num_frames ----
        texts = []
        num_frames = []
        for seg in spec.segments:
            texts.append(seg.text)
            num_frames.append(seg.end_frame - seg.start_frame)

        # ---- Build segment constraints ----
        # Pass origin offset so trajectory/inbetween constraints are translated
        # to the same origin-centered frame as history constraints.
        origin_offset_2d = None
        num_over = 0
        if history_info:
            import torch
            origin_offset_2d = torch.tensor(
                history_info["root_origin_2d_yup"], dtype=torch.float32)
            # Shift segment frame indices by over-gen amount so constraints
            # align with content frames (not history overlap prefix).
            num_over = history_info["num_over_generate"]
            for seg in spec.segments:
                seg.start_frame += num_over
                seg.end_frame += num_over

        # Extract root 2D positions from pose NPZs BEFORE building trajectory
        # constraints so that intermediate waypoints follow the real path
        # through both targets and pose positions.
        pose_anchors = []
        if spec.pose_constraints:
            pose_anchors = service.extract_pose_root2d(
                spec.pose_constraints,
                staged_files=staged_files,
                frame_offset=num_over,
                coord_in=spec.coord_in,
            )
            log.info("[%s] Extracted %d pose anchor(s) for trajectory: %s",
                     req_id, len(pose_anchors),
                     [(f, f"{x:.3f},{z:.3f}") for f, x, z in pose_anchors])

        try:
            segment_constraints = service.build_constraints(
                spec.segments, coord_in=spec.coord_in, staged_files=staged_files,
                origin_offset_2d=origin_offset_2d,
                pose_anchors=pose_anchors,
            )
        except Exception as e:
            log.error("[%s] Failed to build constraints: %s", req_id, e)
            raise HTTPException(status_code=400, detail="Constraint error")

        constraint_lst = history_constraints + segment_constraints

        # ---- Build external pose constraints (overlay on segments) ----
        if spec.pose_constraints:
            try:
                num_over = history_info["num_over_generate"] if history_info else 0
                pose_constraints = service.build_pose_constraints(
                    spec.pose_constraints,
                    staged_files=staged_files,
                    frame_offset=num_over,
                    origin_offset_2d=origin_offset_2d,
                )
                constraint_lst.extend(pose_constraints)
                log.info("[%s] Pose constraints: %d external pose keyframe(s)",
                         req_id, len(spec.pose_constraints))
            except Exception as e:
                log.error("[%s] Failed to build pose constraints: %s", req_id, e)
                raise HTTPException(status_code=400, detail="Pose constraint error")

        # ---- Generate ----
        try:
            t0 = time.time()
            result = service.generate(
                texts=texts,
                num_frames=num_frames,
                constraint_lst=constraint_lst,
                seed=spec.seed,
                diffusion_steps=spec.diffusion_steps,
                cfg_weight=spec.cfg_weight,
                num_samples=spec.num_samples,
                post_processing=spec.post_processing,
                num_transition_frames=spec.num_transition_frames,
                return_format=spec.return_format,
                history_info=history_info,
                first_heading_angle_override=spec.first_heading_angle,
            )
            elapsed = time.time() - t0
            log.info(
                "[%s] Request completed in %.1fs — %d bytes",
                req_id,
                elapsed,
                len(result["npz_bytes"]),
            )
        except Exception as e:
            log.exception("[%s] Generation failed", req_id)
            raise HTTPException(status_code=500, detail="Generation failed")

    finally:
        # Cleanup temp files
        if tmp_dir:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ---- Body2Hands post-processing ----
    # Enrich hand channels only. On ANY failure (unreachable, timeout, bad
    # response, shape mismatch) fall through with the original kimodo NPZ so
    # the user still gets a valid motion — the X-Kimodo-Body2Hands header
    # makes the outcome explicit (no silent behavior change).
    b2h_status = "disabled"
    if BODY2HANDS_ENABLED:
        t_b2h = time.time()
        enriched = await body2hands_client.enrich(
            BODY2HANDS_URL, result["npz_bytes"], timeout=BODY2HANDS_TIMEOUT_SEC,
        )
        if enriched is None:
            b2h_status = "skipped:unreachable_or_failed"
            log.warning("[%s] Body2Hands: %s — returning kimodo output unchanged",
                        req_id, b2h_status)
        else:
            spliced = body2hands_client.splice_hands_only(result["npz_bytes"], enriched)
            if spliced is None:
                b2h_status = "skipped:shape_mismatch"
                log.warning("[%s] Body2Hands: %s — returning kimodo output unchanged",
                            req_id, b2h_status)
            else:
                result["npz_bytes"] = spliced
                b2h_status = "applied"
                log.info("[%s] Body2Hands: applied in %.1fs (%d bytes)",
                         req_id, time.time() - t_b2h, len(spliced))

    # ---- Return NPZ ----
    filename = f"motion_{req_id}.npz"
    return Response(
        content=result["npz_bytes"],
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Kimodo-Body2Hands": b2h_status,
            "X-Kimodo-Meta": json.dumps({
                k: v for k, v in result["meta"].items()
                if k in ("total_frames", "fps", "elapsed_sec")
            }),
        },
    )


# ---------------------------------------------------------------------------
# Error handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("KIMODO_API_PORT", "8020"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
