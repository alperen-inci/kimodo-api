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
    if service.is_loaded:
        return HealthResponse(
            status="ok",
            device=DEVICE,
            model_loaded=True,
            model_name=MODEL_NAME,
            skeleton=service.skeleton.name if service.skeleton else None,
        )
    else:
        return HealthResponse(
            status="not_ready",
            device=DEVICE,
            model_loaded=False,
            detail="Model not loaded. Check startup logs.",
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
        raise HTTPException(status_code=400, detail=f"Spec validation error: {e}")

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
                raise HTTPException(status_code=400, detail=f"History constraint error: {e}")

        # ---- Build texts and num_frames ----
        texts = []
        num_frames = []
        for seg in spec.segments:
            texts.append(seg.text)
            num_frames.append(seg.end_frame - seg.start_frame)

        # ---- Build segment constraints ----
        try:
            segment_constraints = service.build_constraints(
                spec.segments, coord_in=spec.coord_in
            )
        except Exception as e:
            log.error("[%s] Failed to build constraints: %s", req_id, e)
            raise HTTPException(status_code=400, detail=f"Constraint error: {e}")

        constraint_lst = history_constraints + segment_constraints

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
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    finally:
        # Cleanup temp files
        if tmp_dir:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ---- Return NPZ ----
    filename = f"kimodo_motion_{req_id}.npz"
    return Response(
        content=result["npz_bytes"],
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Kimodo-Meta": json.dumps(result["meta"]),
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
        content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("KIMODO_API_PORT", "8020"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
