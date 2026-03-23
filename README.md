# Kimodo API

REST API for Kimodo text-driven and trajectory-constrained body motion generation.

Wraps the [Kimodo](https://github.com/nv-tlabs/kimodo) diffusion model and exposes
a timeline-based API compatible with the DART API request format.

## Quick Start

```bash
# 1. Setup (build Docker images, download weights)
cd kimodo/kimodo-api/
bash setup.sh

# 2. Run (standalone — uses local text encoder, needs ~17GB VRAM)
bash run.sh

# OR run with docker-compose (separate text encoder service)
bash run.sh --compose

# 3. Test
curl http://localhost:8020/health
bash tests/test_requests.sh
```

## Endpoints

| Method | Path                  | Description                          |
|--------|-----------------------|--------------------------------------|
| GET    | `/health`             | Service health and model status      |
| POST   | `/generate/timeline`  | Generate motion from timeline spec   |

## Request Format

Multipart form with `spec_json` field containing a JSON timeline specification:

```bash
curl -X POST http://localhost:8020/generate/timeline \
  -F 'spec_json={
    "fps": 30,
    "seed": 0,
    "segments": [
      {"type": "text", "text": "walk forward", "start_frame": 0, "end_frame": 90}
    ]
  }' \
  -o motion.npz
```

## Segment Types

### Text
```json
{"type": "text", "text": "a person walks forward", "start_frame": 0, "end_frame": 90}
```

### Trajectory
```json
{
  "type": "trajectory",
  "text": "walk to the right",
  "start_frame": 0,
  "end_frame": 150,
  "points": [
    {"frame": 75,  "pos": [1.0, 0.0, 0.96]},
    {"frame": 149, "pos": [2.0, 0.0, 0.96]}
  ]
}
```

Position format: `[x, y, z]` in lzyx (X=right, Y=forward, Z=up).

## Output Format

NPZ file with SMPL-X parameters in Z-up (lzyx) coordinates:

```
poses:           (T, 165)  float32   axis-angle rotations
trans:           (T, 3)    float32   root translation
betas:           (16,)     float32   shape parameters
gender:          str                 "neutral"
mocap_framerate: int64               30
```

Compatible with DART API output — same Blender import workflow.

## Coordinate System

```
lzyx (Z-up, left-handed):
    X = right
    Y = forward
    Z = up
```

## Architecture

```
kimodo-api/
├── app/
│   ├── server.py     FastAPI endpoints
│   ├── schema.py     Pydantic request models
│   ├── service.py    Model loading & inference
│   └── coord.py      Z-up ↔ Y-up conversions
├── Dockerfile
├── setup.sh          Build & download weights
├── run.sh            Run the server
├── docker-compose.yaml
├── docs/
│   └── api_reference.md   Full API reference
└── tests/
    └── test_requests.sh   Curl test suite
```

## Full Documentation

See [docs/api_reference.md](docs/api_reference.md) for:
- Complete schema reference
- All curl examples
- Output format details
- SMPL-X joint index table
- Differences from DART API
- Environment variables
