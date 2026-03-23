# Kimodo API Reference

## Overview

The Kimodo API is a REST service for text-driven and trajectory-constrained body motion
generation using the Kimodo diffusion model. It wraps the `kimodo-smplx-rp` model and
exposes a timeline-based API compatible with the DART API format.

**Base URL:** `http://localhost:8020`

**Output:** SMPL-X NPZ files in Z-up (lzyx) coordinates — directly compatible with
Blender and Unreal Engine SMPL-X workflows.

---

## Coordinate System

All positions use the **lzyx** coordinate system (same as DART API):

```
lzyx (left-handed, Z-up):
    X = right       (+X points right)
    Y = forward     (+Y points forward)
    Z = up          (+Z points up)
    Ground plane: XY (Z = 0)
```

Standing character:
- Pelvis height: ~0.9m (Z ≈ 0.9)
- Head height: ~1.7m (Z ≈ 1.7)
- Feet: Z ≈ 0

---

## Endpoints

### GET /health

Check service status and model readiness.

**Response:**
```json
{
    "status": "ok",
    "device": "cuda",
    "model_loaded": true,
    "model_name": "smplx",
    "skeleton": "smplx22"
}
```

**Example:**
```bash
curl http://localhost:8020/health
```

---

### POST /generate/timeline

Generate motion from a timeline specification.

**Content-Type:** `multipart/form-data`

**Form Fields:**

| Field       | Type   | Required | Description                         |
|-------------|--------|----------|-------------------------------------|
| `spec_json` | string | Yes      | JSON string matching TimelineSpec   |

**Response:** Raw `.npz` file (`application/octet-stream`)

**Response Headers:**
- `Content-Disposition: attachment; filename="kimodo_motion_<id>.npz"`
- `X-Kimodo-Meta: <JSON metadata about the generation>`

---

## Timeline Spec Schema

```json
{
    "model": "smplx",
    "fps": 30,
    "coord_in": "lzyx",
    "coord_out": "lzyx",
    "seed": 0,
    "diffusion_steps": 100,
    "cfg_weight": [2.0, 2.0],
    "num_samples": 1,
    "post_processing": true,
    "num_transition_frames": 5,
    "return_format": "npz",
    "segments": [...]
}
```

### Top-Level Fields

| Field                    | Type       | Default       | Description                                              |
|--------------------------|------------|---------------|----------------------------------------------------------|
| `model`                  | string     | `"smplx"`     | Kimodo model variant                                     |
| `fps`                    | int        | `30`          | Must be 30                                               |
| `coord_in`               | string     | `"lzyx"`      | Input coordinate system                                  |
| `coord_out`              | string     | `"lzyx"`      | Output coordinate system                                 |
| `seed`                   | int        | `0`           | Random seed for reproducibility                          |
| `diffusion_steps`        | int        | `100`         | DDIM denoising steps (more = better quality, slower)     |
| `cfg_weight`             | [float, float] | `[2.0, 2.0]` | Classifier-free guidance [text, constraint]          |
| `num_samples`            | int        | `1`           | Number of motion samples (1-8)                           |
| `post_processing`        | bool       | `true`        | Apply foot-skating cleanup                               |
| `num_transition_frames`  | int        | `5`           | Transition blend frames between segments                 |
| `return_format`          | string     | `"npz"`       | `"npz"` (DART-compatible) or `"amass_npz"`               |
| `segments`               | array      | required      | Ordered list of segments                                 |

### Segment Timing

Each segment uses half-open frame intervals `[start_frame, end_frame)`.
You can specify timing as frames OR seconds (converted with `floor(sec * 30)`):

```json
{"start_frame": 0, "end_frame": 90}
```
or
```json
{"start_sec": 0.0, "end_sec": 3.0}
```

**Rules:**
- First segment must start at frame 0
- Segments must be contiguous: `segments[i].start_frame == segments[i-1].end_frame`

---

## Segment Types

### 1. Text Segment (`type: "text"`)

Generate motion from a text description.

```json
{
    "type": "text",
    "start_frame": 0,
    "end_frame": 90,
    "text": "a person walks forward"
}
```

| Field   | Type   | Required | Description              |
|---------|--------|----------|--------------------------|
| `type`  | string | Yes      | Must be `"text"`         |
| `text`  | string | Yes      | Motion description       |
| Timing  | —      | Yes      | See Segment Timing above |

### 2. Trajectory Segment (`type: "trajectory"`)

Generate motion conditioned on root (pelvis) trajectory waypoints.

```json
{
    "type": "trajectory",
    "start_frame": 0,
    "end_frame": 150,
    "text": "walk to the right",
    "points": [
        {"frame": 75,  "pos": [1.0, 0.0, 0.96]},
        {"frame": 149, "pos": [2.0, 0.0, 0.96]}
    ],
    "joints": [0]
}
```

| Field    | Type     | Required | Description                                          |
|----------|----------|----------|------------------------------------------------------|
| `type`   | string   | Yes      | Must be `"trajectory"`                               |
| `text`   | string   | Yes      | Motion description for text conditioning             |
| `points` | array    | Yes      | Trajectory waypoints (at least 1)                    |
| `joints` | [int]    | No       | Joint indices (default `[0]`, only `[0]` supported)  |
| Timing   | —        | Yes      | See Segment Timing above                             |

**Trajectory Point:**

| Field   | Type       | Description                                     |
|---------|------------|-------------------------------------------------|
| `frame` | int        | Segment-local frame index `[0, end-start)`      |
| `pos`   | [x, y, z]  | Position in coord_in (lzyx: X=right, Y=fwd, Z=up) |

**Notes:**
- `frame` is segment-local (0 = first frame of this segment)
- `pos[2]` (Z/height) is stored but not constrained by Root2D — only XY ground position is enforced
- Currently only joint index 0 (pelvis/root) is supported
- Dense waypoints (more points) produce better trajectory following

---

## Output NPZ Format

### DART-compatible format (`return_format: "npz"`, default)

```
poses            : float32 (T, 165)   axis-angle rotations
                   [0:3]   = global_orient
                   [3:66]  = body_pose (21 joints × 3)
                   [66:69] = jaw_pose
                   [69:75] = eye_pose (left 3 + right 3)
                   [75:165]= hand_pose (left 45 + right 45)
trans            : float32 (T, 3)     root translation in lzyx
betas            : float32 (16,)      SMPL-X shape parameters
gender           : str                "neutral"
mocap_framerate  : int64              30
```

### AMASS format (`return_format: "amass_npz"`)

```
trans            : float32 (T, 3)     root translation
root_orient      : float32 (T, 3)     root orientation (axis-angle)
pose_body        : float32 (T, 63)    body pose (21 joints × 3, axis-angle)
pose_hand        : float32 (T, 90)    hand pose (30 joints × 3)
pose_jaw         : float32 (T, 3)     jaw pose
pose_eye         : float32 (T, 6)     eye pose
betas            : float32 (16,)      shape parameters
gender           : str                "neutral"
surface_model_type : str              "smplx"
mocap_frame_rate : float              30.0
mocap_time_length: float              duration in seconds
```

---

## SMPL-X 22-Joint Index Reference

| Index | Joint Name     | Index | Joint Name      |
|-------|----------------|-------|-----------------|
| 0     | pelvis         | 11    | right_foot      |
| 1     | left_hip       | 12    | neck            |
| 2     | right_hip      | 13    | left_collar     |
| 3     | spine1         | 14    | right_collar    |
| 4     | left_knee      | 15    | head            |
| 5     | right_knee     | 16    | left_shoulder   |
| 6     | spine2         | 17    | right_shoulder  |
| 7     | left_ankle     | 18    | left_elbow      |
| 8     | right_ankle    | 19    | right_elbow     |
| 9     | spine3         | 20    | left_wrist      |
| 10    | left_foot      | 21    | right_wrist     |

---

## Differences from DART API

| Feature               | DART API                   | Kimodo API                          |
|-----------------------|----------------------------|--------------------------------------|
| Model backbone        | MLD + SMPL-X               | Kimodo diffusion + SMPL-X            |
| Trajectory method     | Optimization (gradient)    | Constraint conditioning (diffusion)  |
| Trajectory joints     | Any of 22 joints           | Root (pelvis) only                   |
| Goal reaching         | RL policy                  | Not supported (use trajectory)       |
| Inbetween/keyframe    | Optimization               | Not supported (planned)              |
| Scene interaction     | SDF collision              | Not supported                        |
| Physics post-process  | Optional PhysX pipeline    | Foot-skate cleanup (MotionCorrection)|
| Text encoder          | CLIP                       | LLM2Vec (local or API)              |
| Multi-segment         | Yes (contiguous)           | Yes (contiguous + transitions)       |
| History continuation  | Yes (NPZ upload)           | Not supported (planned)              |
| Output coord system   | lzyx (Z-up)                | lzyx (Z-up)                          |

---

## Curl Examples

### Health Check

```bash
curl -s http://localhost:8020/health | python3 -m json.tool
```

### Text — Walk Forward (3 seconds)

```bash
curl -X POST http://localhost:8020/generate/timeline \
  -F 'spec_json={
    "fps": 30,
    "coord_in": "lzyx",
    "coord_out": "lzyx",
    "seed": 42,
    "diffusion_steps": 100,
    "segments": [
      {
        "type": "text",
        "text": "a person walks forward",
        "start_frame": 0,
        "end_frame": 90
      }
    ]
  }' \
  -o walk_forward.npz
```

### Text — Wave Hello (3 seconds)

```bash
curl -X POST http://localhost:8020/generate/timeline \
  -F 'spec_json={
    "fps": 30,
    "seed": 0,
    "segments": [
      {
        "type": "text",
        "text": "wave hello with right hand",
        "start_sec": 0,
        "end_sec": 3.0
      }
    ]
  }' \
  -o wave_hello.npz
```

### Text — Dance (5 seconds)

```bash
curl -X POST http://localhost:8020/generate/timeline \
  -F 'spec_json={
    "fps": 30,
    "seed": 123,
    "segments": [
      {
        "type": "text",
        "text": "a person dances energetically",
        "start_frame": 0,
        "end_frame": 150
      }
    ]
  }' \
  -o dance.npz
```

### Single Trajectory — Walk Right (5 seconds)

```bash
curl -X POST http://localhost:8020/generate/timeline \
  -F 'spec_json={
    "fps": 30,
    "seed": 0,
    "segments": [
      {
        "type": "trajectory",
        "text": "walk to the right",
        "start_frame": 0,
        "end_frame": 150,
        "points": [
          {"frame": 149, "pos": [2.0, 0.0, 0.96]}
        ]
      }
    ]
  }' \
  -o walk_right.npz
```

### Single Trajectory — Walk Forward with Midpoint (5 seconds)

```bash
curl -X POST http://localhost:8020/generate/timeline \
  -F 'spec_json={
    "fps": 30,
    "seed": 0,
    "segments": [
      {
        "type": "trajectory",
        "text": "walk forward steadily",
        "start_frame": 0,
        "end_frame": 150,
        "points": [
          {"frame": 75,  "pos": [0.0, 1.5, 0.96]},
          {"frame": 149, "pos": [0.0, 3.0, 0.96]}
        ]
      }
    ]
  }' \
  -o walk_forward_trajectory.npz
```

### Single Trajectory — Walk Diagonal (5 seconds)

```bash
curl -X POST http://localhost:8020/generate/timeline \
  -F 'spec_json={
    "fps": 30,
    "seed": 0,
    "segments": [
      {
        "type": "trajectory",
        "text": "walk diagonally",
        "start_frame": 0,
        "end_frame": 150,
        "points": [
          {"frame": 50,  "pos": [1.0, 1.0, 0.96]},
          {"frame": 100, "pos": [2.0, 2.0, 0.96]},
          {"frame": 149, "pos": [3.0, 3.0, 0.96]}
        ]
      }
    ]
  }' \
  -o walk_diagonal.npz
```

### Multiple Trajectory — Walk Right Then Forward (8 seconds)

```bash
curl -X POST http://localhost:8020/generate/timeline \
  -F 'spec_json={
    "fps": 30,
    "seed": 0,
    "segments": [
      {
        "type": "trajectory",
        "text": "walk to the right",
        "start_frame": 0,
        "end_frame": 120,
        "points": [
          {"frame": 60,  "pos": [1.0, 0.0, 0.96]},
          {"frame": 119, "pos": [2.0, 0.0, 0.96]}
        ]
      },
      {
        "type": "trajectory",
        "text": "walk forward",
        "start_frame": 120,
        "end_frame": 240,
        "points": [
          {"frame": 60,  "pos": [2.0, 1.5, 0.96]},
          {"frame": 119, "pos": [2.0, 3.0, 0.96]}
        ]
      }
    ]
  }' \
  -o walk_right_then_forward.npz
```

### Multi-Segment — Text + Trajectory (6 seconds)

```bash
curl -X POST http://localhost:8020/generate/timeline \
  -F 'spec_json={
    "fps": 30,
    "seed": 0,
    "segments": [
      {
        "type": "text",
        "text": "stand up from sitting",
        "start_frame": 0,
        "end_frame": 90
      },
      {
        "type": "trajectory",
        "text": "walk forward",
        "start_frame": 90,
        "end_frame": 180,
        "points": [
          {"frame": 89, "pos": [0.0, 2.0, 0.96]}
        ]
      }
    ]
  }' \
  -o standup_then_walk.npz
```

### Multi-Segment — Three Actions (9 seconds)

```bash
curl -X POST http://localhost:8020/generate/timeline \
  -F 'spec_json={
    "fps": 30,
    "seed": 42,
    "segments": [
      {
        "type": "text",
        "text": "wave hello with right hand",
        "start_frame": 0,
        "end_frame": 90
      },
      {
        "type": "trajectory",
        "text": "walk to the right while looking around",
        "start_frame": 90,
        "end_frame": 180,
        "points": [
          {"frame": 89, "pos": [2.0, 0.0, 0.96]}
        ]
      },
      {
        "type": "text",
        "text": "sit down on a chair",
        "start_frame": 180,
        "end_frame": 270
      }
    ]
  }' \
  -o wave_walk_sit.npz
```

### AMASS Output Format

```bash
curl -X POST http://localhost:8020/generate/timeline \
  -F 'spec_json={
    "fps": 30,
    "return_format": "amass_npz",
    "segments": [
      {
        "type": "text",
        "text": "a person walks forward",
        "start_frame": 0,
        "end_frame": 90
      }
    ]
  }' \
  -o walk_amass.npz
```

### Multiple Samples (3 variations)

```bash
curl -X POST http://localhost:8020/generate/timeline \
  -F 'spec_json={
    "fps": 30,
    "seed": 0,
    "num_samples": 3,
    "segments": [
      {
        "type": "text",
        "text": "a person dances",
        "start_frame": 0,
        "end_frame": 150
      }
    ]
  }' \
  -o dance_3samples.npz
```

---

## Inspecting NPZ Output in Python

```python
import numpy as np

data = np.load("walk_forward.npz", allow_pickle=True)
print("Keys:", list(data.keys()))
print("poses shape:", data["poses"].shape)    # (T, 165)
print("trans shape:", data["trans"].shape)    # (T, 3)
print("betas shape:", data["betas"].shape)   # (16,)
print("gender:", str(data["gender"]))        # "neutral"
print("fps:", int(data["mocap_framerate"]))   # 30
print("duration:", data["poses"].shape[0] / 30, "seconds")
```

---

## Loading in Blender

Use the same Blender SMPL-X import workflow as DART outputs:

```python
# In Blender's Python console or script
import numpy as np
import bpy

data = np.load("walk_forward.npz", allow_pickle=True)
poses = data["poses"]    # (T, 165)
trans = data["trans"]    # (T, 3)

# The output is in Z-up coordinates (lzyx), which is Blender's native frame.
# Import using your SMPL-X addon with these parameters:
# - poses[:, 0:3] = global orientation (axis-angle)
# - poses[:, 3:66] = body pose (axis-angle)
# - trans = root translation
```

---

## Error Responses

All errors return JSON:

```json
{
    "detail": "Human-readable error message"
}
```

| Status | Meaning                                    |
|--------|--------------------------------------------|
| 400    | Invalid request (bad JSON, validation fail)|
| 500    | Internal error (generation failed)         |
| 503    | Model not loaded / service not ready       |

---

## Environment Variables

| Variable              | Default    | Description                           |
|-----------------------|------------|---------------------------------------|
| `KIMODO_MODEL`        | `smplx`    | Model variant to load                 |
| `KIMODO_DEVICE`       | `cuda`     | PyTorch device                        |
| `KIMODO_API_PORT`     | `8020`     | Server port                           |
| `KIMODO_API_LOG_LEVEL`| `INFO`     | Log level (DEBUG, INFO, WARNING, etc) |
| `TEXT_ENCODER_URL`    | (auto)     | Remote text encoder service URL       |
| `TEXT_ENCODER_MODE`   | `auto`     | `auto`, `api`, or `local`             |
| `HF_HOME`            | (default)  | HuggingFace cache directory           |
| `CHECKPOINT_DIR`      | (auto)     | Override model checkpoint location    |
