#!/bin/bash
# Kimodo API test script.
#
# Runs a sequence of curl requests against the Kimodo API and saves outputs.
# Each test prints its status and the output NPZ file path.
#
# Usage:
#   cd kimodo/kimodo-api/
#   bash tests/test_requests.sh                        # default localhost:8020
#   bash tests/test_requests.sh http://remote:8020     # custom URL

set -euo pipefail

BASE_URL="${1:-http://localhost:8020}"
OUT_DIR="/tmp/kimodo_tests"
PASS=0
FAIL=0
TOTAL=0

mkdir -p "$OUT_DIR"/{text,trajectory,multi_segment}

echo "=========================================="
echo " Kimodo API Test Suite"
echo "=========================================="
echo "  Base URL:   $BASE_URL"
echo "  Output dir: $OUT_DIR"
echo ""

run_test() {
    local name="$1"
    local outfile="$2"
    local spec_json="$3"
    TOTAL=$((TOTAL + 1))

    echo "--- Test $TOTAL: $name ---"
    echo "  Output: $outfile"

    local http_code
    http_code=$(curl -s -w '%{http_code}' -o "$outfile" \
        -X POST "$BASE_URL/generate/timeline" \
        -F "spec_json=$spec_json")

    if [[ "$http_code" == "200" ]]; then
        local size
        size=$(stat -c%s "$outfile" 2>/dev/null || stat -f%z "$outfile" 2>/dev/null)
        echo "  Status: PASS (HTTP $http_code, ${size} bytes)"
        PASS=$((PASS + 1))

        # Quick NPZ validation
        python3 -c "
import numpy as np, sys
try:
    d = np.load('$outfile', allow_pickle=True)
    keys = sorted(d.keys())
    print(f'  NPZ keys: {keys}')
    if 'poses' in d:
        print(f'  poses: {d[\"poses\"].shape} ({d[\"poses\"].dtype})')
    if 'trans' in d:
        print(f'  trans: {d[\"trans\"].shape} ({d[\"trans\"].dtype})')
    if 'mocap_framerate' in d:
        fps = int(d['mocap_framerate'])
        T = d['poses'].shape[0] if 'poses' in d else d['trans'].shape[0]
        print(f'  duration: {T/fps:.1f}s ({T} frames @ {fps}fps)')
except Exception as e:
    print(f'  NPZ validation warning: {e}', file=sys.stderr)
" 2>&1 || true
    else
        echo "  Status: FAIL (HTTP $http_code)"
        # Print error body
        cat "$outfile" 2>/dev/null || true
        FAIL=$((FAIL + 1))
    fi
    echo ""
}

# ===========================================================================
# 0. Health check
# ===========================================================================
echo "--- Test 0: Health Check ---"
TOTAL=$((TOTAL + 1))
health_response=$(curl -s "$BASE_URL/health")
echo "  Response: $health_response"
if echo "$health_response" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('status')=='ok' else 1)" 2>/dev/null; then
    echo "  Status: PASS"
    PASS=$((PASS + 1))
else
    echo "  Status: FAIL"
    FAIL=$((FAIL + 1))
fi
echo ""

# ===========================================================================
# 1. Text — Walk Forward
# ===========================================================================
run_test "Text: Walk Forward (3s)" \
    "$OUT_DIR/text/walk_forward.npz" \
    '{
        "fps": 30,
        "seed": 0,
        "diffusion_steps": 50,
        "segments": [
            {"type": "text", "text": "a person walks forward", "start_frame": 0, "end_frame": 90}
        ]
    }'

# ===========================================================================
# 2. Text — Wave Hello
# ===========================================================================
run_test "Text: Wave Hello (3s)" \
    "$OUT_DIR/text/wave_hello.npz" \
    '{
        "fps": 30,
        "seed": 0,
        "diffusion_steps": 50,
        "segments": [
            {"type": "text", "text": "wave hello with right hand", "start_frame": 0, "end_frame": 90}
        ]
    }'

# ===========================================================================
# 3. Text — Dance
# ===========================================================================
run_test "Text: Dance (5s)" \
    "$OUT_DIR/text/dance.npz" \
    '{
        "fps": 30,
        "seed": 123,
        "diffusion_steps": 50,
        "segments": [
            {"type": "text", "text": "a person dances energetically", "start_frame": 0, "end_frame": 150}
        ]
    }'

# ===========================================================================
# 4. Text — Jump
# ===========================================================================
run_test "Text: Jump (2s)" \
    "$OUT_DIR/text/jump.npz" \
    '{
        "fps": 30,
        "seed": 42,
        "diffusion_steps": 50,
        "segments": [
            {"type": "text", "text": "a person jumps in place", "start_frame": 0, "end_frame": 60}
        ]
    }'

# ===========================================================================
# 5. Text — Sit Down
# ===========================================================================
run_test "Text: Sit Down (3s)" \
    "$OUT_DIR/text/sit_down.npz" \
    '{
        "fps": 30,
        "seed": 0,
        "diffusion_steps": 50,
        "segments": [
            {"type": "text", "text": "sit down on a chair", "start_frame": 0, "end_frame": 90}
        ]
    }'

# ===========================================================================
# 6. Trajectory — Walk Right
# ===========================================================================
run_test "Trajectory: Walk Right (5s)" \
    "$OUT_DIR/trajectory/walk_right.npz" \
    '{
        "fps": 30,
        "seed": 0,
        "diffusion_steps": 50,
        "segments": [
            {
                "type": "trajectory",
                "text": "walk to the right",
                "start_frame": 0,
                "end_frame": 150,
                "points": [{"frame": 149, "pos": [2.0, 0.0, 0.96]}]
            }
        ]
    }'

# ===========================================================================
# 7. Trajectory — Walk Forward with midpoint
# ===========================================================================
run_test "Trajectory: Walk Forward with Midpoint (5s)" \
    "$OUT_DIR/trajectory/walk_forward.npz" \
    '{
        "fps": 30,
        "seed": 0,
        "diffusion_steps": 50,
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
    }'

# ===========================================================================
# 8. Trajectory — Walk Diagonal
# ===========================================================================
run_test "Trajectory: Walk Diagonal (5s)" \
    "$OUT_DIR/trajectory/walk_diagonal.npz" \
    '{
        "fps": 30,
        "seed": 0,
        "diffusion_steps": 50,
        "segments": [
            {
                "type": "trajectory",
                "text": "walk diagonally forward and to the right",
                "start_frame": 0,
                "end_frame": 150,
                "points": [
                    {"frame": 50,  "pos": [1.0, 1.0, 0.96]},
                    {"frame": 100, "pos": [2.0, 2.0, 0.96]},
                    {"frame": 149, "pos": [3.0, 3.0, 0.96]}
                ]
            }
        ]
    }'

# ===========================================================================
# 9. Multi-Trajectory — Walk Right Then Forward
# ===========================================================================
run_test "Multi-Trajectory: Right then Forward (8s)" \
    "$OUT_DIR/trajectory/walk_right_then_forward.npz" \
    '{
        "fps": 30,
        "seed": 0,
        "diffusion_steps": 50,
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
    }'

# ===========================================================================
# 10. Multi-Segment — Text + Trajectory
# ===========================================================================
run_test "Multi-Segment: Wave then Walk (6s)" \
    "$OUT_DIR/multi_segment/wave_then_walk.npz" \
    '{
        "fps": 30,
        "seed": 42,
        "diffusion_steps": 50,
        "segments": [
            {
                "type": "text",
                "text": "wave hello with right hand",
                "start_frame": 0,
                "end_frame": 90
            },
            {
                "type": "trajectory",
                "text": "walk forward",
                "start_frame": 90,
                "end_frame": 180,
                "points": [{"frame": 89, "pos": [0.0, 2.0, 0.96]}]
            }
        ]
    }'

# ===========================================================================
# 11. Multi-Segment — Three Actions
# ===========================================================================
run_test "Multi-Segment: Wave + Walk Right + Sit (9s)" \
    "$OUT_DIR/multi_segment/wave_walk_sit.npz" \
    '{
        "fps": 30,
        "seed": 42,
        "diffusion_steps": 50,
        "segments": [
            {
                "type": "text",
                "text": "wave hello with right hand",
                "start_frame": 0,
                "end_frame": 90
            },
            {
                "type": "trajectory",
                "text": "walk to the right",
                "start_frame": 90,
                "end_frame": 180,
                "points": [{"frame": 89, "pos": [2.0, 0.0, 0.96]}]
            },
            {
                "type": "text",
                "text": "sit down on a chair",
                "start_frame": 180,
                "end_frame": 270
            }
        ]
    }'

# ===========================================================================
# Summary
# ===========================================================================
echo "=========================================="
echo " Test Results"
echo "=========================================="
echo "  Total:  $TOTAL"
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo ""
echo "  Output files: $OUT_DIR/"
echo ""

if [[ $FAIL -gt 0 ]]; then
    echo "  SOME TESTS FAILED"
    exit 1
else
    echo "  ALL TESTS PASSED"
    exit 0
fi
