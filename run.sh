#!/bin/bash
# Run the Kimodo API server.
#
# Usage:
#   cd kimodo-api/
#   bash run.sh                # foreground
#   bash run.sh -d             # detached (background)
#   bash run.sh --port 9000    # custom port

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="$SCRIPT_DIR/weights"

PORT="${KIMODO_API_PORT:-8020}"
DEVICE="${KIMODO_DEVICE:-cuda}"
DETACH=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--detach) DETACH="-d"; shift ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Check weights
if [ ! -f "$WEIGHTS_DIR/Kimodo-SMPLX-RP-v1/model.safetensors" ]; then
    echo "ERROR: Model weights not found. Run 'bash setup.sh' first."
    exit 1
fi

# Check docker image
if ! docker image inspect kimodo:genmo &>/dev/null; then
    echo "ERROR: Docker image kimodo:genmo not found. Run 'bash setup.sh' first."
    exit 1
fi

echo "=========================================="
echo " Kimodo API"
echo "=========================================="
echo "  Port:     $PORT"
echo "  Device:   $DEVICE"
echo "  Weights:  $WEIGHTS_DIR"
echo "=========================================="

docker run --gpus all --rm $DETACH \
    --name kimodo_api \
    -p "${PORT}:8020" \
    -v "$SCRIPT_DIR/app":/workspace/kimodo-api/app:ro \
    -v "$WEIGHTS_DIR":/workspace/weights:ro \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    -e CHECKPOINT_DIR=/workspace/weights \
    -e KIMODO_MODEL=smplx \
    -e KIMODO_DEVICE="$DEVICE" \
    -e KIMODO_API_PORT=8020 \
    -e KIMODO_API_LOG_LEVEL=INFO \
    -e PYTHONPATH=/workspace/kimodo \
    -w /workspace/kimodo-api \
    kimodo:genmo \
    python -m uvicorn app.server:app --host 0.0.0.0 --port 8020

if [ -n "$DETACH" ]; then
    echo ""
    echo "API started in background."
    echo "  Health: curl http://localhost:$PORT/health"
    echo "  Logs:   docker logs -f kimodo_api"
    echo "  Stop:   docker stop kimodo_api"
fi
