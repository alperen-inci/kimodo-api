#!/bin/bash
# Kimodo API setup script.
#
# Downloads pre-built Docker image and model weights from Google Drive,
# then starts the API server.
#
# Usage:
#   cd kimodo-api/
#   bash setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Google Drive file IDs
DOCKER_TAR_ID="1iUKp9YGJ9SJ4OtgHzSF8uf9Yx4sajmwi"
WEIGHTS_TAR_ID="1eAoGm7-bMDZtbCeNjKvyM5RVY41w-73o"

DOCKER_TAR="kimodo-genmo-docker.tar.gz"
WEIGHTS_TAR="kimodo-smplx-rp-v1-weights.tar.gz"
WEIGHTS_DIR="$SCRIPT_DIR/weights"

echo "=========================================="
echo " Kimodo API Setup"
echo "=========================================="
echo ""

# ---- 1. Check prerequisites ----
echo "[1/4] Checking prerequisites..."

if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found. Install Docker first."
    exit 1
fi

if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia\|nvidia-container"; then
    echo "WARNING: NVIDIA container runtime not detected."
    echo "         Install nvidia-container-toolkit for GPU support."
fi

if ! command -v gdown &>/dev/null; then
    echo "  Installing gdown..."
    pip install gdown
fi

echo "  Docker: $(docker --version)"
echo ""

# ---- 2. Download and load Docker image ----
echo "[2/4] Docker image (kimodo:genmo)..."

if docker image inspect kimodo:genmo &>/dev/null; then
    echo "  Image already loaded, skipping."
else
    if [ ! -f "$SCRIPT_DIR/$DOCKER_TAR" ]; then
        echo "  Downloading $DOCKER_TAR (~35 GB)..."
        gdown "https://drive.google.com/uc?id=$DOCKER_TAR_ID" -O "$SCRIPT_DIR/$DOCKER_TAR"
    else
        echo "  $DOCKER_TAR already downloaded."
    fi
    echo "  Loading Docker image..."
    gunzip -c "$SCRIPT_DIR/$DOCKER_TAR" | docker load
    echo "  Done. Cleaning up tar..."
    rm -f "$SCRIPT_DIR/$DOCKER_TAR"
fi
echo ""

# ---- 3. Download model weights ----
echo "[3/4] Model weights (Kimodo-SMPLX-RP-v1)..."

if [ -f "$WEIGHTS_DIR/model.safetensors" ]; then
    echo "  Weights already present in $WEIGHTS_DIR"
else
    if [ ! -f "$SCRIPT_DIR/$WEIGHTS_TAR" ]; then
        echo "  Downloading $WEIGHTS_TAR (~1 GB)..."
        gdown "https://drive.google.com/uc?id=$WEIGHTS_TAR_ID" -O "$SCRIPT_DIR/$WEIGHTS_TAR"
    else
        echo "  $WEIGHTS_TAR already downloaded."
    fi
    echo "  Extracting weights..."
    mkdir -p "$WEIGHTS_DIR/Kimodo-SMPLX-RP-v1"
    tar -xzf "$SCRIPT_DIR/$WEIGHTS_TAR" -C "$WEIGHTS_DIR/Kimodo-SMPLX-RP-v1"
    echo "  Done. Cleaning up tar..."
    rm -f "$SCRIPT_DIR/$WEIGHTS_TAR"
fi

echo "  Weights dir contents:"
ls -lh "$WEIGHTS_DIR/Kimodo-SMPLX-RP-v1"/ 2>/dev/null | grep -v "^total"
echo ""

# ---- 4. Verify ----
echo "[4/4] Verifying setup..."

docker image inspect kimodo:genmo &>/dev/null && echo "  Docker image: OK" || echo "  Docker image: MISSING"
[ -f "$WEIGHTS_DIR/Kimodo-SMPLX-RP-v1/model.safetensors" ] && echo "  Model weights: OK" || echo "  Model weights: MISSING"
[ -f "$WEIGHTS_DIR/Kimodo-SMPLX-RP-v1/config.yaml" ] && echo "  Model config:  OK" || echo "  Model config:  MISSING"
[ -d "$WEIGHTS_DIR/Kimodo-SMPLX-RP-v1/stats" ] && echo "  Model stats:   OK" || echo "  Model stats:   MISSING"

echo ""
echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
echo ""
echo "Start the API:"
echo "  bash run.sh"
echo ""
echo "API will be available at: http://localhost:8020"
echo "Health check: curl http://localhost:8020/health"
