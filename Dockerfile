# Kimodo API Docker image.
#
# Builds on top of the kimodo:1.0 base image (which includes PyTorch, CUDA,
# kimodo package, MotionCorrection, text encoder, and all dependencies).
#
# Build:
#   docker build -t kimodo-api:latest -f Dockerfile ..
#
# The build context should be the kimodo repo root (one level up from kimodo-api/).

FROM kimodo:1.0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/kimodo-api

# Install API-specific deps (fastapi, uvicorn, etc.)
COPY kimodo-api/requirements.txt /workspace/kimodo-api/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY kimodo-api/app /workspace/kimodo-api/app

# Default port
ENV KIMODO_API_PORT=8020
ENV KIMODO_DEVICE=cuda
ENV KIMODO_MODEL=smplx
ENV KIMODO_API_LOG_LEVEL=INFO

EXPOSE 8020

CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8020"]
