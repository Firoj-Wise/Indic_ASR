# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    # uv configuration
    UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-dev \
    libsndfile1 \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (frozen, no dev)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy project files
COPY . .

# Install the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Create models directory
RUN mkdir -p models && chmod 777 models

# Expose port
EXPOSE 8000

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]