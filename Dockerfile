# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# - python3-pip, python3-dev: for python
# - libsndfile1, ffmpeg: for librosa/audio processing
# - curl, git: utilities
run apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    libsndfile1 \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install python dependencies
# First copy requirements to cache pip steps
COPY requirements.txt .

# Upgrade pip and install requirements
# We point to the CUDA 11.8 wheel for torch to match the base image
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    python3.10 -m pip install -r requirements.txt

# Copy project files
COPY . .

# Create models directory ensuring permissions
RUN mkdir -p models && chmod 777 models

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
