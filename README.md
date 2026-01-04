# Indic ASR API

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95.1-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Available-2496ED?style=flat-square&logo=docker&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?style=flat-square&logo=nvidia&logoColor=white)

## Overview

This project provides a robust, containerized REST API for automatic speech recognition (ASR) focused on Indian languages. It encapsulates advanced acoustic modeling within a high-performance web server, designed to simplify the integration of speech-to-text capabilities into broader applications.

The system is built on a modular architecture that separates model inference, API routing, and configuration management. It leverages GPU acceleration to ensure low-latency transcription, making it suitable for near real-time use cases.

## Core Capabilities

### High-Performance Inference
The engine utilizes CUDA-optimized kernels to perform rapid audio decoding and feature extraction. By offloading matrix operations to the GPU, the system achieves significant throughput improvements compared to CPU-only execution, especially when processing batch requests.

### Containerization and Isolation
The entire application stack, including system dependencies (ffmpeg, libsndfile) and Python environments, is defined within a Docker container. This ensures complete reproducibility and eliminates "works on my machine" issues. Deep learning weights are persisted via volume mounting, preventing unnecessary redownloads while maintaining container ephemerality.

### Interactive User Interface
A custom-built web interface is served directly by the API. Designed with a modern, minimal aesthetic, it interfaces with the browser's MediaRecorder API to capture audio streams at the correct sampling rate and submit them for processing. This allows for immediate testing and demonstration without the need for external tools like cURL or Postman.

## Architecture

The system follows a layered architecture:

1.  **Transport Layer (FastAPI)**: Handles HTTP request validation, multipart file parsing, and response serialization.
2.  **Service Layer**: Manages the lifecycle of the ASR engine, ensuring models are loaded into memory only once during startup.
3.  **Inference Layer**:
    -   **Preprocessing**: Resamples arbitrary input audio to 16kHz mono.
    -   **Encoding**: Passes normalized audio tensors through a Conformer-based encoder.
    -   **Decoding**: Maps output logits to characters using Connectionist Temporal Classification (CTC).

## Deployment

### Docker (Recommended)

Deployment is managed via Docker Compose, which handles port mapping:

```bash
docker-compose up --build
```

This command will:
1.  Build the image using the NVIDIA CUDA base.
2.  Mount the local `models/` directory to persist downloaded weights.
3.  Expose the API on port 8000.

### Local Execution

For development environments without Docker:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

## API Reference

### Transcribe Audio
**Endpoint**: `POST /transcribe`

Accepts a binary audio file (WAV, MP3, FLAC) and limits processing to the specified language.

**Parameters**:
- `file`: The audio file object.
- `language`: ISO code (e.g., `ne` for Nepali, `hi` for Hindi, `mai` for Maithili).

### Health Check
**Endpoint**: `GET /`

Returns the operational status of the service and links to documentation.

### User Interface
**Endpoint**: `GET /ui`

Renders the HTML5 recording interface.