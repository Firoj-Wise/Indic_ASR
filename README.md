# Indic Conformer ASR API & Collaborative Dashboard

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95.1-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Available-2496ED?style=flat-square&logo=docker&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?style=flat-square&logo=nvidia&logoColor=white)

## Overview

High-performance, containerized ASR service for Indian languages (Nepali, Hindi, Maithili). Features a real-time **Collaborative Dashboard** where multiple clients (browsers, terminal scripts, robots) can stream audio and view synchronized transcripts instantly.

## Architecture

-   **Backend**: FastAPI with WebSockets for low-latency streaming.
-   **Model**: Custom hybrid Conformer-CTC model (ONNX/PyTorch).
-   **Dashboard**: Real-time Web UI for monitoring and control.
-   **Orchestration**: Pipecat integration for headless/terminal clients.

## Quick Start

### 1. Run Server
Mange dependencies with `uv` for speed.
```bash
uv sync
uv run python -m app.main
```

### 2. Access Dashboard
Open **[http://localhost:8000/ui](http://localhost:8000/ui)**
-   **Monitor**: See transcripts from all connected sources.
-   **Control**: Change the global language setting (syncs across all clients).
-   **Test**: Use the browser microphone.

### 3. Run Headless Client (Pipecat)
Run the terminal client to simulate a robot or backend process.
```bash
# Language flag is optional (defaults to global state or 'hi')
uv run scripts/run_pipecat.py --language ne
```
*Note: If testing locally, mute the browser microphone to avoid echo.*

## API Reference

### Service Info
**`GET /info`**
Returns service metadata, version, and usage examples.

### Streaming Transcription
**`WS /transcribe/ws`**
-   **Audio**: Binary PCM16 (16kHz, Mono).
-   **Config**: JSON `{ "type": "config", "language": "ne" }`.
-   **Output**: JSON `{ "type": "transcription", "text": "...", "source": "stream" }`.

### Batch Transcription
**`POST /transcribe`**
Upload a file for full transcription.
```bash
curl -X POST "http://localhost:8000/transcribe?language=ne" -F "file=@audio.wav"
```

## Benchmarking

Evaluate CER/WER on IndicVoices dataset.
```bash
# Benchmark Nepali
python scripts/benchmark.py --language ne --subset Nepali --samples 100
```

## Docker Deployment

Production-ready Dockerfile included.
```bash
docker-compose up --build -d
```
Mounts `models/` volume for persistence.