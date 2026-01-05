# Indic ASR API & Benchmarking

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95.1-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Available-2496ED?style=flat-square&logo=docker&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?style=flat-square&logo=nvidia&logoColor=white)

## Overview

This project provides a robust, containerized REST API for automatic speech recognition (ASR) focused on Indian languages. It encapsulates advanced acoustic modeling within a high-performance web server.

The codebase has been refactored into a modular `app/` structure and includes a comprehensive **benchmarking suite** to evaluate model performance (CER/WER) on the `ai4bharat/indicvoices_r` dataset.

## Directory Structure

```
.
├── app/
│   ├── api/            # API Routers (Endpoints)
│   ├── services/       # Business Logic (Model Loading)
│   ├── utils/          # Config, Logging, Metrics
│   ├── static/         # Frontend UI Assets
│   └── main.py         # Application Entry Point
├── scripts/
│   └── benchmark.py    # Benchmarking Script
├── models/             # Local Model Storage (Volume Mounted)
├── logs/               # Application Logs
├── colab_benchmark.ipynb # Google Colab Notebook
├── compose.yaml        # Docker Compose
└── pyproject.toml      # Project Dependencies (uv)
```

## Getting Started

### 1. Local Development (uv)

This project uses `uv` for ultra-fast dependency management.

```bash
# 1. Install dependencies
uv sync

# 2. Run the server
# The entry point is now app.main:app
uv run uvicorn app.main:app --reload

# 3. Access API
# Swagger UI: http://localhost:8000/docs
# Recorder UI: http://localhost:8000/ui
```

### 2. Docker Deployment

```bash
docker-compose up --build
```
This mounts the local `models/` directory, persisting large weights across container restarts.

## Benchmarking (New!)

You can benchmark the model against the **IndicVoices** dataset (Nepali, Hindi, Maithili).

### Run on Google Colab (Recommended)
1.  Upload `colab_benchmark.ipynb` to Google Colab.
2.  Follow the instructions in the notebook.
3.  **Note**: Ensure you use a T4 GPU runtime.

### Run Locally
```bash
# Example: Benchmark Nepali on the 'Nepali' subset
python scripts/benchmark.py --language ne --subset Nepali --samples 50

# Example: Benchmark Hindi
python scripts/benchmark.py --language hi --subset Hindi
```

Results are saved to `benchmark_results.csv`.

## API Reference

### Transcribe Audio
**Endpoint**: `POST /transcribe`

Accepts a binary audio file (WAV, MP3, FLAC) and language code.

**Parameters**:
- `file`: The audio file object.
- `language`: `ne` (Nepali), `hi` (Hindi), `mai` (Maithili).

### Health Check
**Endpoint**: `GET /`

Returns the operational status.