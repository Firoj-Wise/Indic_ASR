# Evaluation Pipeline for English-to-Devanagari ASR

This directory contains scripts to evaluate the ASR model on English audio inputs where the expected output is Devanagari transliteration.

## Workflow

1.  **Fetch Audio**: Download diverse English audio samples (Common Voice, LibriSpeech).
2.  **Generate Ground Truth**: 
    -   Transcribe English audio using `openai-whisper` (Base English Text).
    -   Transliterate English Text -> Devanagari using `indic-transliteration` (Ground Truth).
3.  **Benchmark**:
    -   Run local `IndicConformerASR` model on the audio.
    -   Compare hypothesis (Model Output) vs Ground Truth.
    -   Calculate CER/WER.

## Datasets

This pipeline uses open-source, public datasets to ensure ease of reproduction:
1.  **LibriSpeech**: Clean, audiobook-based English speech.
2.  **Common Voice 11.0**: Diverse, global English speech (Historical Open Release).

No special access tokens are required for these datasets.

## Running on Colab

If you are running this in Google Colab, you can use the following commands:

```bash
# 1. Install Dependencies
# Downgrade pip to avoid omegaconf metadata error (fairseq issue)
!pip install "pip<24.1"
!apt-get install -y libsndfile1 ffmpeg
!pip install openai-whisper ai4bharat-transliteration ffmpeg-python jiwer soundfile huggingface-hub

# 2. Login to HF (Important for Uploading Results ONLY)
import os
from huggingface_hub import login
token = "hf_..." # Replace with your token
login(token=token)
os.environ["HF_TOKEN"] = token

# 3. Fetch Audio (e.g. 500 samples)
# Uses LibriSpeech + FLEURS (Public)
!python eval/fetch_audio.py --samples 500

# 4. Generate Ground Truth & Run Benchmark (3-Way Evaluation)

# --- 1. NEPALI (ne) ---
!python eval/generate_ground_truth.py --language ne --output_manifest eval/manifest_ne.jsonl
!python eval/benchmark_eval.py --manifest eval/manifest_ne.jsonl --output eval/results_ne.csv --language ne

# --- 2. HINDI (hi) ---
!python eval/generate_ground_truth.py --language hi --output_manifest eval/manifest_hi.jsonl
!python eval/benchmark_eval.py --manifest eval/manifest_hi.jsonl --output eval/results_hi.csv --language hi

# --- 3. MAITHILI (mai) ---
!python eval/generate_ground_truth.py --language mai --output_manifest eval/manifest_mai.jsonl
!python eval/benchmark_eval.py --manifest eval/manifest_mai.jsonl --output eval/results_mai.csv --language mai

# 5. Upload Results to Hugging Face
!python eval/push_to_hub.py --results_csv eval/results_ne.csv --repo_id <your-username>/indic-asr-eval-ne
!python eval/push_to_hub.py --results_csv eval/results_hi.csv --repo_id <your-username>/indic-asr-eval-hi
!python eval/push_to_hub.py --results_csv eval/results_mai.csv --repo_id <your-username>/indic-asr-eval-mai
```

## Scripts

-   `fetch_audio.py`: Downloads and filters audio from Hugging Face Datasets (LibriSpeech/CommonVoice).
-   `generate_ground_truth.py`: Uses Whisper + Transliteration to create the reference text.
-   `benchmark_eval.py`: Runs the standard evaluation loop.
-   `push_to_hub.py`: Uploads results and audio to HF Hub for visualization.