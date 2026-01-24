# Evaluation Pipeline for English-to-Devanagari ASR

This directory contains scripts to evaluate the ASR model on English audio inputs where the expected output is Devanagari transliteration.

## Workflow



1.  **Fetch Audio**: Download diverse English audio samples (LibriSpeech) along with their ground truth English text.
2.  **Generate Ground Truth**: 
    -   Read English transcripts from fetched dataset (Silver Standard).
    -   Transliterate English Text -> Devanagari using `google-transliteration-api` (Ground Truth).
3.  **Benchmark**:
    -   Run local `IndicConformerASR` model on the audio.
    -   Compare hypothesis (Model Output) vs Ground Truth.
    -   Calculate CER/WER.

## Datasets

This pipeline uses open-source, public datasets to ensure ease of reproduction:
1.  **LibriSpeech Clean**: High-quality audiobook speech.
2.  **LibriSpeech Other**: More challenging speech (accents/noise) to test robustness.

No special access tokens are required for these datasets.

## Evaluation Resources

You can reproduce the evaluation results or explore the data using the following resources:

-   **Colab Notebook**: [Run Evaluation on Colab](https://colab.research.google.com/drive/1Tpz1ntTeml-vcz5GZ14SoRjvlQIZrfpL?usp=sharing)
-   **Results & Datasets (HuggingFace)**:
    -   [Maithili (mai)](https://huggingface.co/datasets/SamaFiroz/indic-asr-eval-results-mai)
    -   [Hindi (hi)](https://huggingface.co/datasets/SamaFiroz/indic-asr-eval-results-hi)
    -   [Nepali (ne)](https://huggingface.co/datasets/SamaFiroz/indic-asr-eval-results-ne)

## Challenges & Limitations

-   **Google Transliteration API**: During our evaluation, we observed that the Google Transliteration API did not always produce the expected quality of transliteration for our specific use cases. While it serves as a baseline, we are actively exploring alternative transliteration engines to improve the ground truth generation quality.


## Running on Colab

If you are running this in Google Colab, you can use the following commands:

```bash
# 1. Install Dependencies
!apt-get install -y libsndfile1 ffmpeg
!pip install google-transliteration-api ffmpeg-python jiwer soundfile huggingface-hub onnxruntime-gpu

# 2. Login to HF (Important for Uploading Results ONLY)
import os
from huggingface_hub import login
token = "hf_..." # Replace with your token
login(token=token)
os.environ["HF_TOKEN"] = token

# 3. Fetch Audio (e.g. 500 samples)
# Uses LibriSpeech (Public) - Generates both .wav and .txt files
!python eval/fetch_audio.py --samples 500 --token "$token"

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

-   `fetch_audio.py`: Downloads and filters audio from Hugging Face Datasets (LibriSpeech). Saves .wav and .txt (transcript) files.
-   `generate_ground_truth.py`: Uses English text + Google Transliteration API to create the reference Devanagari text.
-   `benchmark_eval.py`: Runs the standard evaluation loop.
-   `push_to_hub.py`: Uploads results and audio to HF Hub for visualization.