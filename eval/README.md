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

## Running on Colab

If you are running this in Google Colab, you can use the following commands:

```bash
# 1. Install Dependencies
pip install openai-whisper indic-transliteration ffmpeg-python jiwer soundfile

# 2. Fetch Audio (e.g. 500 samples)
python eval/fetch_audio.py --samples 500 --output_dir eval/audio_samples

# 3. Generate Ground Truth Manifest
python eval/generate_ground_truth.py --input_dir eval/audio_samples --output_manifest eval/manifest.jsonl

# 4. Run Benchmark
python eval/benchmark_eval.py --manifest eval/manifest.jsonl --output eval/results.csv
```

## Scripts

-   `fetch_audio.py`: Downloads and filters audio from Hugging Face Datasets.
-   `generate_ground_truth.py`: Uses Whisper + Transliteration to create the reference text.
-   `benchmark_eval.py`: Runs the standard evaluation loop.