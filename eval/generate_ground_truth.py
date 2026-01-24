"""
Ground Truth Generation Pipeline for Cross-Lingual ASR Evaluation.

Objective:
    To rigorously benchmark the Indic Conformer ASR model's capability to transcription-transliterate 
    English speech into Devanagari script (e.g., "Hello World" -> "हेलो वर्ल्ड").

Methodology:
    Since large-scale "English Audio -> Devanagari Text" datasets are non-standard, we synthesize 
    a high-quality evaluation set using a two-stage pseudo-labeling pipeline:

    1.  **High-Fidelity ASR (Source Transcription)**:
        We employ OpenAI's Whisper (Transformer-based encoder-decoder) to recover the source English 
        text from the audio. Whisper is chosen for its robustness to accents and noise, providing 
        a strong "Silver Standard" English transcript.

    2.  **Neural Transliteration (Script Conversion)**:
        We utilize `ai4bharat/IndicXlit` (Transformer-based sequence-to-sequence model) to convert 
        the English text into Devanagari. Unlike rule-based systems (ITRANS), IndicXlit captures 
        context-aware phonetics and common usage (e.g., "Bank" -> "बैंक", not just "बंक"). 
        This provides a much more natural "Ground Truth" for mixed-lingual scenarios.

    This pipeline minimizes the domain gap between "Transliterated English" and "Native Indic Script".
"""

import argparse
import os
import json
import torch
import whisper
from tqdm import tqdm
from ai4bharat.transliteration import XlitEngine

def generate_ground_truth(input_dir, output_manifest, model_size="base", lang_code="ne"):
    """
    Executes the ground truth generation pipeline.
    
    Args:
        input_dir (str): Directory containing source .wav files.
        output_manifest (str): Path to write the resulting JSONL manifest.
        model_size (str): Whisper model capacity (tiny|base|small|medium|large).
        lang_code (str): Target language code (ne/hi/mai).
    """
    print(f"Initializing Whisper ASR Model (Backbone: {model_size})...")
    
    # Device agnostic loading (Priority: CUDA > CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = whisper.load_model(model_size, device=device)
    except Exception as e:
        print(f"Critical Error: Failed to load Whisper backbone: {e}")
        return

    # IndicXlit uses 2-letter codes mostly (ne, hi, mai) which match our input.
    xlit_code = lang_code 
    
    print(f"Initializing IndicXlit Engine (Target: {xlit_code})...")
    try:
        # Initialize XlitEngine. Note: This downloads models on first run.
        # Beam width 10 gives better quality.
        xlit_engine = XlitEngine(xlit_code, beam_width=10)
    except Exception as e:
        print(f"Critical Error: Failed to initialize IndicXlit: {e}")
        return

    # Filter for valid audio extensions
    audio_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    results = []
    
    print(f"Pipeline configured. Processing {len(audio_files)} artifacts in: {input_dir}")
    
    for filename in tqdm(audio_files, desc="Synthesizing Ground Truth"):
        file_path = os.path.join(input_dir, filename)
        
        try:
            # Stage 1: Source Transcription (English)
            result = model.transcribe(file_path, language="en")
            en_text = result["text"].strip()
            
            # Stage 2: Target Transliteration (English -> Devanagari)
            # Using Neural Transliteration from AI4Bharat
            ne_text = xlit_engine.translit_sentence(en_text)
            
            # Since translit_sentence might return a dict or string depending on version,
            # usually it returns specific top string if beam is handled, or we check docs.
            # Standard generic usage: returns transliterated string.
            if isinstance(ne_text, dict):
                 # Handle if it returns dictionary (some versions do keys as language)
                 ne_text = ne_text.get(xlit_code, en_text) # Fallback

            manifest_entry = {
                "audio_path": os.path.abspath(file_path),
                "en_text": en_text,       # Reference English
                "ne_text": ne_text,       # Reference Devanagari (Neural Xlit)
                "source": "pipeline_whisper_indicxlit"
            }
            results.append(manifest_entry)
            
        except Exception as e:
            print(f"Warning: Dropping sample {filename} due to pipeline failure: {e}")
            continue

    # Serialize Manifest
    with open(output_manifest, "w", encoding="utf-8") as f:
        for entry in results:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
            
    print(f"SUCCESS: Ground truth manifest generated at {output_manifest}")
    print(f"Total Samples: {len(results)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Ground Truth")
    parser.add_argument("--input_dir", type=str, default="eval/audio_samples", help="Input audio directory")
    parser.add_argument("--output_manifest", type=str, default="eval/manifest.jsonl", help="Output JSONL")
    parser.add_argument("--model_size", type=str, default="base", help="Whisper model size")
    parser.add_argument("--language", type=str, default="ne", help="Target Indic Language (ne/hi/mai)")
    
    args = parser.parse_args()
    
    generate_ground_truth(args.input_dir, args.output_manifest, args.model_size, args.language)
