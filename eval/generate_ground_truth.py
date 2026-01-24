"""
Ground Truth Generation Pipeline for Cross-Lingual ASR Evaluation.

Objective:
    To rigorously benchmark the Indic Conformer ASR model's capability to transcription-transliterate 
    English speech into Devanagari script (e.g., "Hello World" -> "हेलो वर्ल्ड").

Methodology:
    Since large-scale "English Audio -> Devanagari Text" datasets are non-standard, we synthesize 
    a high-quality evaluation set using a two-stage pseudo-labeling pipeline:

    1.  **Read Source Transcription**:
        We read the "Silver Standard" English transcriptions directly from the dataset (e.g. LibriSpeech)
        which are extracted during the fetch stage.

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

import argparse
import os
import json
import torch
from tqdm import tqdm
try:
    from google.transliteration import transliterate_text
except ImportError:
    transliterate_text = None

def generate_ground_truth(input_dir, output_manifest, lang_code="ne"):
    """
    Executes the ground truth generation pipeline.
    
    Args:
        input_dir (str): Directory containing source .wav files and matching .txt transcripts.
        output_manifest (str): Path to write the resulting JSONL manifest.
        lang_code (str): Target language code (ne/hi/mai).
    """
    print(f"Initializing Transliteration Engine (Target: {lang_code})...")
    
    if transliterate_text is None:
        print("Error: google-transliteration-api library not found. Please install it.")
        return

    # Check for Maithili support or fallback
    # Google Transliteration API supports 'hi', 'ne', etc. 
    # 'mai' (Maithili) might fall back to 'hi' if not explicitly supported, or we can try 'hi' script.
    # We will pass the code as is.
    
    # Filter for valid audio extensions
    audio_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    results = []
    
    print(f"Pipeline configured. Processing {len(audio_files)} artifacts in: {input_dir}")
    
    for filename in tqdm(audio_files, desc="Synthesizing Ground Truth"):
        file_path = os.path.join(input_dir, filename)
        txt_filename = filename.replace(".wav", ".txt")
        txt_path = os.path.join(input_dir, txt_filename)
        
        try:
            # Stage 1: Load Source Transcription (English) from .txt
            if not os.path.exists(txt_path):
                # print(f"Warning: Missing transcript for {filename}, skipping.")
                continue
                
            with open(txt_path, "r", encoding="utf-8") as f:
                en_text = f.read().strip()
                
            if not en_text:
                continue


            # Stage 2: Target Transliteration (English -> Devanagari)
            # Using Google Transliteration API (Unofficial)
            # It returns the exact transliterated string directly
            try:
                ne_text = transliterate_text(en_text, lang_code=lang_code)
            except Exception as e:
                # Fallback if specific code failed, or let it fail
                # For Maithili (mai), if not supported, we can fallback to Hindi (hi) which is same script
                if lang_code == "mai":
                     ne_text = transliterate_text(en_text, lang_code="hi")
                else:
                    raise e
            
            manifest_entry = {
                "audio_path": os.path.abspath(file_path),
                "en_text": en_text,       # Reference English
                "ne_text": ne_text,       # Reference Devanagari 
                "source": "pipeline_librispeech_google_xlit"
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
    # Removed --model_size as Whisper is no longer used
    parser.add_argument("--language", type=str, default="ne", help="Target Indic Language (ne/hi/mai)")
    
    args = parser.parse_args()
    
    generate_ground_truth(args.input_dir, args.output_manifest, args.language)

