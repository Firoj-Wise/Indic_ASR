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
from tqdm import tqdm
try:
    from ai4bharat.transliteration import XlitEngine
except ImportError:
    XlitEngine = None

def generate_ground_truth(input_dir, output_manifest, lang_code="ne"):
    """
    Executes the ground truth generation pipeline.
    
    Args:
        input_dir (str): Directory containing source .wav files and matching .txt transcripts.
        output_manifest (str): Path to write the resulting JSONL manifest.
        lang_code (str): Target language code (ne/hi/mai).
    """
    # IndicXlit uses 2-letter codes mostly (ne, hi, mai) which match our input.
    xlit_code = lang_code 
    
    print(f"Initializing IndicXlit Engine (Target: {xlit_code})...")
    
    if XlitEngine is None:
        print("Error: ai4bharat-transliteration library not found. Please install it.")
        return

    try:
        # Initialize XlitEngine. Note: This downloads models on first run.
        # Beam width 10 gives better quality.
        xlit_engine = XlitEngine(xlit_code, beam_width=10)
    except ValueError as e:
        if "mutable default" in str(e) and "fairseq" in str(e):
             print(f"\n{'='*60}")
             print("CRITICAL COMPATIBILITY ERROR DETECTED")
             print(f"Error: {e}")
             print("-" * 60)
             print("CAUSE: You are running Python 3.9+ (likely 3.11/3.12) with an old version of Fairseq.")
             print("FIX: You MUST upgrade fairseq from source to fix this 'dataclass' issue.")
             print("Run this command in your terminal/notebook cell:")
             print("\n    pip install --upgrade git+https://github.com/facebookresearch/fairseq.git\n")
             print("Then restart the kernel and try again.")
             print("="*60 + "\n")
             return
        else:
            print(f"Critical Error: Failed to initialize IndicXlit: {e}")
            return
    except Exception as e:
        print(f"Critical Error: Failed to initialize IndicXlit: {e}")
        return

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
                "source": "pipeline_librispeech_indicxlit"
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

