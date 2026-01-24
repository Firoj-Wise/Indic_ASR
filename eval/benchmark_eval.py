import argparse
import json
import sys
import os
import pandas as pd
from tqdm import tqdm
import time
from jiwer import cer, wer

# Add project root to path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.load_model import IndicConformerASR

def benchmark_eval(manifest_path, output_csv, language_code="ne"):
    """
    Runs evaluation using the local IndicConformerASR model against the generated manifest.
    """
    print(f"Loading Manifest: {manifest_path}")
    data = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
            
    print(f"Initializing Model (Language: {language_code})...")
    try:
        asr_model = IndicConformerASR()
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    results = []
    total_cer = 0.0
    total_wer = 0.0
    count = 0
    
    print(f"Running Inference on {len(data)} samples...")
    
    for item in tqdm(data):
        audio_path = item['audio_path']
        reference_text = item['ne_text'] # Devanagari Ground Truth
        
        if not os.path.exists(audio_path):
            print(f"Audio not found: {audio_path}")
            continue
            
        start_time = time.time()
        try:
            # Run Inference
            hypothesis_text = asr_model.transcribe(audio_path, language_id=language_code)
        except Exception as e:
            print(f"Inference error on {audio_path}: {e}")
            hypothesis_text = ""
        duration = time.time() - start_time
        
        # Calculate Metrics
        # Handle empty strings to avoid division by zero in libraries sometimes
        if not reference_text.strip():
            curr_cer = 1.0 if hypothesis_text else 0.0
            curr_wer = 1.0 if hypothesis_text else 0.0
        else:
            curr_cer = cer(reference_text, hypothesis_text)
            curr_wer = wer(reference_text, hypothesis_text)
            
        total_cer += curr_cer
        total_wer += curr_wer
        count += 1
        
        results.append({
            "audio_path": audio_path,
            "reference": reference_text,
            "hypothesis": hypothesis_text,
            "cer": curr_cer,
            "wer": curr_wer,
            "duration": duration,
            "english_source": item.get('en_text', '')
        })

    # Summary
    if count > 0:
        avg_cer = total_cer / count
        avg_wer = total_wer / count
        print(f"Evaluation Complete.")
        print(f"Average CER: {avg_cer:.4%}")
        print(f"Average WER: {avg_wer:.4%}")
        
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    else:
        print("No samples processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Eval")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest.jsonl")
    parser.add_argument("--output", type=str, default="eval/results.csv", help="Output CSV")
    parser.add_argument("--language", type=str, default="ne", help="Target language code for model (ne/hi)")
    
    args = parser.parse_args()
    
    benchmark_eval(args.manifest, args.output, args.language)
