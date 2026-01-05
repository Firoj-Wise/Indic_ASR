import argparse
import sys
import os
import torch
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import time

# Add project root to path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.load_model import IndicConformerASR
from app.utils.metrics import calculate_cer, calculate_wer
from app.utils.logger_utils import LOGGER

def benchmark(language: str, subset: str = None, samples: int = 100, output_csv: str = "benchmark_results.csv"):
    """
    Benchmarks the ASR model on a specific subset of IndicVoices.
    
    Args:
        language (str): Language code (e.g., 'ne', 'hi', 'mai').
        subset (str): Dataset subset name (usually capitalized language name, e.g., 'Nepali').
        samples (int): Number of samples to process (default 100). -1 for all.
        output_csv (str): Output CSV file path.
    """
    LOGGER.info(f"Starting benchmarking for Language: {language}, Subset: {subset}")
    
    # Initialize Model
    try:
        asr_model = IndicConformerASR()
    except Exception as e:
        LOGGER.critical(f"Failed to initialize model: {e}")
        return

    # Load Dataset
    # Streaming is safer for Colab RAM
    dataset_name = "ai4bharat/indicvoices_r"
    
    LOGGER.info(f"Loading dataset: {dataset_name} [{subset}, split='test'] (Streaming)...")
    try:
        ds = load_dataset(dataset_name, subset, split="test", streaming=True, trust_remote_code=True)
    except Exception as e:
        LOGGER.error(f"Failed to load dataset: {e}")
        return

    results = []
    
    # Iterate
    # Since it's streaming, we take N samples
    iterator = iter(ds)
    
    total_cer = 0.0
    total_wer = 0.0
    count = 0
    
    # Create a temporary directory for audio files if needed 
    # (But IndicVoices usually streams audio. Let's see if we need to save it to disk for the model)
    # The model.transcribe() takes a path. So we MUST save to disk.
    temp_dir = "temp_bench_audio"
    os.makedirs(temp_dir, exist_ok=True)
    
    LOGGER.info(f"Processing {samples if samples > 0 else 'ALL'} samples...")
    
    try:
        pbar = tqdm(total=samples if samples > 0 else None)
        
        for i, sample in enumerate(iterator):
            if samples > 0 and i >= samples:
                break
                
            # Sample structure: {'id': ..., 'audio': {'path': ..., 'array': ..., 'sampling_rate': ...}, 'transcription': ...}
            audio_data = sample['audio']
            reference_text = sample['transcription']
            
            # Save audio to temp file
            # We use soundfile to save the array
            import soundfile as sf
            
            temp_path = os.path.join(temp_dir, f"sample_{i}.wav")
            sf.write(temp_path, audio_data['array'], audio_data['sampling_rate'])
            
            # Transcribe
            start_time = time.time()
            try:
                hypothesis_text = asr_model.transcribe(temp_path, language_id=language)
            except Exception as e:
                LOGGER.error(f"Inference failed on sample {i}: {e}")
                hypothesis_text = ""
            duration = time.time() - start_time
            
            # Calculate Metrics
            cer = calculate_cer(reference_text, hypothesis_text)
            wer = calculate_wer(reference_text, hypothesis_text)
            
            total_cer += cer
            total_wer += wer
            count += 1
            
            results.append({
                "id": sample.get('id', i),
                "reference": reference_text,
                "hypothesis": hypothesis_text,
                "cer": cer,
                "wer": wer,
                "duration": duration
            })
            
            pbar.update(1)
            pbar.set_description(f"Avg CER: {total_cer/count:.4f}")
            
            # Cleanup temp file
            os.remove(temp_path)
            
    except KeyboardInterrupt:
        LOGGER.info("Benchmarking interrupted by user.")
    except Exception as e:
        LOGGER.error(f"Benchmarking loop error: {e}")
    finally:
        pbar.close()
        if os.path.exists(temp_dir):
             import shutil
             shutil.rmtree(temp_dir)

    # Save Results
    if count > 0:
        avg_cer = total_cer / count
        avg_wer = total_wer / count
        LOGGER.info(f"Benchmarking Complete.")
        LOGGER.info(f"Average CER: {avg_cer:.4%}")
        LOGGER.info(f"Average WER: {avg_wer:.4%}")
        
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        LOGGER.info(f"Results saved to {output_csv}")
    else:
        LOGGER.warning("No samples processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Indic Conformer ASR")
    parser.add_argument("--language", type=str, required=True, help="Language code (e.g., ne, hi, mai)")
    parser.add_argument("--subset", type=str, required=True, help="HF Dataset Subset (e.g., Nepali, Hindi, Maithili)")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to benchmark")
    parser.add_argument("--output", type=str, default="benchmark_results.csv", help="Output CSV path")
    
    args = parser.parse_args()
    
    benchmark(args.language, args.subset, args.samples, args.output)
