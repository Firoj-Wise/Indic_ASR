import argparse
import os
import random
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

def fetch_audio(output_dir, samples_per_source=250):
    """
    Fetches audio from Common Voice and LibriSpeech to create a diverse evaluation set.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Fetching audio to {output_dir}...")

    print(f"Fetching audio to {output_dir}...")

    # Source 1: LibriSpeech (Clean, Audiobooks) - Reliable & Clean
    print("Loading LibriSpeech (clean)...")
    try:
        ls_ds = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
        process_dataset(ls_ds, output_dir, "ls", samples_per_source)
    except Exception as e:
        print(f"Error loading LibriSpeech: {e}")

    # Source 2: Google FLEURS (Diverse, Multi-domain)
    # Using 'en_us' subset. It is open and diverse, unlike Common Voice which is now gated/restricted.
    print("Loading Google FLEURS (en_us)...")
    try:
        # subset="en_us", split="test"
        fl_ds = load_dataset("google/fleurs", "en_us", split="test", streaming=True)
        process_dataset(fl_ds, output_dir, "fl", samples_per_source)
    except Exception as e:
        print(f"Error loading FLEURS: {e}")
        
    print("Done fetching audio.")
        
    print("Done fetching audio.")

def process_dataset(dataset, output_dir, prefix, limit):
    count = 0
    # Use iter explicitly
    iterator = iter(dataset)
    
    pbar = tqdm(total=limit, desc=f"Saving {prefix}")
    
    while count < limit:
        try:
            sample = next(iterator)
            
            # Ensure audio is loaded/decoded
            if 'audio' not in sample:
                continue
                
            audio = sample['audio']
            
            # audio['array'] might be None if decoding failed
            if audio.get('array') is None:
                continue
            
            # Text field varies by dataset
            # LibriSpeech: 'text'
            # Common Voice: 'sentence'
            # FLEURS: 'transcription' or 'raw_transcription'
            original_text = (
                sample.get('text') or 
                sample.get('sentence') or 
                sample.get('transcription') or 
                sample.get('raw_transcription') or 
                ""
            )
            
            # Filter out very short audio
            if len(audio['array']) < 16000 * 1: # < 1 second
                continue
                
            file_name = f"{prefix}_{count:04d}.wav"
            file_path = os.path.join(output_dir, file_name)
            
            # Save as 16kHz wav (standard for most ASR)
            sf.write(file_path, audio['array'], audio['sampling_rate'])
            
            # Save original english text for reference (optional, but good to have)
            # We will generate our own ground truth later, but having the source text is useful for debug.
            txt_path = os.path.join(output_dir, f"{prefix}_{count:04d}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(original_text)

            count += 1
            pbar.update(1)
            
        except StopIteration:
            break
        except Exception as e:
            print(f"Skipping sample due to error: {e}")
            continue
            
    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Audio for Eval")
    parser.add_argument("--output_dir", type=str, default="eval/audio_samples", help="Directory to save audio")
    parser.add_argument("--samples", type=int, default=500, help="Total samples (split roughly evenly between sources)")
    
    args = parser.parse_args()
    
    # Split samples between sources
    per_source = args.samples // 2
    fetch_audio(args.output_dir, samples_per_source=per_source)
