```python
import argparse
import os
import io
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm

def fetch_audio(output_dir, samples_per_source=250, token=None):
    """
    Fetches audio from LibriSpeech and Google FLEURS.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Fetching audio to {output_dir}...")

    if token:
        try:
            from huggingface_hub import login
            login(token=token)
        except ImportError:
            pass

    # Source 1: LibriSpeech (Clean)
    print("Loading LibriSpeech (clean)...")
    try:
        ls_ds = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
        # Force decode=False immediately to avoid backend (torchcodec) issues
        ls_ds = ls_ds.cast_column("audio", Audio(decode=False))
        process_dataset(ls_ds, output_dir, "ls", samples_per_source)
    except Exception as e:
        print(f"Error loading LibriSpeech: {e}")

    # Source 2: Common Voice 11.0 (Diverse)
    print("Loading Common Voice 11.0 (en)...")
    try:
        # User reported 'trust_remote_code' is not supported for this dataset in newer 'datasets' lib.
        # Removing the flag.
        cv_ds = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train", streaming=True)
        cv_ds = cv_ds.cast_column("audio", Audio(decode=False))
        process_dataset(cv_ds, output_dir, "cv", samples_per_source)
    except Exception as e:
        print(f"Error loading Common Voice 11.0: {e}")
        print("Falling back to LibriSpeech 'other' for diversity...")
        try:
             ls_other = load_dataset("librispeech_asr", "other", split="test", streaming=True)
             ls_other = ls_other.cast_column("audio", Audio(decode=False))
             process_dataset(ls_other, output_dir, "ls_other", samples_per_source)
        except Exception as ex:
            print(f"Error loading fallback: {ex}")
        
    print("Done fetching audio.")

def process_dataset(dataset, output_dir, prefix, limit):
    count = 0
    # Already cast to decode=False, so we expect bytes.
    iterator = iter(dataset)
    
    pbar = tqdm(total=limit, desc=f"Saving {prefix}")
    
    while count < limit:
        try:
            sample = next(iterator)
            
            # Now we have bytes!
            audio_dict = sample['audio']
            bytes_data = audio_dict.get('bytes')
            
            array = None
            sampling_rate = None

            # Decode manually using soundfile
            if bytes_data:
                try:
                    array, sampling_rate = sf.read(io.BytesIO(bytes_data))
                except Exception as e:
                    # print(f"Soundfile decoding from bytes failed: {e}")
                    pass # Try fallback
            
            if array is None: # If bytes decoding failed or wasn't available
                # If decode=False failed or not supported, maybe we got array?
                array = audio_dict.get('array')
                sampling_rate = audio_dict.get('sampling_rate')

            if array is None:
                continue

            # Resample? Model usually expects 16k.
            # LibriSpeech is 16k. FLEURS is 16k usually.
            # If not 16k, we should resample. 
            # Ideally use librosa for resampling, but let's assume 16k or accept strictness.
            # But wait, sf.read returns whatever the file is.
            # Let's verify rate.
            if sampling_rate != 16000:
                # quick resample if needed, or skip? 
                # skipping is safer for creating "Standard Dataset" than bad resampling without scipy/librosa
                # But adding librosa dep is fine.
                try:
                    import librosa
                    import numpy as np
                    
                    # sf.read puts data in float64 usually, librosa expects float32
                    if array.dtype != np.float32:
                        array = array.astype(np.float32)

                    if len(array.shape) > 1:
                         array = array.mean(axis=1) # stereo to mono
                    
                    array = librosa.resample(array, orig_sr=sampling_rate, target_sr=16000)
                    sampling_rate = 16000
                except ImportError:
                    # print("Librosa not installed, skipping resampling.")
                    pass
                except Exception as e:
                    # print(f"Resampling failed: {e}")
                    pass
            
            # Filter short audio
            if len(array) < 16000 * 1: # < 1 second
                continue
                
            file_name = f"{prefix}_{count:04d}.wav"
            file_path = os.path.join(output_dir, file_name)
            
            sf.write(file_path, array, sampling_rate)
            
            # Handle text extraction
            original_text = (
                sample.get('text') or 
                sample.get('sentence') or 
                sample.get('transcription') or 
                sample.get('raw_transcription') or 
                ""
            )
            
            if original_text:
                txt_path = os.path.join(output_dir, f"{prefix}_{count:04d}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(original_text)

            count += 1
            pbar.update(1)
            
        except StopIteration:
            break
        except Exception as e:
            # print(f"Skipping: {e}") # Reduce spam
            continue
            
    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Audio for Eval")
    parser.add_argument("--output_dir", type=str, default="eval/audio_samples", help="Directory to save audio")
    parser.add_argument("--samples", type=int, default=500, help="Total samples")
    parser.add_argument("--token", type=str, help="HF Token for login")
    
    args = parser.parse_args()
    
    # Split samples
    per_source = args.samples // 2
    # Use env token if arg not provided
    token = args.token or os.environ.get("HF_TOKEN")
    
    fetch_audio(args.output_dir, samples_per_source=per_source, token=token)
```
