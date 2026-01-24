import argparse
import os
import pandas as pd
from datasets import Dataset, Audio, Features, Value
from huggingface_hub import login

def push_results_to_hub(results_csv, hub_repo_id, token=None):
    """
    Loads the results CSV, maps audio paths, and pushes a Datasets object to HF Hub.
    """
    if token:
        login(token=token)
        
    if not os.path.exists(results_csv):
        print(f"Results file not found: {results_csv}")
        return

    print(f"Loading results from {results_csv}...")
    df = pd.read_csv(results_csv)
    
    # Verify audio paths exist
    # The 'audio_path' column contains absolute local paths. 
    # When creating a Dataset with Audio feature, it embeds the audio.
    
    # We need to construct the dataset
    # Feature definition
    features = Features({
        "audio": Audio(sampling_rate=16000),
        "reference_text": Value("string"),
        "hypothesis_text": Value("string"),
        "english_source": Value("string"),
        "cer": Value("float"),
        "wer": Value("float"),
        "duration": Value("float"),
        "audio_path_local": Value("string") # Keep path for reference
    })
    
    # Prepare data dict
    data_dict = {
        "audio": df['audio_path'].tolist(), # Datasets will load audio from these paths
        "reference_text": df['reference'].tolist(),
        "hypothesis_text": df['hypothesis'].fillna("").tolist(),
        "english_source": df['english_source'].fillna("").tolist(),
        "cer": df['cer'].tolist(),
        "wer": df['wer'].tolist(),
        "duration": df['duration'].tolist(),
        "audio_path_local": df['audio_path'].tolist()
    }
    
    print("Creating HF Dataset...")
    dataset = Dataset.from_dict(data_dict, features=features)
    
    print(f"Pushing to Hub: {hub_repo_id}...")
    dataset.push_to_hub(hub_repo_id, private=True) # Defaults to private for safety
    print(f"Successfully pushed to https://huggingface.co/datasets/{hub_repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push Results to Hub")
    parser.add_argument("--results_csv", type=str, required=True, help="Path to results.csv")
    parser.add_argument("--repo_id", type=str, required=True, help="Target Hub Repo ID (e.g. username/dataset_name)")
    parser.add_argument("--token", type=str, help="HF Token (optional if logged in)")
    
    args = parser.parse_args()
    
    # Use env token if not provided arg
    token = args.token or os.environ.get("HF_TOKEN")
    
    push_results_to_hub(args.results_csv, args.repo_id, token)
