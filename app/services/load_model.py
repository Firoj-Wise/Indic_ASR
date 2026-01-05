import torch
import librosa
from transformers import AutoModel
from app.utils.config.config import Config
from app.utils.logger_utils import LOGGER

class IndicConformerASR:
    def __init__(self):
        """
        Initializes the IndicConformer ASR model resources.
        Uses AutoModel to load the custom IndicASRModel (ONNX-based).
        """
        self.model_id = Config.MODEL_ID
        self.device = Config.DEVICE_CUDA if torch.cuda.is_available() else Config.DEVICE_CPU
        self.token = Config.HF_TOKEN

        LOGGER.info(f"Initializing model: {self.model_id} on device: {self.device}")
        
        if not self.token:
             LOGGER.warning("HF_TOKEN missing. Gated model access may fail.")

        try:
            # The custom model code (model_onnx.py) manages its own downloads via snapshot_download.
            # It unfortunately ignores cache_dir in its from_pretrained method, so it defaults to ~/.cache.
            # We proceed with AutoModel as this is the only correct way to load this custom model.
            load_kwargs = {
                "pretrained_model_name_or_path": self.model_id,
                "trust_remote_code": True,
                "token": self.token
            }

            LOGGER.info("Loading model (this might trigger internal download)...")
            self.model = AutoModel.from_pretrained(**load_kwargs)
            
            # The generic AutoModel wrapping might store the custom model in .model or match it directly
            # Based on model_onnx.py, from_pretrained returns the instance directly.
            
            LOGGER.info("Model loaded successfully.")
            
        except Exception as e:
            LOGGER.error(f"Failed to load model: {e}")
            raise e

    def transcribe(self, audio_path: str, language_id: str = "hi") -> str:
        """
        Transcribes the given audio file into text using the specified language.
        
        Args:
            audio_path: Absolute path to the audio file.
            language_id: ISO language code (e.g., 'hi', 'ne', 'mai').
            
        Returns:
            str: Transcribed text.
        """
        try:
            LOGGER.info(f"Processing {audio_path} [Lang: {language_id}]")
            
            # Load audio using librosa
            # The model's preprocessor expects 16kHz audio
            audio_array, _ = librosa.load(audio_path, sr=Config.SAMPLING_RATE)
            
            # Convert to torch tensor
            audio_tensor = torch.tensor(audio_array).unsqueeze(0) # Batch dimension (1, T)
            
            # Perform inference
            transcription = self.model(audio_tensor, language_id)
            
            if not transcription:
                raise ValueError("Model returned empty transcription.")

            return transcription

        except Exception as e:
            LOGGER.error(f"Inference failed for {audio_path}: {e}")
            raise e
