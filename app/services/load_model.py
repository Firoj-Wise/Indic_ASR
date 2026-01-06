import io
import soundfile as sf
import torch
import librosa
from transformers import AutoModel
from app.utils.config.config import Config
from app.utils.logger_utils import LOGGER
from app.constants import log_msg

class IndicConformerASR:
    def __init__(self):
        """
        Initializes the IndicConformer ASR model resources.
        Uses AutoModel to load the custom IndicASRModel (ONNX-based).
        """
        self.model_id = Config.MODEL_ID
        self.device = Config.DEVICE_CUDA if torch.cuda.is_available() else Config.DEVICE_CPU
        self.token = Config.HF_TOKEN

        LOGGER.info(log_msg.ASR_MODEL_INIT.format(self.model_id, self.device))
        
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

            LOGGER.info(log_msg.ASR_MODEL_LOAD_START)
            self.model = AutoModel.from_pretrained(**load_kwargs)
            
            # The generic AutoModel wrapping might store the custom model in .model or match it directly
            # Based on model_onnx.py, from_pretrained returns the instance directly.
            
            LOGGER.info(log_msg.ASR_MODEL_LOAD_SUCCESS)
            
        except Exception as e:
            LOGGER.error(log_msg.ASR_MODEL_LOAD_FAIL.format(e))
            raise e

    def transcribe(self, audio_data, language_id: str = "hi") -> str:
        """
        Transcribes the given audio data into text using the specified language.
        
        Args:
            audio_data: BytesIO object or path (str) containing audio data.
            language_id: ISO language code (e.g., 'hi', 'ne', 'mai').
            
        Returns:
            str: Transcribed text.
        """
        try:
            # LOGGER.info(log_msg.ASR_PROCESSING_START.format("MEMORY_STREAM", language_id))
            
            # Load audio from bytes or path
            # Using soundfile is faster and supports file-like objects natively
            if isinstance(audio_data, str):
                audio_array, _ = librosa.load(audio_data, sr=Config.SAMPLING_RATE)
            else:
                 # audio_data is likely BytesIO from UploadFile
                 # sf.read returns (data, samplerate)
                 audio_array, sr = sf.read(audio_data)
                 # Resample if needed using librosa (only if SR differs provided)
                 # Ideally client sends correct SR, but safe to resample
                 if sr != Config.SAMPLING_RATE:
                     audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=Config.SAMPLING_RATE)

            # Convert to torch tensor
            audio_tensor = torch.tensor(audio_array).float().unsqueeze(0) # Batch dimension (1, T)
            
            # Perform inference
            transcription = self.model(audio_tensor, language_id)
            
            if not transcription:
                # raise ValueError("Model returned empty transcription.")
                return "" # Return empty string instead of error for VAD silence

            return transcription

        except Exception as e:
            LOGGER.error(f"Inference Logic Error: {e}")
            raise e
