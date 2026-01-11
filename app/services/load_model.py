import torch
import librosa
import numpy as np
from transformers import AutoModel
from app.utils.config.config import Config
from app.utils.logger_utils import LOGGER
from app.constants import log_msg

class IndicConformerASR:
    """
    Handles loading and inference for the Indic Conformer ASR model.
    
    Attributes:
        model_id (str): Hugging Face model ID.
        device (str): Computation device ('cuda' or 'cpu').
        model (AutoModel): Loaded ASR model instance.
    """
    def __init__(self) -> None:
        """
        Initializes the ASR model components.
        
        Raises:
            Exception: If model loading fails.
        """
        self.model_id: str = Config.MODEL_ID
        self.device: str = Config.DEVICE_CUDA if torch.cuda.is_available() else Config.DEVICE_CPU
        self.token: str = Config.HF_TOKEN

        LOGGER.info(log_msg.LOG_ENTRY.format("IndicConformerASR.__init__", f"Device={self.device}"))

        if not self.token:
             LOGGER.warning("HF_TOKEN missing. Gated model access may fail.")

        try:
            load_kwargs = {
                "pretrained_model_name_or_path": self.model_id,
                "trust_remote_code": True,
                "token": self.token
            }

            LOGGER.info(log_msg.ASR_MODEL_LOAD_START)
            self.model = AutoModel.from_pretrained(**load_kwargs)
            LOGGER.info(log_msg.ASR_MODEL_LOAD_SUCCESS)
            
        except Exception as e:
            LOGGER.error(log_msg.ASR_MODEL_LOAD_FAIL.format(e), exc_info=True)
            raise e
        
        LOGGER.info(log_msg.LOG_EXIT.format("IndicConformerASR.__init__"))
        return None

    def transcribe(self, audio_path: str, language_id: str = "hi") -> str:
        """
        Transcribes a local audio file.

        Args:
            audio_path (str): Absolute path to the audio file.
            language_id (str): Language code (e.g., 'hi', 'ne', 'mai'). Defaults to "hi".

        Returns:
            str: The transcribed text.

        Raises:
            Exception: If audio loading or inference fails.
        """
        LOGGER.info(log_msg.LOG_ENTRY.format("transcribe", f"File={audio_path}, Lang={language_id}"))
        try:
            LOGGER.info(log_msg.ASR_PROCESSING_START.format(audio_path, language_id))
            
            # Load audio (16kHz required by model)
            audio_array, _ = librosa.load(audio_path, sr=Config.SAMPLING_RATE)
            audio_tensor = torch.tensor(audio_array).unsqueeze(0) 
            
            result = self.transcribe_tensor(audio_tensor, language_id)
            LOGGER.info(log_msg.LOG_EXIT.format("transcribe"))
            return result

        except Exception as e:
            LOGGER.error(log_msg.ASR_INFERENCE_FAIL.format(audio_path, e), exc_info=True)
            raise e

    def transcribe_tensor(self, audio_tensor: torch.Tensor, language_id: str = "hi") -> str:
        """
        Transcribes a raw audio tensor.

        Args:
            audio_tensor (torch.Tensor): Audio data tensor.
            language_id (str): Language code. Defaults to "hi".

        Returns:
            str: The transcribed text, or empty string if no output.

        Raises:
            Exception: If inference fails.
        """
        # Verbose logging suppressed for tensor to avoid log flooding in streams
        try:
            transcription = self.model(audio_tensor, language_id)
            
            if not transcription:
                return "" 
            
            return transcription
            
        except Exception as e:
            LOGGER.error(log_msg.LOG_ERROR.format("transcribe_tensor", str(e)), exc_info=True)
            raise e
    
    def transcribe_with_timestamps(self, audio: str | np.ndarray, language_id: str = "hi") -> tuple:
        """
        Transcribes audio and returns word-level timestamps.
        
        Args:
            audio: str (path) or np.ndarray (waveform).
        
        Returns:
            tuple: (transcription_text, word_timestamps)
            where word_timestamps = [(word, start_sec, end_sec), ...]
        """
        if isinstance(audio, str):
            audio_array, _ = librosa.load(audio, sr=Config.SAMPLING_RATE)
        else:
            audio_array = audio

        audio_tensor = torch.tensor(audio_array).unsqueeze(0)
    
        # Call model with compute_timestamps='w' for word-level
        result = self.model(audio_tensor, language_id, decoding='ctc', compute_timestamps='w')
    
        transcription = result[0]
        word_timestamps = result[1][0]  # [0] because batch size = 1
    
        return transcription, word_timestamps