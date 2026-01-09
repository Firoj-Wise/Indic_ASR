import torch
import numpy as np
import soundfile as sf
import librosa
from typing import Optional, List, Dict
from app.utils.config.config import Config
from app.utils.logger_utils import LOGGER


class SpeakerDiarizer:
    """Speaker diarization using pyannote/speaker-diarization-community-1."""
    
    def __init__(self) -> None:
        self.device = Config.DEVICE_CUDA if torch.cuda.is_available() else Config.DEVICE_CPU
        self.pipeline = None
        
        if not Config.DIARIZATION_ENABLED:
            LOGGER.info("Speaker diarization disabled.")
            return
        
        try:
            from pyannote.audio import Pipeline
            
            # FIX: Monkeypatch torch.load to use weights_only=False
            # pyannote models are from HuggingFace (trusted source)
            _original_torch_load = torch.load
            
            def _patched_torch_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return _original_torch_load(*args, **kwargs)
            
            torch.load = _patched_torch_load
            
            try:
                self.pipeline = Pipeline.from_pretrained(
                    Config.DIARIZATION_MODEL_ID,
                    token=Config.HF_TOKEN  # pyannote v4 API uses 'token'
                )
                if self.device == "cuda":
                    self.pipeline.to(torch.device("cuda"))
                LOGGER.info(f"Diarization pipeline loaded: {Config.DIARIZATION_MODEL_ID}")
            finally:
                # Restore original torch.load
                torch.load = _original_torch_load
                
        except Exception as e:
            LOGGER.error(f"Diarization load failed: {e}")
            self.pipeline = None
    
    def is_available(self) -> bool:
        return self.pipeline is not None
    
    def diarize(
        self, 
        audio_path: str, 
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> List[Dict]:
        """
        Perform speaker diarization on an audio file.
        
        Uses librosa to load the audio (handles webm, wav, mp3, etc.) and passes 
        waveform directly to avoid torchcodec FFmpeg dependency issues on Windows.
        
        Args:
            audio_path: Path to the audio file
            num_speakers: Optional exact number of speakers (overrides min/max)
            min_speakers: Optional minimum number of speakers
            max_speakers: Optional maximum number of speakers
            
        Returns:
            List of dicts with 'speaker', 'start', 'end' keys
        """
        if not self.is_available():
            return []
        
        try:
            # Load audio using librosa (handles many formats including webm)
            # sr=16000 resamples to 16kHz which pyannote expects
            waveform_np, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
            
            # Convert to torch tensor with shape (1, samples) for mono
            waveform = torch.from_numpy(waveform_np).unsqueeze(0).float()
            
            # Pass waveform dict instead of file path
            audio_input = {"waveform": waveform, "sample_rate": sample_rate}
            
            # Build kwargs based on what's provided
            kwargs = {}
            if num_speakers is not None:
                kwargs["num_speakers"] = num_speakers
            else:
                if min_speakers is not None:
                    kwargs["min_speakers"] = min_speakers
                if max_speakers is not None:
                    kwargs["max_speakers"] = max_speakers
            
            LOGGER.info(f"Running diarization with params: {kwargs}")
            output = self.pipeline(audio_input, **kwargs)
            
            # community-1 model uses output.speaker_diarization property
            # Returns (turn, speaker) tuples
            segments = [
                {"speaker": speaker, "start": round(turn.start, 3), "end": round(turn.end, 3)}
                for turn, speaker in output.speaker_diarization
            ]
            LOGGER.info(f"Diarization found {len(segments)} segments")
            return segments
            
        except Exception as e:
            LOGGER.error(f"Diarization error: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())
            return []