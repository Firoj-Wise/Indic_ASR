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
        self.clock = 0.0 # Time tracking
        
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
        audio_path: str = None, 
        waveform: Optional[torch.Tensor] = None,
        sample_rate: Optional[int] = 16000,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> List[Dict]:
        """
        Perform speaker diarization on an audio file or waveform.
        
        Args:
            audio_path: Path to the audio file (optional if waveform provided)
            waveform: torch.Tensor of shape (channel, time) (optional)
            sample_rate: Sampling rate of the waveform (default 16000)
            ...
        """
        if not self.is_available():
            return []
        
        try:
            if waveform is not None:
                # Use provided waveform
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                audio_input = {"waveform": waveform, "sample_rate": sample_rate}
            elif audio_path:
                # Load audio using librosa
                waveform_np, sr = librosa.load(audio_path, sr=16000, mono=True)
                waveform_t = torch.from_numpy(waveform_np).unsqueeze(0).float()
                audio_input = {"waveform": waveform_t, "sample_rate": sr}
            else:
                 raise ValueError("Either audio_path or waveform must be provided.")
            
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

class StreamingDiarizer:
    """
    Streaming Diarizer using diart for real-time speaker tracking.
    Maintains state across calls for a single session.
    """
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.pipeline = None
        self.last_speaker = None
        self.clock = 0.0 # Time tracking for SlidingWindow
        # Buffer to hold valid audio for diart pipeline (5 seconds usually)
        self.audio_buffer = None 
        
        # Check if diarization is enabled in config
        if not Config.DIARIZATION_ENABLED:
            return

        try:
            from diart import SpeakerDiarization, SpeakerDiarizationConfig
            import diart.models as m
            from pyannote.audio import Model
            from pyannote.core import SlidingWindow, SlidingWindowFeature
            
            # Using standard models (pyannote/segmentation-3.1 & pyannote/embedding)
            # which diart uses by default.
            _original_torch_load = torch.load
            def _patched_torch_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return _original_torch_load(*args, **kwargs)
            torch.load = _patched_torch_load
            
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Custom loaders to bypass diart's use_auth_token issue
                def load_segmentation():
                    model = Model.from_pretrained("pyannote/segmentation", token=Config.HF_TOKEN)
                    # Handle powerset if needed (replicating diart logic)
                    specs = getattr(model, "specifications", None)
                    if specs is not None and getattr(specs, "powerset", False):
                        model = m.PowersetAdapter(model)
                    return model.to(device)

                def load_embedding():
                    model = Model.from_pretrained("pyannote/embedding", token=Config.HF_TOKEN)
                    return model.to(device)

                # Explicitly load models with wrappers
                segmentation = m.SegmentationModel(load_segmentation)
                embedding = m.EmbeddingModel(load_embedding)
                
                # Configure Diart Pipeline
                # step=1.0 match the ASR buffer size for consistency (1s updates)
                # max_speakers=10 as requested
                config = SpeakerDiarizationConfig(
                    segmentation=segmentation,
                    embedding=embedding,
                    device=device,
                    step=1.0, 
                    max_speakers=10,
                    sample_rate=Config.SAMPLING_RATE
                    # tau_active=0.6, # VAD threshold (default 0.6)
                    # rho_update=0.3, # Centroid update rate (default 0.3)
                )
                self.pipeline = SpeakerDiarization(config)
                LOGGER.info("Streaming Diarizer initialized (diart).")
            finally:
                torch.load = _original_torch_load
            
        except Exception as e:
            LOGGER.error(f"Streaming Diarizer init failed: {e}")
            
    def process(self, waveform: torch.Tensor) -> str:
        """
        Process a chunk of audio and return the dominant speaker.
        
        Args:
            waveform: torch.Tensor of shape (1, samples)
            
        Returns:
             str: Speaker ID (e.g., "speaker0", "speaker1") or None
        """
        if self.pipeline is None:
            return None
            
        try:
            from pyannote.core import SlidingWindow, SlidingWindowFeature

            # FORCE TYPE CONVERSION AND DEBUGGING
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.detach().cpu().numpy()
            elif isinstance(waveform, memoryview):
                waveform = np.array(waveform)
            elif not isinstance(waveform, np.ndarray):
                # Try to convert whatever it is (list, etc)
                waveform = np.array(waveform)
            
            # Additional safety: Ensure float32. diart/pyannote models are float32.
            if waveform.dtype != np.float32:
                waveform = waveform.astype(np.float32)

            # diart expects (channels, time) - Ensure 2D
            if waveform.ndim == 1:
                waveform = waveform[np.newaxis, :]

            # Get config parameters
            expected_samples = int(self.pipeline.config.duration * self.pipeline.config.sample_rate)
            
            # Initialize buffer if needed
            if self.audio_buffer is None:
                # Initialize with zeros (silence)
                self.audio_buffer = np.zeros((1, expected_samples), dtype=np.float32)
                
            # Update Rolling Buffer
            # Shift buffer left by incoming chunk size
            new_samples = waveform.shape[1]
            
            if new_samples >= expected_samples:
                 # If new chunk is bigger than buffer, just take the last part
                 self.audio_buffer = waveform[:, -expected_samples:]
            else:
                 # Shift left
                 self.audio_buffer = np.roll(self.audio_buffer, -new_samples, axis=1)
                 # Update end
                 self.audio_buffer[:, -new_samples:] = waveform

            # Wrap in SlidingWindowFeature for diart
            # pyannote.core expects (samples, channels) usually, so we transpose
            data = self.audio_buffer.T # Shape (samples, channels)
            
            # Correct clock calculation
            # Advance clock by the step size (which matches our 1.0s buffer ideally)
            # If input is slightly different, using new_samples is safer for drift
            current_duration_seconds = new_samples / Config.SAMPLING_RATE
            
            # Ensure we step coherently. If diart configured step=1.0, we should ideally step 1.0
            
            resolution = SlidingWindow(start=self.clock, duration=1.0/Config.SAMPLING_RATE, step=1.0/Config.SAMPLING_RATE)
            chunk = SlidingWindowFeature(data, resolution)

            self.clock += current_duration_seconds
            
            # Pass to pipeline
            output = self.pipeline([chunk])
            
            # Output is (Annotation, waveform)
            annotation = output[0][0] # First item in sequence
            
            # We only care about the speaker in the *new* segment (the last part of the window)
            # The annotation covers [start, start+5s]. 
            # The new info is at [end-new_duration, end].
            
            window_end = chunk.extent.end
            new_segment_start = window_end - current_duration_seconds
            
            # FOCUS: Add a small margin to avoid boundary artifacts?
            # Or just take the whole new segment.
            from pyannote.core import Segment
            focus_segment = Segment(new_segment_start, window_end)
            
            # Crop annotation to focus segment
            cropped_annotation = annotation.crop(focus_segment)
            
            # Extract speakers from cropped annotation
            # Filter out very short segments?
            speakers = []
            for segment, track, label in cropped_annotation.itertracks(yield_label=True):
                 # Only count if meaningful duration?
                 if segment.duration > 0.05: # 50ms
                     speakers.append(label)
            
            # LOGGER.info(f"Stream Diarizer: found speakers {speakers} in segment {focus_segment}") # Debug log
            
            if not speakers:
                # Fallback
                if hasattr(self, 'last_speaker') and self.last_speaker:
                     return self.last_speaker
                return "Identifying..." # Better than Unknown for UI
                
            # Return the most frequent speaker in this chunk
            from collections import Counter
            most_common = Counter(speakers).most_common(1)
            if most_common:
                speaker_label = most_common[0][0]
                # Format: speakerN -> Speaker N (or just keep ID)
                # Diart usually returns "speaker0", "speaker1"
                # Let's standardize
                if "speaker" in speaker_label:
                    speaker = speaker_label.replace("speaker", "Speaker ")
                else:
                    speaker = f"Speaker {speaker_label}"
                    
                self.last_speaker = speaker
                return speaker
            
            if hasattr(self, 'last_speaker') and self.last_speaker:
                return self.last_speaker
            return "Identifying..."

        except Exception as e:
            import traceback
            LOGGER.error(f"Streaming diarization error details: {traceback.format_exc()}")
            LOGGER.error(f"Streaming diarization error: {e}")
            # Ensure we don't crash the stream
            return None