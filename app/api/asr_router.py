from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict, Any
import shutil
import os
import numpy as np
import torch
from app.utils.config.config import Config
from app.utils.logger_utils import LOGGER
from app.services.model_registry import model_container
from app.constants.enums import Language
from app.constants import log_msg

router = APIRouter()

@router.post("/transcribe", tags=["ASR"], summary="Transcribe Audio", response_description="Transcription result")
async def transcribe_audio(
    file: UploadFile = File(...), 
    language: Language = Language.hindi
) -> Dict[str, Any]:
    """
    Uploads an audio file and returns the transcription in the selected language.
    """
    asr_model = model_container.get("asr")
    
    if not asr_model:
        LOGGER.error(log_msg.ERR_ASR_MODEL_NOT_FOUND)
        raise HTTPException(status_code=503, detail="ASR Model invalid or not loaded.")

    file_ext = os.path.splitext(file.filename)[1]
    if not file_ext:
        file_ext = ".wav" # Default for blob uploads
        
    temp_path = Config.TEMP_DIR / f"temp_{os.urandom(8).hex()}{file_ext}"

    try:
        LOGGER.info(log_msg.ASR_PROCESSING_START.format(file.filename, language.value))
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        transcription = asr_model.transcribe(str(temp_path), language_id=language.value)
        
        return {
            "filename": file.filename, 
            "transcription": transcription, 
            "language": language.value
        }

    except Exception as e:
        LOGGER.error(f"Processing error: {e}")
        # Ensure we don't return empty, but raise HTTP exception
        raise HTTPException(status_code=500, detail=log_msg.ERR_INTERNAL_PROCESSING)
        
    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass


@router.websocket("/transcribe/ws")
async def transcribe_audio_stream(websocket: WebSocket, language: Language = Language.hindi):
    """
    Streaming ASR WebSocket Endpoint.
    
    Expects raw PCM16 audio bytes (16000Hz, Mono, 16-bit little-endian).
    Accumulates chunks and runs inference periodically.
    """
    await websocket.accept()
    
    asr_model = model_container.get("asr")
    if not asr_model:
        await websocket.close(code=1011, reason="ASR Model not loaded")
        return

    # Buffer for accumulating audio chunks
    # We'll use a simple list of bytes and concat, or pre-allocate bytearray
    audio_buffer = bytearray()
    
    # 16kHz * 2 bytes/sample * 2 seconds = 64000 bytes
    BUFFER_THRESHOLD = 64000 
    
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)
            
            if len(audio_buffer) >= BUFFER_THRESHOLD:
                # Process the buffered audio
                # Convert bytes to numpy float32 array
                # raw bytes -> int16 -> float32
                audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Check for silence or very short length?
                # For now, just transcribe everything
                
                tensor = torch.from_numpy(audio_np).unsqueeze(0) # (1, T)
                
                try:
                    # Run inference
                    # Note: This runs on the main thread loop if not careful. 
                    # Ideally, should run in executor if it blocks too long.
                    # But Conformer is fast enough on GPU for short chunks usually.
                    transcription = asr_model.transcribe_tensor(tensor, language_id=language)
                    
                    if transcription:
                        await websocket.send_json({
                            "type": "transcription",
                            "text": transcription,
                            "language": language
                        })
                        
                except Exception as e:
                    LOGGER.error(f"Streaming inference error: {e}")
                    await websocket.send_json({"type": "error", "message": str(e)})
                
                # Clear buffer after processing
                # Note: In a real streaming scenario ("VAD"), we might keep some overlap 
                # or wait for silence. Here we simple-drain.
                audio_buffer = bytearray()
                
    except WebSocketDisconnect:
        LOGGER.info("WebSocket disconnected")
    except Exception as e:
        LOGGER.error(f"WebSocket connection error: {e}")
        # Try to close if still open
        try:
            await websocket.close()
        except:
            pass
