from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict, Any, List
import shutil
import os
import numpy as np
import torch
import json
from app.utils.config.config import Config
from app.utils.logger_utils import LOGGER
from app.services.model_registry import model_container
from app.constants.enums import Language
from app.constants import log_msg

router = APIRouter()

# --- Global State & Connection Manager ---

class GlobalAppState:
    def __init__(self):
        self.current_language: Language = Language.hindi

app_state = GlobalAppState()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        # Convert dict to JSON string once
        payload = json.dumps(message)
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except Exception:
                # If sending fails, we might want to remove the dead connection, 
                # but disconnect handle usually catches it.
                pass

manager = ConnectionManager()


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
        raise HTTPException(status_code=500, detail=log_msg.ERR_INTERNAL_PROCESSING)
        
    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass


@router.websocket("/transcribe/ws")
async def transcribe_audio_stream(websocket: WebSocket, language: Language = None) -> None:
    """
    Streaming ASR WebSocket Endpoint.
    
    Handles:
        - Binary Audio Data: For transcription.
        - JSON Config Data: For changing global language.
        - Broadcasting: Sends transcripts to ALL connected clients.

    Args:
        websocket (WebSocket): The active WebSocket connection.
        language (Language, optional): Initial language hint. Defaults to None.
    """
    try:
        await manager.connect(websocket)
        logger_context = f"Client={websocket.client.host}:{websocket.client.port}"
        LOGGER.info(log_msg.LOG_ENTRY.format("transcribe_audio_stream", logger_context))
        
        current_lang = app_state.current_language
        await websocket.send_json({"type": "config", "language": current_lang.value})
        
        asr_model = model_container.get("asr")
        if not asr_model:
            LOGGER.error(log_msg.ERR_ASR_MODEL_NOT_FOUND)
            await websocket.close(code=1011, reason="ASR Model not loaded")
            return

        # Buffer for accumulating audio chunks
        audio_buffer = bytearray()
        BUFFER_THRESHOLD = 64000 
        
        while True:
            try:
                message = await websocket.receive()
                
                if message["type"] == "websocket.receive":
                    if "bytes" in message and message["bytes"]:
                        data = message["bytes"]
                        audio_buffer.extend(data)
                        
                        if len(audio_buffer) >= BUFFER_THRESHOLD:
                            # Process buffer: int16 -> float32
                            audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                            tensor = torch.from_numpy(audio_np).unsqueeze(0)
                            
                            try:
                                transcription = asr_model.transcribe_tensor(tensor, language_id=app_state.current_language.value)
                                
                                if transcription:
                                    await manager.broadcast({
                                        "type": "transcription",
                                        "text": transcription,
                                        "language": app_state.current_language.value,
                                        "source": "stream" 
                                    })
                                    
                            except Exception as e:
                                LOGGER.error(log_msg.ASR_INFERENCE_FAIL.format("stream", e))
                                await websocket.send_json({"type": "error", "message": f"Inference failed: {str(e)}"})
                            
                            audio_buffer = bytearray()

                    if "text" in message and message["text"]:
                        try:
                            data = json.loads(message["text"])
                            if data.get("type") == "config":
                                new_lang = data.get("language")
                                if new_lang in [l.value for l in Language]:
                                    LOGGER.info(log_msg.CONFIG_LANG_UPDATE.format(new_lang))
                                    app_state.current_language = Language(new_lang)
                                    await manager.broadcast({
                                        "type": "config",
                                        "language": new_lang
                                    })
                        except json.JSONDecodeError:
                            LOGGER.warning(log_msg.CONFIG_INVALID_JSON)
                        except Exception as e:
                            LOGGER.error(log_msg.CONFIG_PROCESS_ERROR.format(e))
                            await websocket.send_json({"type": "error", "message": "Invalid config data"})

            except WebSocketDisconnect:
                LOGGER.info(log_msg.WS_DISCONNECTED)
                manager.disconnect(websocket)
                break
            except Exception as e:
                LOGGER.error(log_msg.WS_RECEIVE_ERROR.format(e))
                await websocket.send_json({"type": "error", "message": "Internal processing error"})
                break

    except Exception as e:
        LOGGER.error(log_msg.LOG_ERROR.format("transcribe_audio_stream", str(e)), exc_info=True)
        manager.disconnect(websocket)
        raise e
    finally:
        LOGGER.info(log_msg.LOG_EXIT.format("transcribe_audio_stream"))
    return None
