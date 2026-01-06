from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any
import shutil
import os
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

    try:
        # Read file into memory
        audio_bytes = await file.read()
        import io
        audio_file = io.BytesIO(audio_bytes)
            
        # Transcribe directly from memory
        transcription = asr_model.transcribe(audio_file, language_id=language.value)
        
        return {
            "filename": "stream", 
            "transcription": transcription, 
            "language": language.value
        }

    except Exception as e:
        LOGGER.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=log_msg.ERR_INTERNAL_PROCESSING)
