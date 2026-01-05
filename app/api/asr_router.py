from fastapi import APIRouter, UploadFile, File, HTTPException
from enum import Enum
from typing import Dict, Any
import shutil
import os
from app.utils.config.config import Config
from app.utils.logger_utils import LOGGER
from app.services.model_registry import model_container

router = APIRouter()

class Language(str, Enum):
    nepali = "ne"
    hindi = "hi"
    maithili = "mai"

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
        LOGGER.error("ASR Model requested but not found in registry.")
        raise HTTPException(status_code=503, detail="ASR Model invalid or not loaded.")

    file_ext = os.path.splitext(file.filename)[1]
    if not file_ext:
        file_ext = ".wav" # Default for blob uploads
        
    temp_path = Config.TEMP_DIR / f"temp_{os.urandom(8).hex()}{file_ext}"

    try:
        LOGGER.info(f"Received upload: {file.filename}")
        
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
        raise HTTPException(status_code=500, detail="Internal processing error.")
        
    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass
