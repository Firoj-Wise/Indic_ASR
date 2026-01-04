from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from contextlib import asynccontextmanager
import shutil
import os
from enum import Enum
from typing import Dict, Any

from src.config import Config
from src.logger import LOGGER
from src.model_loader import IndicConformerASR

model_container = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    LOGGER.info("Server starting up...")
    try:
        model_container["asr"] = IndicConformerASR()
    except Exception as e:
        LOGGER.critical(f"Startup failure: {e}")
    
    yield
    
    LOGGER.info("Server shutting down...")
    model_container.clear()

app = FastAPI(
    title=Config.API_TITLE,
    version=Config.API_VERSION,
    description="""
    High-performance, containerized ASR inference engine optimized for Indian languages.
    
    This API exposes a CUDA-accelerated conformer model via REST endpoints, facilitating scalable speech-to-text integration.
    """,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Favicon Fix
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Return 204 No Content to silence browser 404s
    return Response(status_code=204)

# 2. Serve Static UI
app.mount("/static", StaticFiles(directory="src/static"), name="static")

@app.get("/ui", tags=["UI"], include_in_schema=False)
async def read_index():
    return FileResponse('src/static/index.html')

class Language(str, Enum):
    nepali = "ne"
    hindi = "hi"
    maithili = "mai"

@app.post("/transcribe", tags=["ASR"], summary="Transcribe Audio", response_description="Transcription result")
async def transcribe_audio(
    file: UploadFile = File(...), 
    language: Language = Language.hindi
) -> Dict[str, Any]:
    """
    Uploads an audio file and returns the transcription in the selected language.
    """
    asr_model = model_container.get("asr")
    
    if not asr_model:
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
        raise HTTPException(status_code=500, detail="Internal processing error.")
        
    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.get("/", tags=["Health"], summary="Health Check")
def home():
    return {
        "status": "online", 
        "message": "IndicConformer ASR API is ready.",
        "ui_url": "http://localhost:8000/ui",
        "docs_url": "http://localhost:8000/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host=Config.API_HOST, 
        port=Config.API_PORT, 
        reload=True
    )