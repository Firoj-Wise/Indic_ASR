from fastapi import APIRouter, Response
from fastapi.responses import FileResponse
from app.utils.config.config import Config

router = APIRouter()

@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@router.get("/ui", tags=["UI"], include_in_schema=False)
async def read_index():
    # Assuming execution from project server root
    return FileResponse('app/static/index.html')

@router.get("/", tags=["Health"], summary="Health Check")
def home():
    return {
        "status": "online", 
        "message": "IndicConformer ASR API is ready.",
        "ui": "http://localhost:8000/ui",
        "docs": "http://localhost:8000/docs",
        "info": "http://localhost:8000/info"
    }

@router.get("/info", tags=["Meta"], summary="Service Information")
def info():
    """
    Returns API metadata and usage examples.
    """
    return {
        "id": "indic-conformer-asr",
        "type": "asr-service",
        "name": "Indic Conformer ASR",
        "version": "1.0.0",
        "supported_languages": ["hi", "ne", "mai"],
        "usage": {
            "curl_example": "curl -X POST 'http://localhost:8000/transcribe?language=ne' -F 'file=@/path/to/audio.wav'",
            "websocket_url": "ws://localhost:8000/transcribe/ws"
        }
    }
