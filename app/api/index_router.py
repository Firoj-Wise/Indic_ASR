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
        "ui_url": f"http://localhost:{Config.API_PORT}/ui", # Localhost hardcoded for browser access usually, or use host
        "docs_url": f"http://localhost:{Config.API_PORT}/docs"
    }
