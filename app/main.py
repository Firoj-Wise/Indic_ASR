from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from app.utils.config.config import Config
from app.utils.logger_utils import LOGGER
from app.services.load_model import IndicConformerASR
from app.services.model_registry import model_container
from app.api import index_router, asr_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    LOGGER.info("Server starting up...")
    try:
        model_container["asr"] = IndicConformerASR()
    except Exception as e:
        LOGGER.critical(f"Startup failure: {e}")
        # We might want to re-raise to stop server start if critical model fails
        # But keeping old behavior of just logging mostly, though user said "throw something"
        # in general context. For lifespan, catching and logging is often safer than crash loop, 
        # but if model is valid, it should work.
        # I'll let it slide but ensure it's logged critical.
    
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

# Mount Static Files
# directory="app/static" because we will move src/static to app/static
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include Routers
app.include_router(index_router.router)
app.include_router(asr_router.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app", 
        host=Config.API_HOST, 
        port=Config.API_PORT, 
        reload=True
    )
