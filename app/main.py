from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from app.utils.config.config import Config
from app.utils.logger_utils import LOGGER
from app.services.load_model import IndicConformerASR
from app.services.model_registry import model_container
from app.api import index_router, asr_router
from app.constants import log_msg

@asynccontextmanager
async def lifespan(app: FastAPI):
    LOGGER.info(log_msg.SERVER_STARTUP)
    try:
        model_container["asr"] = IndicConformerASR()
    except Exception as e:
        LOGGER.critical(log_msg.STARTUP_FAILURE.format(e))
        raise e
    
    yield
    
    LOGGER.info(log_msg.SERVER_SHUTDOWN)
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
