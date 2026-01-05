import os
from pathlib import Path
from dotenv import load_dotenv

# Load env variables
load_dotenv()

class Config:
    # Base Directory (Project Root)
    # app/utils/config/config.py -> app/utils/config -> app/utils -> app -> root
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    
    # Force Hugging Face to use local models directory
    # This ensures "everything is in the container"
    os.environ["HF_HOME"] = str(BASE_DIR / "models")
    
    LOGS_DIR = BASE_DIR / "logs"
    TEMP_DIR = BASE_DIR / "temp_uploads"

    LOGS_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)

    LOG_FILE_PATH = LOGS_DIR / "app.log"
    
    HF_TOKEN = os.getenv("HF_TOKEN")
    MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
    DEVICE_CUDA = "cuda"
    DEVICE_CPU = "cpu"
    
    SAMPLING_RATE = 16000

    API_TITLE = "Indic ASR API"
    API_VERSION = "1.0.0"
    API_HOST = "0.0.0.0"
    API_PORT = 8000
