# Log Messages
SERVER_STARTUP = "Server starting up..."
SERVER_SHUTDOWN = "Server shutting down..."
STARTUP_FAILURE = "Startup failure: {}"

# ASR
ASR_MODEL_INIT = "Initializing model: {} on device: {}"
ASR_MODEL_LOAD_START = "Loading model (this might trigger internal download)..."
ASR_MODEL_LOAD_SUCCESS = "Model loaded successfully."
ASR_MODEL_LOAD_FAIL = "Failed to load model: {}"
ASR_PROCESSING_START = "Processing {} [Lang: {}]"
ASR_INFERENCE_FAIL = "Inference failed for {}: {}"

# API Errors
ERR_ASR_MODEL_NOT_FOUND = "ASR Model requested but not found in registry."
ERR_INTERNAL_PROCESSING = "Internal processing error."

# Metrics
WARN_EMPTY_TEXT_NORM = "Normalization received empty text or None. Returning empty string."
