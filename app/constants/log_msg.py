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

# Generic Entry/Exit
LOG_ENTRY = "Entry: {} | Args: {}"
LOG_EXIT = "Exit: {} | Success"
LOG_ERROR = "Error: {} | Message: {}"

# WebSocket / Pipecat
WS_CONNECTING = "Connecting to ASR WS: {}"
WS_CONNECTED = "WebSocket Connected"
WS_DISCONNECTED = "WebSocket Disconnected"
WS_CLOSED_CLEANLY = "WebSocket closed cleanly"
WS_CONNECTION_CLOSED = "ASR WS Connection closed by server"
WS_RECEIVE_ERROR = "ASR WS Receive error: {}"
WS_SEND_ERROR = "Failed to send audio frame: {}"
WS_MSG_PARSE_ERROR = "Error parsing WS message: {}"

# Tasks
TASK_CANCELLED = "Task cancelled cleanly: {}"
TASK_CANCEL_ERROR = "Error cancelling task: {}"

# Config
CONFIG_LANG_UPDATE = "Global language updated to: {}"
CONFIG_INVALID_JSON = "Received invalid JSON config"
CONFIG_PROCESS_ERROR = "Config processing error: {}"
