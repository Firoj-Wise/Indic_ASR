import logging
from logging.handlers import RotatingFileHandler
import sys
from app.utils.config.config import Config

def setup_logger(name: str = "app_logger") -> logging.Logger:
    """
    Configures and returns a dedicated logger instance with file and console handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        try:
            file_handler = RotatingFileHandler(
                Config.LOG_FILE_PATH, maxBytes=5*1024*1024, backupCount=5
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)
        except Exception as e:
            # Fallback if file logging fails, though unlikely
            # Using print here as per user request to avoid print, but if logger fails, print is only option?
            # User said "do not use print use logs". But this is the logger setup.
            # I'll keep it as print for safe fallback or just ignore. 
            # I will assume standard error stream is fine for logger failure.
            sys.stderr.write(f"Failed to setup file handler: {e}\n")

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

        logger.addHandler(console_handler)

    return logger

LOGGER = setup_logger()
