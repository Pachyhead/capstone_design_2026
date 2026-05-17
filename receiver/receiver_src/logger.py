import logging
from logging import Logger
from pathlib import Path

from config import PROJECT_ROOT

# Define log directory and file path
LOG_DIR = Path(PROJECT_ROOT) / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_logger(name: str = "") -> Logger:
    """
    Configures and returns a logger instance.
    Prevents duplicate handlers if called multiple times.
    """
    if name == "":
        raise ValueError("logger name is empty")
    logger = logging.getLogger(name)

    LOG_FILE = LOG_DIR / f"{name}__processing.log"
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # 1. File Handler (Saves logs to the file)
        file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 2. Console Handler (Prints logs to terminal)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
