import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path


# Constants for log configuration
LOG_DIR_NAME = 'logs'
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3  # Number of backup log files to keep


# Determine project root reliably (repo_root/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
log_dir_path = PROJECT_ROOT / LOG_DIR_NAME
log_dir_path.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir_path / LOG_FILE


def configure_logger():
    """Configures logging with a rotating file handler and a console handler."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    file_handler = RotatingFileHandler(str(log_file_path), maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Prevent adding duplicate handlers if module is reloaded
    has_file = any(isinstance(h, RotatingFileHandler) and getattr(h, 'baseFilename', '') == str(log_file_path) for h in logger.handlers)
    has_console = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    if not has_file:
        logger.addHandler(file_handler)
    if not has_console:
        logger.addHandler(console_handler)


# Configure the logger on import
configure_logger()

