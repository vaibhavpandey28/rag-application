import logging
import os
import sys
from dotenv import load_dotenv

load_dotenv() 

_CONFIGURED = False


def _is_logger_enabled() -> bool:
    value = os.getenv("LOGGER", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _is_color_enabled() -> bool:
    value = os.getenv("LOGGER_COLOR", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


class ColorFormatter(logging.Formatter):
    RESET = "\033[0m"
    COLORS = {
        "DEBUG": "\033[96m",     # Neon Cyan
        "INFO": "\033[95m",     # Bright Mint Green
        "WARNING": "\033[93m",   # Pastel Yellow
        "ERROR": "\033[94m",     # Electric Blue 🌟 (Replaces Red)
        "CRITICAL": "\033[35m",   # Bright Pink
    }

    def format(self, record: logging.LogRecord) -> str:
        # Save original states to prevent breaking other handlers
        original_levelname = record.levelname
        original_name = record.name
        
        # 1. Convert dot notation folder structure into forward slash format
        record.name = original_name.replace(".", "/")
        
        # 2. Inject terminal colors if matching level exists
        color = self.COLORS.get(original_levelname, "")
        if color:
            record.levelname = f"{color}{original_levelname}{self.RESET}"
            
        message = super().format(record)
        
        # Restore original records
        record.levelname = original_levelname
        record.name = original_name
        return message


class PlainFormatter(logging.Formatter):
    """Custom formatter to convert dot notation to slashes for non-color outputs."""
    def format(self, record: logging.LogRecord) -> str:
        original_name = record.name
        record.name = original_name.replace(".", "/")
        message = super().format(record)
        record.name = original_name
        return message


def _configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    if not _is_logger_enabled():
        logging.disable(logging.CRITICAL)
        _CONFIGURED = True
        return

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Changed time format to standard YYYY-MM-DD HH:MM:SS
    log_format = "%(levelname)s:     %(asctime)s - %(name)s  ----> %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Setup both custom formatters with specific timeframe configurations
    if _is_color_enabled():
        formatter = ColorFormatter(fmt=log_format, datefmt=date_format)
    else:
        formatter = PlainFormatter(fmt=log_format, datefmt=date_format)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)
    root.addHandler(handler)

    # Reduce noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_logging()
    return logging.getLogger(name)