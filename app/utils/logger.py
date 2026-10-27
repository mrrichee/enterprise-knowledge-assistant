"""
app/utils/logger.py
====================
Structured Logging Setup

CONCEPT: Why structured logging over print()?
- print() output disappears after your terminal closes
- Logging writes to files, has levels (DEBUG/INFO/WARNING/ERROR),
  timestamps, and source location — essential for debugging production issues
- When a recruiter asks "how do you debug in production?", logging is the answer

USAGE:
    from app.utils.logger import get_logger
    logger = get_logger(__name__)   # __name__ = "app.services.rag_pipeline"

    logger.info("Processing document", extra={"filename": "report.pdf"})
    logger.error("Embedding failed", exc_info=True)  # includes stack trace
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for the given module name.

    RotatingFileHandler keeps logs from growing forever:
    - maxBytes=5MB per file
    - backupCount=3 means keeps eka.log, eka.log.1, eka.log.2
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured — avoid duplicate handlers

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler — INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler — DEBUG and above (catches everything)
    file_handler = RotatingFileHandler(
        LOGS_DIR / "eka.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
