"""
Logging utilities for APEGA.
Contains functions for setting up and configuring logging.
"""

import os
import sys
from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    rotation: str = "10 MB",
    retention: str = "1 week"
) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (defaults to logs/apega_{time}.log)
        rotation: Log rotation policy
        retention: Log retention policy
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(sys.stderr, level=log_level)
    
    # Add file handler if log_file is provided or use default
    if log_file is None:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        log_file = "logs/apega_{time}.log"
    
    logger.add(
        log_file,
        level=log_level,
        rotation=rotation,
        retention=retention,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message} | {extra}",
        backtrace=True,
        diagnose=True
    )
    
    logger.info(f"Logging initialized with level {log_level}")


def get_logger():
    """
    Get the logger instance.
    
    Returns:
        Logger instance
    """
    return logger