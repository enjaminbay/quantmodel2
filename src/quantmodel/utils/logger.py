# utils/logger.py

import logging
import sys
from logging.handlers import RotatingFileHandler
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantmodel.utils.config import Config

def setup_logger(name: str, config=None) -> logging.Logger:
    """Set up logger with both file and console handlers."""
    logger = logging.getLogger(name)

    # Default configuration if no config provided
    log_level = 'INFO'
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file = 'trading_system.log'
    console_output = True

    # Use config if provided
    if config:
        log_level = config.LOG.level
        log_format = config.LOG.format
        log_file = config.LOG.file_name
        console_output = config.LOG.console_output

        # Ensure log directory exists
        log_dir = config.PATHS.log_directory
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, log_file)

    logger.setLevel(logging.getLevelName(log_level))
    formatter = logging.Formatter(log_format)

    # File handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

# Create the default logger
logger = setup_logger('trading_system')

def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return logging.getLogger(f'trading_system.{name}')