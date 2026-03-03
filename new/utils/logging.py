"""
Logging Configuration
"""
import logging
import sys
from .orchestrator_config import Config

def setup_logging():
    """Configure logging for the application"""
    
    # Create formatter
    formatter = logging.Formatter(Config.LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(Config.LOG_LEVEL)
    
    # File handler (optional)
    file_handler = logging.FileHandler("babybionn.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    
    # Create babybionn logger
    logger = get_logger("BabyBIONN")
    logger.info("Logging configured successfully")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Add console handler if not already configured
        formatter = logging.Formatter(Config.LOG_FORMAT)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(Config.LOG_LEVEL)
        logger.addHandler(console_handler)
        logger.setLevel(Config.LOG_LEVEL)
        logger.propagate = False
    
    return logger

# Create a default logger
logger = get_logger(__name__) 
