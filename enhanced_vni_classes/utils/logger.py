# enhanced_vni_classes/utils/logger.py
"""
Logging utilities
"""
import logging
import sys
from typing import Optional
from .vni_config import LogLevel

def get_logger(
    name: str, 
    level = logging.INFO,  # Accepts int, string, or LogLevel
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Get a configured logger
    """
    # Handle LogLevel enum
    if isinstance(level, LogLevel):
        level = level.to_logging_level()
    # Handle string levels
    elif isinstance(level, str):
        level = getattr(logging, level.upper())
    # level is already int if it's logging.INFO, logging.DEBUG, etc.
        
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Use custom format if provided, else default
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler
    logger.addHandler(console_handler)
    
    # Don't propagate to root logger
    logger.propagate = False
    
    return logger

class VNILogger:
    """Enhanced logger for VNIs with domain context"""
    
    def __init__(self, vni_id: str, domain: str):
        self.vni_id = vni_id
        self.domain = domain
        self.logger = get_logger(f"VNI.{domain}.{vni_id}")
    
    def info(self, message: str, extra: Optional[dict] = None):
        """Log info message"""
        full_message = f"[{self.vni_id}] {message}"
        self.logger.info(full_message, extra=extra or {})
    
    def debug(self, message: str, extra: Optional[dict] = None):
        """Log debug message"""
        full_message = f"[{self.vni_id}] {message}"
        self.logger.debug(full_message, extra=extra or {})
    
    def warning(self, message: str, extra: Optional[dict] = None):
        """Log warning message"""
        full_message = f"[{self.vni_id}] {message}"
        self.logger.warning(full_message, extra=extra or {})
    
    def error(self, message: str, extra: Optional[dict] = None):
        """Log error message"""
        full_message = f"[{self.vni_id}] {message}"
        self.logger.error(full_message, extra=extra or {})
    
    def exception(self, message: str, exc_info=True):
        """Log exception"""
        full_message = f"[{self.vni_id}] {message}"
        self.logger.exception(full_message, exc_info=exc_info) 
