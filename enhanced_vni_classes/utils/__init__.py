# enhanced_vni_classes/utils/__init__.py
"""Utility modules for the Enhanced VNI system.
Provides configuration, logging, dependency management, and other utilities."""

import logging
from .vni_config import VNIConfig, ConfigManager, LogLevel
from .logger import get_logger, VNILogger
from .imports import (
    get_dependency_manager,
    check_dependencies,
    import_optional,
    require_feature,
    DependencyManager,
    Dependency
)
# Initialize default logger on import
_default_logger = get_logger(
    name="enhanced_vni_classes",
    level=LogLevel.INFO,  # This now works with the updated get_logger
)

# Version
__version__ = "1.0.0"

__all__ = [
    # Configuration
    "VNIConfig",
    "ConfigManager",
    "LogLevel",
    
    # Logging
    "get_logger",
    "VNILogger",
    
    # Dependency management
    "get_dependency_manager",
    "check_dependencies",
    "import_optional",
    "require_feature",
    "DependencyManager",
    "Dependency",
    
    # Version
    "__version__"
]

# Initialize default logger on import
# This creates a default logger that can be overridden by users
_default_logger = get_logger(
    name="enhanced_vni_classes",
    level=LogLevel.INFO,
    log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def get_version() -> str:
    """Get the version of the enhanced VNI utilities."""
    return __version__

def check_environment() -> dict:
    """
    Check the environment and dependencies for the VNI system.
    
    Returns:
        Dictionary with environment information and dependency status
    """
    import sys
    import platform
    from datetime import datetime
    
    deps = check_dependencies()
    
    return {
        "version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "os": platform.system(),
        "dependencies_available": deps,
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "python_executable": sys.executable,
            "python_path": sys.path[:3],  # First 3 entries
            "cwd": platform.node()
        }
    }

def initialize_default_config(config_dir: str = "vni_config") -> ConfigManager:
    """
    Initialize a default configuration manager.
    
    Args:
        config_dir: Directory to store configuration files
        
    Returns:
        Initialized ConfigManager instance
    """
    return ConfigManager(config_dir=config_dir)

# Export initialization functions
__all__.extend(["get_version", "check_environment", "initialize_default_config"]) 
