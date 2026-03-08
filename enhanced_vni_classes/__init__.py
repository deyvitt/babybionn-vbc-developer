# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

# enhanced_vni_classes/__init__.py
"""
Enhanced VNI Classes - Advanced Virtual Networked Intelligence System

A comprehensive framework for creating intelligent, specialized Virtual Networked Intelligences (VNIs)
with capabilities including learning, collaboration, domain specialization, and neural networking.

Features:
- Multi-domain VNI specialization (Medical, Legal, Technical, etc.)
- Self-learning systems with experience recording
- Intelligent query routing and classification
- VNI collaboration and neural pathways
- Dynamic VNI creation for any domain
- Professional utilities (config, logging, dependency management)

Version: 2.0.0
"""

from .core.base_vni import EnhancedBaseVNI
from .core.capabilities import VNICapabilities, VNIType
from .core.neural_pathway import NeuralPathway
from .core.collaboration import CollaborationRequest, CollaborationResponse, CollaborationStatus
from .core.registry import VNIRegistry

from .modules.knowledge_base import KnowledgeBase, KnowledgeEntry
from .modules.learning_system import LearningSystem, LearningExperience
# REMOVED: from .modules.generation import EnhancedGenerationModule, GenerationStyle
# REMOVED: from .domains.general import EnhancedGenerationModule, GenerationStyle
from .modules.web_search import WebSearch
from .modules.attention import AttentionMechanism, AttentionType, AttentionWeight
from .modules.classifier import (
    DomainClassifier,
    EnhancedDomainClassifier,
    DynamicDomainClassifier,
    ClassificationResult,
    Domain
)

from .managers.vni_manager import VNIManager
from .managers.session_manager import SessionManager, Session

from .domains.medical import MedicalVNI
from .domains.legal import LegalVNI
from .domains.general import GeneralVNI  # ← Only GeneralVNI, not EnhancedGenerationModule
from .domains.dynamic_vni import DynamicVNI
from .domains.technical import (
    TechnicalVNI,
    TechnicalOperActionConfig,
    TechnicalKnowledgeGraph,
    TechnicalReasoningEngine
)

from .utils.vni_config import VNIConfig, ConfigManager, LogLevel
from .utils.logger import get_logger, VNILogger
from .utils.imports import (
    get_dependency_manager,
    check_dependencies,
    import_optional,
    require_feature,
    DependencyManager,
    Dependency
)

# Version information
__version__ = "2.0.0"
__author__ = "VNI Development Team"
__description__ = "Enhanced Virtual Networked Intelligence System with advanced capabilities"

# Initialize dependency manager
_dep_manager = get_dependency_manager()

# Backward compatibility aliases
EnhancedMedicalVNI = MedicalVNI
EnhancedLegalVNI = LegalVNI
EnhancedGeneralVNI = GeneralVNI

# Export main classes
__all__ = [
    # === CORE CLASSES ===
    "EnhancedBaseVNI",
    "VNICapabilities",
    "VNIType",
    "NeuralPathway",
    "CollaborationRequest",
    "CollaborationResponse",
    "CollaborationStatus",
    "VNIRegistry",
    
    # === MODULE CLASSES ===
    "KnowledgeBase",
    "KnowledgeEntry",
    "LearningSystem",
    "LearningExperience",
    # REMOVED: "EnhancedGenerationModule",
    # REMOVED: "GenerationStyle",
    "WebSearch",
    "AttentionMechanism",
    "AttentionType",
    "AttentionWeight",
    "DomainClassifier",
    "EnhancedDomainClassifier",
    "DynamicDomainClassifier",
    "ClassificationResult",
    "Domain",
    
    # === MANAGER CLASSES ===
    "VNIManager",
    "SessionManager",
    "Session",
    
    # === DOMAIN CLASSES ===
    "MedicalVNI",
    "LegalVNI",
    "GeneralVNI",
    "DynamicVNI",
    "TechnicalVNI",
    "TechnicalOperActionConfig",
    "TechnicalKnowledgeGraph",
    "TechnicalReasoningEngine",
    
    # === BACKWARD COMPATIBILITY ALIASES ===
    "EnhancedMedicalVNI",
    "EnhancedLegalVNI",
    "EnhancedGeneralVNI",
    
    # === UTILITY CLASSES ===
    "VNIConfig",
    "ConfigManager",
    "LogLevel",
    "VNILogger",
    "DependencyManager",
    "Dependency",
    
    # === FUNCTIONS ===
    "get_logger",
    "get_dependency_manager",
    "check_dependencies",
    "import_optional",
    "require_feature",
]

# === HELPER FUNCTIONS ===
def get_version() -> str:
    """Get the current version of the enhanced VNI system."""
    return __version__

def check_system_health() -> dict:
    """
    Check the health of the VNI system.
    
    Returns:
        Dictionary with system health information including:
        - version information
        - Python environment details
        - dependency status
        - module availability
        - optional feature status
    """
    import sys
    import platform
    from datetime import datetime
    
    deps = check_dependencies()
    
    health_info = {
        "version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "os": platform.system(),
        "dependencies": deps,
        "timestamp": datetime.now().isoformat(),
        "core_modules_available": all([
            "EnhancedBaseVNI" in globals(),
            "VNIManager" in globals(),
            "DomainClassifier" in globals(),
            "KnowledgeBase" in globals(),
            "LearningSystem" in globals()
        ]),
        "module_status": {
            "core": "EnhancedBaseVNI" in globals(),
            "managers": "VNIManager" in globals() and "SessionManager" in globals(),
            "modules": all(m in globals() for m in ["KnowledgeBase", "LearningSystem"]),  # REMOVED EnhancedGenerationModule
            "domains": all(d in globals() for d in ["MedicalVNI", "LegalVNI", "GeneralVNI"]),
            "utils": all(u in globals() for u in ["VNIConfig", "get_logger", "get_dependency_manager"]),
        },
        "optional_features": {
            "web_search": require_feature("full_web_search"),
            "advanced_attention": require_feature("advanced_attention"),
            "yaml_config": require_feature("yaml_config"),
            "learning_persistence": require_feature("learning_persistence"),
        }
    }
    
    return health_info

def initialize_system(config_dir: str = "vni_config", log_level: str = "INFO") -> tuple:
    """
    Initialize the VNI system with default configuration.
    
    Args:
        config_dir: Directory for configuration files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Tuple of (config_manager, logger, dependency_manager)
    """
    # Setup logging
    logger = get_logger(
        name="enhanced_vni_system",
        level=getattr(LogLevel, log_level.upper(), LogLevel.INFO),
        log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize config manager
    config_manager = ConfigManager(config_dir=config_dir)
    
    # Check dependencies
    dep_manager = get_dependency_manager()
    deps_status = dep_manager.check_feature_availability("full_web_search")
    
    if not deps_status["available"] and deps_status["missing_dependencies"]:
        logger.warning(f"Missing optional dependencies: {deps_status['missing_dependencies']}")
    
    logger.info(f"Enhanced VNI System v{__version__} initialized successfully")
    logger.info(f"Configuration directory: {config_dir}")
    logger.info(f"Log level: {log_level}")
    
    return config_manager, logger, dep_manager

# === BACKWARD COMPATIBILITY FUNCTIONS ===
def get_config(config_name: str = "default") -> VNIConfig:
    """
    Backward compatibility function for get_config.
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        VNIConfig object
    """
    manager = ConfigManager()
    return manager.load_config(f"vni_{config_name}")

# Add helper functions to exports
__all__.extend([
    "get_version", 
    "check_system_health", 
    "initialize_system",
    "get_config"
])

# === INITIALIZATION ===
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Log system initialization
_logger = logging.getLogger(__name__)
_logger.debug(f"Enhanced VNI Classes v{__version__} imported successfully")

# Check for critical missing dependencies on import
try:
    _deps = check_dependencies()
    missing_critical = [dep for dep, available in _deps.items() 
                       if not available and dep in ["json", "typing", "dataclasses", "datetime", "enum"]]
    if missing_critical:
        _logger.error(f"Critical dependencies missing: {missing_critical}")
        raise ImportError(f"Critical dependencies missing: {missing_critical}")
except Exception as e:
    _logger.warning(f"Could not check dependencies on import: {e}")
