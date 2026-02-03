# enhanced_vni_classes/compatibility.py
"""
Compatibility layer for backward compatibility with old imports
"""

import warnings
warnings.warn(
    "Direct imports from enhanced_vni_classes are deprecated. "
    "Please update to use the new modular structure.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export key classes from new structure
from ..core.base_vni import EnhancedBaseVNI
from ..core.capabilities import VNICapabilities, VNIType
from ..managers.vni_manager import VNIManager
from ..domains.medical import MedicalVNI as MedicalVNI, MedicalVNI as EnhancedMedicalVNI
from ..domains.legal import LegalVNI as LegalVNI, LegalVNI as EnhancedLegalVNI
from ..domains.technical import TechnicalVNI
from ..domains.general_multi_vni import GeneralMultiVNI
from ..modules.generation import EnhancedGenerationModule
from ..modules.knowledge_base import KnowledgeBaseLoader
from ..modules.classifier import DynamicDomainClassifier

__all__ = [
    'EnhancedBaseVNI',
    'VNICapabilities',
    'VNIType',
    'VNIManager',
    'MedicalVNI',
    'EnhancedMedicalVNI',
    'LegalVNI',
    'EnhancedLegalVNI',
    'TechnicalVNI',
    'GeneralMultiVNI',
    'EnhancedGenerationModule',
    'KnowledgeBaseLoader',
    'DynamicDomainClassifier'
]
