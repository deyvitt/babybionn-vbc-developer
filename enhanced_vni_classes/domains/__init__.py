"""
Domains package for VNI implementations.
"""
from .base_knowledge_loader import BaseKnowledgeLoader
from .medical import MedicalVNI
from .legal import LegalVNI
from .general import GeneralVNI
from .dynamic_vni import DynamicVNI, EnhancedDomainFactory

__all__ = [
    'BaseKnowledgeLoader',
    'MedicalVNI',
    'LegalVNI', 
    'GeneralVNI',
    'DynamicVNI',
    'EnhancedDomainFactory'
]
