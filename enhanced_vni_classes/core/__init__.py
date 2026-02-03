"""
Core module for enhanced VNI classes
"""
from .base_vni import EnhancedBaseVNI
from .biological_mixin import BiologicalSystemsMixin
from .capabilities import VNICapabilities, VNIType
from .neural_pathway import NeuralPathway
from .collaboration import CollaborationRequest, CollaborationResponse, CollaborationStatus
from .registry import VNIRegistry

__all__ = [
    'EnhancedBaseVNI',
    'BiologicalSystemsMixin',
    'VNICapabilities',
    'VNIType', 
    'NeuralPathway',
    'CollaborationRequest',
    'CollaborationResponse',
    'CollaborationStatus',
    'VNIRegistry'
]
