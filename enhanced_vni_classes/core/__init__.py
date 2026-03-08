# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

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
