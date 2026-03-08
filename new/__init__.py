# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

"""
BabyBIONN - Enhanced Neural Mesh System
Ultimate neural mesh orchestrator combining neural mesh sophistication with clean orchestrator design.
"""

from enhanced_neural_mesh import (
    EnhancedNeuralMeshCore,
    MeshNodeState,
    SynapseType,
    ActivationPulse,
    MeshNode,
    MeshSynapse,
    SynapticPattern,
    NeuralMeshTask,
    CollaborationPatternTracker,
    integrate_enhanced_mesh
)

__all__ = [
    # Main orchestrator
    'EnhancedNeuralMeshCore',
    
    # Neural mesh types and components
    'MeshNodeState',
    'SynapseType',
    'ActivationPulse',
    'MeshNode',
    'MeshSynapse',
    'SynapticPattern',
    
    # Task management
    'NeuralMeshTask',
    
    # Collaboration patterns
    'CollaborationPatternTracker',
    
    # Integration utility
    'integrate_enhanced_mesh'
]

__version__ = "2.0.0"
__author__ = "BabyBIONN Team"
__description__ = "Enhanced Neural Mesh System combining neural mesh sophistication with orchestrator design"

# Create a convenient alias for the main orchestrator
BabyBIONNOrchestrator = EnhancedNeuralMeshCore
EnhancedBabyBIONNOrchestrator = EnhancedNeuralMeshCore

# Export the aliases
__all__.extend(['BabyBIONNOrchestrator', 'EnhancedBabyBIONNOrchestrator'])

# Import convenience for users who might expect the old name
__import__('warnings').warn(
    "EnhancedBabyBIONNOrchestrator is now EnhancedNeuralMeshCore. Use EnhancedNeuralMeshCore for direct access.",
    DeprecationWarning,
    stacklevel=2
)

print(f"🚀 BabyBIONN Enhanced Neural Mesh v{__version__} initialized")
print(f"📦 Main class: EnhancedNeuralMeshCore")
print(f"🔗 Use: 'from new import EnhancedNeuralMeshCore' or 'from new import BabyBIONNOrchestrator'")
