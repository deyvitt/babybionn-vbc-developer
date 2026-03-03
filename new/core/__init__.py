"""
Core BabyBIONN Modules
"""
# Import from enhanced_neural_mesh.py - using the actual class name
from enhanced_neural_mesh import EnhancedNeuralMeshCore

# Import other core modules
from .attention import EnhancedSynapticAttentionBridge
from .autonomy import VNIMessage, AutonomyEngine, AutonomousVNIProtocol
from .routing import SmartActivationRouter, RoutingIntelligence
from .adapters import OrchestratorToVNIManagerAdapter

# Define aliases for backward compatibility
EnhancedBabyBIONNOrchestrator = EnhancedNeuralMeshCore
BabyBIONNOrchestrator = EnhancedNeuralMeshCore

__all__ = [
    # Main orchestrator (actual class)
    'EnhancedNeuralMeshCore',
    
    # Backward compatibility aliases
    'EnhancedBabyBIONNOrchestrator',
    'BabyBIONNOrchestrator',
    
    # Attention module
    'EnhancedSynapticAttentionBridge',
    
    # Autonomy module
    'VNIMessage',
    'AutonomyEngine',
    'AutonomousVNIProtocol',
    
    # Routing module
    'SmartActivationRouter',
    'RoutingIntelligence',
    
    # Adapter module
    'OrchestratorToVNIManagerAdapter'
]

# Optional: Show deprecation warning when aliases are imported
import warnings

def _warn_about_alias(alias_name, actual_name):
    """Warn about deprecated alias usage"""
    warnings.warn(
        f"{alias_name} is an alias for {actual_name}. "
        f"Please use {actual_name} for future compatibility.",
        DeprecationWarning,
        stacklevel=3
    )

# Monkey patch to warn when aliases are accessed
import sys

class _AliasWarningModule(sys.modules[__name__].__class__):
    """Custom module class to warn about deprecated aliases"""
    def __getattr__(self, name):
        if name == 'EnhancedBabyBIONNOrchestrator':
            _warn_about_alias(name, 'EnhancedNeuralMeshCore')
        elif name == 'BabyBIONNOrchestrator':
            _warn_about_alias(name, 'EnhancedNeuralMeshCore')
        return super().__getattr__(name)

# Apply the custom module class
sys.modules[__name__].__class__ = _AliasWarningModule

print("✅ Core BabyBIONN modules loaded successfully")
print("   🧠 Main orchestrator: EnhancedNeuralMeshCore")
print("   🔗 Use: 'from new.core import EnhancedNeuralMeshCore'")
