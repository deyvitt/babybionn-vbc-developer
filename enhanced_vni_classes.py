# This is the main-entry and re-routing mechanism to 'app/enhanced_vni_classes/' system 
#!/usr/bin/env python3
"""
enhanced_vni_classes.py - MAIN ENTRY POINT
This file provides backward compatibility for existing code.
All actual implementation is in the enhanced_vni_classes/ directory.
"""
import time
from typing import Dict, Any, Optional, List

# Import everything from the modular directory
from enhanced_vni_classes.core.base_vni import EnhancedBaseVNI
from enhanced_vni_classes.core.capabilities import VNICapabilities, VNIType
from enhanced_vni_classes.core.neural_pathway import NeuralPathway
from enhanced_vni_classes.core.collaboration import CollaborationRequest
from enhanced_vni_classes.core.registry import VNIRegistry

# Import ALL domain VNIs including DynamicVNI
from enhanced_vni_classes.domains.medical import MedicalVNI
from enhanced_vni_classes.domains.legal import LegalVNI
from enhanced_vni_classes.domains.general import GeneralVNI
from enhanced_vni_classes.domains.dynamic_vni import DynamicVNI  # ← ADD THIS
from enhanced_vni_classes.domains.technical import TechnicalVNI

from enhanced_vni_classes.managers.vni_manager import VNIManager
from enhanced_vni_classes.managers.session_manager import SessionManager

# from enhanced_vni_classes.modules.generation import SharedNLPComponents, SSPConverter

# Import constants from the modular directory's utils
from enhanced_vni_classes.utils.imports import (
    GENERATION_AVAILABLE,
    TORCH_AVAILABLE,
    WEB_SEARCH_AVAILABLE,
    PREDICTIVE_AVAILABLE,
    SENTENCE_TRANSFORMER_AVAILABLE,
    GPT2_AVAILABLE
)

# Backward compatibility aliases (keep these for existing code)
EnhancedMedicalVNI = MedicalVNI
EnhancedLegalVNI = LegalVNI
EnhancedGeneralVNI = GeneralVNI
EnhancedTechnicalVNI = TechnicalVNI


# ========== LLM GATEWAY STUBS (REPLACEMENT FOR generation.py) ==========
class SharedNLPComponents:
    """
    Stub class replacing old SharedNLPComponents.
    Now using external LLM Gateway via llm_gateway.py
    """
    def __init__(self):
        self.llm_gateway = None
        # Try to initialize LLM Gateway
        try:
            from llm_Gateway import get_gateway
            self.llm_gateway = get_gateway()
            print("✅ SharedNLPComponents: LLM Gateway initialized")
        except ImportError as e:
            print(f"⚠️ SharedNLPComponents: LLM Gateway not available: {e}")
    
    def process_text(self, text, context=None):
        """Process text using LLM Gateway"""
        if self.llm_gateway:
            try:
                # Simple prompt for text processing
                prompt = f"Process this text: {text}"
                if context:
                    prompt += f"\nContext: {context}"
                
                response = self.llm_gateway.generate(prompt=prompt, vni_context="general")
                return response.content
            except Exception as e:
                return f"LLM processing error: {str(e)}"
        else:
            return f"[LLM not available] Process: {text}"

class SSPConverter:
    """
    Stub class replacing old SSPConverter.
    Semantic Structure Pattern conversion now via LLM
    """
    def __init__(self):
        self.llm_gateway = None
        try:
            from llm_Gateway import get_gateway
            self.llm_gateway = get_gateway()
            print("✅ SSPConverter: LLM Gateway initialized")
        except ImportError as e:
            print(f"⚠️ SSPConverter: LLM Gateway not available: {e}")
    
    def convert_pattern(self, pattern, target_format="natural_language"):
        """Convert patterns using LLM"""
        if self.llm_gateway:
            try:
                prompt = f"Convert this pattern to {target_format}:\n{pattern}"
                response = self.llm_gateway.generate(
                    prompt=prompt, 
                    vni_context="technical"
                )
                return response.content
            except Exception as e:
                return f"Pattern conversion error: {str(e)}"
        else:
            return f"[LLM not available] Convert pattern: {pattern}"

# Try to import VNI Spawner if it exists
try:
    from enhanced_vni_classes.vni_spawner import VNISpawner
except ImportError:
    # Create placeholder if module doesn't exist
    class VNISpawner:
        """Placeholder for VNI Spawner"""
        pass

# Re-export everything exactly as in the original monolithic file
__all__ = [
    # Core classes
    'EnhancedBaseVNI',
    'VNICapabilities',
    'VNIType',
    'NeuralPathway',
    'CollaborationRequest',
    'VNIRegistry',
    
    # Domain VNIs (both actual names and aliases)
    'MedicalVNI',
    'LegalVNI', 
    'GeneralVNI',
    'DynamicVNI',
    'TechnicalVNI',
    'EnhancedMedicalVNI',
    'EnhancedLegalVNI',
    'EnhancedGeneralVNI',
    
    # Managers
    'VNIManager',
    'SessionManager',

    # VNI Spawner
    'VNISpawner',
    
    # Modules
    'SharedNLPComponents',
    'SSPConverter',
    
    # Constants
    'GENERATION_AVAILABLE',
    'TORCH_AVAILABLE',
    'WEB_SEARCH_AVAILABLE',
    'PREDICTIVE_AVAILABLE',
    'SENTENCE_TRANSFORMER_AVAILABLE',
    'GPT2_AVAILABLE'
]
