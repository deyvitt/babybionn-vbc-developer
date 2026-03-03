"""
Integration with enhanced_vni_classes.py
"""
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def import_enhanced_vni_classes():
    """Import classes from the refactored enhanced_vni_classes structure"""
    try:
        from enhanced_vni_classes.core.base_vni import EnhancedBaseVNI
        from enhanced_vni_classes.managers.vni_manager import VNIManager
        from enhanced_vni_classes.domains.medical import MedicalVNI
        from enhanced_vni_classes.domains.legal import LegalVNI
        
        # Create aliases for compatibility
        EnhancedMedicalVNI = MedicalVNI
        EnhancedLegalVNI = LegalVNI
        
        return {
            'EnhancedBaseVNI': EnhancedBaseVNI,
            'VNIManager': VNIManager,
            'EnhancedMedicalVNI': EnhancedMedicalVNI,
            'EnhancedLegalVNI': EnhancedLegalVNI,
            'MedicalVNI': MedicalVNI,
            'LegalVNI': LegalVNI
        }
    except Exception as e:
        logger.error(f"❌ Failed to import enhanced_vni_classes: {e}")
        # Fallback to simplified versions
        from new.models.vnis import MedicalVNI as SimpleMedicalVNI, LegalVNI as SimpleLegalVNI
        
        return {
            'EnhancedMedicalVNI': SimpleMedicalVNI,
            'EnhancedLegalVNI': SimpleLegalVNI,
            'MedicalVNI': SimpleMedicalVNI,
            'LegalVNI': SimpleLegalVNI
        }
        
    except ImportError as e:
        logger.error(f"❌ Failed to import enhanced_vni_classes: {e}")
        
        # Fallback to the simplified versions we created
        from ..models.vnis import (
            EnhancedMedicalVNI as SimpleMedicalVNI,
            EnhancedLegalVNI as SimpleLegalVNI,
            EnhancedGeneralVNI as SimpleGeneralVNI,
            NeuralPathway as SimpleNeuralPathway
        )
        
        logger.warning("⚠️ Using simplified VNI implementations as fallback")
        
        return {
            'EnhancedMedicalVNI': SimpleMedicalVNI,
            'EnhancedLegalVNI': SimpleLegalVNI,
            'EnhancedGeneralVNI': SimpleGeneralVNI,
            'NeuralPathway': SimpleNeuralPathway
        } 
