"""
Integration with smart_activation_router.py
"""
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def import_smart_activation_router():
    """Import SmartActivationRouter from the original file"""
    try:
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(PROJECT_ROOT))
        
        from neuron.smart_activation_router import SmartActivationRouter
        logger.info("✅ Imported SmartActivationRouter from original file")
        return SmartActivationRouter
        
    except ImportError as e:
        logger.error(f"❌ Failed to import SmartActivationRouter: {e}")
        
        # Use our refactored version
        from ..core.routing import SmartActivationRouter as RefactoredRouter
        logger.warning("⚠️ Using refactored SmartActivationRouter")
        return RefactoredRouter 
