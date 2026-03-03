"""
Integration with other existing modules
"""
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def import_safety_check():
    """Import SafetyManager from safety_check.py"""
    try:
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(PROJECT_ROOT))
        
        from safety_check import SafetyManager
        logger.info("✅ Imported SafetyManager")
        return SafetyManager
        
    except ImportError as e:
        logger.error(f"❌ Failed to import SafetyManager: {e}")
        return None

def import_learning_analytics():
    """Import LearningAnalytics from learning_analytics.py"""
    try:
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(PROJECT_ROOT))
        
        from learning_analytics import LearningAnalytics
        logger.info("✅ Imported LearningAnalytics")
        return LearningAnalytics
        
    except ImportError as e:
        logger.error(f"❌ Failed to import LearningAnalytics: {e}")
        return None

def import_synaptic_visualization():
    """Import SynapticVisualizer from synaptic_visualization.py"""
    try:
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(PROJECT_ROOT))
        
        from synaptic_visualization import SynapticVisualizer
        logger.info("✅ Imported SynapticVisualizer")
        return SynapticVisualizer
        
    except ImportError as e:
        logger.error(f"❌ Failed to import SynapticVisualizer: {e}")
        return None

def import_model_loading():
    """Import model_manager from model_loading.py"""
    try:
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(PROJECT_ROOT))
        
        from model_loading import model_manager
        logger.info("✅ Imported model_manager")
        return model_manager
        
    except ImportError as e:
        logger.error(f"❌ Failed to import model_manager: {e}")
        return None 
