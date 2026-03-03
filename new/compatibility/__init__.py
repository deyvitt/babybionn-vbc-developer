"""
Compatibility layer for existing modules
"""
from enhanced_vni_classes.modules.knowledge_base import KnowledgeBase
from enhanced_vni_classes.modules.classifier import DomainClassifier, DynamicDomainClassifier
from enhanced_vni_classes.modules.learning_system import LearningSystem
from enhanced_vni_classes.domains.general import EnhancedGenerationModule

# Backward compatibility aliases
KnowledgeBaseLoader = KnowledgeBase
TextGenerator = EnhancedGenerationModule
GenerationModule = EnhancedGenerationModule

__all__ = [
    'KnowledgeBaseLoader',
    'KnowledgeBase',
    'DomainClassifier',
    'DynamicDomainClassifier',
    'LearningSystem',
    'TextGenerator',
    'GenerationModule',
    'EnhancedGenerationModule',
    'GenerationStyle'
]
