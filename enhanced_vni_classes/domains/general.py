"""General domain VNI - AGGREGATOR ONLY VERSION
NO generation - only prepares data for aggregator
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib
import logging

from ..core.base_vni import EnhancedBaseVNI
from ..core.capabilities import VNICapabilities, VNIType
from .base_knowledge_loader import BaseKnowledgeLoader

logger = logging.getLogger(__name__)

class GeneralVNI(BaseKnowledgeLoader, EnhancedBaseVNI):
    """
    General domain VNI - NO generation, only prepares data for aggregator
    All response generation is handled by Aggregator + LLM Gateway
    """
    
    def __init__(
        self,
        instance_id: str = "general_001",
        vni_config: Dict[str, Any] = None,
        auto_load_knowledge: bool = True
    ):
        # Generate VNI ID if not provided
        if instance_id is None:
            timestamp = int(datetime.now().timestamp() * 1000) % 10000
            instance_id = f"general_vni_{timestamp}"
        
        self.instance_id = instance_id
        self.vni_type = "general"
        self.domain = "general"
        
        # Set up capabilities - NO generation!
        capabilities = VNICapabilities(
            domains=["general", "conversation", "information", "assistance"],  # ← Use domains parameter
            can_search=True,
            can_learn=True,
            can_collaborate=True,
            max_context_length=4096,
            special_abilities=[
                "conversational",
                "task_assistance", 
                "knowledge_synthesis",
                "general_purpose"
            ],
            vni_type= 'general' #VNIType.GENERAL  # or 'general' if VNIType not available
        )
        
        # Initialize base classes
        BaseKnowledgeLoader.__init__(self)
        EnhancedBaseVNI.__init__(
            self,
            instance_id=instance_id,
            domain="general",
            capabilities=capabilities,
            name=f"GeneralVNI-{instance_id}",
            description="General purpose VNI - prepares data for aggregator",
            vni_type="general"
        )
        
        # Simple knowledge base
        self.general_knowledge = self._initialize_knowledge()
        
        # Load knowledge from files if enabled
        if auto_load_knowledge:
            self.load_domain_knowledge("general")
        
        logger.info(f"✅ GeneralVNI '{instance_id}' initialized (NO generation, aggregator-only)")
    
    def _initialize_knowledge(self) -> Dict[str, Any]:
        """Initialize general knowledge base."""
        return {
            "categories": [
                "science", "history", "technology", "arts", "culture",
                "health", "education", "sports", "travel", "food"
            ],
            "conversation_topics": [
                "current events", "technology trends", "science discoveries",
                "book recommendations", "movie reviews", "travel destinations",
                "health tips", "learning resources", "hobbies", "personal development"
            ]
        }
    
    async def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process general query - ONLY prepares data for aggregator.
        NO response generation - aggregator handles that with LLM Gateway.
        """
        context = context or {}
        
        # Simple query classification
        classification = self._classify_query(query)
        confidence = self._calculate_confidence(query, classification)
        
        # Get conversation history if available
        conversation_history = context.get('conversation_history', [])
        
        # Prepare generation data for aggregator
        generation_data = {
            'needs_generation': True,  # Signal to aggregator
            'domain': 'general',
            'query': query,
            'confidence': confidence,
            'query_type': classification.get('type', 'general'),
            'topics': classification.get('topics', []),
            'context': {
                'conversation_length': len(conversation_history),
                'is_follow_up': len(conversation_history) > 0,
                'query_complexity': self._assess_complexity(query)
            },
            'style_suggestions': {
                'tone': self._suggest_tone(query, classification),
                'temperature': 0.7,  # Balanced for general conversation
                'max_tokens': 300
            }
        }
        
        # Prepare complete output for aggregator
        opinion_text = (
            f"General query. Type: {classification.get('type', 'general')}. "
            f"Topics: {', '.join(classification.get('topics', ['none']))}. "
            f"Confidence: {confidence:.0%}."
        )

        result = {
            'vni_id': self.instance_id,
            'vni_type': 'general',
            'domain': 'general',
            'confidence_score': confidence,
            'opinion_text': opinion_text,
            'generation_data': generation_data,
            'analysis': {
                'classification': classification,
                'word_count': len(query.split()),
                'has_question': '?' in query
            },
            'vni_metadata': {
                'vni_id': self.instance_id,
                'success': True,
                'processing_time': 0.01,
                'timestamp': datetime.now().isoformat()
            }
        }
        return result
    async def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Alias for process method to maintain compatibility with mesh_core. This just calls process()."""
        return await self.process(query, context)

    def _classify_query(self, query: str) -> Dict[str, Any]:
        """Simple query classification."""
        query_lower = query.lower()
        
        # Determine query type
        query_type = "general"
        if any(word in query_lower for word in ["how to", "tutorial", "guide", "steps"]):
            query_type = "instructional"
        elif any(word in query_lower for word in ["what is", "definition", "meaning", "explain"]):
            query_type = "explanatory"
        elif any(word in query_lower for word in ["why", "reason", "cause"]):
            query_type = "analytical"
        elif any(word in query_lower for word in ["compare", "difference", "vs", "versus"]):
            query_type = "comparative"
        elif any(word in query_lower for word in ["best", "top", "recommend", "suggest"]):
            query_type = "recommendation"
        elif any(word in query_lower for word in ["hi", "hello", "hey", "how are you"]):
            query_type = "greeting"
        
        # Detect topics
        topics = []
        for category in self.general_knowledge.get("categories", []):
            if category in query_lower:
                topics.append(category)
        
        return {
            'type': query_type,
            'topics': topics[:3],  # Limit to top 3 topics
            'primary_topic': topics[0] if topics else 'general'
        }
    
    def _calculate_confidence(self, query: str, classification: Dict[str, Any]) -> float:
        """Calculate confidence score."""
        confidence = 0.5  # Base confidence
        
        # Adjust based on query length
        word_count = len(query.split())
        if 3 <= word_count <= 20:
            confidence += 0.2
        elif word_count > 20:
            confidence += 0.1
        
        # Adjust based on query type
        query_type = classification.get('type', 'general')
        if query_type in ['explanatory', 'instructional']:
            confidence += 0.2
        elif query_type == 'greeting':
            confidence += 0.3
        
        # Adjust based on topic detection
        if classification.get('topics'):
            confidence += 0.1
        
        return min(confidence, 0.95)
    
    def _assess_complexity(self, query: str) -> float:
        """Assess query complexity (0.0 to 1.0)."""
        words = len(query.split())
        sentences = query.count('.') + query.count('?') + query.count('!')
        return min(1.0, (words / 50) + (sentences / 5))
    
    def _suggest_tone(self, query: str, classification: Dict[str, Any]) -> str:
        """Suggest response tone for aggregator."""
        query_type = classification.get('type', 'general')
        
        tone_map = {
            'greeting': 'friendly',
            'instructional': 'helpful',
            'explanatory': 'clear',
            'analytical': 'balanced',
            'comparative': 'objective',
            'recommendation': 'suggestive',
            'general': 'conversational'
        }
        
        return tone_map.get(query_type, 'conversational')
