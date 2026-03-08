#neuron/mock_response.py
"""Simple mock response provider for development and testing"""
import random
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger("mock_provider")

class MockResponseProvider:
    """Provides mock responses when real VNIs/LLMs aren't available"""
    
    def __init__(self, config=None):
        self.config = config
        self.interaction_count = 0
        self.start_confidence = float(os.getenv("MOCK_CONFIDENCE_START", "0.7"))
        self.confidence_increment = float(os.getenv("MOCK_CONFIDENCE_INCREMENT", "0.01"))
        logger.info(f"✅ MockResponseProvider initialized (start confidence: {self.start_confidence})")
    
    def generate_response(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate a mock response that looks real enough for the system to learn from"""
        self.interaction_count += 1
        
        query_lower = query.lower()
        
        # Domain detection (simple version)
        domain = "general"
        if any(word in query_lower for word in ['medical', 'health', 'doctor', 'pain', 'fever', 'diabetes', 'symptom']):
            domain = "medical"
        elif any(word in query_lower for word in ['legal', 'law', 'contract', 'court', 'rights', 'attorney']):
            domain = "legal"
        elif any(word in query_lower for word in ['code', 'programming', 'python', 'software', 'computer', 'debug', 'algorithm']):
            domain = "technical"
        
        # Calculate confidence that increases with interactions
        confidence = min(0.95, self.start_confidence + (self.interaction_count * self.confidence_increment))
        
        # Generate appropriate mock response
        if domain == "medical":
            response = self._medical_response(query)
        elif domain == "legal":
            response = self._legal_response(query)
        elif domain == "technical":
            response = self._technical_response(query)
        else:
            response = self._general_response(query)
        
        # Return in format that matches what VNIs would return
        return {
            'response': response,
            'confidence': confidence,
            'domain': domain,
            'vni_metadata': {
                'vni_id': f"mock_{domain}_vni_{self.interaction_count}",
                'success': True,
                'domain': domain,
                'mock_mode': True,
                'interaction_count': self.interaction_count
            },
            'learning_data': {
                'query': query[:100],
                'domain': domain,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _medical_response(self, query: str) -> str:
        responses = [
            f"Regarding your question about {query[:50]}... This appears to be a health-related inquiry. For educational purposes, I can share that {self._get_medical_fact()}. Remember to consult with a healthcare professional for personal medical advice.",
            
            f"Health questions are important. While I can't provide medical advice, I can tell you that {self._get_medical_fact()}. Would you like more general information about this topic?",
            
            f"I understand you're asking about {query[:50]}... In general terms, {self._get_medical_fact()}. This is educational information only."
        ]
        return random.choice(responses)
    
    def _get_medical_fact(self) -> str:
        facts = [
            "many common symptoms can have multiple causes",
            "prevention is often as important as treatment",
            "lifestyle factors significantly impact health outcomes",
            "early detection of health issues leads to better outcomes",
            "medications should always be taken as prescribed by a doctor"
        ]
        return random.choice(facts)
    
    def _legal_response(self, query: str) -> str:
        responses = [
            f"Regarding your legal question about {query[:50]}... Laws vary by jurisdiction. Generally speaking, {self._get_legal_fact()}. For your specific situation, consulting with an attorney would be best.",
            
            f"Legal matters are complex. In general terms, {self._get_legal_fact()}. An attorney can provide advice tailored to your specific circumstances.",
            
            f"I understand you're asking about {query[:50]}... While I can't provide legal advice, I can explain that {self._get_legal_fact()}. Would you like me to suggest resources for finding legal help?"
        ]
        return random.choice(responses)
    
    def _get_legal_fact(self) -> str:
        facts = [
            "contracts require offer, acceptance, and consideration",
            "statute of limitations varies by jurisdiction and case type",
            "legal precedents can vary significantly between different courts",
            "documentation is crucial in legal matters",
            "many legal disputes can be resolved through mediation"
        ]
        return random.choice(facts)
    
    def _technical_response(self, query: str) -> str:
        responses = [
            f"Great technical question about {query[:50]}! In programming, {self._get_technical_fact()}. Would you like me to explain with a concrete example?",
            
            f"Technical queries like this are fascinating! {self._get_technical_fact()}. Let me know if you'd like me to elaborate on any specific aspect.",
            
            f"I appreciate your technical curiosity. {self._get_technical_fact()}. This is a fundamental concept that many developers encounter."
        ]
        return random.choice(responses)
    
    def _get_technical_fact(self) -> str:
        facts = [
            "algorithms are step-by-step procedures for solving problems",
            "data structures organize and store data efficiently",
            "debugging is the process of finding and fixing errors in code",
            "version control helps track changes in code over time",
            "APIs allow different software applications to communicate"
        ]
        return random.choice(facts)
    
    def _general_response(self, query: str) -> str:
        responses = [
            f"That's an interesting question: {query[:50]}... I'm here to help explore it with you.",
            
            f"I'd be happy to discuss {query[:50]}... with you. What aspects are you most curious about?",
            
            f"Thanks for asking! Let me think about {query[:50]}... This is a topic worth exploring together."
        ]
        return random.choice(responses)
