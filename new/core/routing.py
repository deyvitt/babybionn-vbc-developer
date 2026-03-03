"""
Query Routing and Smart Activation
"""
import logging
from typing import Dict, List, Any, Optional
import random

logger = logging.getLogger(__name__)

class SmartActivationRouter:
    """Intelligent VNI activation router"""
    
    def __init__(self):
        self.activation_threshold = 0.3
        self.domain_keywords = self._initialize_domain_keywords()
    
    def _initialize_domain_keywords(self) -> Dict[str, set]:
        """Initialize domain-specific keywords"""
        return {
            'medical': {
                'medical', 'health', 'symptom', 'treatment', 'medicine', 'doctor', 'patient',
                'hospital', 'disease', 'diagnosis', 'pain', 'therapy', 'clinical', 'physical',
                'blood', 'urinating', 'emergency', 'urgent', 'fever', 'headache'
            },
            'legal': {
                'legal', 'law', 'contract', 'rights', 'agreement', 'lawyer', 'court',
                'case', 'judge', 'legal', 'regulation', 'compliance', 'liability'
            },
            'general': {
                'code', 'programming', 'technical', 'general', 'system', 'algorithm', 'software',
                'python', 'java', 'database', 'api', 'framework', 'development', 'debug'
            }
        }
    
    def analyze_query(self, query_text: str) -> Dict[str, float]:
        """Analyze query text and return domain scores"""
        words = query_text.lower().split()
        
        scores = {domain: 0.0 for domain in self.domain_keywords}
        
        for domain, keywords in self.domain_keywords.items():
            for word in words:
                for keyword in keywords:
                    if keyword in word or word in keyword:
                        scores[domain] += 1
        
        total = sum(scores.values()) + 0.001  # Avoid division by zero
        return {domain: score / total for domain, score in scores.items()}
    
    def select_vnis(self, attention_scores: Dict[str, float]) -> List[str]:
        """Select VNIs based on attention scores"""
        if not attention_scores:
            return []
        
        # Return VNIs with scores above threshold
        selected = [vni_id for vni_id, score in attention_scores.items() 
                   if score >= self.activation_threshold]
        
        # If none above threshold, return top 2
        if not selected:
            sorted_scores = sorted(attention_scores.items(), key=lambda x: x[1], reverse=True)
            selected = [vni_id for vni_id, score in sorted_scores[:2]]
        
        return selected

class RoutingIntelligence:
    """Advanced routing intelligence with complexity analysis"""
    
    @staticmethod
    def analyze_query_complexity(query: str) -> int:
        """Analyze query complexity (1-10 scale)"""
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        complexity = 0
        
        # Word count factor
        if len(words) > 20:
            complexity += 3
        elif len(words) > 10:
            complexity += 2
        elif len(words) > 5:
            complexity += 1
        
        # Question complexity
        if '?' in query:
            complexity += 1
        if any(word in query_lower for word in ['how', 'why', 'what', 'when', 'where']):
            complexity += 1
        
        # Urgency detection
        urgency_words = ['urgent', 'emergency', 'help', 'immediately', 'now', 'asap']
        if any(word in query_lower for word in urgency_words):
            complexity += 2
        
        # Medical emergency detection
        medical_urgency = ['blood', 'pain', 'symptom', 'fever', 'headache', 'chest pain']
        if any(word in query_lower for word in medical_urgency):
            complexity += 2
        
        return min(10, max(1, complexity))
    
    @staticmethod
    def identify_relevant_domains(query: str) -> List[str]:
        """Identify relevant domains for a query"""
        query_lower = query.lower()
        domains = []
        
        medical_keywords = ['medical', 'health', 'doctor', 'hospital', 'disease', 'medicine']
        legal_keywords = ['legal', 'law', 'lawyer', 'court', 'contract', 'agreement']
        technical_keywords = ['technical', 'code', 'programming', 'system', 'software']
        
        if any(keyword in query_lower for keyword in medical_keywords):
            domains.append("medical")
        if any(keyword in query_lower for keyword in legal_keywords):
            domains.append("legal")
        if any(keyword in query_lower for keyword in technical_keywords):
            domains.append("general")
        
        if not domains:
            domains.append("general")
        
        return domains
    
    @staticmethod
    def enhanced_smart_route_query(orchestrator, input_text: str) -> List[str]:
        """Enhanced routing that checks VNI should_handle"""
        logger.info(f"🔍 ROUTING CHECK for: '{input_text[:50]}...'")
        
        activated_vnis = []
        
        # Ask each VNI if they can handle this
        for vni_id, vni in orchestrator.vni_instances.items():
            if hasattr(vni, 'should_handle'):
                try:
                    if vni.should_handle(input_text):
                        logger.info(f"   ✅ {vni_id} says: I can handle this!")
                        activated_vnis.append(vni_id)
                    else:
                        logger.info(f"   ❌ {vni_id} says: Not for me")
                except Exception as e:
                    logger.error(f"   ⚠️  {vni_id} error: {e}")
        
        # If any VNIs claim it, return them
        if activated_vnis:
            return activated_vnis
        
        # Fallback to general VNI
        logger.warning(f"   ⚠️  No VNI claimed it, using general_0")
        return ['general_0'] 
