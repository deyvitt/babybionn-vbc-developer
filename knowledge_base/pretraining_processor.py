# neuron/reinforcement_learning/training/pretraining_processor.py
"""
BabyBIONN Pretraining Processor
Converts domain knowledge JSON files into synaptic patterns for VNI initialization
"""

import json
import logging
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
from pathlib import Path

logger = logging.getLogger("babybionn_pretrainer")

@dataclass
class PretrainingConfig:
    """Configuration for pretraining process"""
    base_confidence: float = 0.7
    synaptic_strength_multiplier: float = 1.2
    pattern_complexity_threshold: float = 0.6
    max_concepts_per_domain: int = 1000
    knowledge_base_path: str = "knowledge_bases"

class ConceptProcessor:
    """Process individual concepts into synaptic patterns"""
    
    def __init__(self, config: PretrainingConfig):
        self.config = config
        
    def process_concept(self, concept_name: str, concept_data: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Convert a concept into synaptic pattern format"""
        
        # Generate unique pattern ID
        pattern_id = self._generate_pattern_id(domain, concept_name, concept_data['response'])
        
        # Calculate initial synaptic strength based on confidence
        base_strength = concept_data.get('confidence', self.config.base_confidence)
        synaptic_strength = base_strength * self.config.synaptic_strength_multiplier
        
        # Calculate pattern complexity
        complexity = self._calculate_pattern_complexity(concept_data['response'])
        
        return {
            'pattern_id': pattern_id,
            'concept_name': concept_name,
            'domain': domain,
            'response_template': concept_data['response'],
            'initial_confidence': concept_data.get('confidence', 0.7),
            'synaptic_strength': min(synaptic_strength, 1.0),  # Cap at 1.0
            'complexity': complexity,
            'usage_count': concept_data.get('usage_count', 0),
            'metadata': {
                'created_at': time.time(),
                'source': 'pretraining',
                'content_hash': self._hash_content(concept_data['response'])
            }
        }
    
    def _generate_pattern_id(self, domain: str, concept: str, response: str) -> str:
        """Generate unique pattern identifier"""
        content_hash = hashlib.md5(response.encode()).hexdigest()[:8]
        clean_concept = ''.join(c for c in concept.lower() if c.isalnum())[:20]
        return f"{domain}_{clean_concept}_{content_hash}"
    
    def _calculate_pattern_complexity(self, response: str) -> float:
        """Calculate complexity of response pattern (0-1 scale)"""
        word_count = len(response.split())
        sentence_count = response.count('.') + response.count('!') + response.count('?')
        
        # Normalize factors
        word_factor = min(word_count / 50, 1.0)  # Max 50 words = 1.0
        sentence_factor = min(sentence_count / 3, 1.0)  # Max 3 sentences = 1.0
        
        return (word_factor + sentence_factor) / 2
    
    def _hash_content(self, content: str) -> str:
        """Create content hash for duplicate detection"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

class ReasoningPatternProcessor:
    """Process reasoning patterns into VNI associations"""
    
    def process_reasoning_pattern(self, pattern_name: str, pattern_data: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Convert reasoning pattern into VNI association format"""
        
        return {
            'pattern_type': 'reasoning',
            'name': pattern_name,
            'domain': domain,
            'trigger_conditions': self._extract_trigger_conditions(pattern_data),
            'response_template': pattern_data['response'],
            'confidence': pattern_data.get('confidence', 0.8),
            'priority': self._calculate_priority(pattern_name),
            'associations': self._generate_associations(domain, pattern_data)
        }
    
    def _extract_trigger_conditions(self, pattern_data: Dict[str, Any]) -> List[str]:
        """Extract trigger conditions from pattern data"""
        # This would analyze the response template for trigger words
        response = pattern_data.get('response', '').lower()
        triggers = []
        
        # Simple keyword extraction - enhance with NLP in production
        emergency_triggers = ['emergency', 'immediate', 'serious', 'urgent', 'call 911']
        for trigger in emergency_triggers:
            if trigger in response:
                triggers.append(trigger)
                
        return triggers if triggers else ['general_trigger']
    
    def _calculate_priority(self, pattern_name: str) -> int:
        """Calculate execution priority for reasoning patterns"""
        priority_map = {
            'emergency': 100,
            'symptom_emergency': 90,
            'critical': 80,
            'warning': 70,
            'general': 50
        }
        return priority_map.get(pattern_name.lower(), 50)
    
    def _generate_associations(self, domain: str, pattern_data: Dict[str, Any]) -> List[str]:
        """Generate VNI associations for this pattern"""
        base_associations = [f"medical_0", "legal_0", "general_0"]
        
        # Add cross-domain associations based on content
        response = pattern_data.get('response', '').lower()
        if 'legal' in response or 'contract' in response:
            base_associations.append("legal_0")
        if 'technical' in response or 'system' in response:
            base_associations.append("general_0")
            
        return list(set(base_associations))

class BabyBIONNPretrainer:
    """
    Main pretraining orchestrator
    Converts domain knowledge JSON into synaptic patterns for VNI initialization
    """
    
    def __init__(self, vni_manager, rl_system, config: PretrainingConfig = None):
        self.vni_manager = vni_manager
        self.rl_system = rl_system
        self.config = config or PretrainingConfig()
        
        self.concept_processor = ConceptProcessor(self.config)
        self.reasoning_processor = ReasoningPatternProcessor()
        
        # Ensure knowledge base directory exists
        Path(self.config.knowledge_base_path).mkdir(exist_ok=True)
        
        logger.info("BabyBIONN Pretrainer initialized")
    
    def pretrain_domain(self, domain: str, pretrain_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main pretraining method - processes domain knowledge into synaptic patterns
        """
        logger.info(f"Starting pretraining for domain: {domain}")
        
        analytics = {
            'domain': domain,
            'start_time': time.time(),
            'concepts_processed': 0,
            'patterns_created': 0,
            'reasoning_patterns': 0,
            'errors': []
        }
        
        try:
            # Process concepts
            synaptic_patterns = {}
            if 'concepts' in pretrain_data:
                for concept_name, concept_data in pretrain_data['concepts'].items():
                    try:
                        pattern = self.concept_processor.process_concept(
                            concept_name, concept_data, domain
                        )
                        synaptic_patterns[pattern['pattern_id']] = pattern
                        analytics['concepts_processed'] += 1
                    except Exception as e:
                        error_msg = f"Error processing concept {concept_name}: {str(e)}"
                        analytics['errors'].append(error_msg)
                        logger.error(error_msg)
            
            # Process reasoning patterns
            reasoning_patterns = {}
            if 'reasoning_patterns' in pretrain_data:
                for pattern_name, pattern_data in pretrain_data['reasoning_patterns'].items():
                    try:
                        reasoning_pattern = self.reasoning_processor.process_reasoning_pattern(
                            pattern_name, pattern_data, domain
                        )
                        reasoning_patterns[pattern_name] = reasoning_pattern
                        analytics['reasoning_patterns'] += 1
                    except Exception as e:
                        error_msg = f"Error processing reasoning pattern {pattern_name}: {str(e)}"
                        analytics['errors'].append(error_msg)
                        logger.error(error_msg)
            
            # Process response templates
            response_templates = pretrain_data.get('response_templates', {})
            
            # Initialize VNIs with pretrained patterns
            self._initialize_vnis(domain, synaptic_patterns, reasoning_patterns, response_templates)
            
            # Initialize RL system with pretrained synaptic strengths
            self._initialize_rl_system(domain, synaptic_patterns, reasoning_patterns)
            
            # Save to knowledge base
            self._save_knowledge_base(domain, synaptic_patterns, reasoning_patterns, response_templates)
            
            analytics['patterns_created'] = len(synaptic_patterns)
            analytics['end_time'] = time.time()
            analytics['success'] = True
            analytics['processing_time'] = analytics['end_time'] - analytics['start_time']
            
            logger.info(f"Pretraining completed for {domain}: {analytics['concepts_processed']} concepts, "
                       f"{analytics['patterns_created']} patterns, {analytics['reasoning_patterns']} reasoning patterns")
            
        except Exception as e:
            analytics['success'] = False
            analytics['errors'].append(f"Pretraining failed: {str(e)}")
            logger.error(f"Pretraining failed for domain {domain}: {e}")
        
        return analytics
    
    def _initialize_vnis(self, domain: str, synaptic_patterns: Dict, reasoning_patterns: Dict, response_templates: Dict):
        """Initialize VNIs with pretrained patterns"""
        vni_id = f"VNI_{domain}_001"
        
        # Check if VNI exists, create if not
        if not self._vni_exists(vni_id):
            self._create_vni(vni_id, domain)
        
        # Initialize VNI with pretrained knowledge
        # This would integrate with your enhanced_vni_classes
        self._load_patterns_into_vni(vni_id, synaptic_patterns, reasoning_patterns, response_templates)
    
    def _vni_exists(self, vni_id: str) -> bool:
        """Check if VNI exists in the system"""
        # This would check your VNI manager
        # For now, simple implementation
        return hasattr(self.vni_manager, 'vnis') and vni_id in self.vni_manager.vnis
    
    def _create_vni(self, vni_id: str, domain: str):
        """Create a new VNI for the domain"""
        from neuron.reinforcement_learning.vni_core import VNIType
        
        domain_type_map = {
            'medical': VNIType.MEDICAL,
            'legal': VNIType.LEGAL,
            'general': VNIType.GENERAL,
            'technical': VNIType.TECHNICAL
        }
        
        vni_type = domain_type_map.get(domain, VNIType.GENERAL)
        self.vni_manager.register_vni(vni_id, vni_type)
        
        logger.info(f"Created new VNI: {vni_id} for domain: {domain}")
    
    def _load_patterns_into_vni(self, vni_id: str, synaptic_patterns: Dict, reasoning_patterns: Dict, response_templates: Dict):
        """Load pretrained patterns into specific VNI"""
        # This method would integrate with your enhanced_vni_classes
        # For now, we'll store in VNI metadata
        
        if vni_id in self.vni_manager.vnis:
            self.vni_manager.vnis[vni_id]['pretrained_patterns'] = synaptic_patterns
            self.vni_manager.vnis[vni_id]['reasoning_patterns'] = reasoning_patterns
            self.vni_manager.vnis[vni_id]['response_templates'] = response_templates
            
            logger.info(f"Loaded {len(synaptic_patterns)} patterns into {vni_id}")
    
    def _initialize_rl_system(self, domain: str, synaptic_patterns: Dict, reasoning_patterns: Dict):
        """Initialize RL system with pretrained synaptic strengths"""
        vni_id = f"VNI_{domain}_001"
        
        for pattern_id, pattern_data in synaptic_patterns.items():
            # Set initial synaptic strength in RL system
            self.rl_system.rl_engine.synaptic_memory.update_synaptic_strength(
                vni_id, pattern_id, pattern_data['synaptic_strength']
            )
        
        logger.info(f"Initialized RL system with {len(synaptic_patterns)} synaptic strengths for {vni_id}")
    
    def _save_knowledge_base(self, domain: str, synaptic_patterns: Dict, reasoning_patterns: Dict, response_templates: Dict):
        """Save pretrained knowledge to file"""
        knowledge_data = {
            'domain': domain,
            'timestamp': time.time(),
            'synaptic_patterns': synaptic_patterns,
            'reasoning_patterns': reasoning_patterns,
            'response_templates': response_templates,
            'metadata': {
                'total_patterns': len(synaptic_patterns),
                'total_reasoning_patterns': len(reasoning_patterns),
                'version': '1.0'
            }
        }
        
        filename = f"{self.config.knowledge_base_path}/{domain}_knowledge_base.json"
        
        with open(filename, 'w') as f:
            json.dump(knowledge_data, f, indent=2)
        
        logger.info(f"Saved knowledge base to: {filename}")
    
    def get_knowledge_status(self) -> Dict[str, Any]:
        """Get status of all knowledge bases"""
        knowledge_path = Path(self.config.knowledge_base_path)
        status = {'knowledge_bases': {}}
        
        for kb_file in knowledge_path.glob('*_knowledge_base.json'):
            try:
                with open(kb_file, 'r') as f:
                    kb_data = json.load(f)
                
                domain = kb_data['domain']
                status['knowledge_bases'][domain] = {
                    'concepts': len(kb_data.get('synaptic_patterns', {})),
                    'patterns': len(kb_data.get('reasoning_patterns', {})),
                    'last_updated': kb_data['timestamp'],
                    'file': str(kb_file)
                }
            except Exception as e:
                logger.error(f"Error reading knowledge base {kb_file}: {e}")
        
        return status

# Utility function to create pretrainer instance
def create_pretrainer(vni_manager, rl_system) -> BabyBIONNPretrainer:
    """Factory function to create pretrainer instance"""
    config = PretrainingConfig(
        base_confidence=0.7,
        synaptic_strength_multiplier=1.3,
        knowledge_base_path="knowledge_bases"
    )
    
    return BabyBIONNPretrainer(vni_manager, rl_system, config) 