# vni_rl_integration.py
"""
Integration module connecting the reinforcement learning system with VNIs
Creates a complete learning loop where VNIs can evolve based on feedback
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import time
from collections import defaultdict
import json

from reinforcement_learning import BabyBIONNReinforcementSystem, RLConfig
# Import the minimal VNI core implementation
try:
    from vni_core import VNIManager, VNIType, VNIResponse, VNIStimulus
except ImportError:
    # Fallback to the implementation above if separate file doesn't exist
    from dataclasses import dataclass, field
    from typing import Dict, List, Any, Optional
    from enum import Enum
    
    class VNIType(Enum):
        GENERAL = "general"
        MEDICAL = "medical"
        LEGAL = "legal"
        TECHNICAL = "technical"
        HEALTH = "health"
        COMPLIANCE = "compliance"
        SOFTWARE = "software"
        BIO = "bio"
        ETHICS = "ethics"
        MATH = "math"
    
    @dataclass
    class VNIStimulus:
        content: Any
        stimulus_type: str = "general"
        metadata: Dict[str, Any] = field(default_factory=dict)
        timestamp: float = 0.0
        
        def __post_init__(self):
            if self.timestamp == 0:
                import time
                self.timestamp = time.time()
    
    @dataclass
    class VNIResponse:
        content: Any
        confidence: float
        response_type: str
        metadata: Dict[str, Any] = field(default_factory=dict)
        processing_time: float = 0.0
    
    class VNIManager:
        def __init__(self):
            self.vnis = {}
            
        def register_vni(self, vni_id: str, vni_type: VNIType, processor_func=None):
            self.vnis[vni_id] = {
                'type': vni_type,
                'processor': processor_func or self._default_processor
            }
        
        def process_stimulus(self, vni_id: str, stimulus: VNIStimulus) -> VNIResponse:
            if vni_id not in self.vnis:
                raise ValueError(f"VNI {vni_id} not found")
            
            import time
            start_time = time.time()
            
            processor = self.vnis[vni_id]['processor']
            content = processor(stimulus, vni_id)
            
            processing_time = time.time() - start_time
            
            return VNIResponse(
                content=content,
                confidence=0.8,  # Default confidence
                response_type=self.vnis[vni_id]['type'].value,
                processing_time=processing_time
            )
        
        def _default_processor(self, stimulus: VNIStimulus, vni_id: str) -> Any:
            return f"Processed by {vni_id}: {stimulus.content}"
        
logger = logging.getLogger("babybionn_rl_integration")

@dataclass
class VNIRLIntegrationConfig:
    """Configuration for VNI-RL integration"""
    rl_config: RLConfig = RLConfig()
    feedback_delay: float = 0.1  # Delay before processing feedback
    max_context_vnis: int = 5  # Maximum number of context VNIs to consider
    learning_threshold: float = 0.7  # Confidence threshold for considering learning
    pattern_extraction_window: int = 10  # Number of responses to analyze for pattern extraction

class VNIPatternExtractor:
    """Extracts patterns from VNI responses for RL learning"""
    
    def __init__(self, config: VNIRLIntegrationConfig):
        self.config = config
        self.response_history = defaultdict(list)
        self.pattern_templates = {}
        
    def extract_pattern_from_response(self, vni_id: str, response: VNIResponse) -> Dict[str, Any]:
        """Extract learning pattern from VNI response"""
        pattern = {
            'vni_id': vni_id,
            'pattern_id': self._generate_pattern_id(response),
            'response_type': response.response_type,
            'confidence': response.confidence,
            'content_hash': hash(str(response.content)),
            'timestamp': time.time(),
            'metadata': {
                'complexity': self._calculate_complexity(response),
                'novelty': self._calculate_novelty(vni_id, response),
                'effectiveness': 0.5  # Initial assumption, updated via RL
            }
        }
        
        # Store in history
        self.response_history[vni_id].append(pattern)
        if len(self.response_history[vni_id]) > self.config.pattern_extraction_window:
            self.response_history[vni_id].pop(0)
            
        return pattern
    
    def _generate_pattern_id(self, response: VNIResponse) -> str:
        """Generate unique pattern ID from response characteristics"""
        components = [
            response.response_type,
            str(response.confidence)[:4],
            str(hash(str(response.content)))[-6:]
        ]
        return f"pattern_{'_'.join(components)}"
    
    def _calculate_complexity(self, response: VNIResponse) -> float:
        """Calculate complexity of response (0-1 scale)"""
        content_str = str(response.content)
        # Simple heuristic: longer, more structured responses are more complex
        length_factor = min(len(content_str) / 1000, 1.0)  # Normalize by length
        structure_factor = 1.0 if hasattr(response, 'structured_data') else 0.3
        return (length_factor + structure_factor) / 2
    
    def _calculate_novelty(self, vni_id: str, response: VNIResponse) -> float:
        """Calculate how novel this response is for the VNI"""
        recent_patterns = self.response_history[vni_id][-5:]  # Last 5 responses
        if not recent_patterns:
            return 1.0  # Maximum novelty if no history
            
        current_hash = hash(str(response.content))
        matching_patterns = sum(1 for p in recent_patterns 
                              if p['content_hash'] == current_hash)
        
        return 1.0 - (matching_patterns / len(recent_patterns))

class VNILearningOrchestrator:
    """
    Orchestrates the learning loop between VNIs and RL system
    Manages stimulus processing, response evaluation, and reinforcement
    """
    
    def __init__(self, vni_manager: VNIManager, config: VNIRLIntegrationConfig = None):
        self.vni_manager = vni_manager
        self.config = config or VNIRLIntegrationConfig()
        self.rl_system = BabyBIONNReinforcementSystem(self.config.rl_config)
        self.pattern_extractor = VNIPatternExtractor(self.config)
        
        # Learning state tracking
        self.learning_sessions = 0
        self.performance_metrics = defaultdict(list)
        self.active_context = {}
        
        logger.info("VNI Learning Orchestrator initialized")
    
    def process_stimulus_with_learning(self, stimulus: VNIStimulus) -> Dict[str, Any]:
        """Process stimulus through VNIs with RL-guided pattern selection"""
        
        # Determine target VNIs based on stimulus
        target_vnis = self._select_target_vnis(stimulus)
        context_vnis = self._get_context_vnis(stimulus, target_vnis)
        
        # Prepare RL stimulus
        rl_stimulus = {
            'target_vnis': target_vnis,
            'context_vnis': context_vnis,
            'input_data': {
                'stimulus_type': stimulus.stimulus_type,
                'content': stimulus.content,
                'metadata': stimulus.metadata
            },
            'session_type': 'vni_learning'
        }
        
        # Get RL-guided responses
        rl_response = self.rl_system.stimulate_system(rl_stimulus)
        
        # Execute actual VNI processing with RL-guided patterns
        vni_responses = {}
        learning_patterns = {}
        
        for vni_id, rl_vni_response in rl_response['responses'].items():
            if vni_id in target_vnis:
                # Apply RL-guided pattern to actual VNI processing
                vni_response = self._apply_rl_pattern_to_vni(
                    vni_id, rl_vni_response, stimulus
                )
                vni_responses[vni_id] = vni_response
                
                # Extract learning pattern
                pattern = self.pattern_extractor.extract_pattern_from_response(
                    vni_id, vni_response
                )
                learning_patterns[vni_id] = pattern
                
                # Record activation for RL
                self.rl_system.rl_engine.synaptic_memory.record_activation(
                    vni_id, rl_vni_response['pattern_used'], 
                    vni_response.confidence
                )
        
        # Store session context for later feedback
        session_context = {
            'session_id': self.learning_sessions,
            'stimulus': stimulus,
            'rl_stimulus_context': rl_response['stimulus_context'],
            'vni_responses': vni_responses,
            'learning_patterns': learning_patterns,
            'target_vnis': target_vnis,
            'context_vnis': context_vnis,
            'timestamp': time.time()
        }
        
        self.active_context[self.learning_sessions] = session_context
        self.learning_sessions += 1
        
        return {
            'session_id': session_context['session_id'],
            'vni_responses': vni_responses,
            'rl_guidance': rl_response,
            'learning_metadata': {
                'patterns_used': learning_patterns,
                'exploration_rates': {
                    vni_id: resp.get('exploration', False) 
                    for vni_id, resp in rl_response['responses'].items()
                }
            }
        }
    
    def provide_learning_feedback(self, session_id: int, feedback_data: Dict[str, Any]):
        """Provide feedback for a learning session"""
        
        if session_id not in self.active_context:
            logger.warning(f"Session {session_id} not found for feedback")
            return
        
        session_context = self.active_context[session_id]
        
        # Calculate reward based on feedback
        reward_value = self._calculate_reward_value(session_context, feedback_data)
        
        # Apply reinforcement
        if reward_value > 0:
            self.rl_system.provide_feedback(
                session_id, reward_value,
                {
                    'target_vnis': session_context['target_vnis'],
                    'feedback_data': feedback_data,
                    'performance_metrics': self._calculate_performance_metrics(session_context)
                }
            )
        else:
            self.rl_system.punish_system(
                session_id, abs(reward_value),
                {
                    'target_vnis': session_context['target_vnis'],
                    'feedback_data': feedback_data
                }
            )
        
        # Update performance metrics
        self._update_performance_metrics(session_id, reward_value, feedback_data)
        
        # Clean up
        del self.active_context[session_id]
        
        logger.info(f"Learning feedback applied to session {session_id}: reward={reward_value:.3f}")
    
    def _select_target_vnis(self, stimulus: VNIStimulus) -> List[str]:
        """Select which VNIs should process this stimulus"""
        # This would integrate with your existing VNI selection logic
        target_vnis = []
        
        # Example selection logic - adapt based on your VNI manager
        stimulus_type = getattr(stimulus, 'stimulus_type', 'general')
        content = getattr(stimulus, 'content', '')
        
        # Simple type-based routing
        if 'medical' in stimulus_type.lower() or any(med_term in content.lower() 
                                                   for med_term in ['patient', 'treatment', 'diagnosis']):
            target_vnis.extend(['VNI_medical_001', 'VNI_health_001'])
        
        if 'legal' in stimulus_type.lower() or any(legal_term in content.lower() 
                                                 for legal_term in ['contract', 'law', 'legal']):
            target_vnis.extend(['VNI_legal_001', 'VNI_compliance_001'])
            
        if 'technical' in stimulus_type.lower() or any(tech_term in content.lower() 
                                                     for tech_term in ['system', 'code', 'technical']):
            target_vnis.extend(['VNI_technical_001', 'VNI_software_001'])
        
        # Fallback to base VNI
        if not target_vnis:
            target_vnis = ['VNI_base_001']
            
        return target_vnis[:self.config.max_context_vnis]
    
    def _get_context_vnis(self, stimulus: VNIStimulus, target_vnis: List[str]) -> List[str]:
        """Get context VNIs that might provide relevant associations"""
        # Start with target VNIs as context
        context_vnis = target_vnis.copy()
        
        # Add VNIs that are frequently co-activated with targets
        # This would be enhanced with your VNI association graphs
        association_map = {
            'VNI_medical_001': ['VNI_health_001', 'VNI_bio_001'],
            'VNI_legal_001': ['VNI_compliance_001', 'VNI_ethics_001'],
            'VNI_technical_001': ['VNI_software_001', 'VNI_math_001'],
        }
        
        for vni in target_vnis:
            if vni in association_map:
                context_vnis.extend(association_map[vni])
        
        # Remove duplicates and limit size
        return list(set(context_vnis))[:self.config.max_context_vnis]
    
    def _apply_rl_pattern_to_vni(self, vni_id: str, rl_response: Dict[str, Any], 
                                stimulus: VNIStimulus) -> VNIResponse:
        """Apply RL-guided pattern to actual VNI processing"""
        
        # Check if this is an exploratory pattern
        if rl_response.get('exploration', False):
            # For exploration, let VNI try something new
            return self._exploratory_vni_processing(vni_id, stimulus, rl_response)
        else:
            # For exploitative, use RL-guided approach
            return self._pattern_guided_vni_processing(vni_id, stimulus, rl_response)
    
    def _exploratory_vni_processing(self, vni_id: str, stimulus: VNIStimulus,
                                  rl_response: Dict[str, Any]) -> VNIResponse:
        """Let VNI try exploratory processing"""
        # This encourages novel approaches and discovery
        enhanced_stimulus = VNIStimulus(
            content=stimulus.content,
            stimulus_type=f"exploratory_{stimulus.stimulus_type}",
            metadata={
                **stimulus.metadata,
                'rl_guidance': {
                    'pattern': rl_response['pattern_used'],
                    'exploration': True,
                    'novelty_boost': 0.7
                }
            }
        )
        
        # Process with VNI - it should handle exploratory mode
        response = self.vni_manager.process_stimulus(vni_id, enhanced_stimulus)
        
        # Mark as exploratory for learning
        response.metadata['learning_context'] = {
            'pattern_used': rl_response['pattern_used'],
            'exploration': True,
            'novelty_score': rl_response.get('novelty_score', 0.8)
        }
        
        return response
    
    def _pattern_guided_vni_processing(self, vni_id: str, stimulus: VNIStimulus,
                                     rl_response: Dict[str, Any]) -> VNIResponse:
        """Use RL pattern to guide VNI processing"""
        
        # Enhance stimulus with RL pattern guidance
        guided_stimulus = VNIStimulus(
            content=stimulus.content,
            stimulus_type=stimulus.stimulus_type,
            metadata={
                **stimulus.metadata,
                'rl_guidance': {
                    'pattern': rl_response['pattern_used'],
                    'confidence': rl_response['confidence'],
                    'exploration': False,
                    'pattern_strength': rl_response.get('pattern_strength', 0.5)
                }
            }
        )
        
        # Process with pattern-guided approach
        response = self.vni_manager.process_stimulus(vni_id, guided_stimulus)
        
        # Add learning context
        response.metadata['learning_context'] = {
            'pattern_used': rl_response['pattern_used'],
            'rl_confidence': rl_response['confidence'],
            'exploration': False
        }
        
        return response
    
    def _calculate_reward_value(self, session_context: Dict[str, Any], 
                              feedback_data: Dict[str, Any]) -> float:
        """Calculate reward value based on feedback and performance"""
        
        base_reward = feedback_data.get('quality_score', 0.5)
        
        # Adjust based on response characteristics
        vni_responses = session_context['vni_responses']
        
        performance_factors = []
        for vni_id, response in vni_responses.items():
            # Factor 1: Confidence alignment
            confidence_factor = response.confidence if base_reward > 0.5 else (1 - response.confidence)
            
            # Factor 2: Response complexity (appropriateness)
            complexity = len(str(response.content)) / 1000  # Normalized
            complexity_factor = 1.0 - abs(0.7 - complexity)  # Peak at medium complexity
            
            # Factor 3: Novelty (if exploration was used)
            learning_ctx = response.metadata.get('learning_context', {})
            if learning_ctx.get('exploration', False) and base_reward > 0.6:
                novelty_bonus = learning_ctx.get('novelty_score', 0.5) * 0.3
            else:
                novelty_bonus = 0.0
                
            performance_factors.append(
                (confidence_factor + complexity_factor) / 2 + novelty_bonus
            )
        
        avg_performance = sum(performance_factors) / len(performance_factors) if performance_factors else 0.5
        
        # Combine base reward with performance factors
        final_reward = (base_reward * 0.7) + (avg_performance * 0.3)
        
        # Scale to -1 to 1 range
        return (final_reward - 0.5) * 2
    
    def _calculate_performance_metrics(self, session_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate detailed performance metrics for learning"""
        metrics = {}
        vni_responses = session_context['vni_responses']
        
        for vni_id, response in vni_responses.items():
            metrics[vni_id] = {
                'confidence': response.confidence,
                'response_time': getattr(response, 'processing_time', 0),
                'content_complexity': len(str(response.content)),
                'pattern_effectiveness': 0.5  # Will be updated via RL
            }
        
        return metrics
    
    def _update_performance_metrics(self, session_id: int, reward: float, 
                                  feedback_data: Dict[str, Any]):
        """Update long-term performance tracking"""
        for vni_id, metrics in feedback_data.get('vni_metrics', {}).items():
            self.performance_metrics[vni_id].append({
                'session_id': session_id,
                'reward': reward,
                'timestamp': time.time(),
                'metrics': metrics
            })
            
            # Keep only recent history
            if len(self.performance_metrics[vni_id]) > 100:
                self.performance_metrics[vni_id].pop(0)
    
    def get_learning_state(self) -> Dict[str, Any]:
        """Get complete learning system state"""
        rl_state = self.rl_system.get_system_state()
        
        return {
            'learning_sessions': self.learning_sessions,
            'performance_metrics': {
                vni_id: {
                    'total_sessions': len(sessions),
                    'avg_reward': sum(s['reward'] for s in sessions) / len(sessions) if sessions else 0,
                    'recent_trend': self._calculate_recent_trend(sessions[-10:] if sessions else [])
                }
                for vni_id, sessions in self.performance_metrics.items()
            },
            'rl_system_state': rl_state,
            'pattern_extraction': {
                'total_patterns': sum(len(patterns) for patterns in self.pattern_extractor.response_history.values()),
                'vnis_with_patterns': list(self.pattern_extractor.response_history.keys())
            }
        }
    
    def _calculate_recent_trend(self, recent_sessions: List[Dict]) -> float:
        """Calculate performance trend from recent sessions"""
        if len(recent_sessions) < 2:
            return 0.0
        
        rewards = [s['reward'] for s in recent_sessions]
        # Simple linear trend calculation
        return (rewards[-1] - rewards[0]) / len(rewards)

# Example usage and integration with existing VNI system
def integrate_with_existing_vnis(vni_manager: VNIManager) -> VNILearningOrchestrator:
    """Create and configure learning orchestrator with existing VNI manager"""
    
    # Configure RL for VNI learning
    rl_config = RLConfig(
        learning_rate=0.15,
        exploration_rate=0.25,
        discount_factor=0.85,
        reward_scale=1.5,
        synaptic_decay_rate=0.98
    )
    
    integration_config = VNIRLIntegrationConfig(rl_config=rl_config)
    
    # Create learning orchestrator
    orchestrator = VNILearningOrchestrator(vni_manager, integration_config)
    
    logger.info("VNI-RL integration completed successfully")
    return orchestrator

# Demonstration
def demonstrate_vni_learning_loop():
    """Demonstrate the complete VNI learning loop"""
    
    # This would use your actual VNI manager
    # For demonstration, we'll create a mock
    class MockVNIManager:
        def process_stimulus(self, vni_id: str, stimulus):
            # Mock response
            class MockResponse:
                def __init__(self, content, confidence=0.8):
                    self.content = content
                    self.confidence = confidence
                    self.response_type = "processed"
                    self.metadata = {}
                    self.processing_time = 0.1
                    
            return MockResponse(f"Processed by {vni_id}: {stimulus.content}")
    
    # Create integrated system
    vni_manager = MockVNIManager()
    learning_system = integrate_with_existing_vnis(vni_manager)
    
    print("=== VNI Reinforcement Learning Loop Demonstration ===")
    
    # Create sample stimulus
    class MockStimulus:
        def __init__(self, content, stimulus_type="general"):
            self.content = content
            self.stimulus_type = stimulus_type
            self.metadata = {}
    
    # Process stimulus with learning
    stimulus = MockStimulus(
        "Patient shows symptoms of fever and requires medical attention. "
        "Also, review the treatment consent contract."
    )
    
    response = learning_system.process_stimulus_with_learning(stimulus)
    print(f"\nLearning Session {response['session_id']} completed:")
    
    for vni_id, vni_response in response['vni_responses'].items():
        print(f"  {vni_id}: {vni_response.content} (conf: {vni_response.confidence:.2f})")
    
    # Provide feedback
    feedback = {
        'quality_score': 0.8,  # Good response
        'vni_metrics': {
            'VNI_medical_001': {'relevance': 0.9, 'accuracy': 0.8},
            'VNI_legal_001': {'relevance': 0.7, 'accuracy': 0.6}
        }
    }
    
    learning_system.provide_learning_feedback(response['session_id'], feedback)
    
    # Show learning state
    state = learning_system.get_learning_state()
    print(f"\nLearning System State:")
    print(f"Total Sessions: {state['learning_sessions']}")
    print(f"VNIs with Learning: {list(state['performance_metrics'].keys())}")
    print(f"RL Exploration Rate: {state['rl_system_state']['learning_stats']['exploration_rate']:.3f}")

if __name__ == "__main__":
    demonstrate_vni_learning_loop() 