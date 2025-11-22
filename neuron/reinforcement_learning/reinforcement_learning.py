# reinforcement_learning.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import time
from collections import defaultdict, deque
import random

logger = logging.getLogger("babybionn_rl")

@dataclass
class RLConfig:
    """Configuration for reinforcement learning system"""
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    exploration_rate: float = 0.3
    exploration_decay: float = 0.995
    min_exploration: float = 0.01
    memory_size: int = 10000
    batch_size: int = 32
    reward_scale: float = 1.0
    punishment_scale: float = 1.0
    synaptic_decay_rate: float = 0.99

class SynapticMemory:
    """Tracks synaptic patterns and their reinforcement history"""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.synaptic_strengths = defaultdict(float)  # (vni_id, pattern_id) -> strength
        self.association_strengths = defaultdict(float)  # (vni1, vni2) -> strength
        self.reward_history = deque(maxlen=config.memory_size)
        self.activation_patterns = {}  # vni_id -> list of pattern activations
        
    def record_activation(self, vni_id: str, pattern_id: str, activation_level: float):
        """Record VNI activation pattern"""
        if vni_id not in self.activation_patterns:
            self.activation_patterns[vni_id] = []
        
        self.activation_patterns[vni_id].append({
            'pattern_id': pattern_id,
            'activation_level': activation_level,
            'timestamp': time.time()
        })
        
        # Keep only recent activations
        if len(self.activation_patterns[vni_id]) > 100:
            self.activation_patterns[vni_id].pop(0)
    
    def get_synaptic_strength(self, vni_id: str, pattern_id: str) -> float:
        """Get current synaptic strength for a pattern"""
        return self.synaptic_strengths.get((vni_id, pattern_id), 0.1)  # Default strength
    
    def get_association_strength(self, vni1: str, vni2: str) -> float:
        """Get association strength between two VNIs"""
        return self.association_strengths.get((vni1, vni2), 0.1)  # Default association
    
    def update_synaptic_strength(self, vni_id: str, pattern_id: str, delta: float):
        """Update synaptic strength with reinforcement"""
        key = (vni_id, pattern_id)
        current_strength = self.synaptic_strengths.get(key, 0.1)
        new_strength = current_strength + delta
        self.synaptic_strengths[key] = max(0.0, min(1.0, new_strength))  # Clamp to [0,1]
    
    def update_association_strength(self, vni1: str, vni2: str, delta: float):
        """Update association strength between VNIs"""
        key = (vni1, vni2)
        current_strength = self.association_strengths.get(key, 0.1)
        new_strength = current_strength + delta
        self.association_strengths[key] = max(0.0, min(1.0, new_strength))
    
    def apply_synaptic_decay(self):
        """Apply forgetting mechanism to prevent saturation"""
        for key in list(self.synaptic_strengths.keys()):
            self.synaptic_strengths[key] *= self.config.synaptic_decay_rate
            if self.synaptic_strengths[key] < 0.01:  # Remove very weak synapses
                del self.synaptic_strengths[key]
        
        for key in list(self.association_strengths.keys()):
            self.association_strengths[key] *= self.config.synaptic_decay_rate
            if self.association_strengths[key] < 0.01:
                del self.association_strengths[key]

class RewardSignal:
    """Manages reward/punishment signals and their propagation"""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.reward_queue = deque()
        self.punishment_queue = deque()
        
    def add_reward(self, reward_value: float, target_vnis: List[str] = None, 
                  context: Dict[str, Any] = None):
        """Add a positive reward signal"""
        self.reward_queue.append({
            'value': reward_value * self.config.reward_scale,
            'target_vnis': target_vnis,
            'context': context,
            'timestamp': time.time(),
            'type': 'reward'
        })
    
    def add_punishment(self, punishment_value: float, target_vnis: List[str] = None,
                      context: Dict[str, Any] = None):
        """Add a negative reward (punishment) signal"""
        self.punishment_queue.append({
            'value': punishment_value * self.config.punishment_scale * -1,
            'target_vnis': target_vnis,
            'context': context,
            'timestamp': time.time(),
            'type': 'punishment'
        })
    
    def get_pending_signals(self) -> List[Dict[str, Any]]:
        """Get all pending reward/punishment signals"""
        signals = []
        while self.reward_queue:
            signals.append(self.reward_queue.popleft())
        while self.punishment_queue:
            signals.append(self.punishment_queue.popleft())
        return signals

class VNIReinforcementEngine:
    """Core engine that applies reinforcement learning to VNIs"""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.synaptic_memory = SynapticMemory(config)
        self.reward_signal = RewardSignal(config)
        self.exploration_rate = config.exploration_rate
        
        # Neural network for value estimation
        self.value_estimator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()  # Output between -1 and 1
        )
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=config.memory_size)
        
    def stimulate_vni(self, vni_id: str, input_data: Dict[str, Any], 
                     context_vnis: List[str] = None) -> Dict[str, Any]:
        """Stimulate a VNI and get its response with exploration"""
        
        # Apply exploration: sometimes choose random pattern
        if random.random() < self.exploration_rate:
            return self._exploratory_response(vni_id, input_data)
        
        # Otherwise use learned patterns
        return self._exploitative_response(vni_id, input_data, context_vnis)
    
    def _exploratory_response(self, vni_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate exploratory response by trying new patterns"""
        # This encourages the system to discover new solutions
        exploratory_pattern = f"exploratory_{int(time.time())}_{random.randint(1000,9999)}"
        
        return {
            'vni_id': vni_id,
            'pattern_used': exploratory_pattern,
            'response': self._generate_exploratory_output(input_data),
            'confidence': 0.3,  # Low confidence for exploration
            'exploration': True
        }
    
    def _exploitative_response(self, vni_id: str, input_data: Dict[str, Any],
                             context_vnis: List[str] = None) -> Dict[str, Any]:
        """Generate response using learned patterns"""
        
        # Get strongest patterns for this VNI
        patterns = self._get_strongest_patterns(vni_id, limit=5)
        
        if not patterns:
            return self._exploratory_response(vni_id, input_data)
        
        # Select pattern based on strength and context
        selected_pattern = self._select_pattern_by_context(patterns, input_data, context_vnis)
        
        # Record activation
        self.synaptic_memory.record_activation(vni_id, selected_pattern['pattern_id'], 1.0)
        
        return {
            'vni_id': vni_id,
            'pattern_used': selected_pattern['pattern_id'],
            'response': self._generate_pattern_based_output(selected_pattern, input_data),
            'confidence': selected_pattern['strength'],
            'exploration': False
        }
    
    def _get_strongest_patterns(self, vni_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get strongest synaptic patterns for a VNI"""
        patterns = []
        for (vni, pattern_id), strength in self.synaptic_memory.synaptic_strengths.items():
            if vni == vni_id:
                patterns.append({
                    'pattern_id': pattern_id,
                    'strength': strength,
                    'vni_id': vni_id
                })
        
        patterns.sort(key=lambda x: x['strength'], reverse=True)
        return patterns[:limit]
    
    def _select_pattern_by_context(self, patterns: List[Dict[str, Any]], 
                                 input_data: Dict[str, Any], context_vnis: List[str]) -> Dict[str, Any]:
        """Select pattern considering context and associations"""
        if not context_vnis:
            return patterns[0]  # Just use strongest pattern
        
        # Calculate context-aware scores
        scored_patterns = []
        for pattern in patterns:
            score = pattern['strength']
            
            # Boost score if pattern has strong associations with context VNIs
            for context_vni in context_vnis:
                association_strength = self.synaptic_memory.get_association_strength(
                    pattern['vni_id'], context_vni
                )
                score += association_strength * 0.2  # Association boost
            
            scored_patterns.append((score, pattern))
        
        # Select pattern with highest context-aware score
        scored_patterns.sort(key=lambda x: x[0], reverse=True)
        return scored_patterns[0][1]
    
    def apply_reinforcement(self, response_result: Dict[str, Any], 
                          reward_value: float, stimulus_context: Dict[str, Any]):
        """Apply reinforcement learning to strengthen/weaken synaptic patterns"""
        
        vni_id = response_result['vni_id']
        pattern_used = response_result['pattern_used']
        
        # Calculate reinforcement delta
        base_delta = reward_value * self.config.learning_rate
        
        # Apply to synaptic strength
        self.synaptic_memory.update_synaptic_strength(vni_id, pattern_used, base_delta)
        
        # Also reinforce associations with context VNIs
        context_vnis = stimulus_context.get('active_vnis', [])
        for context_vni in context_vnis:
            if context_vni != vni_id:
                association_delta = base_delta * 0.5  # Smaller association reinforcement
                self.synaptic_memory.update_association_strength(vni_id, context_vni, association_delta)
        
        # Store experience for later learning
        experience = {
            'vni_id': vni_id,
            'pattern_used': pattern_used,
            'reward': reward_value,
            'context': stimulus_context,
            'timestamp': time.time()
        }
        self.experience_buffer.append(experience)
        
        # Update exploration rate
        self._update_exploration_rate()
        
        logger.info(f"Applied reinforcement: {reward_value:.3f} to {vni_id}:{pattern_used}")
    
    def _update_exploration_rate(self):
        """Gradually reduce exploration rate"""
        self.exploration_rate = max(
            self.config.min_exploration,
            self.exploration_rate * self.config.exploration_decay
        )
    
    def process_reward_signals(self):
        """Process all pending reward/punishment signals"""
        signals = self.reward_signal.get_pending_signals()
        
        for signal in signals:
            # Find recent activations that match the signal context
            recent_activations = self._find_relevant_activations(signal)
            
            for activation in recent_activations:
                self.apply_reinforcement(activation, signal['value'], signal.get('context', {}))
    
    def _find_relevant_activations(self, signal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find VNI activations relevant to a reward signal"""
        relevant_activations = []
        target_vnis = signal.get('target_vnis')
        
        if target_vnis:
            # Specific VNIs targeted
            for vni_id in target_vnis:
                if vni_id in self.synaptic_memory.activation_patterns:
                    recent_activations = self.synaptic_memory.activation_patterns[vni_id][-5:]  # Last 5
                    for activation in recent_activations:
                        relevant_activations.append({
                            'vni_id': vni_id,
                            'pattern_used': activation['pattern_id'],
                            'activation_level': activation['activation_level']
                        })
        else:
            # No specific targets - use all recent activations
            for vni_id, activations in self.synaptic_memory.activation_patterns.items():
                recent_activations = activations[-3:]  # Last 3 per VNI
                for activation in recent_activations:
                    relevant_activations.append({
                        'vni_id': vni_id,
                        'pattern_used': activation['pattern_id'],
                        'activation_level': activation['activation_level']
                    })
        
        return relevant_activations
    
    def _generate_exploratory_output(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate output for exploratory responses"""
        return {
            'type': 'exploratory',
            'content': 'Exploring new pattern...',
            'confidence': 0.3,
            'novelty_score': random.uniform(0.7, 0.9)
        }
    
    def _generate_pattern_based_output(self, pattern: Dict[str, Any], 
                                    input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate output based on learned pattern"""
        return {
            'type': 'pattern_based',
            'content': f"Response based on pattern: {pattern['pattern_id']}",
            'confidence': pattern['strength'],
            'pattern_strength': pattern['strength'],
            'novelty_score': 0.1  # Low novelty for established patterns
        }
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the learning system"""
        total_synapses = len(self.synaptic_memory.synaptic_strengths)
        total_associations = len(self.synaptic_memory.association_strengths)
        
        avg_synaptic_strength = np.mean(list(self.synaptic_memory.synaptic_strengths.values())) if total_synapses > 0 else 0
        avg_association_strength = np.mean(list(self.synaptic_memory.association_strengths.values())) if total_associations > 0 else 0
        
        return {
            'total_synapses': total_synapses,
            'total_associations': total_associations,
            'avg_synaptic_strength': avg_synaptic_strength,
            'avg_association_strength': avg_association_strength,
            'exploration_rate': self.exploration_rate,
            'experience_buffer_size': len(self.experience_buffer),
            'reward_history_size': len(self.synaptic_memory.reward_history)
        }

class BabyBIONNReinforcementSystem:
    """Main reinforcement learning system for BabyBIONN"""
    
    def __init__(self, config: RLConfig = None):
        self.config = config or RLConfig()
        self.rl_engine = VNIReinforcementEngine(self.config)
        
        # Track VNI performance
        self.vni_performance = defaultdict(list)
        self.learning_sessions = 0
        
        logger.info("BabyBIONN Reinforcement System initialized")
    
    def stimulate_system(self, stimulus: Dict[str, Any]) -> Dict[str, Any]:
        """Present a stimulus to the system and get response"""
        
        target_vnis = stimulus.get('target_vnis', [])
        context_vnis = stimulus.get('context_vnis', [])
        input_data = stimulus.get('input_data', {})
        
        responses = {}
        
        # Stimulate each target VNI
        for vni_id in target_vnis:
            response = self.rl_engine.stimulate_vni(vni_id, input_data, context_vnis)
            responses[vni_id] = response
        
        # Record stimulus context
        stimulus_context = {
            'target_vnis': target_vnis,
            'context_vnis': context_vnis,
            'input_data': input_data,
            'timestamp': time.time(),
            'session_id': self.learning_sessions
        }
        
        return {
            'responses': responses,
            'stimulus_context': stimulus_context,
            'session_id': self.learning_sessions
        }
    
    def provide_feedback(self, session_id: int, reward_value: float, 
                        feedback_context: Dict[str, Any] = None):
        """Provide feedback for a learning session"""
        
        # Find the session context (in real implementation, we'd store sessions)
        # For now, we'll apply to recent activations
        
        self.rl_engine.reward_signal.add_reward(
            reward_value, 
            feedback_context.get('target_vnis'),
            feedback_context
        )
        
        # Process the reward signal
        self.rl_engine.process_reward_signals()
        
        # Apply synaptic decay periodically
        if self.learning_sessions % 10 == 0:
            self.rl_engine.synaptic_memory.apply_synaptic_decay()
        
        self.learning_sessions += 1
        logger.info(f"Feedback provided: {reward_value:.3f} for session {session_id}")
    
    def punish_system(self, session_id: int, punishment_value: float,
                     punishment_context: Dict[str, Any] = None):
        """Provide punishment feedback"""
        
        self.rl_engine.reward_signal.add_punishment(
            punishment_value,
            punishment_context.get('target_vnis'),
            punishment_context
        )
        
        self.rl_engine.process_reward_signals()
        self.learning_sessions += 1
        logger.info(f"Punishment applied: {punishment_value:.3f} for session {session_id}")
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get complete system state"""
        learning_stats = self.rl_engine.get_learning_stats()
        
        return {
            'learning_sessions': self.learning_sessions,
            'learning_stats': learning_stats,
            'vni_performance': dict(self.vni_performance),
            'config': {
                'learning_rate': self.config.learning_rate,
                'exploration_rate': self.rl_engine.exploration_rate,
                'discount_factor': self.config.discount_factor
            }
        }

# Demonstration and testing
def test_reinforcement_system():
    """Test the reinforcement learning system"""
    
    config = RLConfig(
        learning_rate=0.2,
        exploration_rate=0.4,
        reward_scale=2.0,
        punishment_scale=1.5
    )
    
    rl_system = BabyBIONNReinforcementSystem(config)
    
    print("=== BabyBIONN Reinforcement Learning System Test ===\n")
    
    # Test stimulus 1
    stimulus_1 = {
        'target_vnis': ['VNI_medical_001', 'VNI_legal_001'],
        'context_vnis': ['VNI_base_001'],
        'input_data': {'text': 'Patient has fever and contract needs review'},
        'session_type': 'multi_domain'
    }
    
    response_1 = rl_system.stimulate_system(stimulus_1)
    print("Stimulus 1 Response:")
    for vni_id, response in response_1['responses'].items():
        print(f"  {vni_id}: {response['pattern_used']} (conf: {response['confidence']:.2f})")
    
    # Provide positive feedback
    rl_system.provide_feedback(
        response_1['session_id'], 
        0.8,  # Good response
        {'target_vnis': ['VNI_medical_001', 'VNI_legal_001']}
    )
    
    # Test stimulus 2
    stimulus_2 = {
        'target_vnis': ['VNI_technical_001'],
        'input_data': {'text': 'System performance issue'},
        'session_type': 'technical'
    }
    
    response_2 = rl_system.stimulate_system(stimulus_2)
    print("\nStimulus 2 Response:")
    for vni_id, response in response_2['responses'].items():
        print(f"  {vni_id}: {response['pattern_used']} (conf: {response['confidence']:.2f})")
    
    # Provide negative feedback
    rl_system.punish_system(
        response_2['session_id'],
        -0.6,  # Poor response
        {'target_vnis': ['VNI_technical_001']}
    )
    
    # Show system state
    state = rl_system.get_system_state()
    print(f"\nSystem State:")
    print(f"Learning Sessions: {state['learning_sessions']}")
    print(f"Total Synapses: {state['learning_stats']['total_synapses']}")
    print(f"Exploration Rate: {state['learning_stats']['exploration_rate']:.3f}")
    
    return rl_system

if __name__ == "__main__":
    test_reinforcement_system() 