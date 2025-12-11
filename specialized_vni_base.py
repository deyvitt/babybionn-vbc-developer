# specialized_vni_base.py
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class SpecializedVNIState:
    """State for tracking dynamic adaptation"""
    topic: str
    usage_count: int = 0
    success_rate: float = 0.5
    learned_patterns: List[Dict] = None
    adaptation_weights: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        if self.learned_patterns is None:
            self.learned_patterns = []

class SpecializedVNIBase(nn.Module):
    """Base class for all specialized VNIs with dynamic adaptation capabilities"""
    
    def __init__(self, topic_name: str, config: Dict = None):
        super().__init__()
        self.topic_name = topic_name
        self.config = config or {}
        
        # State tracking for dynamic learning
        self.state = SpecializedVNIState(topic=topic_name)
        
        # Dynamic adaptation network (can be extended by subclasses)
        self.dynamic_adapter = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.Tanh()
        )
        
        # Pattern memory for learned adaptations
        self.pattern_memory = []
        
        # Effectiveness metrics
        self.adaptation_strength = 0.3  # How much to adapt (0-1)
    
    def forward(self, base_features: Dict, input_data: Any) -> Dict:
        """Main processing method - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement forward method")
    
    def apply_dynamic_adaptation(self, base_result: Dict, context: Dict) -> Dict:
        """Apply learned dynamic adaptations to the base result"""
        if self.adaptation_strength < 0.1 or not self.pattern_memory:
            return base_result
        
        # Extract relevant features for adaptation
        adaptation_features = self.extract_adaptation_features(base_result, context)
        
        if adaptation_features is None:
            return base_result
        
        # Apply dynamic adaptation
        with torch.no_grad():
            adaptation = self.dynamic_adapter(adaptation_features)
        
        # Blend base result with adaptation
        adapted_result = self.blend_results(base_result, adaptation)
        
        return adapted_result
    
    def extract_adaptation_features(self, base_result: Dict, context: Dict) -> Optional[torch.Tensor]:
        """Extract features for dynamic adaptation"""
        # Default implementation - can be overridden
        if 'semantic_features' in base_result:
            return base_result['semantic_features']
        elif 'tensor' in base_result:
            return base_result['tensor']
        return None
    
    def blend_results(self, base_result: Dict, adaptation: torch.Tensor) -> Dict:
        """Blend base result with dynamic adaptation"""
        # Simple blending - can be made more sophisticated
        adapted_result = base_result.copy()
        
        if 'tensor' in base_result:
            # Blend tensors
            alpha = self.adaptation_strength
            adapted_result['tensor'] = (1 - alpha) * base_result['tensor'] + alpha * adaptation
        
        # Mark as adapted
        adapted_result['dynamic_adaptation_applied'] = True
        adapted_result['adaptation_strength'] = self.adaptation_strength
        
        return adapted_result
    
    def learn_from_interaction(self, input_data: Any, result: Dict, success_metric: float):
        """Learn from interaction to improve future adaptations"""
        self.state.usage_count += 1
        
        # Update success rate
        current_rate = self.state.success_rate
        learning_rate = 0.1
        self.state.success_rate = (1 - learning_rate) * current_rate + learning_rate * success_metric
        
        # If successful, store pattern for reinforcement
        if success_metric > 0.7:
            pattern = {
                'input_pattern': self.extract_pattern(input_data),
                'result_pattern': self.extract_pattern(result),
                'success_metric': success_metric,
                'context': self.get_context_snapshot()
            }
            self.pattern_memory.append(pattern)
            
            # Keep only recent patterns
            if len(self.pattern_memory) > 50:
                self.pattern_memory.pop(0)
            
            # Gradually increase adaptation strength if successful
            self.adaptation_strength = min(0.8, self.adaptation_strength + 0.02)
        
        elif success_metric < 0.3:
            # Decrease adaptation strength if poor result
            self.adaptation_strength = max(0.1, self.adaptation_strength - 0.05)
    
    def extract_pattern(self, data: Any) -> Dict:
        """Extract reusable pattern from data"""
        if isinstance(data, torch.Tensor):
            return {'type': 'tensor', 'shape': list(data.shape)}
        elif isinstance(data, dict):
            return {'type': 'dict', 'keys': list(data.keys())}
        elif isinstance(data, str):
            return {'type': 'text', 'length': len(data)}
        else:
            return {'type': str(type(data))}
    
    def get_context_snapshot(self) -> Dict:
        """Get current context snapshot for learning"""
        return {
            'topic': self.topic_name,
            'adaptation_strength': self.adaptation_strength,
            'usage_count': self.state.usage_count,
            'success_rate': self.state.success_rate
        }
    
    def save_state(self, path: str):
        """Save learned state to disk"""
        state_dict = {
            'topic': self.topic_name,
            'state': self.state.__dict__,
            'pattern_memory': self.pattern_memory,
            'adaptation_strength': self.adaptation_strength
        }
        
        with open(path, 'w') as f:
            json.dump(state_dict, f, indent=2)
    
    def load_state(self, path: str):
        """Load learned state from disk"""
        with open(path, 'r') as f:
            state_dict = json.load(f)
        
        self.state = SpecializedVNIState(**state_dict['state'])
        self.pattern_memory = state_dict['pattern_memory']
        self.adaptation_strength = state_dict['adaptation_strength']
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return capabilities of this specialized VNI"""
        return {
            'topic': self.topic_name,
            'has_dynamic_adaptation': True,
            'adaptation_strength': self.adaptation_strength,
            'learned_patterns': len(self.pattern_memory),
            'usage_count': self.state.usage_count,
            'success_rate': self.state.success_rate
        } 
