# neuron/aggregator_attn.py
"""Simple attention helper for aggregator - uses existing demoHybridAttention"""
import logging
import torch
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class AggregatorAttention:
    """
    Simple attention helper for aggregator.
    Uses demoHybridAttention to focus on and learn from VNI outputs.
    """
    
    def __init__(self, dim: int = 256, num_heads: int = 8):
        # Lazy import to avoid circular dependencies
        from neuron.demoHybridAttention import DemoHybridAttention
        
        self.attention = DemoHybridAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=128,
            use_sliding=True,
            use_global=True,
            use_hierarchical=False  # Keep it simple
        )
        self.attention_history = []
        
    def analyze_outputs(self, vni_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze multiple VNI outputs and return attention weights.
        This is what the aggregator uses to focus on important parts.
        """
        if not vni_outputs:
            return {'attention_weights': {}, 'focus': 'none', 'confidence': 0.0}
        
        # Convert outputs to simple feature vectors for attention
        features = self._outputs_to_features(vni_outputs)
        
        # Let the attention mechanism do its magic
        with torch.no_grad():
            # Self-attention on the features
            attended, weights = self.attention(
                query=features,
                key=features,
                value=features
            )
        
        # Extract attention weights per VNI
        attention_weights = {}
        vni_list = list(vni_outputs.keys())
        
        for i, vni_id in enumerate(vni_list):
            if i < weights.shape[-1]:
                # Average attention this VNI received from all others
                attention_weights[vni_id] = float(weights[0, :, i].mean().item())
        
        # Normalize weights
        total = sum(attention_weights.values())
        if total > 0:
            attention_weights = {k: v/total for k, v in attention_weights.items()}
        
        # Determine primary focus
        if attention_weights:
            primary_focus = max(attention_weights.items(), key=lambda x: x[1])[0]
        else:
            primary_focus = 'none'
        
        result = {
            'attention_weights': attention_weights,
            'primary_focus': primary_focus,
            'vni_count': len(vni_outputs),
            'confidence': float(weights.mean().item()) if weights.numel() > 0 else 0.5,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store for learning
        self.attention_history.append({
            'vni_count': len(vni_outputs),
            'primary_focus': primary_focus,
            'weights': attention_weights,
            'timestamp': result['timestamp']
        })
        
        # Keep history manageable
        if len(self.attention_history) > 100:
            self.attention_history = self.attention_history[-100:]
        
        return result
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from attention history for Hebbian learning"""
        if not self.attention_history:
            return {'patterns': [], 'collaborations': {}}
        
        # Track which VNIs often get high attention together
        collaborations = {}
        for record in self.attention_history[-50:]:  # Last 50 interactions
            weights = record.get('weights', {})
            high_attention = [vni for vni, w in weights.items() if w > 0.3]
            
            for i, v1 in enumerate(high_attention):
                for v2 in high_attention[i+1:]:
                    key = tuple(sorted([v1, v2]))
                    collaborations[key] = collaborations.get(key, 0) + 1
        
        # Find patterns
        patterns = []
        for (v1, v2), count in collaborations.items():
            if count > 5:  # Appeared together frequently
                patterns.append({
                    'vni_pair': [v1, v2],
                    'collaboration_strength': count / 50,
                    'type': 'frequent_collaboration'
                })
        
        return {
            'patterns': patterns,
            'collaborations': collaborations,
            'total_interactions': len(self.attention_history)
        }
    
    def _outputs_to_features(self, outputs: Dict[str, Dict]) -> torch.Tensor:
        """Convert VNI outputs to feature tensors for attention - with proper dimension"""
        features = []
        
        # Target dimension for the attention mechanism (should match DemoHybridAttention.dim)
        target_dim = 256  # This should match the dim parameter in AggregatorAttention __init__
        
        for vni_id, output in outputs.items():
            # Create base features
            confidence = output.get('confidence_score', 0.5)
            has_generation = 1.0 if 'generation_data' in output else 0.0
            
            # Extract text length if available
            text_length = 0.0
            if 'generation_data' in output:
                query = output['generation_data'].get('query', '')
                text_length = min(1.0, len(query) / 500)
            
            # Domain indicator
            domain = output.get('domain', 'general')
            domain_vec = [1.0 if domain == d else 0.0 for d in ['medical', 'legal', 'technical', 'general']]
            
            # Combine base features (7 features)
            base_feat = [confidence, has_generation, text_length] + domain_vec
            
            # Expand to target_dim by repeating and adding noise/variance
            # This ensures we have enough dimensions for the attention mechanism
            repeats = target_dim // len(base_feat) + 1
            expanded = (base_feat * repeats)[:target_dim]
            
            # Add small random noise to create variance between features
            # This helps the attention mechanism distinguish between different VNIs
            noise = torch.randn(target_dim) * 0.01
            final_feat = torch.tensor(expanded, dtype=torch.float32) + noise
            
            features.append(final_feat)
        
        # Stack all features
        if features:
            feature_tensor = torch.stack(features).unsqueeze(0)  # [1, num_vnis, target_dim]
        else:
            feature_tensor = torch.zeros(1, 1, target_dim)
        
        return feature_tensor
