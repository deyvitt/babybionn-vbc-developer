# neuron/demoHybridAttention.py - Simplified hybrid attention for BabyBIONN demo
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict, Any
# from neurosynaptic_tensor_encoder import NeurosynapticEventEncoder

class DemoHybridAttention(nn.Module):
    """
    Simplified hybrid attention for BabyBIONN demo
    - Keeps: hierarchical, sliding window, memory tokens, multi-modal fusion
    - Removes: quantum components, content-aware selection, C++ extensions
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 256,
        use_sliding: bool = True,
        use_global: bool = True,
        use_hierarchical: bool = True,
        global_token_ratio: float = 0.05,
        memory_tokens: int = 16,
        multi_modal: bool = True,
    ):
        super().__init__()
        # Validate parameters
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        
        # Feature flags
        self.use_sliding = use_sliding
        self.use_global = use_global
        self.use_hierarchical = use_hierarchical
        self.multi_modal = multi_modal
        
        # Configuration parameters
        self.global_token_ratio = global_token_ratio
        self.memory_tokens = memory_tokens
        
        # Persistent memory (simplified)
        self.persistent_memory = nn.Parameter(torch.zeros(1, memory_tokens, dim))
        nn.init.normal_(self.persistent_memory, mean=0.0, std=0.02)
        
        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Multi-modal support (keep for demo)
        if multi_modal:
            self.modal_gate = nn.Linear(dim, 1)
            self.modal_projection = nn.Linear(dim * 2, dim)

    def _create_sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create sliding window attention mask with memory tokens"""
        total_len = seq_len + self.memory_tokens
        mask = torch.full((total_len, total_len), float('-inf'), device=device)
        
        # Memory tokens attend to everything and are attended by everything
        mask[:self.memory_tokens, :] = 0
        mask[:, :self.memory_tokens] = 0
        
        # Sliding window attention for content tokens
        for i in range(self.memory_tokens, total_len):
            start = max(self.memory_tokens, i - self.window_size // 2)
            end = min(total_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 0
                
        return mask

    def _select_global_tokens(self, key_layer: torch.Tensor, ratio: float = None) -> torch.Tensor:
        """Select global tokens based on importance scoring (simplified)"""
        if ratio is None:
            ratio = self.global_token_ratio
            
        # Skip memory tokens
        content_keys = key_layer[:, self.memory_tokens:, :]
        seq_len = content_keys.size(1)
        num_global_tokens = max(1, int(seq_len * ratio))
        
        # Simple scoring by magnitude
        scores = torch.norm(content_keys, dim=-1).mean(dim=0)  # [seq]
        
        # Select top-k
        _, indices = torch.topk(scores, k=min(num_global_tokens, seq_len))
        return indices

    def _core_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                       mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard multi-head attention without quantum components"""
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
            
        # Compute attention weights
        weights = torch.softmax(scores, dim=-1)
        
        # Apply to values
        context = torch.matmul(weights, v)
        
        return context, weights

    def _apply_memory_augmented_attention(self, query: torch.Tensor, key: torch.Tensor, 
                                        value: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Memory-augmented attention with sliding window"""
        batch_size, seq_len, _ = query.size()
        
        # Expand memory tokens to batch size
        memory = self.persistent_memory.expand(batch_size, -1, -1)
        
        # Prepend memory tokens to sequence
        query_with_memory = torch.cat([memory, query], dim=1)
        key_with_memory = torch.cat([memory, key], dim=1)
        value_with_memory = torch.cat([memory, value], dim=1)
        
        # Project queries, keys, values
        q = self.q_proj(query_with_memory)
        k = self.k_proj(key_with_memory)
        v = self.v_proj(value_with_memory)
        
        # Reshape for multi-head attention: [batch, heads, seq, head_dim]
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Create attention mask
        full_seq_len = seq_len + self.memory_tokens
        if attention_mask is None:
            mask = self._create_sliding_window_mask(seq_len, query.device)
            mask = mask.unsqueeze(0).unsqueeze(1)  # [1, 1, full_seq, full_seq]
        else:
            # Expand provided mask to include memory tokens
            mask = torch.full(
                (batch_size, full_seq_len, full_seq_len), 
                float('-inf'), 
                device=query.device
            )
            mask[:, self.memory_tokens:, self.memory_tokens:] = attention_mask
            mask = mask.unsqueeze(1)  # [batch, 1, full_seq, full_seq]
            
        # Compute attention
        context, _ = self._core_attention(q, k, v, mask)
        
        # Reshape back: [batch, seq, dim]
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, full_seq_len, self.dim)
        
        # Remove memory tokens from output
        context = context[:, self.memory_tokens:]
        
        # Final projection
        return self.out_proj(context)

    def _apply_hierarchical_attention(self, query: torch.Tensor, key: torch.Tensor, 
                                    value: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Hierarchical attention for long sequences"""
        batch_size, seq_len, _ = query.size()
        chunk_size = min(512, seq_len)  # Adjustable chunk size
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        
        chunk_outputs = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, seq_len)
            
            # Extract chunk
            q_chunk = query[:, start_idx:end_idx, :]
            k_chunk = key[:, start_idx:end_idx, :]
            v_chunk = value[:, start_idx:end_idx, :]
            
            # Project
            q_proj = self.q_proj(q_chunk)
            k_proj = self.k_proj(k_chunk)
            v_proj = self.v_proj(v_chunk)
            
            # Reshape for multi-head attention
            q_proj = q_proj.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k_proj = k_proj.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v_proj = v_proj.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Apply chunk-specific mask if provided
            chunk_mask = None
            if attention_mask is not None:
                chunk_mask = attention_mask[:, start_idx:end_idx, start_idx:end_idx]
                chunk_mask = chunk_mask.unsqueeze(1)  # [batch, 1, chunk_seq, chunk_seq]
            
            # Compute attention for this chunk
            context, _ = self._core_attention(q_proj, k_proj, v_proj, chunk_mask)
            
            # Reshape back
            context = context.transpose(1, 2).contiguous()
            context = context.view(batch_size, -1, self.dim)
            chunk_outputs.append(context)
            
        # Concatenate all chunks
        output = torch.cat(chunk_outputs, dim=1)
        
        # Final projection
        return self.out_proj(output)

    def _apply_standard_attention(self, query: torch.Tensor, key: torch.Tensor, 
                                value: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard multi-head attention for short sequences"""
        batch_size, seq_len, _ = query.size()
        
        # Project
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply mask if provided
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask.unsqueeze(1)  # [batch, 1, seq, seq]
        
        # Compute attention
        context, _ = self._core_attention(q, k, v, attn_mask)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous()
        output = self.out_proj(context.view(batch_size, seq_len, self.dim))
        
        return output

    def _fuse_multi_modal(self, text_output: torch.Tensor, visual_output: torch.Tensor) -> torch.Tensor:
        """Fuse text and visual modalities with gating mechanism"""
        batch_size, seq_len, dim = text_output.shape
        
        # Calculate modality gating weights
        text_gate = torch.sigmoid(self.modal_gate(text_output))  # [batch, seq, 1]
        visual_gate = torch.sigmoid(self.modal_gate(visual_output))  # [batch, seq, 1]
        
        # Normalize gates to sum to 1
        total_gate = text_gate + visual_gate + 1e-8
        text_gate = text_gate / total_gate
        visual_gate = visual_gate / total_gate
        
        # Apply gating and combine features
        text_weighted = text_output * text_gate
        visual_weighted = visual_output * visual_gate
        
        # Concatenate and project to original dimension
        combined = torch.cat([text_weighted, visual_weighted], dim=-1)
        return self.modal_projection(combined)

    def _select_attention_strategy(self, seq_len: int) -> str:
        """Simple strategy selection based on sequence length"""
        if self.use_hierarchical and seq_len > 2048:
            return "hierarchical"
        elif self.use_sliding and seq_len > self.window_size:
            return "sliding"
        else:
            return "standard"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        visual_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with automatic strategy selection
        
        Args:
            query: [batch_size, seq_len, dim]
            key: [batch_size, seq_len, dim] 
            value: [batch_size, seq_len, dim]
            attention_mask: Optional [batch_size, seq_len, seq_len]
            visual_input: Optional [batch_size, seq_len, dim] for multi-modal
            
        Returns:
            output: [batch_size, seq_len, dim]
            attention_weights: [batch_size, num_heads, seq_len, seq_len] (placeholder)
        """
        batch_size, seq_len, _ = query.size()
        
        # Select attention strategy based on sequence length
        strategy = self._select_attention_strategy(seq_len)
        
        # Apply selected attention strategy
        if strategy == "hierarchical":
            output = self._apply_hierarchical_attention(query, key, value, attention_mask)
        elif strategy == "sliding":
            output = self._apply_memory_augmented_attention(query, key, value, attention_mask)
        else:  # standard
            output = self._apply_standard_attention(query, key, value, attention_mask)
        
        # Multi-modal fusion if visual input provided
        if self.multi_modal and visual_input is not None:
            # Ensure visual_input has same sequence length (through padding/truncation if needed)
            if visual_input.size(1) != seq_len:
                # Simple handling: take first seq_len tokens or pad with zeros
                if visual_input.size(1) > seq_len:
                    visual_input = visual_input[:, :seq_len, :]
                else:
                    padding = torch.zeros(batch_size, seq_len - visual_input.size(1), 
                                        self.dim, device=visual_input.device)
                    visual_input = torch.cat([visual_input, padding], dim=1)
            
            output = self._fuse_multi_modal(output, visual_input)
        
        # Return attention weights (placeholder - can implement if needed for visualization)
        attention_weights = torch.zeros(
            batch_size, self.num_heads, seq_len, seq_len, 
            device=query.device
        )
        
        return output, attention_weights

    def get_attention_info(self) -> Dict[str, Any]:
        """Get information about current attention configuration"""
        return {
            "dim": self.dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "window_size": self.window_size,
            "memory_tokens": self.memory_tokens,
            "use_hierarchical": self.use_hierarchical,
            "use_sliding": self.use_sliding,
            "multi_modal": self.multi_modal
        }
    
# neurosyntaptic_tensor_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import math

class NeurosynapticEventEncoder(nn.Module):
    """
    Converts neurosynaptic events/time-series into Q/K/V tensors
    for hybrid attention processing
    """
    
    def __init__(
        self,
        input_dim: int = 256,  # Number of input features per event
        hidden_dim: int = 512,  # Dimension for Q/K/V projections
        num_heads: int = 8,
        max_sequence_length: int = 1024,
        temporal_window: int = 100,  # ms window for event grouping
        use_relative_position: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.temporal_window = temporal_window
        self.max_sequence_length = max_sequence_length
        
        # Event feature extraction
        self.event_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Temporal encoding (for event timing)
        self.temporal_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )
        
        # Relative position encoding for temporal relationships
        if use_relative_position:
            self.relative_position_bias = nn.Parameter(
                torch.zeros(2 * max_sequence_length - 1, num_heads)
            )
        
        # Q/K/V projections specifically for neurosynaptic data
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Spike rate normalization
        self.spike_rate_norm = nn.LayerNorm(hidden_dim)
        
        # Dynamic connectivity projection
        self.connectivity_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        print(f"Neurosynaptic Event Encoder initialized: {input_dim} -> {hidden_dim}")

    def _create_temporal_encoding(self, timestamps: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Create temporal encoding for event timing"""
        # Normalize timestamps to [0, 1] range
        if timestamps.max() > timestamps.min():
            normalized_times = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
        else:
            normalized_times = torch.zeros_like(timestamps)
        
        # Create sinusoidal temporal encoding
        positions = torch.arange(0, seq_len, dtype=torch.float, device=timestamps.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2, device=timestamps.device).float() * 
                           -(math.log(10000.0) / self.hidden_dim))
        
        temporal_encoding = torch.zeros(seq_len, self.hidden_dim, device=timestamps.device)
        temporal_encoding[:, 0::2] = torch.sin(positions * div_term)
        temporal_encoding[:, 1::2] = torch.cos(positions * div_term)
        
        return temporal_encoding.unsqueeze(0)  # [1, seq_len, hidden_dim]

    def _group_events_by_window(
        self, 
        events: Dict[str, torch.Tensor], 
        window_size: int = None
    ) -> Dict[str, torch.Tensor]:
        """Group neurosynaptic events into temporal windows"""
        if window_size is None:
            window_size = self.temporal_window
        
        # Extract event data
        timestamps = events.get('timestamps', torch.tensor([0.0]))
        features = events.get('features', torch.randn(1, self.input_dim))
        neuron_ids = events.get('neuron_ids', torch.tensor([0]))
        spike_amplitudes = events.get('amplitudes', torch.ones(1))
        
        # If we have multiple events, group them
        if len(timestamps) > 1:
            # Sort events by timestamp
            sorted_indices = torch.argsort(timestamps)
            timestamps = timestamps[sorted_indices]
            features = features[sorted_indices]
            neuron_ids = neuron_ids[sorted_indices]
            spike_amplitudes = spike_amplitudes[sorted_indices]
            
            # Create temporal windows
            num_windows = max(1, len(timestamps) // window_size)
            windowed_features = []
            
            for i in range(num_windows):
                start_idx = i * window_size
                end_idx = min((i + 1) * window_size, len(features))
                
                if end_idx > start_idx:
                    # Average features in this window (could use other aggregations)
                    window_feature = features[start_idx:end_idx].mean(dim=0)
                    windowed_features.append(window_feature)
            
            if windowed_features:
                features = torch.stack(windowed_features)
                timestamps = timestamps[::window_size][:len(windowed_features)]
        
        return {
            'timestamps': timestamps,
            'features': features,
            'neuron_ids': neuron_ids,
            'amplitudes': spike_amplitudes
        }

    def _encode_event_features(self, events: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode raw event features into dense representations"""
        features = events['features']
        timestamps = events['timestamps']
        
        # Encode basic event features
        event_encoded = self.event_encoder(features)
        
        # Encode temporal information
        temporal_features = self.temporal_encoder(timestamps.unsqueeze(-1))
        
        # Combine event and temporal features
        combined_features = torch.cat([
            event_encoded, 
            temporal_features.repeat(1, event_encoded.size(-1) // temporal_features.size(-1))
        ], dim=-1)
        
        # Apply spike rate normalization if we have amplitude data
        if 'amplitudes' in events:
            amplitudes = events['amplitudes'].unsqueeze(-1)
            combined_features = combined_features * amplitudes
            combined_features = self.spike_rate_norm(combined_features)
        
        return combined_features

    def _create_connectivity_matrix(
        self, 
        neuron_ids: torch.Tensor, 
        seq_len: int
    ) -> torch.Tensor:
        """Create connectivity-aware attention biases"""
        # Simple connectivity: neurons that fire close in time might be connected
        connectivity = torch.eye(seq_len, device=neuron_ids.device)
        
        if len(neuron_ids) > 1:
            # Enhanced connectivity based on temporal proximity
            for i in range(seq_len):
                for j in range(seq_len):
                    if i != j:
                        # Simple heuristic: neurons firing close in time have stronger connectivity
                        time_diff = abs(i - j)
                        connectivity[i, j] = 1.0 / (1.0 + time_diff)
        
        return connectivity

    def forward(
        self, 
        neurosynaptic_events: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert neurosynaptic events into Q, K, V tensors
        
        Args:
            neurosynaptic_events: Dict containing:
                - 'timestamps': event timestamps [batch_size, seq_len] or [num_events]
                - 'features': event features [batch_size, seq_len, input_dim] or [num_events, input_dim]
                - 'neuron_ids': source neuron identifiers
                - 'amplitudes': spike amplitudes
            attention_mask: Optional attention mask
            return_components: Whether to return intermediate components
            
        Returns:
            Q, K, V tensors ready for hybrid attention
        """
        
        # Group events into temporal windows
        grouped_events = self._group_events_by_window(neurosynaptic_events)
        
        # Encode event features
        encoded_events = self._encode_event_features(grouped_events)
        
        seq_len = encoded_events.size(0) if len(encoded_events.shape) > 1 else 1
        
        # Add temporal encoding
        temporal_encoding = self._create_temporal_encoding(
            grouped_events['timestamps'], 
            seq_len
        )
        
        # Combine event encoding with temporal information
        if len(encoded_events.shape) == 1:
            encoded_events = encoded_events.unsqueeze(0)
        
        if encoded_events.size(0) == temporal_encoding.size(1):
            context_encoded = encoded_events + temporal_encoding.squeeze(0)
        else:
            # Handle dimension mismatch by broadcasting or truncation
            min_len = min(encoded_events.size(0), temporal_encoding.size(1))
            context_encoded = encoded_events[:min_len] + temporal_encoding.squeeze(0)[:min_len]
        
        # Ensure we have a batch dimension
        if len(context_encoded.shape) == 2:
            context_encoded = context_encoded.unsqueeze(0)  # [1, seq_len, hidden_dim]
        
        # Create connectivity-aware attention biases
        connectivity_matrix = self._create_connectivity_matrix(
            grouped_events.get('neuron_ids', torch.arange(seq_len)),
            context_encoded.size(1)
        )
        
        # Project to Q, K, V
        Q = self.q_proj(context_encoded)
        K = self.k_proj(context_encoded) 
        V = self.v_proj(context_encoded)
        
        # Apply connectivity biases to keys
        K = K + connectivity_matrix.unsqueeze(-1) * 0.1  # Small connectivity influence
        
        if return_components:
            components = {
                'encoded_events': encoded_events,
                'temporal_encoding': temporal_encoding,
                'context_encoded': context_encoded,
                'connectivity_matrix': connectivity_matrix,
                'grouped_events': grouped_events
            }
            return Q, K, V, components
        
        return Q, K, V

    def create_event_batch_from_stream(
        self,
        event_stream: List[Dict[str, Any]],
        batch_size: int = 32,
        max_sequence_length: int = None
    ) -> Dict[str, torch.Tensor]:
        """
        Convert a stream of neurosynaptic events into batched tensor format
        """
        if max_sequence_length is None:
            max_sequence_length = self.max_sequence_length
        
        # Extract features from event stream
        timestamps = []
        features = []
        neuron_ids = []
        amplitudes = []
        
        for event in event_stream:
            timestamps.append(event.get('timestamp', 0.0))
            features.append(event.get('features', torch.zeros(self.input_dim)))
            neuron_ids.append(event.get('neuron_id', 0))
            amplitudes.append(event.get('amplitude', 1.0))
        
        # Convert to tensors
        batch_data = {
            'timestamps': torch.tensor(timestamps, dtype=torch.float),
            'features': torch.stack(features) if features else torch.zeros(0, self.input_dim),
            'neuron_ids': torch.tensor(neuron_ids, dtype=torch.long),
            'amplitudes': torch.tensor(amplitudes, dtype=torch.float)
        }
        
        return batch_data

# Integration with your existing system
class NeurosynapticHybridAttention(nn.Module):
    """
    Complete system: Event encoding + Hybrid Attention + Activation Routing
    """
    
    def __init__(
        self,
        event_input_dim: int = 256,
        hidden_dim: int = 512,
        num_heads: int = 8,
        **attention_kwargs
    ):
        super().__init__()
        
        self.event_encoder = NeurosynapticEventEncoder(
            input_dim=event_input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
        
        self.hybrid_attention = DemoHybridAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            **attention_kwargs
        )
        
        # Optional: Connect to your activation router
        self.activation_router = None  # Can initialize your SmartActivationRouter here
        
    def forward(
        self,
        neurosynaptic_events: Dict[str, torch.Tensor],
        visual_input: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        # Step 1: Convert events to Q/K/V tensors
        Q, K, V = self.event_encoder(neurosynaptic_events, attention_mask)
        
        # Step 2: Process through hybrid attention
        attention_output, attention_weights = self.hybrid_attention(
            query=Q,
            key=K, 
            value=V,
            attention_mask=attention_mask,
            visual_input=visual_input
        )
        
        # Step 3: Optional routing through activation system
        if self.activation_router is not None:
            # Convert attention output to baseVNI format expected by router
            baseVNI_output = self._attention_to_basevni(attention_output, attention_weights)
            routing_result = self.activation_router(baseVNI_output)
            return routing_result
        
        return {
            'attention_output': attention_output,
            'attention_weights': attention_weights,
            'QKV_tensors': (Q, K, V)
        }
    
    def _attention_to_basevni(self, attention_output: torch.Tensor, 
                            attention_weights: torch.Tensor) -> Dict[str, Any]:
        """Convert attention output to baseVNI format for the router"""
        return {
            'abstraction_levels': {
                'cognitive': {
                    'tensor': attention_output.mean(dim=1),  # Sequence-level representation
                    'concepts': ['neurosynaptic_encoding', 'temporal_pattern', 'connectivity'],
                    'intent': 'pattern_analysis'
                },
                'structural': {
                    'tensor': attention_output  # Full sequence representation
                }
            },
            'topic_classification': {
                'neurosynaptic': 0.9,
                'temporal': 0.8,
                'structural': 0.7
            },
            'primary_topic': 'neurosynaptic'
        }

# Usage Example
def demonstrate_neurosynaptic_processing():
    """Show how to use the neurosynaptic event encoder"""
    
    # Initialize the system
    neurosyntaptic_attention = NeurosynapticHybridAttention(
        event_input_dim=256,
        hidden_dim=512,
        num_heads=8,
        window_size=256,
        use_hierarchical=True
    )
    
    # Create sample neurosynaptic events (simulating real data)
    sample_events = {
        'timestamps': torch.tensor([0.0, 1.5, 3.2, 4.8, 6.1]),  # milliseconds
        'features': torch.randn(5, 256),  # Event features (spike patterns, etc.)
        'neuron_ids': torch.tensor([101, 205, 101, 308, 205]),  # Source neurons
        'amplitudes': torch.tensor([1.2, 0.8, 1.5, 0.9, 1.1])  # Spike amplitudes
    }
    
    # Process through the complete system
    with torch.no_grad():
        result = neurosyntaptic_attention(sample_events)
    
    print("Neurosynaptic Processing Result:")
    print(f"Input events: {len(sample_events['timestamps'])}")
    print(f"Attention output shape: {result['attention_output'].shape}")
    print(f"Q/K/V shapes: {[t.shape for t in result['QKV_tensors']]}")
    
    return result

if __name__ == "__main__":
    demonstrate_neurosynaptic_processing()


# Usage Example
"""# Initialize for demo
attention = DemoHybridAttention(
    dim=512,
    num_heads=8,
    window_size=256,
    use_hierarchical=True,
    multi_modal=True
)

# Example forward pass
batch_size, seq_len, dim = 2, 1024, 512
query = torch.randn(batch_size, seq_len, dim)
key = torch.randn(batch_size, seq_len, dim)
value = torch.randn(batch_size, seq_len, dim)

# Optional visual input for multi-modal
visual_input = torch.randn(batch_size, seq_len, dim)

output, weights = attention(query, key, value, visual_input=visual_input)
print(f"Input: {query.shape} -> Output: {output.shape}")
# Output: torch.Size([2, 1024, 512]) -> torch.Size([2, 1024, 512])
# 
# Key Features Preserved:
✅ Hierarchical Attention - For sequences > 2048 tokens

✅ Sliding Window + Memory - For medium sequences (256-2048 tokens)

✅ Standard Attention - For short sequences (<256 tokens)

✅ Multi-Modal Fusion - Text + visual input fusion

✅ Memory Tokens - Persistent context across sequences

✅ Automatic Strategy Selection - Based on sequence length

What's Removed:
❌ Quantum components and entanglement

❌ Content-aware profile selection

❌ C++ extensions and complex dependencies

❌ Global token selection complexity""" 
