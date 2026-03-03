# neuron/shared/synaptic_config.py
"""
Unified configuration for synaptic learning components.
Extracted from aggregator.py to serve as single config source.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import timedelta


@dataclass
class SynapticConfig:
    """Unified configuration for the synaptic learning system"""
    
    # ==================== IDENTITY ====================
    aggregator_id: str = "unified_aggregator"
    
    # ==================== CONFLICT RESOLUTION ====================
    consensus_threshold: float = 0.7
    conflict_resolution_strategy: str = "confidence_weighted"
    min_confidence_threshold: float = 0.4
    
    # ==================== OUTPUT CONTROL ====================
    max_output_length: int = 500
    enable_cross_domain_synthesis: bool = True
    enable_conflict_detection: bool = True
    enable_confidence_calibration: bool = True
    
    # ==================== BIOLOGICAL ROUTING ====================
    enable_biological_routing: bool = False  # or True
    attention_routing_weight: float = 0.25
    activation_routing_weight: float = 0.25
    memory_routing_weight: float = 0.20
    hebbian_routing_weight: float = 0.30
    
    # ==================== SYNAPTIC LEARNING ====================
    enable_hebbian_learning: bool = True
    learning_rate: float = 0.1
    decay_rate: float = 0.01
    strengthening_threshold: float = 0.7
    weakening_threshold: float = 0.4
    pruning_threshold: float = 0.1
    max_connection_strength: float = 1.0
    min_connection_strength: float = 0.0
    
    # ==================== ORCHESTRATION ====================
    enable_auto_spawning: bool = True
    enable_visualization: bool = True
    max_clusters: int = 10
    session_timeout_hours: int = 2
    session_timeout: timedelta = field(default_factory=lambda: timedelta(hours=2))
    
    # ==================== PERFORMANCE ====================
    history_size: int = 1000
    session_history_size: int = 50
    pattern_detection_window: int = 100
    
    # ==================== NEURAL NETWORKS ====================
    embedding_dim: int = 512
    hidden_dim: int = 256
    dropout_rate: float = 0.1
    
    # ==================== VISUALIZATION ====================
    figure_size: Tuple[int, int] = (12, 8)
    visualization_dpi: int = 150
    animation_fps: int = 2
    
    # ==================== EMERGENT LEARNING ====================
    enable_meta_learning: bool = True
    optimization_interval: int = 100  # queries
    performance_evaluation_window: int = 50
    hyperparameter_optimization: bool = True
    
    # ==================== DOMAIN DETECTION ====================
    domain_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        'medical': ['medical', 'health', 'doctor', 'patient', 'symptom', 'diagnosis', 'treatment', 'hospital'],
        'legal': ['legal', 'law', 'contract', 'rights', 'agreement', 'court', 'liability', 'compliance'],
        'technical': ['code', 'programming', 'software', 'debug', 'error', 'bug', 'algorithm', 'technical', 'api'],
        'analytical': ['analyze', 'compare', 'evaluate', 'assess', 'statistics', 'data', 'analysis', 'trend', 'pattern'],
        'financial': ['financial', 'finance', 'investment', 'stock', 'market', 'portfolio', 'risk', 'return'],
        'creative': ['creative', 'story', 'narrative', 'design', 'art', 'write', 'compose', 'imagine'],
        'research': ['research', 'study', 'experiment', 'hypothesis', 'methodology', 'findings']
    })
    
    # ==================== CONNECTION CLASSIFICATION ====================
    connection_type_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'collaborative_threshold': 0.7,
        'competitive_threshold': 0.3,
        'sequential_detection_window': 5
    })
    
    # ==================== STDP PARAMETERS ====================
    stdp_time_constant: float = 20.0  # seconds
    stdp_strengthening_factor: float = 1.0
    stdp_weakening_factor: float = 0.5
    
    def __post_init__(self):
        """Post-initialization to set derived fields"""
        self.session_timeout = timedelta(hours=self.session_timeout_hours)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
    
    @classmethod
    def create_default(cls) -> 'SynapticConfig':
        """Create default configuration"""
        return cls()
    
    @classmethod
    def create_optimized_for_performance(cls) -> 'SynapticConfig':
        """Create configuration optimized for performance"""
        return cls(
            enable_visualization=False,
            history_size=500,
            session_history_size=25,
            pattern_detection_window=50,
            optimization_interval=200
        )
    
    @classmethod
    def create_optimized_for_learning(cls) -> 'SynapticConfig':
        """Create configuration optimized for learning"""
        return cls(
            learning_rate=0.15,
            decay_rate=0.005,
            pattern_detection_window=200,
            enable_visualization=True,
            optimization_interval=50
        )


# Default configuration instance
DEFAULT_CONFIG = SynapticConfig() 
