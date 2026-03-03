# neuron/shared/constants.py
"""
Constants and default values for Neuron subsystem.
Used across aggregator, learning engine, and visualization.
"""

from typing import Dict, List, Tuple

# ==================== DEFAULT VALUES ====================

DEFAULT_LEARNING_RATE = 0.1
DEFAULT_CONSENSUS_THRESHOLD = 0.7
DEFAULT_HISTORY_SIZE = 1000
DEFAULT_EMBEDDING_DIM = 512
DEFAULT_HIDDEN_DIM = 256

# ==================== CONNECTION THRESHOLDS ====================

STRONG_CONNECTION_THRESHOLD = 0.7
WEAK_CONNECTION_THRESHOLD = 0.3
PRUNING_THRESHOLD = 0.1

# Connection type thresholds
COLLABORATIVE_THRESHOLD = 0.7
COMPETITIVE_THRESHOLD = 0.3
SEQUENTIAL_WINDOW = 5

# ==================== TIME CONSTANTS ====================

SESSION_TIMEOUT_HOURS = 2
PRUNING_CHECK_INTERVAL = 50  # learning cycles
ACTIVITY_TRACE_WINDOW = 20
PERFORMANCE_HISTORY_SIZE = 50

# ==================== VISUALIZATION CONSTANTS ====================

# Color mapping for different domain types
COLOR_MAP: Dict[str, str] = {
    'medical': '#FF6B6B',      # Red
    'legal': '#4ECDC4',        # Teal
    'technical': '#45B7D1',    # Blue
    'general': '#96CEB4',      # Green
    'analytical': '#FFEAA7',   # Yellow
    'financial': '#A29BFE',    # Purple
    'creative': '#FD79A8',     # Pink
    'research': '#00B894'      # Mint
}

# Node sizes by importance
NODE_SIZES: Dict[str, int] = {
    'primary': 1000,
    'secondary': 800,
    'tertiary': 600,
    'default': 500
}

# Visualization defaults
FIGURE_SIZE: Tuple[int, int] = (12, 8)
DPI: int = 150
ANIMATION_FPS: int = 2

# ==================== DOMAIN DETECTION ====================

# Domain keywords for query analysis
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    'medical': [
        'medical', 'health', 'doctor', 'patient', 'hospital', 'clinic',
        'symptom', 'diagnosis', 'treatment', 'medicine', 'drug', 'therapy',
        'pain', 'fever', 'injury', 'surgery', 'recovery', 'prescription'
    ],
    'legal': [
        'legal', 'law', 'contract', 'agreement', 'rights', 'obligation',
        'liability', 'compliance', 'regulation', 'court', 'attorney',
        'lawsuit', 'dispute', 'clause', 'termination', 'breach'
    ],
    'technical': [
        'code', 'programming', 'software', 'application', 'system',
        'algorithm', 'database', 'network', 'server', 'api', 'interface',
        'debug', 'error', 'bug', 'optimization', 'performance', 'security'
    ],
    'analytical': [
        'analyze', 'analysis', 'evaluate', 'assessment', 'compare',
        'statistics', 'data', 'metrics', 'trend', 'pattern', 'correlation',
        'regression', 'forecast', 'prediction', 'insight', 'dashboard'
    ],
    'financial': [
        'financial', 'finance', 'money', 'investment', 'stock', 'market',
        'portfolio', 'risk', 'return', 'profit', 'loss', 'revenue',
        'cost', 'budget', 'valuation', 'trading', 'dividend'
    ],
    'creative': [
        'creative', 'creativity', 'story', 'narrative', 'design',
        'art', 'write', 'composition', 'imagine', 'innovation',
        'original', 'unique', 'inspiration', 'concept', 'theme'
    ],
    'research': [
        'research', 'study', 'experiment', 'investigation', 'survey',
        'hypothesis', 'methodology', 'findings', 'results', 'conclusion',
        'publication', 'paper', 'thesis', 'dissertation', 'academic'
    ]
}

# ==================== CONFIDENCE LEVELS ====================

CONFIDENCE_LEVELS: Dict[str, Tuple[float, float]] = {
    'very_high': (0.9, 1.0),
    'high': (0.7, 0.9),
    'medium': (0.4, 0.7),
    'low': (0.2, 0.4),
    'very_low': (0.0, 0.2)
}

CONSENSUS_LEVELS: Dict[str, Tuple[float, float]] = {
    'strong': (0.8, 1.0),
    'moderate': (0.6, 0.8),
    'weak': (0.4, 0.6),
    'none': (0.0, 0.4)
}

# ==================== ERROR MESSAGES ====================

ERROR_MESSAGES: Dict[str, str] = {
    'no_vni_outputs': "Unable to generate analysis. No successful VNI outputs.",
    'aggregation_failed': "System error during aggregation: {error}",
    'learning_disabled': "Hebbian learning is currently disabled.",
    'visualization_failed': "Failed to create visualization: {error}",
    'pattern_not_found': "No learned patterns found for this context.",
    'cluster_not_found': "Cluster '{cluster_id}' not found."
}

# ==================== FILE PATHS ====================

DEFAULT_FILE_PATHS: Dict[str, str] = {
    'synaptic_network': 'synaptic_network.json',
    'aggregator_state': 'aggregator_state.pkl',
    'visualization': 'synaptic_network.png',
    'animation': 'synaptic_evolution.gif',
    'patterns': 'learned_patterns.json',
    'metrics': 'learning_metrics.json'
}

# ==================== NEURAL NETWORK ARCHITECTURE ====================

# Consensus calculator architecture
CONSENSUS_NETWORK_LAYERS: List[Tuple[str, Dict]] = [
    ('linear', {'in_features': 8, 'out_features': 128}),
    ('relu', {}),
    ('dropout', {'p': 0.1}),
    ('linear', {'in_features': 128, 'out_features': 64}),
    ('relu', {}),
    ('linear', {'in_features': 64, 'out_features': 1}),
    ('sigmoid', {})
]

# Conflict detector architecture
CONFLICT_NETWORK_LAYERS: List[Tuple[str, Dict]] = [
    ('linear', {'in_features': 512, 'out_features': 256}),
    ('relu', {}),
    ('dropout', {'p': 0.1}),
    ('linear', {'in_features': 256, 'out_features': 128}),
    ('relu', {}),
    ('linear', {'in_features': 128, 'out_features': 64}),
    ('tanh', {})
]

# ==================== PERFORMANCE METRICS ====================

METRIC_NAMES: List[str] = [
    'response_quality',
    'connection_strength',
    'consensus_score',
    'conflict_count',
    'activation_count',
    'pattern_usage',
    'cluster_performance',
    'learning_rate'
]

# Trend detection thresholds
TREND_THRESHOLDS: Dict[str, float] = {
    'improving': 0.01,
    'declining': -0.01,
    'significant_change': 0.05
}

# ==================== CONTEXT HASHING ====================

CONTEXT_HASH_FIELDS: List[str] = [
    'query_complexity',
    'detected_domains',
    'session_id',
    'timestamp_hour'  # Rounded to hour for pattern matching
]

# ==================== SPECIAL CONSTANTS ====================

# Magic numbers and special values
EPSILON = 1e-8  # Small value to avoid division by zero
MAX_FLOAT = 1e10
MIN_FLOAT = -1e10

# STDP parameters
STDP_TAU = 20.0  # Time constant in seconds
STDP_MAX_TIME_DIFF = 10.0  # Maximum time difference to consider

# Pattern matching
MIN_PATTERN_LENGTH = 2
MAX_PATTERN_LENGTH = 10
PATTERN_QUALITY_THRESHOLD = 0.7

# Export all constants
__all__ = [
    'COLOR_MAP',
    'DOMAIN_KEYWORDS',
    'DEFAULT_LEARNING_RATE',
    'STRONG_CONNECTION_THRESHOLD',
    'WEAK_CONNECTION_THRESHOLD',
    'SESSION_TIMEOUT_HOURS',
    'CONFIDENCE_LEVELS',
    'CONSENSUS_LEVELS',
    'ERROR_MESSAGES',
    'DEFAULT_FILE_PATHS',
    'METRIC_NAMES',
    'TREND_THRESHOLDS',
    'STDP_TAU'
] 
