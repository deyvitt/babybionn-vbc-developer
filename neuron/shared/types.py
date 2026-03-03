# neuron/shared/types.py
"""
Common data types for Neuron subsystem.
Extracted from aggregator.py and other files to avoid duplication.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict, deque as collections_deque


class ConnectionType(Enum):
    """Types of VNI connections"""
    COLLABORATIVE = "collaborative"  # VNIs that work well together
    COMPETITIVE = "competitive"      # VNIs that conflict
    SEQUENTIAL = "sequential"        # VNIs in a processing chain
    COMPLEMENTARY = "complementary"  # VNIs that fill gaps for each other


@dataclass
class SynapticConnection:
    """Represents a synaptic connection between two VNIs"""
    source_vni: str
    target_vni: str
    strength: float = 0.5
    connection_type: ConnectionType = ConnectionType.COLLABORATIVE
    activation_count: int = 0
    success_count: int = 0
    last_activated: Optional[datetime] = None
    created: datetime = field(default_factory=datetime.now)
    
    # Hebbian learning metrics
    pre_activation: float = 0.0
    post_activation: float = 0.0
    correlation: float = 0.0
    
    # Performance history
    performance_history: List[float] = field(default_factory=list)
    
    # Context patterns
    successful_contexts: Set[str] = field(default_factory=set)
    failed_contexts: Set[str] = field(default_factory=set)
    
    def success_rate(self) -> float:
        """Calculate success rate of this connection"""
        if self.activation_count == 0:
            return 0.5
        return self.success_count / self.activation_count
    
    def average_performance(self) -> float:
        """Calculate average performance"""
        if not self.performance_history:
            return 0.5
        return sum(self.performance_history) / len(self.performance_history)
    
    def context_success_rate(self, context_hash: str) -> float:
        """Calculate success rate for specific context"""
        if context_hash in self.successful_contexts:
            successful_in_context = len(self.successful_contexts)
            total_in_context = successful_in_context + len(self.failed_contexts)
            return successful_in_context / total_in_context if total_in_context > 0 else 0.5
        return 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'source': self.source_vni,
            'target': self.target_vni,
            'strength': self.strength,
            'connection_type': self.connection_type.value,
            'activation_count': self.activation_count,
            'success_count': self.success_count,
            'success_rate': self.success_rate(),
            'last_activated': self.last_activated.isoformat() if self.last_activated else None,
            'created': self.created.isoformat(),
            'context_successful': len(self.successful_contexts),
            'context_failed': len(self.failed_contexts)
        }


@dataclass
class LearningMetrics:
    """Metrics for tracking learning and system performance"""
    spontaneous_activations: int = 0
    correlation_clusters: int = 0
    connections_strengthened: int = 0
    connections_weakened: int = 0
    connections_pruned: int = 0
    connections_created: int = 0
    total_learning_cycles: int = 0
    total_queries_processed: int = 0
    last_learning: Optional[datetime] = None
    learning_rate: float = 0.1
    
    # Pattern detection
    patterns_detected: int = 0
    successful_patterns: int = 0
    
    # Performance metrics
    avg_response_quality: float = 0.0
    response_quality_history: List[float] = field(default_factory=list)


@dataclass
class ClusterPerformance:
    """Performance metrics for VNI clusters"""
    cluster_id: str
    avg_response_time: float = 0.0
    success_rate: float = 0.0
    avg_confidence: float = 0.5
    total_queries: int = 0
    specialization: str = "general"
    last_active: Optional[datetime] = None
    performance_history: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'cluster_id': self.cluster_id,
            'avg_response_time': self.avg_response_time,
            'success_rate': self.success_rate,
            'avg_confidence': self.avg_confidence,
            'total_queries': self.total_queries,
            'specialization': self.specialization,
            'last_active': self.last_active.isoformat() if self.last_active else None,
            'performance_history_size': len(self.performance_history)
        }


@dataclass
class VNIContribution:
    """Contribution analysis from a VNI"""
    vni_id: str
    domain: str
    confidence: float
    contribution_score: float
    contribution_level: str  # 'primary' or 'secondary'


@dataclass
class ConsensusResult:
    """Consensus calculation result"""
    consensus_level: str  # 'strong', 'moderate', 'weak', 'none'
    consensus_score: float
    average_confidence: float
    domain_distribution: Dict[str, int]
    domain_consensus: Dict[str, float]
    agreeing_vnis: List[str]
    total_vnis: int


@dataclass  
class ConflictAnalysis:
    """Conflict analysis result"""
    vni1: str
    vni2: str
    conflict_level: str  # 'major_conflict', 'minor_conflict', 'complementary', 'no_conflict'
    confidence: float
    conflicting_aspects: List[str]
    similarity_score: float
    semantic_overlap: float
    domain_comparison: str
    confidence_difference: float


@dataclass
class SynapticInsights:
    """Insights from synaptic learning"""
    available: bool
    active_connections: List[Dict[str, Any]]
    network_statistics: Dict[str, Any]
    strongest_connections: List[Dict[str, Any]]
    avg_connection_strength: float
    strength_trend: Dict[str, Any]


@dataclass
class PerformanceTrend:
    """Performance trend analysis"""
    trend: str  # 'improving', 'declining', 'stable', 'insufficient_data'
    slope: float
    current_avg: float
    window_size: int


@dataclass
class PatternRecord:
    """Record of a learned pattern"""
    vnis: List[str]
    quality: float
    context_hash: str
    pattern_hash: str
    timestamp: str
    usage_count: int = 1
    last_used: Optional[str] = None


# Type aliases for better readability
ConnectionDict = Dict[Tuple[str, str], SynapticConnection]
PatternDict = Dict[str, List[Tuple[List[str], float]]]
HistoryDeque = collections_deque 
