# File: bionn-demo-chatbot/neuron/metaCognitive_enhancer.py
"""
MetaCognitiveEnhancer - 200-line focused module that enhances your existing aggregator's meta-cognition
Adds pattern recognition, connection analysis, and optimization suggestions without bloat
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import json
import hashlib

logger = logging.getLogger("meta_cognitive")

@dataclass
class PatternRecord:
    """Compact pattern record for successful VNI collaborations"""
    vni_combination: Tuple[str, ...]  # Sorted tuple of VNI IDs
    domain_combination: Tuple[str, ...]  # Sorted tuple of domains
    avg_confidence: float
    avg_quality: float
    usage_count: int = 0
    last_used: datetime = field(default_factory=datetime.now)
    success_score: float = 0.0

class MetaCognitiveAnalyzer:
    """
    Lightweight analyzer that works WITH your existing aggregator
    Enhances meta-cognition by analyzing connection patterns and suggesting optimizations
    """
    
    def __init__(self, hebbian_engine, max_patterns: int = 100):
        """
        Initialize with reference to existing Hebbian engine
        
        Args:
            hebbian_engine: Your existing HebbianLearningEngine instance
            max_patterns: Maximum patterns to track
        """
        self.hebbian_engine = hebbian_engine
        self.max_patterns = max_patterns
        
        # Pattern database (what VNI combinations work well together)
        self.patterns: Dict[str, PatternRecord] = {}
        
        # Performance tracking
        self.recent_performance = deque(maxlen=50)
        self.domain_interaction_matrix = defaultdict(lambda: defaultdict(int))
        
        # Optimization cache
        self._optimization_cache = {}
        self._cache_ttl = timedelta(minutes=5)
        self._last_cache_update = datetime.now()
    
    def analyze_hebbian_connections(self) -> Dict[str, Any]:
        """
        Analyze existing Hebbian connections to extract meta-cognitive insights
        
        Returns:
            Dictionary with connection analysis insights
        """
        connections = getattr(self.hebbian_engine, 'connections', {})
        
        if not connections:
            return {"status": "no_connections", "message": "No Hebbian connections to analyze"}
        
        # Extract connection strengths
        strengths = []
        connection_types = defaultdict(int)
        
        for conn_key, conn in connections.items():
            strengths.append(conn.strength)
            
            # Count connection types
            if hasattr(conn, 'connection_type'):
                connection_types[conn.connection_type.value] += 1
        
        # Calculate statistics
        avg_strength = np.mean(strengths) if strengths else 0
        std_strength = np.std(strengths) if len(strengths) > 1 else 0
        
        # Find strongest connections
        strong_connections = [
            (conn_key, conn.strength) 
            for conn_key, conn in connections.items() 
            if conn.strength > avg_strength + std_strength
        ]
        strong_connections.sort(key=lambda x: x[1], reverse=True)
        
        # Find learning patterns
        learning_patterns = self._extract_learning_patterns(connections)
        
        return {
            "total_connections": len(connections),
            "avg_connection_strength": float(avg_strength),
            "strength_variability": float(std_strength),
            "connection_type_distribution": dict(connection_types),
            "top_strong_connections": strong_connections[:5],
            "learning_patterns": learning_patterns,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _extract_learning_patterns(self, connections: Dict) -> List[Dict[str, Any]]:
        """Extract patterns from connection evolution"""
        patterns = []
        
        # Group by VNI pairs (simplified)
        vni_pairs = defaultdict(list)
        
        for conn_key, conn in connections.items():
            if hasattr(conn, 'source_vni') and hasattr(conn, 'target_vni'):
                pair_key = tuple(sorted([conn.source_vni, conn.target_vni]))
                vni_pairs[pair_key].append(conn)
        
        # Analyze each pair
        for pair, pair_connections in list(vni_pairs.items()):
            if len(pair_connections) >= 2:
                # Check for strengthening/weakening patterns
                strengths = [c.strength for c in pair_connections]
                avg_strength = np.mean(strengths)
                
                # Look for trends (simplified)
                if len(strengths) >= 3:
                    recent_trend = self._calculate_trend(strengths[-3:])
                    pattern_type = self._classify_pattern(strengths, avg_strength)
                    
                    patterns.append({
                        "vni_pair": pair,
                        "avg_strength": float(avg_strength),
                        "interaction_count": len(pair_connections),
                        "pattern_type": pattern_type,
                        "recent_trend": recent_trend
                    })
        
        return patterns[:10]  # Top 10 patterns
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from recent values"""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend
        if values[-1] > values[0] + 0.1:
            return "strengthening"
        elif values[-1] < values[0] - 0.1:
            return "weakening"
        return "stable"
    
    def _classify_pattern(self, strengths: List[float], avg_strength: float) -> str:
        """Classify the learning pattern"""
        variability = np.std(strengths) if len(strengths) > 1 else 0
        
        if avg_strength > 0.7:
            return "strong_cooperative"
        elif avg_strength < 0.3:
            return "weak_or_competitive"
        elif variability > 0.2:
            return "volatile_interaction"
        else:
            return "stable_partnership"
    
    def record_interaction_outcome(self, 
                                  activated_vnis: List[str],
                                  vni_outputs: Dict[str, Dict],
                                  overall_quality: float,
                                  query_context: Dict[str, Any]):
        """
        Record an interaction outcome for pattern learning
        
        Called after each query processing to learn what works
        """
        if len(activated_vnis) < 2:
            return  # Only track multi-VNI interactions
        
        # Create pattern key
        sorted_vnis = tuple(sorted(activated_vnis))
        pattern_key = hashlib.md5(str(sorted_vnis).encode()).hexdigest()[:16]
        
        # Extract domains if available
        domains = []
        for vni_id in activated_vnis:
            if vni_id in vni_outputs and 'vni_metadata' in vni_outputs[vni_id]:
                domains.append(vni_outputs[vni_id]['vni_metadata'].get('domain', 'unknown'))
        
        sorted_domains = tuple(sorted(set(domains))) if domains else tuple()
        
        # Update or create pattern
        if pattern_key in self.patterns:
            pattern = self.patterns[pattern_key]
            pattern.usage_count += 1
            pattern.last_used = datetime.now()
            
            # Update success score (weighted moving average)
            pattern.success_score = 0.8 * pattern.success_score + 0.2 * overall_quality
        else:
            # Create new pattern
            pattern = PatternRecord(
                vni_combination=sorted_vnis,
                domain_combination=sorted_domains,
                avg_confidence=self._calculate_avg_confidence(vni_outputs),
                avg_quality=overall_quality,
                usage_count=1,
                success_score=overall_quality
            )
            self.patterns[pattern_key] = pattern
        
        # Update domain interaction matrix
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                d1, d2 = domains[i], domains[j]
                self.domain_interaction_matrix[d1][d2] += 1
                self.domain_interaction_matrix[d2][d1] += 1
        
        # Prune old patterns if needed
        if len(self.patterns) > self.max_patterns:
            self._prune_patterns()
        
        # Update performance tracking
        self.recent_performance.append(overall_quality)
        
        # Clear cache since new data is available
        self._optimization_cache = {}
    
    def _calculate_avg_confidence(self, vni_outputs: Dict[str, Dict]) -> float:
        """Calculate average confidence from VNI outputs"""
        confidences = []
        for vni_id, output in vni_outputs.items():
            if 'confidence_score' in output:
                confidences.append(output['confidence_score'])
        return np.mean(confidences) if confidences else 0.5
    
    def _prune_patterns(self):
        """Remove least useful patterns"""
        # Sort by usage count and success score
        pattern_items = list(self.patterns.items())
        pattern_items.sort(key=lambda x: (
            x[1].usage_count * 0.3 + 
            x[1].success_score * 0.7
        ))
        
        # Keep top max_patterns
        self.patterns = dict(pattern_items[-self.max_patterns:])
    
    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """
        Generate optimization suggestions based on learned patterns
        
        Returns:
            List of actionable suggestions
        """
        # Check cache
        cache_key = "suggestions"
        current_time = datetime.now()
        
        if (cache_key in self._optimization_cache and 
            current_time - self._last_cache_update < self._cache_ttl):
            return self._optimization_cache[cache_key]
        
        suggestions = []
        
        # Analyze patterns for suggestions
        successful_patterns = [
            p for p in self.patterns.values() 
            if p.success_score > 0.7 and p.usage_count >= 2
        ]
        
        # 1. Suggest VNI combinations that work well
        if successful_patterns:
            # Sort by success score
            successful_patterns.sort(key=lambda p: p.success_score, reverse=True)
            
            top_pattern = successful_patterns[0]
            suggestions.append({
                "type": "VNI_COLLABORATION",
                "priority": "high",
                "message": f"VNI combination {top_pattern.vni_combination[:3]} works well (success: {top_pattern.success_score:.1%})",
                "action": f"Consider routing similar queries to these VNIs together",
                "confidence": top_pattern.success_score
            })
        
        # 2. Analyze domain synergies
        domain_synergies = []
        for domain1, interactions in self.domain_interaction_matrix.items():
            for domain2, count in interactions.items():
                if count >= 3 and domain1 != domain2:
                    domain_synergies.append((domain1, domain2, count))
        
        domain_synergies.sort(key=lambda x: x[2], reverse=True)
        
        if domain_synergies:
            top_synergy = domain_synergies[0]
            suggestions.append({
                "type": "DOMAIN_SYNERGY",
                "priority": "medium",
                "message": f"Domains '{top_synergy[0]}' and '{top_synergy[1]}' frequently collaborate ({top_synergy[2]} times)",
                "action": "Encourage more cross-domain queries between these domains",
                "confidence": min(top_synergy[2] / 10, 0.9)  # Cap confidence
            })
        
        # 3. Check for underutilized connections
        connection_analysis = self.analyze_hebbian_connections()
        if connection_analysis.get("total_connections", 0) > 10:
            avg_strength = connection_analysis.get("avg_connection_strength", 0)
            
            if avg_strength < 0.3:
                suggestions.append({
                    "type": "CONNECTION_STRENGTH",
                    "priority": "medium",
                    "message": f"Average connection strength is low ({avg_strength:.2f})",
                    "action": "Consider more diverse query routing to strengthen connections",
                    "confidence": 0.7
                })
        
        # 4. Recent performance trend
        if len(self.recent_performance) >= 5:
            recent_avg = np.mean(list(self.recent_performance)[-5:])
            older_avg = np.mean(list(self.recent_performance)[-10:-5]) if len(self.recent_performance) >= 10 else recent_avg
            
            if recent_avg < older_avg - 0.1:
                suggestions.append({
                    "type": "PERFORMANCE_TREND",
                    "priority": "high",
                    "message": f"Recent response quality declined ({older_avg:.2f} → {recent_avg:.2f})",
                    "action": "Review recent query patterns and consider adjusting routing",
                    "confidence": 0.8
                })
        
        # Cache results
        self._optimization_cache[cache_key] = suggestions
        self._last_cache_update = current_time
        
        return suggestions
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of learned patterns"""
        total_patterns = len(self.patterns)
        
        if total_patterns == 0:
            return {"status": "no_patterns_learned"}
        
        # Calculate statistics
        success_scores = [p.success_score for p in self.patterns.values()]
        usage_counts = [p.usage_count for p in self.patterns.values()]
        
        # Find most successful patterns
        successful_patterns = [
            {
                "vnis": list(p.vni_combination)[:5],  # First 5 VNIs
                "domains": list(p.domain_combination),
                "success_score": p.success_score,
                "usage_count": p.usage_count
            }
            for p in self.patterns.values() 
            if p.success_score > 0.6 and p.usage_count >= 2
        ]
        successful_patterns.sort(key=lambda x: x["success_score"], reverse=True)
        
        return {
            "total_patterns": total_patterns,
            "avg_success_score": float(np.mean(success_scores)) if success_scores else 0,
            "total_interactions": sum(usage_counts),
            "successful_patterns_count": len(successful_patterns),
            "top_patterns": successful_patterns[:5],  # Top 5 patterns
            "domain_interactions": {
                domain: dict(interactions)
                for domain, interactions in self.domain_interaction_matrix.items()
                if sum(interactions.values()) >= 3
            }
        }

# ==================== INTEGRATION HELPERS ====================

def integrate_with_aggregator(aggregator_instance) -> MetaCognitiveAnalyzer:
    """
    Easy integration function - call this from your aggregator's __init__
    
    Usage in aggregator.py:
        self.meta_cognitive = integrate_with_aggregator(self)
    
    Then call after each learning cycle:
        self.meta_cognitive.record_interaction_outcome(...)
    """
    # Get the hebbian_engine from aggregator
    hebbian_engine = getattr(aggregator_instance, 'hebbian_engine', None)
    
    if hebbian_engine is None:
        logger.warning("No Hebbian engine found in aggregator. Meta-cognitive analysis limited.")
        # Create a minimal analyzer anyway
        hebbian_engine = type('DummyEngine', (), {'connections': {}})()
    
    # Create analyzer
    analyzer = MetaCognitiveAnalyzer(hebbian_engine)
    
    logger.info("✅ MetaCognitiveAnalyzer integrated with aggregator")
    return analyzer

def get_enhanced_insights(aggregator_instance, meta_analyzer: MetaCognitiveAnalyzer) -> Dict[str, Any]:
    """
    Get enhanced insights combining original aggregator metrics with meta-cognition
    
    Call this from your get_performance_report() or get_system_overview()
    """
    # Get original metrics
    original_report = {}
    if hasattr(aggregator_instance, 'get_performance_report'):
        original_report = aggregator_instance.get_performance_report()
    
    # Get meta-cognitive insights
    connection_analysis = meta_analyzer.analyze_hebbian_connections()
    pattern_summary = meta_analyzer.get_pattern_summary()
    suggestions = meta_analyzer.get_optimization_suggestions()
    
    return {
        "original_metrics": original_report,
        "meta_cognitive_analysis": {
            "connection_insights": connection_analysis,
            "learned_patterns": pattern_summary,
            "optimization_suggestions": suggestions,
            "pattern_database_size": len(meta_analyzer.patterns),
            "recent_performance_trend": list(meta_analyzer.recent_performance)[-10:] if meta_analyzer.recent_performance else []
        },
        "enhanced_timestamp": datetime.now().isoformat(),
        "system_status": "enhanced_with_meta_cognition"
    } 
