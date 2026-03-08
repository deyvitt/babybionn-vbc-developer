# bionn-demo-chatbot/neuron/aggregator.py
"""Unified Response Aggregator with Full Hebbian Synaptic Learning
Implements true synaptic plasticity for VNI-to-VNI connections
"""
import sys
import json
import uuid
import torch
import pickle
import asyncio
import hashlib
import inspect
import logging
import numpy as np
from enum import Enum
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from neuron.vni_memory import VniMemory
from datetime import datetime, timedelta
from collections import defaultdict, deque
from .aggregatorAttn import AggregatorAttention
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Set
from llm_Gateway import get_gateway, LLMConfig, LLMProvider
from neuron.shared.reasoning_output import ReasoningOutput, VNIOpinion
from .demoHybridAttention import DemoHybridAttention, HybridAttentionEngine
from .smart_activation_router import SmartActivationRouter, FunctionRegistry
from .metaCognitive_enhancer import integrate_with_aggregator, get_enhanced_insights
from neuron.shared.types import (
    ConnectionType, SynapticConnection, LearningMetrics, 
    ClusterPerformance, ConsensusResult, ConflictAnalysis,
    SynapticInsights, PerformanceTrend, VNIContribution, PatternRecord
)
from neuron.shared.synaptic_config import SynapticConfig as AggregatorConfig, DEFAULT_CONFIG
from neuron.shared.constants import (
    COLOR_MAP, DOMAIN_KEYWORDS, DEFAULT_LEARNING_RATE,
    STRONG_CONNECTION_THRESHOLD, WEAK_CONNECTION_THRESHOLD,
    SESSION_TIMEOUT_HOURS, CONFIDENCE_LEVELS, CONSENSUS_LEVELS,
    ERROR_MESSAGES, DEFAULT_FILE_PATHS, METRIC_NAMES, TREND_THRESHOLDS,
    STDP_TAU, FIGURE_SIZE
)

import re
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aggregator")

# ==================== GREETING PREPROCESSOR ====================
class GreetingPreprocessor:
    """Catches greetings before VNI routing"""
    
    GREETING_PATTERNS = [
        (r'\b(hello|hi|hey|greetings|howdy|hiya)\b', 0.9),
        (r'\b(what.*your name|who.*are you)\b', 0.95),
        (r'\b(my name is|i am|i\'m)\b', 0.8),
        (r'\b(good morning|good afternoon|good evening)\b', 0.85),
        (r'\b(how are you|how do you do)\b', 0.8),
        (r'\b(hi there|hello there)\b', 0.9),
        (r'\b(yo|sup|what\'s up)\b', 0.7),
        (r'\b(nice to meet you|pleased to meet you)\b', 0.8),
        (r'\b(welcome|greeting|salutation)\b', 0.7)
    ]
    
    @staticmethod
    def is_greeting(text: str) -> bool:
        """Check if text is a greeting"""
        text_lower = text.lower().strip()
        if not text_lower:
            return False
            
        for pattern, _ in GreetingPreprocessor.GREETING_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False
    
    @staticmethod
    def get_response(text: str) -> Dict[str, Any]:
        """Generate appropriate greeting response"""
        from datetime import datetime
        
        text_lower = text.lower()
        
        # Extract name if present
        name = None
        name_match = re.search(r'my name is (\w+)|i am (\w+)|i\'m (\w+)', text_lower)
        if name_match:
            name = name_match.group(1) or name_match.group(2) or name_match.group(3)
        
        # Custom responses based on query
        if re.search(r'what.*your name|who.*are you', text_lower):
            response = random.choice([
                "Hello! I'm BabyBIONN, an enhanced neural mesh system with specialized VNIs for medical, legal, and general knowledge.",
                "I'm BabyBIONN, your neural mesh assistant with specialized knowledge in medical, legal, and technical domains.",
                "Greetings! I'm BabyBIONN, a collaborative neural network ready to assist you."
            ])
        
        elif name:
            response = random.choice([
                f"Nice to meet you, {name.title()}! I'm BabyBIONN. How can I assist you today?",
                f"Hello {name.title()}! I'm BabyBIONN. What would you like to discuss?",
                f"Pleased to meet you, {name.title()}! I'm BabyBIONN, ready to help with your questions."
            ])
        
        elif re.search(r'how are you', text_lower):
            response = random.choice([
                "I'm functioning optimally, thank you! Ready to assist with medical, legal, or general questions.",
                "All neural pathways are active and ready! How can I help you today?",
                "System status: excellent! Enhanced neural mesh is online and learning.",
                "Operating at full capacity! What can I assist you with?"
            ])
        
        elif re.search(r'good morning|good afternoon|good evening', text_lower):
            time_match = re.search(r'(good morning|good afternoon|good evening)', text_lower)
            time_greeting = time_match.group(1)
            response = f"{time_greeting.title()}! I'm BabyBIONN. How can I help you today?"
        
        else:
            response = random.choice([
                "Hello! How can I help you today?",
                "Hi there! I'm BabyBIONN. What would you like to discuss?",
                "Greetings! I'm ready to assist with medical, legal, or general questions.",
                "Hello! I'm BabyBIONN. What's on your mind?",
                "Hi! I'm your BabyBIONN assistant. How can I assist?"
            ])
        
        return {
            "response": response,
            "confidence": 0.95,
            "sources": ["greeting_preprocessor"],
            "timestamp": datetime.now().isoformat(),
            "learning_applied": False,
            "greeting_detected": True
        }
# ================= END OF GREETING PREPROCESSOR =================
    
class HebbianLearningEngine:
    """
    Implements Hebbian learning for VNI connections
    "Neurons that fire together, wire together"
    """
    
    def __init__(self, config: AggregatorConfig):
        self.config = config
        
        # Connection matrix: {(vni1, vni2): SynapticConnection}
        self.connections: Dict[Tuple[str, str], SynapticConnection] = {}
        
        # Activity traces for temporal correlation
        self.activity_traces: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        
        # Pattern library
        self.successful_patterns: List[Dict[str, Any]] = []
        self.failed_patterns: List[Dict[str, Any]] = []
        
        # Context patterns
        self.context_patterns: Dict[str, List[Tuple[List[str], float]]] = defaultdict(list)
        
        # Learning history
        self.learning_history: deque = deque(maxlen=config.history_size)
        
        # Connection type classifier
        self.connection_type_classifier = self._build_type_classifier()
        
        logger.info("🧠 HebbianLearningEngine initialized")
    
    def _build_type_classifier(self) -> Dict[str, Any]:
        """Build a simple classifier for connection types"""
        return {
            'collaborative_threshold': 0.7,
            'competitive_threshold': 0.3,
            'sequential_detection_window': 5
        }
    
    def get_or_create_connection(self, vni1: str, vni2: str) -> SynapticConnection:
        """Get existing connection or create new one"""
        key = tuple(sorted([vni1, vni2]))
        
        if key not in self.connections:
            self.connections[key] = SynapticConnection(
                source_vni=key[0],
                target_vni=key[1],
                strength=0.5
            )
            logger.debug(f"🔗 Created connection: {key[0]} <-> {key[1]}")
        
        return self.connections[key]
    
    def classify_connection_type(self, connection: SynapticConnection) -> ConnectionType:
        """Classify connection type based on metrics"""
        success_rate = connection.success_rate()
        avg_performance = connection.average_performance()
        
        if success_rate > self.connection_type_classifier['collaborative_threshold']:
            return ConnectionType.COLLABORATIVE
        elif success_rate < self.connection_type_classifier['competitive_threshold']:
            return ConnectionType.COMPETITIVE
        elif self._is_sequential_pattern(connection):
            return ConnectionType.SEQUENTIAL
        else:
            return ConnectionType.COMPLEMENTARY
    
    def _is_sequential_pattern(self, connection: SynapticConnection) -> bool:
        """Check if connection shows sequential activation patterns"""
        # Check temporal patterns in activity traces
        source_traces = list(self.activity_traces.get(connection.source_vni, []))
        target_traces = list(self.activity_traces.get(connection.target_vni, []))
        
        if len(source_traces) < 2 or len(target_traces) < 2:
            return False
        
        # Check if source consistently activates before target
        sequential_count = 0
        for i in range(min(len(source_traces), len(target_traces))):
            if source_traces[i]['time'] < target_traces[i]['time']:
                sequential_count += 1
        
        return sequential_count / min(len(source_traces), len(target_traces)) > 0.7
    
    def hebbian_update(
        self, 
        vni1: str, 
        vni2: str, 
        pre_activation: float, 
        post_activation: float,
        outcome_quality: float,
        context_hash: str = ""
    ):
        """
        Apply Hebbian learning rule: ΔW = η * pre * post * outcome
        
        Args:
            vni1: First VNI ID
            vni2: Second VNI ID
            pre_activation: Activation level of first VNI
            post_activation: Activation level of second VNI
            outcome_quality: Quality of the result (0.0 to 1.0)
            context_hash: Hash of the query context
        """
        connection = self.get_or_create_connection(vni1, vni2)
        
        # Hebbian learning rule with outcome modulation
        correlation = pre_activation * post_activation * outcome_quality
        
        # Update connection strength
        delta_w = self.config.learning_rate * correlation
        
        # Apply natural decay
        decay = self.config.decay_rate * connection.strength
        
        # Update strength with bounds
        new_strength = connection.strength + delta_w - decay
        connection.strength = np.clip(
            new_strength,
            self.config.min_connection_strength,
            self.config.max_connection_strength
        )
        
        # Update connection metadata
        connection.pre_activation = pre_activation
        connection.post_activation = post_activation
        connection.correlation = correlation
        connection.activation_count += 1
        connection.last_activated = datetime.now()
        
        # Track performance
        connection.performance_history.append(outcome_quality)
        if len(connection.performance_history) > 50:
            connection.performance_history = connection.performance_history[-50:]
        
        # Update success count and context tracking
        if outcome_quality > self.config.strengthening_threshold:
            connection.success_count += 1
            if context_hash:
                connection.successful_contexts.add(context_hash)
        elif outcome_quality < self.config.weakening_threshold and context_hash:
            connection.failed_contexts.add(context_hash)
        
        # Update connection type classification
        connection.connection_type = self.classify_connection_type(connection)
        
        # Log learning event
        self.learning_history.append({
            'timestamp': datetime.now().isoformat(),
            'vni1': vni1,
            'vni2': vni2,
            'delta_w': delta_w,
            'new_strength': connection.strength,
            'correlation': correlation,
            'outcome': outcome_quality,
            'context_hash': context_hash
        })
        
        logger.debug(
            f"📚 Hebbian update: {vni1} <-> {vni2} | "
            f"Strength: {connection.strength:.3f} | "
            f"ΔW: {delta_w:+.3f} | "
            f"Type: {connection.connection_type.value}"
        )
    
    def stdp_update(
        self,
        vni1: str,
        vni2: str,
        time_diff: float,
        outcome_quality: float,
        context_hash: str = ""
    ):
        """
        Spike-Timing-Dependent Plasticity (STDP)
        Strengthen connections where VNI1 fires before VNI2 if outcome is good
        
        Args:
            vni1: First VNI (earlier activation)
            vni2: Second VNI (later activation)
            time_diff: Time difference in seconds (positive if vni1 before vni2)
            outcome_quality: Quality of outcome
            context_hash: Hash of the query context
        """
        connection = self.get_or_create_connection(vni1, vni2)
        
        # STDP window function (exponential)
        tau = 20.0  # Time constant in seconds
        
        if time_diff > 0:  # Pre before post - strengthen if good outcome
            stdp_factor = np.exp(-time_diff / tau) * outcome_quality
        else:  # Post before pre - weaken
            stdp_factor = -np.exp(time_diff / tau) * 0.5
        
        delta_w = self.config.learning_rate * stdp_factor
        
        new_strength = connection.strength + delta_w
        connection.strength = np.clip(
            new_strength,
            self.config.min_connection_strength,
            self.config.max_connection_strength
        )
        
        # Update context tracking for STDP
        if outcome_quality > self.config.strengthening_threshold and context_hash:
            connection.successful_contexts.add(context_hash)
        elif outcome_quality < self.config.weakening_threshold and context_hash:
            connection.failed_contexts.add(context_hash)
        
        logger.debug(
            f"⏱️  STDP update: {vni1} -> {vni2} | "
            f"Δt: {time_diff:.2f}s | "
            f"Strength: {connection.strength:.3f}"
        )
    
    def learn_from_interaction(
        self,
        activated_vnis: List[str],
        vni_outputs: Dict[str, Dict[str, Any]],
        overall_quality: float,
        query_context: Dict[str, Any]
    ):
        """Learn from a complete interaction using all activated VNIs
        Args:
            activated_vnis: List of VNI IDs that were activated
            vni_outputs: Dictionary of VNI outputs with confidence scores
            overall_quality: Overall quality of the response (0.0 to 1.0)
            query_context: Context of the query"""
        if len(activated_vnis) < 2:
            return  # Need at least 2 VNIs to form connections
        
        # Extract activation levels (using confidence as proxy)
        activations = {
            vni_id: vni_outputs.get(vni_id, {}).get('confidence_score', 0.5)
            for vni_id in activated_vnis
        }
        
        # Update activity traces
        timestamp = datetime.now()
        for vni_id in activated_vnis:
            self.activity_traces[vni_id].append({
                'time': timestamp,
                'activation': activations[vni_id],
                'context_hash': self._hash_context(query_context)
            })
        
        # Create context hash for this interaction
        context_hash = self._hash_context(query_context)
        
        # Apply Hebbian learning to all pairs
        for i, vni1 in enumerate(activated_vnis):
            for vni2 in activated_vnis[i+1:]:
                self.hebbian_update(
                    vni1, vni2,
                    activations[vni1],
                    activations[vni2],
                    overall_quality,
                    context_hash
                )
        
        # Apply STDP for sequential activations with actual timing
        for i in range(len(activated_vnis) - 1):
            vni1 = activated_vnis[i]
            vni2 = activated_vnis[i + 1]
            
            # Get actual timing from activity traces
            time_diff = 0.1  # Default small time difference
            if vni1 in self.activity_traces and vni2 in self.activity_traces:
                v1_traces = list(self.activity_traces[vni1])
                v2_traces = list(self.activity_traces[vni2])
                if v1_traces and v2_traces:
                    time_diff = (v2_traces[-1]['time'] - v1_traces[-1]['time']).total_seconds()
            
            self.stdp_update(vni1, vni2, time_diff, overall_quality, context_hash)
        
        # Detect and store patterns
        if overall_quality > self.config.strengthening_threshold:
            pattern_quality = self._store_successful_pattern(activated_vnis, query_context, overall_quality)
            # Also store in context patterns
            if context_hash:
                self.context_patterns[context_hash].append((activated_vnis, pattern_quality))
        elif overall_quality < self.config.weakening_threshold:
            self._store_failed_pattern(activated_vnis, query_context, overall_quality)
        
        # Update connection types based on new data
        self._update_connection_types()
        
        # Prune weak connections periodically
        if len(self.learning_history) % 50 == 0:
            self.prune_weak_connections()

        # Implementing meta cognitive:
        if hasattr(self, 'meta_cognitive'):
            self.meta_cognitive.record_interaction_outcome(
                activated_vnis=activated_vnis,
                vni_outputs=vni_outputs,
                overall_quality=overall_quality,
                query_context=query_context
            )

    def _store_successful_pattern(
        self,
        vni_sequence: List[str],
        context: Dict[str, Any],
        quality: float
    ) -> float:
        """Store a successful VNI activation pattern"""
        pattern_hash = self._hash_pattern(vni_sequence, context)
        pattern = {
            'vnis': vni_sequence,
            'quality': quality,
            'context_hash': self._hash_context(context),
            'pattern_hash': pattern_hash,
            'timestamp': datetime.now().isoformat(),
            'usage_count': 1
        }
        
        # Check if pattern already exists
        for existing in self.successful_patterns:
            if existing['pattern_hash'] == pattern_hash:
                existing['usage_count'] += 1
                existing['quality'] = (existing['quality'] + quality) / 2
                return existing['quality']
        
        self.successful_patterns.append(pattern)
        logger.debug(f"✅ Stored successful pattern: {' -> '.join(vni_sequence)}")
        return quality
    
    def _store_failed_pattern(
        self,
        vni_sequence: List[str],
        context: Dict[str, Any],
        quality: float
    ):
        """Store a failed VNI activation pattern to avoid in future"""
        pattern_hash = self._hash_pattern(vni_sequence, context)
        pattern = {
            'vnis': vni_sequence,
            'quality': quality,
            'context_hash': self._hash_context(context),
            'pattern_hash': pattern_hash,
            'timestamp': datetime.now().isoformat()
        }
        
        self.failed_patterns.append(pattern)
        
        # Keep only recent failures
        if len(self.failed_patterns) > 100:
            self.failed_patterns = self.failed_patterns[-100:]
        
        logger.debug(f"❌ Stored failed pattern: {' -> '.join(vni_sequence)}")
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Create hash of context for pattern matching"""
        # Create a stable hash based on context features
        context_str = ""
        if 'query_complexity' in context:
            context_str += f"complexity:{context['query_complexity']:.2f}"
        if 'detected_domains' in context:
            context_str += f"|domains:{sorted(context['detected_domains'])}"
        if 'session_id' in context:
            context_str += f"|session:{context['session_id']}"
        
        return hashlib.md5(context_str.encode()).hexdigest()[:16]
    
    def _hash_pattern(self, vni_sequence: List[str], context: Dict[str, Any]) -> str:
        """Create hash for a specific pattern"""
        pattern_str = "->".join(sorted(vni_sequence))
        context_str = self._hash_context(context)
        return hashlib.md5(f"{pattern_str}|{context_str}".encode()).hexdigest()[:16]
    
    def _update_connection_types(self):
        """Update connection types based on current metrics"""
        for connection in self.connections.values():
            connection.connection_type = self.classify_connection_type(connection)
    
    def prune_weak_connections(self):
        """Remove connections that have become too weak"""
        to_prune = []
        
        for key, conn in self.connections.items():
            # Prune if strength is below threshold and not recently used
            if conn.strength < self.config.pruning_threshold:
                if conn.last_activated:
                    time_since_use = (datetime.now() - conn.last_activated).total_seconds()
                    if time_since_use > 3600:  # 1 hour
                        to_prune.append(key)
                else:
                    to_prune.append(key)
        
        for key in to_prune:
            del self.connections[key]
            logger.debug(f"✂️  Pruned connection: {key[0]} <-> {key[1]}")
        
        if to_prune:
            logger.info(f"🗑️  Pruned {len(to_prune)} weak connections")
    
    def suggest_vni_routing(
        self,
        query_context: Dict[str, Any],
        available_vnis: List[str]
        ) -> List[Tuple[str, float]]:
        """Use Hebbian connection strengths to suggest VNIs"""
        if not available_vnis:
            return []
        
        context_hash = self._hash_context(query_context)
    
        # Calculate Hebbian-influenced scores
        vni_scores = {}
            
        # Check for matching successful patterns for this context
        if context_hash in self.context_patterns:
            patterns = self.context_patterns[context_hash]
            # Sort patterns by quality
            patterns.sort(key=lambda x: x[1], reverse=True)
            
            for pattern_vnis, pattern_quality in patterns:
                # Filter to available VNIs
                suggestions = [(vni, pattern_quality) for vni in pattern_vnis
                             if vni in available_vnis]
                if suggestions:
                    logger.debug(f"📋 Using context-specific pattern with quality {pattern_quality:.2f}")
                    return suggestions
        
        # Check for matching successful patterns (general)
        for pattern in sorted(self.successful_patterns, 
                            key=lambda p: p['quality'] * p['usage_count'], 
                            reverse=True):
            # Return VNIs from successful pattern
            suggestions = [(vni, pattern['quality']) for vni in pattern['vnis']
                         if vni in available_vnis]
            if suggestions:
                logger.debug(f"📋 Using learned pattern with quality {pattern['quality']:.2f}")
                return suggestions
        
        # Calculate scores based on connection strengths with context awareness
        vni_scores = {}
        
        for vni in available_vnis:
            score = 0.5  # Base score
            connection_count = 0
            
            # Look at connections with other VNIs
            for other_vni in available_vnis:
                if vni == other_vni:
                    continue
                
                key = tuple(sorted([vni, other_vni]))
                if key in self.connections:
                    conn = self.connections[key]
                    # Use context-specific success rate if available
                    context_success = conn.context_success_rate(context_hash)
                    base_success = conn.success_rate()
                    success_rate = context_success if context_hash in conn.successful_contexts or context_hash in conn.failed_contexts else base_success
                    
                    # Weight by connection type
                    type_weight = {
                        ConnectionType.COLLABORATIVE: 1.2,
                        ConnectionType.COMPLEMENTARY: 1.1,
                        ConnectionType.SEQUENTIAL: 1.0,
                        ConnectionType.COMPETITIVE: 0.8
                    }.get(conn.connection_type, 1.0)
                    
                    score += conn.strength * success_rate * type_weight
                    connection_count += 1
            
            # Normalize by connection count
            if connection_count > 0:
                score /= connection_count
            
            vni_scores[vni] = score
        
        # Sort by score
        suggestions = sorted(vni_scores.items(), key=lambda x: x[1], reverse=True)
        
        return suggestions
    
    def get_strongest_connections(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get the strongest learned connections"""
        sorted_connections = sorted(
            self.connections.values(),
            key=lambda c: c.strength * c.success_rate() * (1 + len(c.successful_contexts)/10),
            reverse=True
        )
        
        return [
            {
                'source': conn.source_vni,
                'target': conn.target_vni,
                'strength': conn.strength,
                'success_rate': conn.success_rate(),
                'context_successful': len(conn.successful_contexts),
                'context_failed': len(conn.failed_contexts),
                'activations': conn.activation_count,
                'type': conn.connection_type.value,
                'last_activated': conn.last_activated.isoformat() if conn.last_activated else None
            }
            for conn in sorted_connections[:top_k]
        ]
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        total_connections = len(self.connections)
        strong_connections = sum(1 for c in self.connections.values() 
                                if c.strength > 0.7)
        weak_connections = sum(1 for c in self.connections.values() 
                              if c.strength < 0.3)
        
        avg_strength = (sum(c.strength for c in self.connections.values()) / total_connections
                       if total_connections > 0 else 0.0)
        
        # Connection type distribution
        type_distribution = defaultdict(int)
        for conn in self.connections.values():
            type_distribution[conn.connection_type.value] += 1
        
        return {
            'total_connections': total_connections,
            'strong_connections': strong_connections,
            'weak_connections': weak_connections,
            'average_strength': avg_strength,
            'successful_patterns': len(self.successful_patterns),
            'failed_patterns': len(self.failed_patterns),
            'learning_events': len(self.learning_history),
            'pruning_threshold': self.config.pruning_threshold,
            'connection_types': dict(type_distribution),
            'context_patterns': sum(len(v) for v in self.context_patterns.values())
        }
    
    def visualize_network(self) -> str:
        """Create ASCII visualization of the network"""
        if not self.connections:
            return "No connections to visualize"
        
        lines = ["🕸️  VNI Synaptic Network", "=" * 60]
        
        # Get all unique VNIs
        vnis = set()
        for conn in self.connections.values():
            vnis.add(conn.source_vni)
            vnis.add(conn.target_vni)
        
        lines.append(f"VNIs: {len(vnis)} | Connections: {len(self.connections)}")
        lines.append("")
        
        # Connection type summary
        type_counts = defaultdict(int)
        for conn in self.connections.values():
            type_counts[conn.connection_type.value] += 1
        
        lines.append("Connection Types:")
        for conn_type, count in sorted(type_counts.items()):
            lines.append(f"  • {conn_type}: {count}")
        
        lines.append("")
        
        # Show strongest connections
        strongest = self.get_strongest_connections(8)
        lines.append("Strongest Connections:")
        for i, conn in enumerate(strongest, 1):
            strength_bar = "█" * int(conn['strength'] * 10)
            context_info = f"[Ctx✓:{conn['context_successful']} ✗:{conn['context_failed']}]"
            lines.append(
                f"{i:2d}. {conn['source']:15} <-> {conn['target']:15}: "
                f"{strength_bar:10} {conn['strength']:.3f} "
                f"({conn['success_rate']:.0%}) {conn['type'][:4]} {context_info}"
            )
        
        return "\n".join(lines)
    
    def save_network(self, filename: str):
        """Save the learned network"""
        network_data = {
            'connections': {
                str(k): asdict(v) for k, v in self.connections.items()
            },
            'successful_patterns': self.successful_patterns,
            'failed_patterns': self.failed_patterns,
            'context_patterns': dict(self.context_patterns),
            'save_time': datetime.now().isoformat(),
            'config': asdict(self.config)
        }
        
        with open(filename, 'w') as f:
            json.dump(network_data, f, indent=2, default=str)
        
        logger.info(f"💾 Saved synaptic network to {filename}")
        return filename
    
    def load_network(self, filename: str):
        """Load a previously learned network"""
        try:
            with open(filename, 'r') as f:
                network_data = json.load(f)
            
            # Reconstruct connections
            self.connections = {}
            for key_str, conn_dict in network_data['connections'].items():
                key = eval(key_str)  # Convert string back to tuple
                
                # Handle connection type enum
                if 'connection_type' in conn_dict:
                    conn_dict['connection_type'] = ConnectionType(conn_dict['connection_type'])
                
                conn = SynapticConnection(**conn_dict)
                self.connections[key] = conn
            
            self.successful_patterns = network_data['successful_patterns']
            self.failed_patterns = network_data['failed_patterns']
            self.context_patterns = defaultdict(list, network_data.get('context_patterns', {}))
            
            logger.info(f"📂 Loaded synaptic network from {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to load network: {e}")
            return False

# ==================== NEURAL COMPONENTS ====================
class ConsensusCalculator(nn.Module):
    """Calculate consensus between multiple VNI outputs using neural scoring"""
    def __init__(self, config: AggregatorConfig):
        super().__init__()
        self.config = config
        dropout_rate = getattr(config, 'dropout_rate', 0.1)        
        self.consensus_network = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def calculate_consensus(self, vni_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall consensus between VNI outputs"""
        if not vni_outputs:
            return self._empty_consensus()
        
        confidence_scores = {vni_id: output.get('confidence_score', 0.5) 
                           for vni_id, output in vni_outputs.items()}
        domains = {vni_id: self._get_domain(vni_id) for vni_id in vni_outputs}
        
        total_confidence = sum(confidence_scores.values())
        if total_confidence == 0:
            return self._empty_consensus()
        
        avg_confidence = total_confidence / len(confidence_scores)
        domain_counts = defaultdict(int)
        for domain in domains.values():
            domain_counts[domain] += 1
        
        consensus_features = self._extract_consensus_features(vni_outputs, confidence_scores, domains)
        with torch.no_grad():
            consensus_tensor = torch.tensor(consensus_features, dtype=torch.float32).unsqueeze(0)
            neural_consensus = self.consensus_network(consensus_tensor).item()
        
        combined_consensus = (neural_consensus + avg_confidence) / 2
        
        if combined_consensus > 0.8:
            consensus_level = 'strong'
        elif combined_consensus > 0.6:
            consensus_level = 'moderate'
        elif combined_consensus > 0.4:
            consensus_level = 'weak'
        else:
            consensus_level = 'none'
        
        agreeing_vnis = [vni_id for vni_id, conf in confidence_scores.items() 
                        if conf >= avg_confidence]
        
        # Calculate domain consensus
        domain_consensus = {}
        for domain in set(domains.values()):
            domain_vnis = [vni_id for vni_id, d in domains.items() if d == domain]
            if domain_vnis:
                domain_confidences = [confidence_scores[vni_id] for vni_id in domain_vnis]
                domain_consensus[domain] = sum(domain_confidences) / len(domain_confidences)
        
        return {
            'consensus_level': consensus_level,
            'consensus_score': combined_consensus,
            'average_confidence': avg_confidence,
            'domain_distribution': dict(domain_counts),
            'domain_consensus': domain_consensus,
            'agreeing_vnis': agreeing_vnis,
            'total_vnis': len(vni_outputs)
        }
    
    def _extract_consensus_features(self, vni_outputs, confidence_scores, domains) -> List[float]:
        """Extract features for neural consensus calculation"""
        confidences = list(confidence_scores.values())
        
        features = [
            np.mean(confidences),  # Average confidence
            np.std(confidences),   # Confidence variance
            max(confidences),      # Maximum confidence
            min(confidences),      # Minimum confidence
            len(set(domains.values())) / max(len(domains), 1),  # Domain diversity
            sum(1 for o in vni_outputs.values() 
                if o.get('vni_metadata', {}).get('success', False)) / max(len(vni_outputs), 1),  # Success rate
            np.median(confidences) if confidences else 0.0,  # Median confidence
            len([c for c in confidences if c > 0.7]) / max(len(confidences), 1)  # High confidence ratio
        ]
        
        return features[:8]
    
    def _get_domain(self, vni_id: str) -> str:
        """Extract domain from VNI ID"""
        vni_id_lower = vni_id.lower()
        if 'medical' in vni_id_lower or 'med' in vni_id_lower:
            return 'medical'
        elif 'legal' in vni_id_lower or 'law' in vni_id_lower:
            return 'legal'
        elif 'technical' in vni_id_lower or 'tech' in vni_id_lower:
            return 'technical'
        elif 'analyt' in vni_id_lower:
            return 'analytical'
        elif 'creative' in vni_id_lower:
            return 'creative'
        elif 'financial' in vni_id_lower or 'finance' in vni_id_lower:
            return 'financial'
        return 'general'
    
    def _empty_consensus(self) -> Dict[str, Any]:
        """Return empty consensus result"""
        return {
            'consensus_level': 'none',
            'consensus_score': 0.0,
            'average_confidence': 0.0,
            'domain_distribution': {},
            'domain_consensus': {},
            'agreeing_vnis': [],
            'total_vnis': 0
        }

# ==================== ENHANCED MAIN AGGREGATOR ====================
class UnifiedAggregator(nn.Module):
    """Unified response aggregator with full Hebbian synaptic learning
    Main entry point for all aggregation operations with learning capabilities"""
    def __init__(self, config: AggregatorConfig = None, vni_manager=None):
        super().__init__()
        # If no config provided, create one with Hebbian learning ENABLED
        if config is None:
            from neuron.shared.synaptic_config import SynapticConfig
            config = SynapticConfig(
                aggregator_id="unified_aggregator",
                # === CORE AGGREGATION ===
                consensus_threshold=0.7,
                conflict_resolution_strategy="confidence_weighted",
                min_confidence_threshold=0.4,
                max_output_length=500,
                enable_cross_domain_synthesis=True,
                enable_conflict_detection=True,
                enable_confidence_calibration=True,
                
                # === BIOLOGICAL ROUTING (DISABLED) ===
                enable_biological_routing=False,
                attention_routing_weight=0.25,
                activation_routing_weight=0.25,
                memory_routing_weight=0.2,
                hebbian_routing_weight=0.3,
                
                # === HEBBIAN LEARNING (ENABLED) ===
                enable_hebbian_learning=True,
                learning_rate=0.1,
                decay_rate=0.01,
                strengthening_threshold=0.7,
                weakening_threshold=0.4,
                pruning_threshold=0.1,
                max_connection_strength=1.0,
                min_connection_strength=0.0,
                
                # === AUTO SPAWNING ===
                enable_auto_spawning=True,
                enable_visualization=True,
                max_clusters=10,
                
                # === SESSION MANAGEMENT ===
                session_timeout_hours=2,
                history_size=1000,
                session_history_size=50,
                pattern_detection_window=100,
                
                # === NEURAL NETWORK CONFIG ===
                embedding_dim=512,
                hidden_dim=256,
                dropout_rate=0.1,
                figure_size=(12, 8),
                visualization_dpi=150,
                animation_fps=2,
                
                # === META LEARNING ===
                enable_meta_learning=True,
                optimization_interval=100,
                performance_evaluation_window=50,
                hyperparameter_optimization=True,
                
                # === DOMAIN KEYWORDS (keep your existing ones) ===
                domain_keywords={
                    'medical': ['medical', 'health', 'doctor', 'patient', 'symptom', 'diagnosis', 'treatment', 'hospital'],
                    'legal': ['legal', 'law', 'contract', 'rights', 'agreement', 'court', 'liability', 'compliance'],
                    'technical': ['code', 'programming', 'software', 'debug', 'error', 'bug', 'algorithm', 'technical', 'api'],
                    'analytical': ['analyze', 'compare', 'evaluate', 'assess', 'statistics', 'data', 'analysis', 'trend', 'pattern'],
                    'financial': ['financial', 'finance', 'investment', 'stock', 'market', 'portfolio', 'risk', 'return'],
                    'creative': ['creative', 'story', 'narrative', 'design', 'art', 'write', 'compose', 'imagine'],
                    'research': ['research', 'study', 'experiment', 'hypothesis', 'methodology', 'findings']
                },
                
                # === CONNECTION TYPE THRESHOLDS ===
                connection_type_thresholds={
                    'collaborative_threshold': 0.7,
                    'competitive_threshold': 0.3,
                    'sequential_detection_window': 5
                },
                
                # === STDP CONFIG ===
                stdp_time_constant=20.0,
                stdp_strengthening_factor=1.0,
                stdp_weakening_factor=0.5
            )
        self.config = config

        # Initialize aggregator's attention mechanism
        self.agg_attention = AggregatorAttention(dim=256, num_heads=8)
    
        # SINGLE SOURCE OF TRUTH for LLM
        self.llm_gateway = get_gateway(getattr(config, 'llm_configs', None))
          
        # Neural components
        self.conflict_detector = ConflictDetector(self.config)
        self.consensus_calculator = ConsensusCalculator(self.config)
        self.response_synthesizer = ResponseSynthesizer(self.config, self.llm_gateway)
        
        # Hebbian learning engine
        self.hebbian_engine = HebbianLearningEngine(self.config)        
        self.greeting_preprocessor = GreetingPreprocessor()
        self.hybrid_attention = DemoHybridAttention(dim=256, num_heads=8)  # Main attention system
        self.attention_engine = HybridAttentionEngine(config)  # Attention engine
        self.activation_router = SmartActivationRouter(config)
        self.vni_memory = VniMemory(
            vni_id="neural_mesh_aggregator",  # Provide actual vni_id
            storage_manager=None,
            config=config  # Pass config separately if needed
        )
        # To enable "Thinking of 'Thinking' in BabyBIONN"
        self.meta_cognitive = integrate_with_aggregator(self)
        self.p2p_node = None   # will be set from main.py if P2P enabled

        # Orchestration components
        self.vni_manager = vni_manager
        self.neural_mesh = None
        self.visualizer = None
        
        # Cluster management
        self.clusters: Dict[str, Dict] = {}
        self.cluster_performance: Dict[str, ClusterPerformance] = {}
        
        # Session management
        self.sessions: Dict[str, Dict] = {}
        self.session_timeout = timedelta(hours=self.config.session_timeout_hours)
        
        # Metrics
        self.learning_metrics = LearningMetrics()
        self.routing_table: Dict[str, List[str]] = defaultdict(list)
        self.query_history = deque(maxlen=self.config.history_size)
        
        # Performance tracking
        self.response_quality_history = deque(maxlen=100)
        self.connection_strength_history = deque(maxlen=100)
        
        # Initialize if VNI manager provided
        if self.vni_manager:
            self._initialize_from_vni_manager()
        
        logger.info(f"✅ UnifiedAggregator with Hebbian learning initialized: {self.config.aggregator_id}")
    
    def _initialize_from_vni_manager(self):
        """Initialize clusters from VNI manager"""
        if not self.vni_manager:
            return
        
        if hasattr(self.vni_manager, 'vni_instances'):
            for vni_id in self.vni_manager.vni_instances.keys():
                domain = self._get_domain(vni_id)
                cluster_id = f"{domain}_cluster"
                
                if cluster_id not in self.clusters:
                    self.clusters[cluster_id] = {
                        'type': domain,
                        'instance_ids': [],
                        'created': datetime.now(),
                        'specializations': [domain],
                        'total_responses': 0,
                        'successful_responses': 0
                    }
                    self.cluster_performance[cluster_id] = ClusterPerformance(
                        cluster_id=cluster_id,
                        specialization=domain
                    )
                
                self.clusters[cluster_id]['instance_ids'].append(vni_id)
                self.routing_table[domain].append(cluster_id)
        
        logger.info(f"📋 Initialized {len(self.clusters)} clusters from VNI manager")

    def _extract_biological_states(self, vni_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Extract biological telemetry from VNI outputs"""
        biological_states = {}
        
        # First, get overall attention analysis for all outputs
        overall_attention = self.agg_attention.analyze_outputs(vni_outputs)
        
        for vni_id, output in vni_outputs.items():
            # Try to extract biological state from vni_metadata
            vni_metadata = output.get('vni_metadata', {})
            biological_data = vni_metadata.get('biological_state', {})
            
            if biological_data:
                # VNI already provides biological state
                biological_states[vni_id] = biological_data
            else:
                # Infer biological state from available data
                biological_states[vni_id] = self._infer_biological_state(vni_id, output, overall_attention)
        
        return biological_states
    
    def _infer_biological_state(self, vni_id: str, output: Dict[str, Any], 
                            overall_attention: Dict[str, Any]) -> Dict[str, Any]:
        """Infer biological state when not explicitly provided"""
        confidence = output.get('confidence_score', 0.5)
        domain = self._get_domain(vni_id)
        
        # Get this VNI's attention weight from overall analysis
        vni_attention_weight = overall_attention.get('attention_weights', {}).get(vni_id, 0.5)
        
        attention = {
            'weight': vni_attention_weight,
            'focus': overall_attention.get('primary_focus', 'general'),
            'overall_confidence': overall_attention.get('confidence', 0.5),
            'domain_focus': vni_attention_weight * confidence  # Add domain_focus for neurochemical
        }
        
        # Use FunctionRegistry for activation
        activation = self.activation_router.calculate_vni_activation(
            vni_id=vni_id,
            output=output,
            confidence=confidence
        )
    
        # Get memory state using available methods
        if hasattr(self.vni_memory, 'get_memory_stats'):
            # Use get_memory_stats if available
            memory_stats = self.vni_memory.get_memory_stats()
            memory = {
                'similar_cases_recalled': memory_stats.get('total_memories', 0) if memory_stats else 0,
                'retention_strength': 0.5,
                'context_match': confidence * 0.8,
                'memory_recall_level': confidence * 0.7
            }
        elif hasattr(self.vni_memory, 'retrieve_similar'):
            # Try to retrieve similar memories
            try:
                similar = self.vni_memory.retrieve_similar(
                    query=output.get('response', ''),
                    category='interactions',
                    max_results=3
                )
                memory_hits = len(similar) if similar else 0
                memory = {
                    'similar_cases_recalled': memory_hits,
                    'retention_strength': min(1.0, memory_hits * 0.3),
                    'context_match': confidence * 0.8,
                    'memory_recall_level': confidence * 0.7
                }
            except Exception as e:
                logger.debug(f"Memory retrieval failed: {e}")
                memory = {
                    'similar_cases_recalled': 0,
                    'retention_strength': 0.5,
                    'context_match': confidence * 0.5,
                    'memory_recall_level': confidence * 0.5
                }
        else:
            # Simple fallback memory state
            memory = {
                'similar_cases_recalled': 0,
                'retention_strength': 0.5,
                'context_match': confidence * 0.5,
                'memory_recall_level': confidence * 0.5
            }
        
        # ✅ FIX: Define neurochemical HERE, outside all conditionals
        neurochemical = {
            'dopamine': confidence * 0.7,
            'acetylcholine': attention.get('domain_focus', 0.5) * 0.8,
            'serotonin': (confidence + activation.get('current', 0.5)) / 2,
            'norepinephrine': activation.get('response_energy', 0.5),
            'overall_neurochemical_balance': (confidence + activation.get('current', 0.5)) / 2
        }
        
        # Try to use hybrid_attention if available (optional enhancement)
        if hasattr(self, 'hybrid_attention') and hasattr(self.hybrid_attention, 'infer_neurochemical_state'):
            try:
                neurochemical = self.hybrid_attention.infer_neurochemical_state(
                    confidence=confidence,
                    attention=attention,
                    activation=activation
                )
            except Exception as e:
                logger.debug(f"Hybrid attention neurochemical failed, using fallback: {e}")
        
        return {
            'attention': attention,
            'activation': activation,
            'memory': memory,
            'neurochemical': neurochemical,
            'inferred': True,
            'components_used': ['AggregatorAttention', 'FunctionRegistry', 'VniMemory']
        }

    def _build_reasoning_output(self, query: str, successful_outputs: Dict, consensus: Dict, conflicts: List) -> ReasoningOutput:
        """Construct a ReasoningOutput object from the successful VNI outputs and analysis."""
        vni_opinions = []
        for vni_id, output in successful_outputs.items():
            # Extract the fields we added to each VNI
            opinion = VNIOpinion(
                vni_id=vni_id,
                domain=output.get('domain', self._get_domain(vni_id)),
                confidence=output.get('confidence_score', 0.5),
                opinion_text=output.get('opinion_text', ''),
                metadata=output.get('vni_metadata', {})
            )
            vni_opinions.append(opinion)
    
        # Determine primary domain (most frequent domain among high‑confidence opinions)
        domain_counts = {}
        for op in vni_opinions:
            if op.confidence > 0.6:   # threshold for counting
                domain_counts[op.domain] = domain_counts.get(op.domain, 0) + 1
        primary_domain = max(domain_counts.items(), key=lambda x: x[1])[0] if domain_counts else 'general'
    
        # Consensus summary from the consensus analysis
        consensus_level = consensus.get('consensus_level', 'unknown')
        consensus_score = consensus.get('consensus_score', 0.0)
        agreeing_vnis = consensus.get('agreeing_vnis', [])
        total_vnis = consensus.get('total_vnis', len(vni_opinions))
        if total_vnis > 0:
            consensus_summary = f"{len(agreeing_vnis)} out of {total_vnis} experts agree. Consensus level: {consensus_level}."
        else:
            consensus_summary = "No consensus data."
    
        # Retrieve memory snippets if possible (optional)
        memory_snippets = []
        if hasattr(self, 'vni_memory') and self.vni_memory:
            try:
                # Assuming a method to retrieve similar past interactions exists
                memories = self.vni_memory.retrieve_similar(query, category='interactions', max_results=3)
                if memories:
                    memory_snippets = [mem.get('query', '') for mem in memories]
            except Exception as e:
                logger.debug(f"Could not retrieve memory snippets: {e}")
    
        # Overall confidence – can use the consensus score
        overall_confidence = consensus_score
    
        return ReasoningOutput(
            query=query,
            primary_domain=primary_domain,
            vni_opinions=vni_opinions,
            consensus_level=consensus_level,
            consensus_summary=consensus_summary,
            memory_snippets=memory_snippets,
            overall_confidence=overall_confidence
        )

    def _infer_attention_patterns(self, output: Dict[str, Any], domain: str) -> Dict[str, float]:
        """Infer attention focus areas from output"""
        attention_patterns = {
            'domain_focus': 0.7,
            'confidence_alignment': 0.6,
            'detail_level': 0.5,
            'cross_referencing': 0.3
        }
        
        # Extract advice text
        advice_text = ""
        advice_fields = ['medical_advice', 'legal_advice', 'technical_advice', 
                        'financial_advice', 'analytical_advice', 'response']
        
        for field in advice_fields:
            if field in output and output[field]:
                advice_text = str(output[field])
                break
        
        if advice_text:
            # Analyze text characteristics
            words = advice_text.split()
            sentences = advice_text.count('.') + advice_text.count('?') + advice_text.count('!')
            
            # Domain focus - how much is about its specialty
            domain_keywords = {
                'medical': ['patient', 'symptom', 'treatment', 'diagnosis', 'health'],
                'legal': ['law', 'rights', 'contract', 'liability', 'legal'],
                'technical': ['code', 'system', 'algorithm', 'technical', 'software'],
                'financial': ['investment', 'risk', 'financial', 'money', 'market'],
                'analytical': ['analysis', 'data', 'statistical', 'trend', 'compare']
            }
            
            domain_focus_score = 0.5
            if domain in domain_keywords:
                matches = sum(1 for word in words if word.lower() in domain_keywords[domain])
                if matches > 0:
                    domain_focus_score = min(0.9, matches / len(words) * 3)
            
            attention_patterns['domain_focus'] = domain_focus_score
            
            # Detail level based on response length and structure
            if len(words) > 50:
                attention_patterns['detail_level'] = 0.8
            elif len(words) > 20:
                attention_patterns['detail_level'] = 0.6
            else:
                attention_patterns['detail_level'] = 0.4
        
        return attention_patterns

    def _infer_activation_level(self, energetic_text, output: Dict[str, Any], confidence: float) -> Dict[str, Any]:
        """Infer activation level from output characteristics"""
        # Activation typically correlates with confidence
        base_activation = confidence
        
        # Adjust based on response characteristics
        response_energy = 0.5
        advice_text = ""
        
        for field in ['medical_advice', 'legal_advice', 'technical_advice', 'response']:
            if field in output:
                advice_text = str(output[field])
                break
        
        if advice_text:
            # Check for energetic language
            energetic_keywords = ['must', 'should', 'recommend', 'critical', 'important', 'urgent']
            energetic_count = sum(1 for word in energetic_text.split() 
                                 if word.lower() in energetic_keywords)
            
            if energetic_count > 0:
                response_energy = min(0.9, 0.5 + (energetic_count * 0.1))
        
        return {
            'current': base_activation * 0.8 + response_energy * 0.2,
            'decay_rate': 0.01 + (1 - base_activation) * 0.02,  # Higher confidence = slower decay
            'response_energy': response_energy,
            'sustained_focus': base_activation > 0.7
        }
    
    def suggest_vni_routing_biological(self, query_context: Dict[str, Any], available_vnis: List[str]) -> List[Tuple[str, float]]:
        """Biological-aware VNI routing using all systems"""
        
        # First, try Hebbian routing from the engine
        hebbian_suggestions = self.hebbian_engine.suggest_vni_routing(query_context, available_vnis)
        
        # If no biological systems or they're disabled, return Hebbian suggestions
        if not hasattr(self, 'hybrid_attention') or not self.config.enable_biological_routing:
            return hebbian_suggestions
        
        # Get biological predictions from each system
        biological_suggestions = {}
        
        try:
            # 1. Attention-based suggestions (if available)
            if hasattr(self.hybrid_attention, 'suggest_vnis_by_attention'):
                attention_suggestions = self.hybrid_attention.suggest_vnis_by_attention(
                    query_context=query_context,
                    available_vnis=available_vnis
                )
            else:
                attention_suggestions = hebbian_suggestions  # Fallback
                
            # 2. Activation-based suggestions (if available)
            if hasattr(self.activation_router, 'predict_best_vnis'):
                activation_suggestions = self.activation_router.predict_best_vnis(
                    query_context=query_context,
                    available_vnis=available_vnis
                )
            else:
                activation_suggestions = dict(hebbian_suggestions)  # Convert to dict
                
            # 3. Memory-based suggestions (if available)
            if hasattr(self.vni_memory, 'suggest_vnis_by_memory'):
                memory_suggestions = self.vni_memory.suggest_vnis_by_memory(
                    query_context=query_context,
                    available_vnis=available_vnis
                )
            else:
                memory_suggestions = hebbian_suggestions  # Fallback
                
        except Exception as e:
            logger.warning(f"Biological routing failed: {e}, falling back to Hebbian")
            return hebbian_suggestions
        
        # Combine all suggestions with weights
        for vni in available_vnis:
            total_score = 0.0
            weight_sum = 0.0
        
            # Attention weight: 25%
            attention_score = next((s for v, s in attention_suggestions if v == vni), 0.5)
            total_score += attention_score * 0.25
            weight_sum += 0.25
            
            # Activation weight: 25%
            activation_score = activation_suggestions.get(vni, 0.5)
            total_score += activation_score * 0.25
            weight_sum += 0.25
            
            # Memory weight: 20%
            memory_score = next((s for v, s in memory_suggestions if v == vni), 0.5)
            total_score += memory_score * 0.20
            weight_sum += 0.20
            
            # Hebbian weight: 30% (slightly higher as baseline)
            hebbian_score = next((s for v, s in hebbian_suggestions if v == vni), 0.5)
            total_score += hebbian_score * 0.30
            weight_sum += 0.30
            
            # Normalize
            biological_suggestions[vni] = total_score / weight_sum if weight_sum > 0 else 0.5
        
        # Return sorted
        return sorted(biological_suggestions.items(), key=lambda x: x[1], reverse=True)    

    def _infer_memory_engagement(self, output: Dict[str, Any], domain: str) -> Dict[str, float]:
        """Infer memory system engagement"""
        # Check for evidence of memory recall
        memory_keywords = ['recall', 'remember', 'similar', 'previous', 'past', 'experience']
        evidence_keywords = ['evidence', 'study', 'research', 'data', 'shows']
        
        advice_text = ""
        for field in ['medical_advice', 'legal_advice', 'technical_advice', 'response']:
            if field in output:
                advice_text = str(output[field]).lower()
                break
        
        memory_recall = 0.3  # Base level
        retention_strength = 0.5
        
        if advice_text:
            # Count memory-related terms
            memory_terms = sum(1 for keyword in memory_keywords if keyword in advice_text)
            evidence_terms = sum(1 for keyword in evidence_keywords if keyword in advice_text)
            
            memory_recall = min(0.9, 0.3 + (memory_terms * 0.2) + (evidence_terms * 0.1))
            
            # Retention strength based on detail and specificity
            word_count = len(advice_text.split())
            if word_count > 30:
                retention_strength = 0.7
            elif word_count > 15:
                retention_strength = 0.6
        
        return {
            'similar_cases_recalled': max(1, int(memory_recall * 3)),
            'retention_strength': retention_strength,
            'context_match': 0.6,  # Assuming moderate context match
            'memory_recall_level': memory_recall
        }

    def _infer_neurochemical_state(self, confidence: float, attention: Dict, activation: Dict) -> Dict[str, float]:
        """Infer neurochemical state based on other biological signals"""
        # Dopamine - reward signal (correlates with confidence)
        dopamine = min(0.9, confidence * 0.8)
        
        # Acetylcholine - focus signal (correlates with attention focus)
        acetylcholine = attention.get('domain_focus', 0.5) * 0.7
        
        # Serotonin - certainty/stability signal
        serotonin = min(0.8, confidence * 0.6 + activation.get('current', 0.5) * 0.2)
        
        # Norepinephrine - arousal/alertness
        norepinephrine = activation.get('response_energy', 0.5) * 0.8
        
        return {
            'dopamine': dopamine,
            'acetylcholine': acetylcholine,
            'serotonin': serotonin,
            'norepinephrine': norepinephrine,
            'overall_neurochemical_balance': (dopamine + acetylcholine + serotonin + norepinephrine) / 4
        }
    async def handle_remote_query(self, query_msg: Dict[str, Any]) -> Dict[str, Any]:
        """Process a query received from another VBC."""
        query = query_msg.get('query_text', '')
        context = {'remote': True, 'session_id': query_msg.get('session_id', 'remote')}
        
        # Use existing method to get local opinions (you may need to adapt)
        # We assume you have a method like _get_local_opinions – if not, you can call process_query_advanced.
        # For simplicity, we'll use a placeholder.
        # In a real implementation, you would invoke the local VNIs.
        local_opinions = await self._get_local_opinions(query, context)  # You may need to create this helper
        
        # Synthesize a simple response (you can reuse the synthesizer)
        from neuron.shared.reasoning_output import ReasoningOutput  # adjust import as needed
        consensus = {'consensus_score': 0.8}  # placeholder
        conflicts = []
        final_response = self.response_synthesizer.synthesize_response(
            local_opinions, consensus, conflicts
        )
        
        return {
            'query_id': query_msg['query_id'],
            'response': final_response,
            'confidence': 0.8,
            'vni_domains': [self._get_domain(vni_id) for vni_id in local_opinions.keys()]
        }
    
    async def _query_remote_vnis(self, query: str, context: Dict, candidates: List[Dict]) -> List[Dict]:
        """Send queries to remote VBCs and collect responses."""
        if not self.p2p_node:
            return []
        tasks = []
        for cand in candidates:
            msg = {
                'query_id': str(uuid.uuid4()),
                'query_text': query,
                'session_id': context.get('session_id', ''),
                'target_domain': cand.get('domain')
            }
            tasks.append(self.p2p_node.send_message(cand['peer_id'], '/babybionn/query/1.0.0', msg))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, dict) and 'error' not in r]
    
    async def _get_local_opinions(self, query: str, context: Dict) -> Dict[str, Dict[str, Any]]:
        """Query all local VNIs and return their raw outputs.
        This is used when handling remote queries or for local opinion gathering."""
        if not self.vni_manager:
            logger.warning("No VNI manager available for local opinions")
            return {}
    
        opinions = {}
        tasks = []
    
        # Iterate over all VNI instances (or select based on context)
        for vni_id, vni in self.vni_manager.vni_instances.items():
            # Optionally, you could filter by domain if context provides one
            # For now, query all
            if hasattr(vni, 'process_query'):
                if inspect.iscoroutinefunction(vni.process_query):
                    tasks.append(vni.process_query(query=query, context=context))
                else:
                    # If it's synchronous, run in thread to avoid blocking
                    tasks.append(asyncio.to_thread(vni.process_query, query, context))
            elif hasattr(vni, 'process_async'):
                tasks.append(vni.process_async(query, context))
            elif hasattr(vni, 'process'):
                if inspect.iscoroutinefunction(vni.process):
                    tasks.append(vni.process(query, context))
                else:
                    tasks.append(asyncio.to_thread(vni.process, query, context))
            else:
                logger.debug(f"VNI {vni_id} has no suitable process method, skipping")
    
        # Run all VNI calls concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
        # Collect successful results
        for vni_id, result in zip(self.vni_manager.vni_instances.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"VNI {vni_id} failed: {result}")
                continue
            # Convert the result to the format expected by the aggregator
            # This may need adjustment based on what your VNIs return
            opinions[vni_id] = self._format_vni_output(vni_id, result)
    
        logger.info(f"Collected {len(opinions)} local opinions for remote query")
        return opinions
    
    def _format_vni_output(self, vni_id: str, raw_output: Any) -> Dict[str, Any]:
        """Convert a raw VNI output to the aggregator's expected format."""
        if isinstance(raw_output, dict):
            # Already a dict – ensure it has required fields
            output = {
                'response': raw_output.get('response', str(raw_output)),
                'confidence_score': raw_output.get('confidence', 0.5),
                'vni_metadata': raw_output.get('vni_metadata', {}),
            }
            # Ensure vni_metadata has success flag
            if 'vni_metadata' not in output:
                output['vni_metadata'] = {}
            if 'success' not in output['vni_metadata']:
                output['vni_metadata']['success'] = True  # assume success if we got a dict
            return output
        else:
            # String or other type – wrap it
            return {
                'response': str(raw_output),
                'confidence_score': 0.5,
                'vni_metadata': {'vni_id': vni_id, 'success': True}
            }
            
    def _make_json_serializable(self, obj):
        """Recursively convert numpy types to native Python types."""
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(i) for i in obj]
        elif hasattr(obj, '__dict__'):  # for dataclasses and custom objects
            return self._make_json_serializable(vars(obj))
        else:
            return obj

    def forward(self, router_results: Dict[str, Any]) -> Dict[str, Any]:
        """Main aggregation pipeline with Hebbian learning. Compatible with original aggregator interface"""
        try:
            # === EXTRACT QUERY FROM ROUTER RESULTS ===
            query = ""
            if 'query' in router_results:
                query = router_results['query']
            elif 'original_query' in router_results:
                query = router_results['original_query']
            elif 'query_context' in router_results:
                query_context = router_results.get('query_context', {})
                query = query_context.get('query', '')
            
            # === CHECK IF THIS IS A GREETING ===
            if query and self.greeting_preprocessor.is_greeting(query):
                logger.info(f"🎯 Forward method detected greeting: '{query}'")
                response = self.greeting_preprocessor.get_response(query)
                return {
                    'final_response': response["response"],
                    'aggregation_analysis': {
                        'greeting_detected': True,
                        'consensus_analysis': {'consensus_level': 'strong', 'consensus_score': 0.95},
                        'conflict_analysis': [],
                        'vni_contributions': [],
                        'domain_coverage': {},
                        'learning_applied': False
                    },
                    'confidence_metrics': {
                        'overall_confidence': 0.95,
                        'consensus_confidence': 0.95,
                        'vni_confidence_distribution': {'mean': 0.95, 'std': 0.0},
                        'response_quality_trend': {'trend': 'stable'}
                    },
                    'synaptic_learning': {'available': False, 'greeting_handled': True},
                    'processing_metadata': {
                        'total_vnis_processed': 0,
                        'successful_vnis': 0,
                        'conflicts_detected': 0,
                        'cross_domain_synthesis': False,
                        'hebbian_learning_active': False,
                        'learning_cycles': self.learning_metrics.total_learning_cycles
                    },
                    'aggregator_metadata': {
                        'aggregator_id': self.config.aggregator_id,
                        'processing_stages': ['greeting_preprocessor'],
                        'success': True,
                        'timestamp': response["timestamp"]
                    }
                }        
            execution_results = router_results.get('execution_results', {})

            # === ADD DEBUG HERE - BEFORE FILTERING ===
            logger.info("🔍 RAW VNI OUTPUTS BEFORE FILTERING:")
            
            for vni_id, result in execution_results.items():
                logger.info(f"   VNI: {vni_id}")
                logger.info(f"      All keys: {list(result.keys())}")
                
                # Check for vni_metadata specifically
                if 'vni_metadata' in result:
                    logger.info(f"      vni_metadata keys: {list(result['vni_metadata'].keys())}")
                    logger.info(f"      success value: {result['vni_metadata'].get('success', 'NOT FOUND')}")
                else:
                    logger.info(f"      ❌ NO vni_metadata found!")
                
                # Also check if there's any success flag elsewhere
                if 'success' in result:
                    logger.info(f"      success at top level: {result['success']}")
            
            # === END DEBUG ===
                        
            # Filter successful outputs
            successful_outputs = {
                vni_id: result for vni_id, result in execution_results.items()
                if result.get('vni_metadata', {}).get('success', False)
            }
            
            if not successful_outputs:
                return self._generate_no_output_response(router_results)
            
            # Get activated VNI list
            activated_vnis = list(successful_outputs.keys())
            
            # Conflict detection
            conflicts = self.conflict_detector.detect_conflicts(successful_outputs)
            
            # Extract biological states
            biological_states = self._extract_biological_states(successful_outputs)

            # Enhanced consensus calculation with biology
            consensus = self.calculate_consensus_with_biology(successful_outputs, biological_states)
            
            # Response synthesis
            final_response = self.response_synthesizer.synthesize_response(
                successful_outputs, consensus, conflicts
            )
            
            # Calculate overall quality for learning
            overall_confidence = self._calculate_overall_confidence(
                successful_outputs, consensus, conflicts
            )
            
            # **ENHANCED: Hebbian learning with biological awareness**
            if self.config.enable_hebbian_learning and len(activated_vnis) >= 2:
                query_context = router_results.get('query_context', {})
                
                # Enhanced learning call
                self.learn_from_biological_interaction(
                    activated_vnis=activated_vnis,
                    vni_outputs=successful_outputs,
                    biological_states=biological_states,
                    overall_quality=overall_confidence,
                    consensus_analysis=consensus,    
                    query_context=query_context
                )
                
                # Update metrics
                self.learning_metrics.total_learning_cycles += 1
                self.learning_metrics.last_learning = datetime.now()
                self.learning_metrics.total_queries_processed += 1
                
                if overall_confidence > self.config.strengthening_threshold:
                    self.learning_metrics.connections_strengthened += len(activated_vnis) - 1
                elif overall_confidence < self.config.weakening_threshold:
                    self.learning_metrics.connections_weakened += len(activated_vnis) - 1
                
                # Update response quality metrics
                self.learning_metrics.avg_response_quality = (
                    (self.learning_metrics.avg_response_quality * 
                     (self.learning_metrics.total_queries_processed - 1) + overall_confidence) /
                    self.learning_metrics.total_queries_processed
                )
                self.learning_metrics.response_quality_history.append(overall_confidence)
            
            # Update response quality history
            self.response_quality_history.append(overall_confidence)
            
            # Get synaptic insights
            synaptic_insights = self._get_synaptic_insights(activated_vnis)
            
            # Update cluster performance
            self._update_cluster_performance_from_outputs(successful_outputs)
            
            result = {
                'final_response': final_response,
                'aggregation_analysis': {
                    'consensus_analysis': consensus,
                    'conflict_analysis': conflicts,
                    'vni_contributions': self._analyze_vni_contributions(successful_outputs),
                    'domain_coverage': self._analyze_domain_coverage(successful_outputs),
                    'learning_applied': self.config.enable_hebbian_learning
                },
                'confidence_metrics': {
                    'overall_confidence': overall_confidence,
                    'consensus_confidence': consensus.get('consensus_score', 0.0),
                    'vni_confidence_distribution': self._get_confidence_distribution(successful_outputs),
                    'response_quality_trend': self._get_response_quality_trend()
                },
                'synaptic_learning': synaptic_insights,
                'processing_metadata': {
                    'total_vnis_processed': len(execution_results),
                    'successful_vnis': len(successful_outputs),
                    'conflicts_detected': len(conflicts),
                    'cross_domain_synthesis': self._is_cross_domain(successful_outputs),
                    'hebbian_learning_active': self.config.enable_hebbian_learning,
                    'learning_cycles': self.learning_metrics.total_learning_cycles
                },
                'aggregator_metadata': {
                    'aggregator_id': self.config.aggregator_id,
                    'processing_stages': ['conflict_detection', 'consensus_calculation', 
                                        'response_synthesis', 'hebbian_learning'],
                    'success': True,
                    'response_length': len(final_response),
                    'timestamp': datetime.now().isoformat()
                }
            }
            # Convert any numpy types to Python types
            result = self._make_json_serializable(result)
            return result
            
        except Exception as e:
            logger.error(f"❌ Aggregation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._generate_error_output(str(e))
    
    async def process_query_advanced(
        self, 
        query: str, 
        session_id: str = "default",
        context: Optional[Dict] = None,
        use_learning: bool = True
    ) -> Dict[str, Any]:
        """Advanced query processing with Hebbian learning and optimization"""
        logger.info(f"🔨 Processing query: '{query[:50]}...'")

        # === ADD GREETING CHECK HERE ===
        if self.greeting_preprocessor.is_greeting(query):
            logger.info(f"🎯 Detected greeting, using greeting handler")
            response = self.greeting_preprocessor.get_response(query)
            # Add session info to response
            response["session_id"] = session_id
            return response
          
        # Get/create session
        session = self._get_or_create_session(session_id)
        
        # Build context
        full_context = {
            'session_id': session_id,
            'session_data': session,
            'timestamp': datetime.now().isoformat(),
            'query_complexity': self._assess_query_complexity(query),
            'detected_domains': self._detect_query_domains(query),
            **(context or {})
        }
        
        # **HEBBIAN ROUTING: Use learned connections to suggest optimal VNIs**
        available_vnis = list(self.vni_manager.vni_instances.keys()) if self.vni_manager else []
        
        if use_learning and self.config.enable_hebbian_learning and available_vnis:
            # Use Hebbian routing (biological routing is disabled)
            suggested_routing = self.hebbian_engine.suggest_vni_routing(full_context, available_vnis)
            routing_type = "hebbian"
            
            if suggested_routing:
                logger.info(f"🧠 Using {routing_type} routing: {[vni for vni, _ in suggested_routing[:3]]}")
                full_context['suggested_vnis'] = [vni for vni, score in suggested_routing]
                full_context['routing_confidence'] = suggested_routing[0][1] if suggested_routing else 0.5
                full_context['routing_type'] = routing_type     
                
                # Select clusters based on learned connections
                target_clusters = self._select_clusters_with_learning(query, full_context)
        
        # Auto-spawn if needed
        if self.config.enable_auto_spawning:
            spawned = await self._auto_spawn_vnis(query, full_context)
            if spawned:
                logger.info(f"🎯 Auto-spawned {len(spawned)} VNIs")
        
        # Route through VNI manager (with synaptic suggestions)
        if self.vni_manager:
            response = self.vni_manager.route_query(
                query=query,
                context=full_context,
                session_id=session_id
            )
            
            # Enhance with learning metrics
            response['learning_metrics'] = {
                'learning_cycles': self.learning_metrics.total_learning_cycles,
                'connections_strengthened': self.learning_metrics.connections_strengthened,
                'connections_weakened': self.learning_metrics.connections_weakened,
                'avg_response_quality': self.learning_metrics.avg_response_quality
            }
        else:
            response = {
                'response': 'VNI manager not available',
                'confidence': 0.0,
                'sources': []
            }
        
        # Update cluster performance
        self._update_cluster_performance(target_clusters, response)
        
        # Enhance response with synaptic insights
        enhanced = self._enhance_response_with_synaptic_insights(response, full_context)
        
        # Update session
        self._update_session_history(session_id, query, enhanced)
        
        # Cleanup
        self._cleanup_expired_sessions()
        
        self.learning_metrics.total_queries_processed += 1
        
        logger.info(f"✅ Query processed. Confidence: {enhanced.get('confidence', 0):.2f}")
        return enhanced
    
    def calculate_consensus_with_biology(self, vni_outputs, biological_states):
        """Enhanced consensus calculation using biological states"""
        consensus = self.consensus_calculator.calculate_consensus(vni_outputs)
        
        # Add biological consensus metrics
        if biological_states:
            attention_scores = []
            activation_scores = []
            
            for vni_id, bio_state in biological_states.items():
                attention = bio_state.get('attention', {})
                activation = bio_state.get('activation', {})
                
                if 'domain_focus' in attention:
                    attention_scores.append(attention['domain_focus'])
                
                if 'current' in activation:
                    activation_scores.append(activation['current'])
            
            if attention_scores:
                consensus['biological_attention_alignment'] = np.std(attention_scores) < 0.2
                consensus['avg_attention_focus'] = float(np.mean(attention_scores))
            
            if activation_scores:
                consensus['activation_synchronization'] = np.std(activation_scores) < 0.15
                consensus['avg_activation_level'] = float(np.mean(activation_scores))
        
        return consensus
    
    def _select_clusters_with_learning(self, query: str, context: Dict) -> List[str]:
        """Select clusters using learned connection patterns"""
        query_lower = query.lower()
        
        # First, try to use learned patterns
        if self.config.enable_hebbian_learning:
            available_vnis = list(self.vni_manager.vni_instances.keys()) if self.vni_manager else []
            if available_vnis:
                suggestions = self.hebbian_engine.suggest_vni_routing(context, available_vnis)
                if suggestions:
                    # Extract domains from suggested VNIs
                    suggested_domains = {self._get_domain(vni) for vni, _ in suggestions[:3]}
                    selected_clusters = []
                    for domain in suggested_domains:
                        if domain in self.routing_table:
                            selected_clusters.extend(self.routing_table[domain])
                    if selected_clusters:
                        return list(set(selected_clusters))[:3]
        
        # Fall back to keyword matching if no learned patterns
        return self._select_clusters_for_query(query, context)
    
    def _select_clusters_for_query(self, query: str, context: Dict) -> List[str]:
        """Select optimal clusters for query using keyword matching"""
        query_lower = query.lower()
        domain_keywords = {
            'medical': ['medical', 'health', 'doctor', 'hospital', 'pain', 'fever', 'treatment', 'symptom', 'diagnosis'],
            'legal': ['legal', 'law', 'contract', 'rights', 'agreement', 'court', 'attorney', 'liability'],
            'technical': ['code', 'programming', 'software', 'debug', 'error', 'bug', 'algorithm', 'technical'],
            'analytical': ['analyze', 'compare', 'evaluate', 'assess', 'statistics', 'data', 'analysis', 'trend'],
            'financial': ['financial', 'finance', 'investment', 'stock', 'market', 'portfolio', 'risk'],
            'creative': ['creative', 'story', 'narrative', 'design', 'art', 'write', 'compose'],
            'general': []
        }
        
        selected = []
        for domain, keywords in domain_keywords.items():
            if not keywords or any(kw in query_lower for kw in keywords):
                if domain in self.routing_table:
                    selected.extend(self.routing_table[domain])
        
        selected = list(set(selected))
        
        if not selected or (len(selected) == 1 and 'general' in selected[0]):
            general = [cid for cid, c in self.clusters.items() if c['type'] == 'general']
            general.sort(key=lambda cid: self.cluster_performance[cid].avg_confidence, reverse=True)
            selected = general[:2] if general else ['general_cluster']
        
        return selected
    
    def _get_synaptic_insights(self, activated_vnis: List[str]) -> Dict[str, Any]:
        """Get insights from synaptic connections"""
        if not self.config.enable_hebbian_learning or len(activated_vnis) < 2:
            return {'available': False}
        
        # Get connection strengths for activated VNIs
        connection_strengths = []
        for i, vni1 in enumerate(activated_vnis):
            for vni2 in activated_vnis[i+1:]:
                key = tuple(sorted([vni1, vni2]))
                if key in self.hebbian_engine.connections:
                    conn = self.hebbian_engine.connections[key]
                    connection_strengths.append({
                        'vni_pair': f"{vni1} <-> {vni2}",
                        'strength': conn.strength,
                        'success_rate': conn.success_rate(),
                        'type': conn.connection_type.value,
                        'activation_count': conn.activation_count,
                        'context_successful': len(conn.successful_contexts),
                        'context_failed': len(conn.failed_contexts)
                    })
        
        learning_stats = self.hebbian_engine.get_learning_statistics()
        
        # Calculate connection strength trend
        avg_strength = np.mean([c['strength'] for c in connection_strengths]) if connection_strengths else 0.0
        self.connection_strength_history.append(avg_strength)
        
        return {
            'available': True,
            'active_connections': connection_strengths,
            'network_statistics': learning_stats,
            'strongest_connections': self.hebbian_engine.get_strongest_connections(5),
            'avg_connection_strength': avg_strength,
            'strength_trend': self._get_connection_strength_trend()
        }

    def learn_from_biological_interaction(self, activated_vnis, vni_outputs, biological_states, overall_quality, consensus_analysis, query_context):
        """Enhanced learning with all biological systems – no response generation, only learning."""
        if consensus_analysis is None:
            # Calculate a simple consensus analysis from available data
            consensus_analysis = {
                'consensus_level': 'moderate' if overall_quality > 0.6 else 'weak',
                'consensus_score': overall_quality,
                'domain_distribution': {},
                'domain_consensus': {},
                'agreeing_vnis': activated_vnis if overall_quality > 0.5 else []
            }
    
        # Extract biological patterns
        attention_patterns = {}
        activation_patterns = {}
        memory_patterns = {}
        
        for vni_id in activated_vnis:
            if vni_id in biological_states:
                bio = biological_states[vni_id]
                attention_patterns[vni_id] = bio.get('attention', {})
                activation_patterns[vni_id] = bio.get('activation', {})
                memory_patterns[vni_id] = bio.get('memory', {})
        
        # Apply standard Hebbian learning
        self.hebbian_engine.learn_from_interaction(
            activated_vnis=activated_vnis,
            vni_outputs=vni_outputs,
            overall_quality=overall_quality,
            query_context=query_context
        )
        
        # Update all biological systems with learned patterns
        # 1. Update Hybrid Attention
        self.hybrid_attention.learn_from_interaction(
            vni_sequence=activated_vnis,
            attention_patterns=attention_patterns,
            outcome_quality=overall_quality,
            query_context=query_context
        )
        
        # 2. Update Attention Engine
        self.attention_engine.update_attention_patterns(
            vni_ids=activated_vnis,
            patterns=attention_patterns,
            success=overall_quality > 0.7
        )
        
        # 3. Update Activation Router
        self.activation_router.update_activation_patterns(
            vni_ids=activated_vnis,
            activation_levels=activation_patterns,
            outcome=overall_quality
        )
        
        # 4. Update VNI Memory
        self.vni_memory.record_interaction_pattern(
            vni_ids=activated_vnis,
            outputs=vni_outputs,
            biological_states=biological_states,
            quality=overall_quality
        )
        
        logger.debug(f"🧬 Updated all biological systems with {len(activated_vnis)} VNIs, quality: {overall_quality:.3f}")
        # No return – the method is called for side effects only
        
    def _get_response_quality_trend(self) -> Dict[str, Any]:
        """Get response quality trend over time"""
        if len(self.response_quality_history) < 2:
            return {'trend': 'insufficient_data', 'slope': 0.0}
        
        quality_values = list(self.response_quality_history)
        x = np.arange(len(quality_values))
        slope, _ = np.polyfit(x, quality_values, 1)
        
        if slope > 0.01:
            trend = 'improving'
        elif slope < -0.01:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'slope': float(slope),
            'current_avg': float(np.mean(quality_values[-10:])) if len(quality_values) >= 10 else float(np.mean(quality_values)),
            'window_size': len(quality_values)
        }
    
    def _get_connection_strength_trend(self) -> Dict[str, Any]:
        """Get connection strength trend over time"""
        if len(self.connection_strength_history) < 2:
            return {'trend': 'insufficient_data', 'slope': 0.0}
        
        strength_values = list(self.connection_strength_history)
        x = np.arange(len(strength_values))
        slope, _ = np.polyfit(x, strength_values, 1)
        
        if slope > 0.005:
            trend = 'strengthening'
        elif slope < -0.005:
            trend = 'weakening'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'slope': float(slope),
            'current_avg': float(np.mean(strength_values[-10:])) if len(strength_values) >= 10 else float(np.mean(strength_values)),
            'window_size': len(strength_values)
        }
    
    def _enhance_response_with_synaptic_insights(self, response: Dict, context: Dict) -> Dict:
        """Enhance response with synaptic learning insights"""
        enhanced = response.copy()
        
        if self.config.enable_hebbian_learning:
            enhanced['synaptic_learning'] = {
                'total_connections': len(self.hebbian_engine.connections),
                'learning_cycles': self.learning_metrics.total_learning_cycles,
                'connections_strengthened': self.learning_metrics.connections_strengthened,
                'connections_weakened': self.learning_metrics.connections_weakened,
                'connections_pruned': self.learning_metrics.connections_pruned,
                'patterns_detected': self.learning_metrics.patterns_detected,
                'avg_response_quality': self.learning_metrics.avg_response_quality,
                'connection_strength_trend': self._get_connection_strength_trend()
            }
        
        enhanced['system_metrics'] = {
            'total_clusters': len(self.clusters),
            'active_sessions': len(self.sessions),
            'queries_processed': self.learning_metrics.total_queries_processed,
            'learning_cycles': self.learning_metrics.total_learning_cycles,
            'response_quality_trend': self._get_response_quality_trend()
        }
        
        # Add cluster performance info
        enhanced['cluster_performance'] = {
            cluster_id: {
                'type': cluster['type'],
                'total_queries': self.cluster_performance[cluster_id].total_queries,
                'avg_confidence': self.cluster_performance[cluster_id].avg_confidence,
                'success_rate': self.cluster_performance[cluster_id].success_rate
            }
            for cluster_id, cluster in self.clusters.items()
        }
        
        return enhanced
    
    def _calculate_overall_confidence(self, successful_outputs, consensus, conflicts) -> float:
        """Calculate overall confidence score with enhanced metrics"""
        vni_confidences = [o.get('confidence_score', 0.5) for o in successful_outputs.values()]
        base_confidence = sum(vni_confidences) / len(vni_confidences) if vni_confidences else 0.5
        
        consensus_score = consensus.get('consensus_score', 0.5)
        
        # Enhanced conflict penalty calculation
        conflict_penalty = 0.0
        for conflict in conflicts:
            if conflict['conflict_level'] == 'major_conflict':
                penalty = 0.3
            elif conflict['conflict_level'] == 'minor_conflict':
                penalty = 0.1
            else:
                penalty = 0.0
            
            # Adjust penalty based on confidence difference
            confidence_diff = abs(conflict.get('confidence_diff', 0))
            penalty *= (1 + confidence_diff)
            conflict_penalty += penalty
        
        conflict_penalty = min(conflict_penalty, 0.6)  # Cap penalty
        
        # Calculate domain diversity bonus
        domains = {self._get_domain(vni_id) for vni_id in successful_outputs.keys()}
        domain_diversity = len(domains) / max(len(successful_outputs), 1)
        diversity_bonus = 0.1 * domain_diversity
        
        overall = (base_confidence + consensus_score) / 2
        overall = max(0.0, overall - conflict_penalty + diversity_bonus)
        
        return min(overall, 1.0)
    
    def _analyze_vni_contributions(self, successful_outputs) -> List[Dict[str, Any]]:
        """Analyze contributions from each VNI with enhanced metrics"""
        contributions = []
        
        for vni_id, output in successful_outputs.items():
            confidence = output.get('confidence_score', 0.5)
            domain = self._get_domain(vni_id)
            
            # Calculate contribution score
            contribution_score = confidence
            
            # Adjust based on domain uniqueness in this response
            same_domain_count = sum(1 for vni in successful_outputs.keys() 
                                  if self._get_domain(vni) == domain)
            if same_domain_count == 1:
                contribution_score *= 1.2  # Bonus for unique domain
            
            contributions.append({
                'vni_id': vni_id,
                'domain': domain,
                'confidence': confidence,
                'contribution_score': contribution_score,
                'contribution_level': 'primary' if confidence > 0.7 else 'secondary'
            })
        
        return sorted(contributions, key=lambda x: x['contribution_score'], reverse=True)
    
    def _analyze_domain_coverage(self, successful_outputs) -> Dict[str, Any]:
        """Analyze domain coverage with enhanced metrics"""
        domain_counts = defaultdict(int)
        domain_confidences = defaultdict(list)
        
        for vni_id in successful_outputs.keys():
            domain = self._get_domain(vni_id)
            domain_counts[domain] += 1
            domain_confidences[domain].append(successful_outputs[vni_id].get('confidence_score', 0.5))
        
        coverage_analysis = {}
        for domain in domain_counts:
            coverage_analysis[domain] = {
                'count': domain_counts[domain],
                'avg_confidence': np.mean(domain_confidences[domain]) if domain_confidences[domain] else 0.0,
                'coverage_ratio': domain_counts[domain] / len(successful_outputs) if successful_outputs else 0.0
            }
        
        return coverage_analysis
    
    def _get_confidence_distribution(self, successful_outputs) -> Dict[str, Any]:
        """Get confidence score distribution with enhanced statistics"""
        confidences = [o.get('confidence_score', 0.5) for o in successful_outputs.values()]
        
        if not confidences:
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0, 'median': 0.0, 'iqr': 0.0}
        
        confidences_array = np.array(confidences)
        
        return {
            'min': float(np.min(confidences_array)),
            'max': float(np.max(confidences_array)),
            'mean': float(np.mean(confidences_array)),
            'std': float(np.std(confidences_array)),
            'median': float(np.median(confidences_array)),
            'iqr': float(np.percentile(confidences_array, 75) - np.percentile(confidences_array, 25)),
            'high_confidence_ratio': float(np.sum(confidences_array > 0.7) / len(confidences_array)),
            'low_confidence_ratio': float(np.sum(confidences_array < 0.4) / len(confidences_array))
        }
    
    def _is_cross_domain(self, successful_outputs) -> bool:
        """Check if this is cross-domain synthesis"""
        domains = {self._get_domain(vni_id) for vni_id in successful_outputs.keys()}
        return len(domains) >= 2
    
    def _get_domain(self, vni_id: str) -> str:
        """Extract domain from VNI ID"""
        vni_id_lower = vni_id.lower()
        if 'medical' in vni_id_lower or 'med' in vni_id_lower:
            return 'medical'
        elif 'legal' in vni_id_lower or 'law' in vni_id_lower:
            return 'legal'
        elif 'technical' in vni_id_lower or 'tech' in vni_id_lower:
            return 'technical'
        elif 'analyt' in vni_id_lower:
            return 'analytical'
        elif 'creative' in vni_id_lower:
            return 'creative'
        elif 'financial' in vni_id_lower or 'finance' in vni_id_lower:
            return 'financial'
        elif 'research' in vni_id_lower:
            return 'research'
        return 'general'
    
    async def _auto_spawn_vnis(self, query: str, context: Dict) -> List[str]:
        """Auto-spawn VNIs based on query needs with learning considerations"""
        if not self.config.enable_auto_spawning:
            return []
        
        detected_domains = self._detect_query_domains(query)
        existing_domains = {c['type'] for c in self.clusters.values()}
        missing_domains = [d for d in detected_domains if d not in existing_domains]
        
        # Also consider domains that have poor performance
        poor_performance_domains = []
        for domain in existing_domains:
            domain_clusters = [cid for cid, c in self.clusters.items() if c['type'] == domain]
            if domain_clusters:
                avg_success = np.mean([self.cluster_performance[cid].success_rate 
                                     for cid in domain_clusters])
                if avg_success < 0.4:
                    poor_performance_domains.append(domain)
        
        domains_to_spawn = list(set(missing_domains + poor_performance_domains[:2]))
        
        spawned = []
        for domain in domains_to_spawn:
            if len(self.clusters) >= self.config.max_clusters:
                logger.warning(f"⚠️ Max clusters reached ({self.config.max_clusters})")
                break
            
            try:
                timestamp = datetime.now().strftime('%H%M%S')
                cluster_id = f"{domain}_auto_{timestamp}"
                instance_id = f"{domain}_auto_001"
                
                if self.vni_manager:
                    self.vni_manager.create_vni(domain, instance_id)
                
                self.clusters[cluster_id] = {
                    'type': domain,
                    'instance_ids': [instance_id],
                    'created': datetime.now(),
                    'specializations': [domain],
                    'auto_spawned': True,
                    'total_responses': 0,
                    'successful_responses': 0
                }
                
                self.routing_table[domain].append(cluster_id)
                self.cluster_performance[cluster_id] = ClusterPerformance(
                    cluster_id=cluster_id,
                    specialization=domain
                )
                
                spawned.append(cluster_id)
                logger.info(f"🔄 Auto-spawned cluster: {cluster_id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to auto-spawn {domain}: {e}")
        
        return spawned
    
    def _detect_query_domains(self, query: str) -> List[str]:
        """Detect domains present in query with enhanced detection"""
        query_lower = query.lower()
        domains = []
        
        domain_keywords = {
            'medical': ['medical', 'health', 'doctor', 'patient', 'symptom', 'diagnosis', 'treatment', 'hospital'],
            'legal': ['legal', 'law', 'contract', 'rights', 'agreement', 'court', 'liability', 'compliance'],
            'technical': ['code', 'programming', 'software', 'debug', 'error', 'bug', 'algorithm', 'technical', 'api'],
            'analytical': ['analyze', 'compare', 'evaluate', 'assess', 'statistics', 'data', 'analysis', 'trend', 'pattern'],
            'financial': ['financial', 'finance', 'investment', 'stock', 'market', 'portfolio', 'risk', 'return'],
            'creative': ['creative', 'story', 'narrative', 'design', 'art', 'write', 'compose', 'imagine']
        }
        
        # Score domains based on keyword matches
        domain_scores = defaultdict(int)
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    domain_scores[domain] += 1
        
        # Add domains with score > 0
        for domain, score in domain_scores.items():
            if score > 0:
                domains.append(domain)
        
        # Also consider domain combinations
        if 'medical' in domains and 'legal' in domains:
            domains.append('medico_legal')
        
        return domains if domains else ['general']
    
    def _assess_query_complexity(self, query: str) -> float:
        """Assess query complexity (0.0 to 1.0) with enhanced metrics"""
        words = len(query.split())
        sentences = query.count('.') + query.count('?') + query.count('!')
        
        # Count domain-specific terms
        domain_terms = sum(1 for word in query.lower().split() 
                          if any(term in word for term in 
                                ['medical', 'legal', 'technical', 'financial', 'analy']))
        
        # Count question words
        question_words = sum(1 for word in query.lower().split() 
                            if word in ['what', 'why', 'how', 'when', 'where', 'which'])
        
        # Enhanced complexity calculation
        complexity = min(1.0, 
                        (words / 150) + 
                        (sentences / 15) + 
                        (domain_terms / 10) + 
                        (question_words / 5))
        
        return complexity
    
    def _update_cluster_performance(self, clusters: List[str], response: Dict):
        """Update cluster performance metrics"""
        confidence = response.get('confidence', 0.5)
        
        for cluster_id in clusters:
            if cluster_id in self.cluster_performance:
                perf = self.cluster_performance[cluster_id]
                perf.total_queries += 1
                perf.avg_confidence = (
                    (perf.avg_confidence * (perf.total_queries - 1) + confidence) 
                    / perf.total_queries
                )
                perf.last_active = datetime.now()
                
                if confidence > 0.6:
                    perf.success_rate = (
                        (perf.success_rate * (perf.total_queries - 1) + 1) 
                        / perf.total_queries
                    )
                
                # Update performance history
                perf.performance_history.append(confidence)
                if len(perf.performance_history) > 50:
                    perf.performance_history = perf.performance_history[-50:]
                
                # Update cluster statistics
                if cluster_id in self.clusters:
                    self.clusters[cluster_id]['total_responses'] += 1
                    if confidence > 0.6:
                        self.clusters[cluster_id]['successful_responses'] += 1
    
    def _update_cluster_performance_from_outputs(self, successful_outputs: Dict[str, Dict[str, Any]]):
        """Update cluster performance from VNI outputs"""
        for vni_id, output in successful_outputs.items():
            confidence = output.get('confidence_score', 0.5)
            domain = self._get_domain(vni_id)
            
            # Find cluster for this VNI
            for cluster_id, cluster in self.clusters.items():
                if cluster['type'] == domain and vni_id in cluster['instance_ids']:
                    if cluster_id in self.cluster_performance:
                        perf = self.cluster_performance[cluster_id]
                        perf.total_queries += 1
                        perf.avg_confidence = (
                            (perf.avg_confidence * (perf.total_queries - 1) + confidence) 
                            / perf.total_queries
                        )
                        break
    
    def _get_or_create_session(self, session_id: str) -> Dict:
        """Get or create session with enhanced tracking"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'created': datetime.now(),
                'last_activity': datetime.now(),
                'interaction_count': 0,
                'history': deque(maxlen=self.config.session_history_size),
                'preferences': {},
                'query_complexities': [],
                'response_qualities': [],
                'domains_used': defaultdict(int)
            }
        else:
            self.sessions[session_id]['last_activity'] = datetime.now()
            self.sessions[session_id]['interaction_count'] += 1
        
        return self.sessions[session_id]
    
    def _update_session_history(self, session_id: str, query: str, response: Dict):
        """Update session history with enhanced metrics"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Extract domains from response sources
            domains = set()
            if 'sources' in response:
                for source in response['sources']:
                    domains.add(self._get_domain(source))
            
            for domain in domains:
                session['domains_used'][domain] += 1
            
            # Update complexity and quality
            complexity = self._assess_query_complexity(query)
            session['query_complexities'].append(complexity)
            if len(session['query_complexities']) > 50:
                session['query_complexities'] = session['query_complexities'][-50:]
            
            confidence = response.get('confidence', 0.5)
            session['response_qualities'].append(confidence)
            if len(session['response_qualities']) > 50:
                session['response_qualities'] = session['response_qualities'][-50:]
            
            session['history'].append({
                'query': query,
                'response': response.get('response', '')[:200],
                'confidence': confidence,
                'sources': response.get('sources', []),
                'domains': list(domains),
                'timestamp': datetime.now().isoformat()
            })
    
    def _cleanup_expired_sessions(self):
        """Cleanup expired sessions"""
        now = datetime.now()
        expired = [
            sid for sid, sess in self.sessions.items()
            if (now - sess['last_activity']) > self.session_timeout
        ]
        
        for sid in expired:
            # Save session learning before deletion
            self._save_session_learnings(sid)
            del self.sessions[sid]
            logger.debug(f"🗑️  Cleaned up expired session: {sid}")
    
    def _save_session_learnings(self, session_id: str):
        """Save learnings from session before cleanup"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if session['interaction_count'] > 5:
                # Extract patterns from session
                domains_used = dict(session['domains_used'])
                avg_complexity = np.mean(session['query_complexities']) if session['query_complexities'] else 0.0
                avg_quality = np.mean(session['response_qualities']) if session['response_qualities'] else 0.0
                
                logger.debug(f"💾 Saved learnings from session {session_id}: "
                           f"{session['interaction_count']} interactions, "
                           f"avg quality: {avg_quality:.2f}")
    
    def _generate_no_output_response(self, router_results) -> Dict[str, Any]:
        """Generate response when no outputs available"""
        return {
            'final_response': "Unable to generate analysis. No successful VNI outputs.",
            'aggregation_analysis': {
                'consensus_analysis': self.consensus_calculator._empty_consensus(),
                'conflict_analysis': [],
                'vni_contributions': [],
                'domain_coverage': {},
                'learning_applied': False
            },
            'confidence_metrics': {
                'overall_confidence': 0.0,
                'consensus_confidence': 0.0,
                'vni_confidence_distribution': {
                    'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0,
                    'median': 0.0, 'iqr': 0.0, 'high_confidence_ratio': 0.0, 'low_confidence_ratio': 0.0
                },
                'response_quality_trend': {'trend': 'insufficient_data', 'slope': 0.0}
            },
            'synaptic_learning': {'available': False},
            'processing_metadata': {
                'total_vnis_processed': 0,
                'successful_vnis': 0,
                'conflicts_detected': 0,
                'cross_domain_synthesis': False,
                'hebbian_learning_active': self.config.enable_hebbian_learning,
                'learning_cycles': self.learning_metrics.total_learning_cycles
            },
            'aggregator_metadata': {
                'aggregator_id': self.config.aggregator_id,
                'success': False,
                'error': 'No successful VNI outputs',
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _generate_error_output(self, error_msg: str) -> Dict[str, Any]:
        """Generate error output with enhanced information"""
        return {
            'final_response': f"System error during aggregation: {error_msg}",
            'aggregator_metadata': {
                'aggregator_id': self.config.aggregator_id,
                'success': False,
                'error': error_msg,
                'timestamp': datetime.now().isoformat(),
                'learning_cycles': self.learning_metrics.total_learning_cycles,
                'total_queries': self.learning_metrics.total_queries_processed
            }
        }
    
    # ==================== ENHANCED PUBLIC API ====================
    
    def get_synaptic_network_status(self) -> Dict[str, Any]:
        """Get complete synaptic network status with enhanced metrics"""
        learning_stats = self.hebbian_engine.get_learning_statistics()
        
        return {
            'hebbian_learning_enabled': self.config.enable_hebbian_learning,
            'learning_statistics': learning_stats,
            'strongest_connections': self.hebbian_engine.get_strongest_connections(15),
            'learning_metrics': {
                'total_cycles': self.learning_metrics.total_learning_cycles,
                'connections_strengthened': self.learning_metrics.connections_strengthened,
                'connections_weakened': self.learning_metrics.connections_weakened,
                'connections_pruned': self.learning_metrics.connections_pruned,
                'connections_created': self.learning_metrics.connections_created,
                'patterns_detected': self.learning_metrics.patterns_detected,
                'successful_patterns': self.learning_metrics.successful_patterns,
                'total_queries': self.learning_metrics.total_queries_processed,
                'avg_response_quality': self.learning_metrics.avg_response_quality,
                'last_learning': self.learning_metrics.last_learning.isoformat() if self.learning_metrics.last_learning else None
            },
            'connection_strength_trend': self._get_connection_strength_trend(),
            'response_quality_trend': self._get_response_quality_trend(),
            'context_patterns_count': sum(len(v) for v in self.hebbian_engine.context_patterns.values())
        }
    
    def visualize_synaptic_network(self, detailed: bool = False) -> str:
        """Visualize the synaptic network with optional detail"""
        if detailed:
            return self._visualize_detailed_network()
        return self.hebbian_engine.visualize_network()
    
    def _visualize_detailed_network(self) -> str:
        """Create detailed network visualization"""
        lines = ["🧠 DETAILED SYNAPTIC NETWORK ANALYSIS", "=" * 70]
        
        learning_stats = self.hebbian_engine.get_learning_statistics()
        lines.append(f"Total Connections: {learning_stats['total_connections']}")
        lines.append(f"Strong Connections (>0.7): {learning_stats['strong_connections']}")
        lines.append(f"Weak Connections (<0.3): {learning_stats['weak_connections']}")
        lines.append(f"Average Strength: {learning_stats['average_strength']:.3f}")
        lines.append("")
        
        # Connection type breakdown
        lines.append("Connection Type Distribution:")
        for conn_type, count in learning_stats.get('connection_types', {}).items():
            percentage = (count / learning_stats['total_connections'] * 100) if learning_stats['total_connections'] > 0 else 0
            lines.append(f"  • {conn_type}: {count} ({percentage:.1f}%)")
        
        lines.append("")
        
        # Top patterns
        lines.append("Most Successful Patterns:")
        successful_patterns = self.hebbian_engine.successful_patterns
        sorted_patterns = sorted(successful_patterns, 
                               key=lambda p: p['quality'] * p.get('usage_count', 1), 
                               reverse=True)
        
        for i, pattern in enumerate(sorted_patterns[:5], 1):
            vnis_str = ' -> '.join(pattern['vnis'][:3])
            if len(pattern['vnis']) > 3:
                vnis_str += f" (+{len(pattern['vnis']) - 3})"
            lines.append(f"{i}. {vnis_str} | Quality: {pattern['quality']:.3f} | Uses: {pattern.get('usage_count', 1)}")
        
        lines.append("")
        
        # Performance metrics
        lines.append("Performance Metrics:")
        lines.append(f"  • Total Learning Cycles: {self.learning_metrics.total_learning_cycles}")
        lines.append(f"  • Avg Response Quality: {self.learning_metrics.avg_response_quality:.3f}")
        lines.append(f"  • Connections Strengthened: {self.learning_metrics.connections_strengthened}")
        lines.append(f"  • Connections Weakened: {self.learning_metrics.connections_weakened}")
        
        # Connection strength trend
        trend = self._get_connection_strength_trend()
        lines.append(f"  • Connection Trend: {trend['trend']} (slope: {trend['slope']:.4f})")
        
        return "\n".join(lines)
    
    def save_synaptic_network(self, filename: str = "synaptic_network.json"):
        """Save learned synaptic network"""
        self.hebbian_engine.save_network(filename)
        logger.info(f"💾 Saved synaptic network to {filename}")
        return filename
    
    def load_synaptic_network(self, filename: str = "synaptic_network.json"):
        """Load previously learned synaptic network"""
        success = self.hebbian_engine.load_network(filename)
        if success:
            logger.info(f"📂 Loaded synaptic network from {filename}")
        return success
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        learning_stats = self.hebbian_engine.get_learning_statistics() if self.config.enable_hebbian_learning else {}
        
        return {
            'aggregator_id': self.config.aggregator_id,
            'total_clusters': len(self.clusters),
            'active_sessions': len(self.sessions),
            'hebbian_learning_enabled': self.config.enable_hebbian_learning,
            'learning_metrics': {
                'total_cycles': self.learning_metrics.total_learning_cycles,
                'queries_processed': self.learning_metrics.total_queries_processed,
                'connections_strengthened': self.learning_metrics.connections_strengthened,
                'connections_weakened': self.learning_metrics.connections_weakened,
                'connections_pruned': self.learning_metrics.connections_pruned,
                'connections_created': self.learning_metrics.connections_created,
                'patterns_detected': self.learning_metrics.patterns_detected,
                'successful_patterns': self.learning_metrics.successful_patterns,
                'avg_response_quality': self.learning_metrics.avg_response_quality,
                'spontaneous_activations': self.learning_metrics.spontaneous_activations
            },
            'synaptic_network': learning_stats,
            'cluster_types': {cid: c['type'] for cid, c in self.clusters.items()},
            'auto_spawning_enabled': self.config.enable_auto_spawning,
            'visualization_enabled': self.config.enable_visualization,
            'config': {
                'learning_rate': self.config.learning_rate,
                'strengthening_threshold': self.config.strengthening_threshold,
                'weakening_threshold': self.config.weakening_threshold,
                'pruning_threshold': self.config.pruning_threshold
            }
        }
    
    def get_cluster_status(self, cluster_id: str = None) -> Dict[str, Any]:
        """Get cluster status with enhanced metrics"""
        if cluster_id:
            if cluster_id not in self.clusters:
                return {'error': f'Cluster {cluster_id} not found'}
            
            cluster = self.clusters[cluster_id]
            perf = self.cluster_performance.get(cluster_id, ClusterPerformance(cluster_id))
            
            # Calculate performance trend
            performance_trend = 'stable'
            if len(perf.performance_history) >= 5:
                recent = perf.performance_history[-5:]
                older = perf.performance_history[-10:-5] if len(perf.performance_history) >= 10 else perf.performance_history[:5]
                if older:
                    recent_avg = np.mean(recent)
                    older_avg = np.mean(older)
                    if recent_avg > older_avg + 0.05:
                        performance_trend = 'improving'
                    elif recent_avg < older_avg - 0.05:
                        performance_trend = 'declining'
            
            return {
                'cluster_id': cluster_id,
                'type': cluster['type'],
                'instance_ids': cluster['instance_ids'],
                'created': cluster['created'].isoformat(),
                'auto_spawned': cluster.get('auto_spawned', False),
                'performance': {
                    'total_queries': perf.total_queries,
                    'avg_confidence': perf.avg_confidence,
                    'success_rate': perf.success_rate,
                    'last_active': perf.last_active.isoformat() if perf.last_active else None,
                    'performance_trend': performance_trend,
                    'performance_history_size': len(perf.performance_history)
                },
                'statistics': {
                    'total_responses': cluster.get('total_responses', 0),
                    'successful_responses': cluster.get('successful_responses', 0),
                    'success_rate': (cluster.get('successful_responses', 0) / 
                                   cluster.get('total_responses', 1) if cluster.get('total_responses', 0) > 0 else 0.0)
                }
            }
        else:
            return {cid: self.get_cluster_status(cid) for cid in self.clusters}
    
    def save_state(self, filename: str = "aggregator_state.pkl"):
        """Save complete aggregator state including synaptic network"""
        state = {
            'clusters': self.clusters,
            'sessions': self.sessions,
            'learning_metrics': asdict(self.learning_metrics),
            'cluster_performance': {cid: asdict(perf) for cid, perf in self.cluster_performance.items()},
            'routing_table': dict(self.routing_table),
            'config': asdict(self.config),
            'save_time': datetime.now()
        }
        
        # Add Hebbian engine data if enabled
        if self.config.enable_hebbian_learning:
            state['hebbian_engine'] = {
                'connections': {
                    str(k): asdict(v) for k, v in self.hebbian_engine.connections.items()
                },
                'successful_patterns': self.hebbian_engine.successful_patterns,
                'failed_patterns': self.hebbian_engine.failed_patterns,
                'context_patterns': dict(self.hebbian_engine.context_patterns)
            }
        
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"💾 Saved complete state (including synaptic network) to {filename}")
        return filename
    
    def load_state(self, filename: str = "aggregator_state.pkl"):
        """Load complete aggregator state including synaptic network"""
        try:
            with open(filename, 'rb') as f:
                state = pickle.load(f)
            
            self.clusters = state['clusters']
            self.sessions = state['sessions']
            self.learning_metrics = LearningMetrics(**state['learning_metrics'])
            
            # Load cluster performance
            self.cluster_performance = {}
            for cid, perf_dict in state['cluster_performance'].items():
                self.cluster_performance[cid] = ClusterPerformance(**perf_dict)
            
            self.routing_table = defaultdict(list, state['routing_table'])
            
            # Restore Hebbian engine if present
            if self.config.enable_hebbian_learning and 'hebbian_engine' in state:
                hebbian_data = state['hebbian_engine']
                
                # Reconstruct connections
                self.hebbian_engine.connections = {}
                for key_str, conn_dict in hebbian_data['connections'].items():
                    key = eval(key_str)
                    
                    # Handle connection type enum
                    if 'connection_type' in conn_dict:
                        conn_dict['connection_type'] = ConnectionType(conn_dict['connection_type'])
                    
                    # Handle sets for contexts
                    if 'successful_contexts' in conn_dict and isinstance(conn_dict['successful_contexts'], list):
                        conn_dict['successful_contexts'] = set(conn_dict['successful_contexts'])
                    if 'failed_contexts' in conn_dict and isinstance(conn_dict['failed_contexts'], list):
                        conn_dict['failed_contexts'] = set(conn_dict['failed_contexts'])
                    
                    conn = SynapticConnection(**conn_dict)
                    self.hebbian_engine.connections[key] = conn
                
                self.hebbian_engine.successful_patterns = hebbian_data.get('successful_patterns', [])
                self.hebbian_engine.failed_patterns = hebbian_data.get('failed_patterns', [])
                self.hebbian_engine.context_patterns = defaultdict(list, hebbian_data.get('context_patterns', {}))
            
            logger.info(f"📂 Loaded complete state (including synaptic network) from {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False
    
    def get_aggregator_capabilities(self) -> Dict[str, Any]:
        """Get aggregator capabilities with enhanced features"""
        return {
            'aggregator_type': 'unified_hebbian_aggregator_v2',
            'description': 'Enhanced unified system with Hebbian synaptic learning and context-aware routing',
            'version': '2.0.0',
            'capabilities': [
                'Multi-VNI output integration with neural consensus',
                'Advanced conflict detection with semantic analysis',
                'Confidence-weighted response synthesis',
                'Cross-domain response generation',
                'Full Hebbian learning (ΔW = η * pre * post * outcome)',
                'STDP (Spike-Timing-Dependent Plasticity)',
                'Connection strength tracking with context awareness',
                'Pattern recognition and storage with usage tracking',
                'Adaptive routing based on learned connections',
                'Context-aware connection suggestions',
                'Synaptic pruning of weak connections',
                'Connection type classification (collaborative, competitive, sequential, complementary)',
                'Session management with learning persistence',
                'Auto-spawning VNIs based on query needs',
                'Performance tracking with trend analysis',
                'Response quality monitoring and improvement',
                'Cluster performance optimization',
                'Comprehensive visualization capabilities'
            ],
            'learning_features': [
                'Context-aware Hebbian learning with outcome modulation',
                'Temporal correlation tracking (STDP)',
                'Connection type auto-classification',
                'Pattern library with context hashing',
                'Performance trend analysis',
                'Response quality feedback loop',
                'Domain-specific connection optimization',
                'Session-based learning patterns'
            ],
            'neural_components': [
                'ConsensusCalculator with neural network scoring',
                'ConflictDetector with semantic similarity analysis',
                'ResponseSynthesizer with domain-aware generation',
                'HebbianLearningEngine with STDP'
            ],
            'input_types': ['vni_execution_results', 'router_results', 'query', 'session_context'],
            'output_types': [
                'final_response', 
                'aggregation_analysis', 
                'confidence_metrics', 
                'synaptic_insights',
                'system_metrics',
                'learning_metrics'
            ],
            'configurable_parameters': [
                'learning_rate', 'decay_rate', 'strengthening_threshold', 'weakening_threshold',
                'pruning_threshold', 'consensus_threshold', 'max_clusters', 'session_timeout'
            ]
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report"""
        learning_stats = self.hebbian_engine.get_learning_statistics() if self.config.enable_hebbian_learning else {}
        
        # Calculate cluster performance statistics
        cluster_stats = []
        for cluster_id, perf in self.cluster_performance.items():
            cluster_stats.append({
                'cluster_id': cluster_id,
                'type': self.clusters[cluster_id]['type'],
                'total_queries': perf.total_queries,
                'success_rate': perf.success_rate,
                'avg_confidence': perf.avg_confidence
            })
        
        # Sort clusters by performance
        cluster_stats.sort(key=lambda x: x['success_rate'] * x['avg_confidence'], reverse=True)
        
        return {
            'performance_summary': {
                'total_queries': self.learning_metrics.total_queries_processed,
                'avg_response_quality': self.learning_metrics.avg_response_quality,
                'learning_cycles': self.learning_metrics.total_learning_cycles,
                'active_sessions': len(self.sessions),
                'total_clusters': len(self.clusters),
                'total_connections': learning_stats.get('total_connections', 0) if self.config.enable_hebbian_learning else 0
            },
            'learning_effectiveness': {
                'connections_strengthened': self.learning_metrics.connections_strengthened,
                'connections_weakened': self.learning_metrics.connections_weakened,
                'connections_pruned': self.learning_metrics.connections_pruned,
                'patterns_detected': self.learning_metrics.patterns_detected,
                'successful_patterns': self.learning_metrics.successful_patterns,
                'connection_strength_trend': self._get_connection_strength_trend(),
                'response_quality_trend': self._get_response_quality_trend()
            },
            'cluster_performance': cluster_stats[:10],  # Top 10 clusters
            'system_health': {
                'memory_usage': 'normal',  # Placeholder for actual memory monitoring
                'processing_speed': 'optimal',
                'learning_active': self.config.enable_hebbian_learning,
                'auto_spawning_active': self.config.enable_auto_spawning,
                'last_learning': self.learning_metrics.last_learning.isoformat() if self.learning_metrics.last_learning else 'never'
            },
            'recommendations': self._generate_performance_recommendations()
        }
    
    def get_enhanced_diagnostics(self) -> Dict[str, Any]:
        """Get enhanced diagnostics with meta-cognitive insights
        Returns:
            Dictionary with original metrics + meta-cognitive analysis"""
        if hasattr(self, 'meta_cognitive'):
            return get_enhanced_insights(self, self.meta_cognitive)
        else:
            # Fallback to original diagnostics
            return self.get_performance_report()

    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        # Check cluster performance
        for cluster_id, perf in self.cluster_performance.items():
            if perf.total_queries > 10 and perf.success_rate < 0.4:
                recommendations.append(
                    f"Consider optimizing or replacing cluster '{cluster_id}' "
                    f"(success rate: {perf.success_rate:.1%})"
                )
        
        # Check learning effectiveness
        if self.config.enable_hebbian_learning:
            stats = self.hebbian_engine.get_learning_statistics()
            if stats.get('average_strength', 0.5) < 0.3:
                recommendations.append(
                    "Overall connection strength is low. Consider adjusting learning rate "
                    f"(current: {self.config.learning_rate}) or strengthening threshold."
                )
        
        # Check response quality
        if self.learning_metrics.total_queries_processed > 20:
            trend = self._get_response_quality_trend()
            if trend['trend'] == 'declining' and abs(trend['slope']) > 0.02:
                recommendations.append(
                    f"Response quality is declining (slope: {trend['slope']:.4f}). "
                    "Consider reviewing recent queries and responses."
                )
        
        if not recommendations:
            recommendations.append("System performance is optimal. No recommendations at this time.")
        
        return recommendations

    # ==================== BACKWARD COMPATIBILITY ====================
    async def aggregate_response(self, query: str, session_id: str, context: dict = None):
        """Backward compatible wrapper for process_query_advanced"""
        logger.info(f"🔗 aggregate_response called for: '{query[:50]}...'")
        
        # === ADD DEBUG HERE ===
        logger.info("🔍 DEBUG: aggregate_response method called")
        logger.info(f"   query: {query[:100]}")
        logger.info(f"   session_id: {session_id}")
        logger.info(f"   context keys: {list(context.keys()) if context else 'None'}")
        
        # Check for greetings first
        if hasattr(self, 'greeting_preprocessor') and self.greeting_preprocessor.is_greeting(query):
            logger.info(f"🎯 Greeting detected in aggregate_response")
            response = self.greeting_preprocessor.get_response(query)
            response["session_id"] = session_id
            return response
        
        result = await self.process_query_advanced(
            query=query,
            session_id=session_id,
            context=context,
            use_learning=True
        )
        
        # === ADD DEBUG HERE ===
        logger.info(f"✅ aggregate_response completed")
        logger.info(f"   result keys: {list(result.keys())}")
        
        return result
# ==================== ENHANCED COMPONENTS ====================

class ConflictDetector(nn.Module):
    """Enhanced neural network for detecting and classifying conflicts between outputs"""
    def __init__(self, config: AggregatorConfig): 
        super().__init__()
        self.config = config
        
        # Use getattr with defaults in case config doesn't have these
        embedding_dim = getattr(config, 'embedding_dim', 512)
        hidden_dim = getattr(config, 'hidden_dim', 256)
        dropout_rate = getattr(config, 'dropout_rate', 0.1)
        
        self.conflict_detector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64),
            nn.Tanh()
        )
        
        self.conflict_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # 4 classes: no_conflict, minor, major, complementary
            nn.Softmax(dim=-1)
        )
    
    def detect_conflicts(self, vni_outputs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect conflicts between different VNI outputs with enhanced analysis"""
        conflicts = []
        vni_ids = list(vni_outputs.keys())
        
        if len(vni_ids) < 2:
            return conflicts
        
        for i in range(len(vni_ids)):
            for j in range(i + 1, len(vni_ids)):
                vni1_id, vni2_id = vni_ids[i], vni_ids[j]
                vni1_output = vni_outputs[vni1_id]
                vni2_output = vni_outputs[vni2_id]
                
                conflict_analysis = self._analyze_output_conflict(vni1_output, vni2_output)
                
                if conflict_analysis['level'] != 'no_conflict':
                    conflicts.append({
                        'vni1': vni1_id,
                        'vni2': vni2_id,
                        'conflict_level': conflict_analysis['level'],
                        'confidence': conflict_analysis['confidence'],
                        'conflicting_aspects': conflict_analysis['aspects'],
                        'similarity_score': conflict_analysis['similarity_score'],
                        'semantic_overlap': conflict_analysis['semantic_overlap'],
                        'domain_comparison': f"{self._get_domain(vni1_id)} vs {self._get_domain(vni2_id)}",
                        'confidence_difference': conflict_analysis['confidence_difference']
                    })
        
        # Sort by conflict severity
        conflict_weights = {'major_conflict': 3, 'minor_conflict': 2, 'complementary': 1, 'no_conflict': 0}
        conflicts.sort(key=lambda x: conflict_weights[x['conflict_level']], reverse=True)
        
        return conflicts
    
    def _analyze_output_conflict(self, output1: Dict[str, Any], output2: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conflict between two VNI outputs with enhanced metrics"""
        # Extract advice/text from outputs
        advice1 = self._extract_advice(output1)
        advice2 = self._extract_advice(output2)
        
        confidence1 = output1.get('confidence_score', 0.5)
        confidence2 = output2.get('confidence_score', 0.5)
        confidence_difference = abs(confidence1 - confidence2)
        
        # Calculate multiple similarity metrics
        word_similarity = self._calculate_word_similarity(advice1, advice2)
        semantic_similarity = self._calculate_semantic_similarity(advice1, advice2)
        overall_similarity = (word_similarity + semantic_similarity) / 2
        
        # Enhanced conflict classification
        if overall_similarity > 0.85:
            conflict_level = 'no_conflict'
            confidence = 0.9
        elif overall_similarity > 0.65:
            if confidence_difference > 0.4:
                conflict_level = 'minor_conflict'
                confidence = 0.75
            else:
                conflict_level = 'complementary'
                confidence = 0.8
        elif overall_similarity > 0.4:
            conflict_level = 'minor_conflict'
            confidence = 0.7
        else:
            conflict_level = 'major_conflict'
            confidence = 0.85
        
        # Adjust based on confidence difference
        if confidence_difference > 0.3 and conflict_level == 'no_conflict':
            conflict_level = 'minor_conflict'
            confidence = max(confidence, 0.7)
        
        # Identify conflicting aspects
        aspects = []
        if word_similarity < 0.5:
            aspects.append('semantic_meaning')
        if confidence_difference > 0.2:
            aspects.append('confidence_levels')
        if self._has_contradictory_keywords(advice1, advice2):
            aspects.append('keyword_contradiction')
        
        return {
            'level': conflict_level,
            'confidence': confidence,
            'aspects': aspects,
            'similarity_score': overall_similarity,
            'semantic_overlap': semantic_similarity,
            'confidence_difference': confidence_difference
        }
    
    def _extract_advice(self, output: Dict[str, Any]) -> str:
        """Extract advice/text from output"""
        advice_fields = ['medical_advice', 'legal_advice', 'technical_advice', 
                        'financial_advice', 'analytical_advice', 'response']
        
        for field in advice_fields:
            if field in output and output[field]:
                return str(output[field])
        
        return ""
    
    def _calculate_word_similarity(self, text1: str, text2: str) -> float:
        """Calculate word-based similarity"""
        words1 = set(str(text1).lower().split())
        words2 = set(str(text2).lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using word vectors (simplified version)"""
        # This is a simplified version. In production, you would use actual word embeddings
        words1 = str(text1).lower().split()
        words2 = str(text2).lower().split()
        
        if not words1 or not words2:
            return 0.0
        
        # Simple semantic groups
        semantic_groups = {
            'medical': ['health', 'doctor', 'patient', 'treatment', 'medicine', 'hospital'],
            'legal': ['law', 'contract', 'right', 'legal', 'court', 'agreement'],
            'technical': ['code', 'program', 'software', 'technical', 'system', 'algorithm'],
            'financial': ['money', 'finance', 'investment', 'stock', 'market', 'price']
        }
        
        # Count matches in semantic groups
        matches = 0
        total_possible = min(len(words1), len(words2))
        
        for i in range(min(len(words1), len(words2))):
            word1 = words1[i]
            word2 = words2[i]
            
            # Check if words are in same semantic group
            group1 = next((group for group, words in semantic_groups.items() 
                          if word1 in words), None)
            group2 = next((group for group, words in semantic_groups.items() 
                          if word2 in words), None)
            
            if group1 and group2 and group1 == group2:
                matches += 1
            elif word1 == word2:
                matches += 1
        
        return matches / total_possible if total_possible > 0 else 0.0
    
    def _has_contradictory_keywords(self, text1: str, text2: str) -> bool:
        """Check for contradictory keywords"""
        contradictions = [
            ('yes', 'no'), ('true', 'false'), ('correct', 'incorrect'),
            ('should', "shouldn't"), ('recommend', 'avoid'), ('increase', 'decrease')
        ]
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        for word1, word2 in contradictions:
            if (word1 in text1_lower and word2 in text2_lower) or \
               (word2 in text1_lower and word1 in text2_lower):
                return True
        
        return False
    
    def _get_domain(self, vni_id: str) -> str:
        """Extract domain from VNI ID"""
        vni_id_lower = vni_id.lower()
        if 'medical' in vni_id_lower or 'med' in vni_id_lower:
            return 'medical'
        elif 'legal' in vni_id_lower or 'law' in vni_id_lower:
            return 'legal'
        elif 'technical' in vni_id_lower or 'tech' in vni_id_lower:
            return 'technical'
        elif 'analyt' in vni_id_lower:
            return 'analytical'
        elif 'financial' in vni_id_lower or 'finance' in vni_id_lower:
            return 'financial'
        elif 'creative' in vni_id_lower:
            return 'creative'
        return 'general'

class ResponseSynthesizer(nn.Module):
    """Enhanced synthesizer for final response from multiple VNI outputs"""
    def __init__(self, config: AggregatorConfig, llm_gateway=None):
        super().__init__()
        self.config = config
        self.llm_gateway = llm_gateway

        embedding_dim = getattr(config, 'embedding_dim', 512)
        hidden_dim = getattr(config, 'hidden_dim', 256)
        dropout_rate = getattr(config, 'dropout_rate', 0.1)
                
        self.response_generator = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64),
            nn.Tanh()
        )
    
    def synthesize_response(self, vni_outputs: Dict[str, Dict[str, Any]],
                          consensus: Dict[str, Any],
                          conflicts: List[Dict[str, Any]]) -> str:
        """Synthesize final response from all VNI outputs with enhanced logic"""
        if not vni_outputs:
            return "Unable to generate response: No VNI outputs available."
         # Try LLM Gateway first if available
        if self.llm_gateway:
            try:
                return self._synthesize_with_llm(vni_outputs, consensus, conflicts)
            except Exception as e:
                logger.warning(f"LLM synthesis failed, falling back to template: {e}")
                # Fall through to template-based synthesis

        # FIXED: Call the template method and store result in 'response'
        response = self._synthesize_with_templates(vni_outputs, consensus, conflicts)
            
        # Truncate if too long
        if len(response) > self.config.max_output_length:
            response = response[:self.config.max_output_length-3] + "..."
        
        return response
    
    def _synthesize_with_llm(self, vni_outputs: Dict[str, Dict[str, Any]],
                           consensus: Dict[str, Any],
                           conflicts: List[Dict[str, Any]]) -> str:
        """Use LLM Gateway to intelligently synthesize responses"""
               
        domain_advice = self._extract_domain_advice(vni_outputs)
        
        # Build context for LLM
        context = self._build_llm_context(domain_advice, consensus, conflicts)
        
        # Determine VNI context for appropriate system prompt
        primary_domain = self._get_primary_domain(vni_outputs)
        
        # Build system prompt based on consensus level
        system_prompt = self._build_system_prompt(consensus['consensus_level'], conflicts)
        
        # Create LLM config
        llm_config = LLMConfig(
            provider=LLMProvider.DEEPSEEK,  # Default to Deepseek
            model="deepseek-chat",
            temperature=self._get_temperature_for_consensus(consensus['consensus_level']),
            max_tokens=1000,
            system_prompt=system_prompt
        )
        
        # Generate response
        response = self.llm_gateway.generate(
            prompt=context,
            vni_context=primary_domain,
            config=llm_config
        )
        return response.content
    
    def _build_llm_context(self, domain_advice: Dict[str, List[Dict[str, Any]]],
                          consensus: Dict[str, Any],
                          conflicts: List[Dict[str, Any]]) -> str:
        """Build structured context for LLM synthesis"""
        
        context_parts = []
        context_parts.append("=== VNI OUTPUT ANALYSIS ===")
        context_parts.append(f"Consensus Level: {consensus['consensus_level']}")
        context_parts.append(f"Consensus Score: {consensus['consensus_score']:.2%}")
        context_parts.append(f"Total VNIs: {consensus['total_vnis']}")
        
        # Add domain perspectives
        context_parts.append("\n=== EXPERT PERSPECTIVES BY DOMAIN ===")
        for domain, advice_list in domain_advice.items():
            context_parts.append(f"\n[{domain.upper()} DOMAIN]")
            for i, advice in enumerate(advice_list[:2], 1):  # Top 2 per domain
                context_parts.append(f"Expert {i} (Confidence: {advice['confidence']:.0%}):")
                context_parts.append(f"  {advice['advice']}")
        
        # Add conflicts if any
        if conflicts:
            context_parts.append("\n=== IDENTIFIED CONFLICTS ===")
            major_conflicts = [c for c in conflicts if c['conflict_level'] == 'major_conflict']
            minor_conflicts = [c for c in conflicts if c['conflict_level'] == 'minor_conflict']
            
            if major_conflicts:
                context_parts.append(f"Major Conflicts: {len(major_conflicts)}")
                for i, conflict in enumerate(major_conflicts[:3], 1):
                    context_parts.append(f"  {i}. {conflict['vni1']} vs {conflict['vni2']}")
                    context_parts.append(f"     Aspects: {', '.join(conflict['conflicting_aspects'])}")
            
            if minor_conflicts:
                context_parts.append(f"Minor Conflicts: {len(minor_conflicts)}")
        
        # Add domain distribution
        context_parts.append("\n=== DOMAIN COVERAGE ===")
        domain_dist = consensus.get('domain_distribution', {})
        for domain, count in domain_dist.items():
            context_parts.append(f"  • {domain}: {count} VNI(s)")
        
        # Add instruction for LLM
        context_parts.append("\n=== INSTRUCTION ===")
        context_parts.append("Based on the above expert analyses, synthesize a coherent, helpful response.")
        context_parts.append("Guidelines:")
        context_parts.append("1. Integrate information from all relevant domains")
        context_parts.append("2. Acknowledge conflicts transparently but constructively")
        context_parts.append("3. Weight opinions by confidence scores")
        context_parts.append("4. Be concise but comprehensive")
        context_parts.append("5. Use natural, conversational language")
        context_parts.append("6. If medical/legal advice, include appropriate disclaimers")
        
        return "\n".join(context_parts)
    
    def _build_system_prompt(self, consensus_level: str, conflicts: List) -> str:
        """Build appropriate system prompt based on situation"""
        
        base_prompt = "You are BabyBIONN's Response Synthesizer, integrating multiple expert VNI outputs."
        
        prompts = {
            'strong': f"{base_prompt} Strong consensus detected. Synthesize a confident, unified response that clearly presents the converged expert opinions.",
            'moderate': f"{base_prompt} Moderate consensus with some variations. Present a balanced view that acknowledges different perspectives while highlighting areas of agreement.",
            'weak': f"{base_prompt} Weak consensus detected. Present the range of expert opinions, noting the uncertainty and suggesting paths forward.",
            'none': f"{base_prompt} Significant disagreements present. Acknowledge the conflicts transparently, present all major viewpoints, and recommend gathering more information."
        }
        
        system_prompt = prompts.get(consensus_level, base_prompt)
        
        # Add conflict-specific guidance
        if conflicts:
            major_count = len([c for c in conflicts if c['conflict_level'] == 'major_conflict'])
            if major_count > 0:
                system_prompt += f" There are {major_count} major disagreements to address transparently."
        
        return system_prompt
    
    def _get_temperature_for_consensus(self, consensus_level: str) -> float:
        """Get appropriate temperature based on consensus level"""
        temperature_map = {
            'strong': 0.2,   # Lower temp = more focused/factual when confident
            'moderate': 0.4,  # Balanced
            'weak': 0.6,      # More creative when uncertain
            'none': 0.7       # Most creative when handling conflicts
        }
        return temperature_map.get(consensus_level, 0.4)
    
    def _get_primary_domain(self, vni_outputs: Dict[str, Dict[str, Any]]) -> str:
        """Determine primary domain from VNI outputs"""
        domain_counts = defaultdict(int)
        domain_confidences = defaultdict(list)
        
        for vni_id, output in vni_outputs.items():
            domain = self._get_domain(vni_id)
            domain_counts[domain] += 1
            domain_confidences[domain].append(output.get('confidence_score', 0.5))
        
        # Score domains by weighted combination of count and avg confidence
        domain_scores = {}
        for domain in domain_counts:
            avg_confidence = np.mean(domain_confidences[domain]) if domain_confidences[domain] else 0.5
            domain_scores[domain] = domain_counts[domain] * avg_confidence
        
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return 'general'
    
    # ========== EXISTING TEMPLATE-BASED METHODS (KEEP FOR FALLBACK) ==========
    def _synthesize_with_templates(self, vni_outputs: Dict[str, Dict[str, Any]],
                                  consensus: Dict[str, Any],
                                  conflicts: List[Dict[str, Any]]) -> str:
        """Original template-based synthesis - kept as fallback"""
        domain_advice = self._extract_domain_advice(vni_outputs)
        consensus_level = consensus['consensus_level']
        
        if consensus_level == 'strong':
            return self._build_consensus_response(domain_advice, consensus, conflicts)
        elif consensus_level == 'moderate':
            return self._build_balanced_response(domain_advice, consensus, conflicts)
        elif consensus_level == 'weak':
            return self._build_weak_consensus_response(domain_advice, consensus, conflicts)
        else:
            return self._build_conflict_response(domain_advice, consensus, conflicts)

    def _extract_domain_advice(self, vni_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract and group advice by domain"""
        domain_advice = defaultdict(list)
        
        for vni_id, output in vni_outputs.items():
            domain = self._get_domain(vni_id)
            
            advice = self._extract_advice(output)
            confidence = output.get('confidence_score', 0.5)
            
            domain_advice[domain].append({
                'advice': advice,
                'confidence': confidence,
                'vni_id': vni_id,
                'domain': domain
            })
        
        # Sort each domain's advice by confidence
        for domain in domain_advice:
            domain_advice[domain].sort(key=lambda x: x['confidence'], reverse=True)
        
        return dict(domain_advice)
    
    def _extract_advice(self, output: Dict[str, Any]) -> str:
        """Extract advice from output"""
        advice_fields = ['medical_advice', 'legal_advice', 'technical_advice',
                        'financial_advice', 'analytical_advice', 'creative_advice',
                        'response']
        
        for field in advice_fields:
            if field in output and output[field]:
                advice = str(output[field]).strip()
                if advice:
                    return advice
        
        return "No specific advice provided."
    
    def _build_consensus_response(self, domain_advice, consensus, conflicts) -> str:
        """Build response when there's strong consensus"""
        parts = ["Based on comprehensive analysis with strong consensus among experts:"]
        
        # Add domain perspectives
        for domain, advice_list in domain_advice.items():
            if advice_list:
                best_advice = advice_list[0]
                parts.append(f"\n{domain.title()} Analysis: {best_advice['advice']}")
                if len(advice_list) > 1:
                    parts.append(f"  (Supported by {len(advice_list)-1} additional {domain} analyses)")
        
        parts.append(f"\nOverall Confidence: {consensus['consensus_score']:.1%}")
        parts.append("Multiple expert analyses converge on this assessment.")
        
        return " ".join(parts)
    
    def _build_balanced_response(self, domain_advice, consensus, conflicts) -> str:
        """Build balanced response for moderate consensus"""
        parts = ["Analysis reveals multiple expert perspectives:"]
        
        sorted_domains = sorted(domain_advice.items(), 
                              key=lambda x: max(a['confidence'] for a in x[1]), 
                              reverse=True)
        
        for domain, advice_list in sorted_domains:
            if advice_list:
                best_advice = advice_list[0]
                parts.append(f"\n{domain.title()} Perspective: {best_advice['advice']}")
                confidence_note = f" (confidence: {best_advice['confidence']:.0%})"
                parts[-1] += confidence_note
        
        if conflicts:
            major_conflicts = [c for c in conflicts if c['conflict_level'] == 'major_conflict']
            if major_conflicts:
                parts.append(f"\nNote: {len(major_conflicts)} major disagreements were identified and reconciled.")
        
        parts.append("\nThis integrated view considers all available expert opinions, "
                    "weighted by their confidence levels.")
        
        return " ".join(parts)
    
    def _build_weak_consensus_response(self, domain_advice, consensus, conflicts) -> str:
        """Build response for weak consensus"""
        parts = ["Analysis reveals a complex situation with limited consensus among experts:"]
        
        for domain, advice_list in domain_advice.items():
            if advice_list:
                best_advice = advice_list[0]
                parts.append(f"\n{domain.title()} View: {best_advice['advice']}")
        
        if conflicts:
            conflict_count = len([c for c in conflicts if c['conflict_level'] in ['major_conflict', 'minor_conflict']])
            parts.append(f"\nKey Finding: {conflict_count} significant disagreements between expert analyses.")
            parts.append("This suggests the situation may require additional specialist consultation.")
        
        parts.append(f"\nOverall analysis confidence: {consensus['consensus_score']:.1%}")
        parts.append("Consider gathering more information or consulting additional experts.")
        
        return " ".join(parts)
    
    def _build_conflict_response(self, domain_advice, consensus, conflicts) -> str:
        """Build response when there are significant conflicts"""
        parts = ["Analysis reveals significant disagreements among expert opinions:"]
        
        for domain, advice_list in domain_advice.items():
            if advice_list:
                best_advice = advice_list[0]
                parts.append(f"\n{domain.title()} Position: {best_advice['advice']}")
        
        if conflicts:
            major_conflicts = [c for c in conflicts if c['conflict_level'] == 'major_conflict']
            if major_conflicts:
                parts.append(f"\n⚠️  Critical Conflicts: {len(major_conflicts)} major disagreements identified.")
                parts.append("These conflicting views suggest fundamentally different interpretations.")
        
        parts.append(f"\nOverall Consensus Level: Low ({consensus['consensus_score']:.1%})")
        parts.append("\nRecommendation: This complex scenario may require:")
        parts.append("  1. Additional expert consultation")
        parts.append("  2. More detailed information gathering")
        parts.append("  3. Consideration of multiple possible scenarios")
        
        return " ".join(parts)
    
    def _get_domain(self, vni_id: str) -> str:
        """Extract domain from VNI ID"""
        vni_id_lower = vni_id.lower()
        if 'medical' in vni_id_lower or 'med' in vni_id_lower:
            return 'medical'
        elif 'legal' in vni_id_lower or 'law' in vni_id_lower:
            return 'legal'
        elif 'technical' in vni_id_lower or 'tech' in vni_id_lower:
            return 'technical'
        elif 'analyt' in vni_id_lower:
            return 'analytical'
        elif 'financial' in vni_id_lower or 'finance' in vni_id_lower:
            return 'financial'
        elif 'creative' in vni_id_lower:
            return 'creative'
        elif 'research' in vni_id_lower:
            return 'research'
        return 'general'

# ==================== BACKWARD COMPATIBILITY ====================
class ResponseAggregator(UnifiedAggregator):
    """Backward compatible alias for UnifiedAggregator"""
    pass

# ==================== DEMO & TEST FUNCTIONS ====================
async def demo_enhanced_aggregator():
    """Demonstrate the enhanced aggregator with Hebbian learning"""
    print("=" * 80)
    print("🧠 ENHANCED HEBBIAN AGGREGATOR DEMONSTRATION")
    print("=" * 80)
    
    # Create enhanced config
    config = AggregatorConfig(
        aggregator_id="enhanced_demo_001",
        enable_hebbian_learning=True,
        learning_rate=0.15,
        decay_rate=0.005,
        strengthening_threshold=0.75,
        weakening_threshold=0.35,
        pruning_threshold=0.15,
        enable_visualization=True,
        enable_auto_spawning=True,
        max_clusters=8
    ) 
    config.enable_biological_routing = True    
    # Create aggregator
    aggregator = UnifiedAggregator(config)
    
    print("\n📊 System Overview:")
    overview = aggregator.get_system_overview()
    print(f"   • Aggregator ID: {overview['aggregator_id']}")
    print(f"   • Hebbian Learning: {'Enabled' if overview['hebbian_learning_enabled'] else 'Disabled'}")
    print(f"   • Learning Rate: {overview['config']['learning_rate']}")
    print(f"   • Max Clusters: {config.max_clusters}")
    
    # Simulate some learning scenarios
    print("\n🧪 Simulating Learning Scenarios...")
    print("-" * 80)
    
    test_scenarios = [
        {
            'name': 'Medical-Legal Collaboration',
            'execution_results': {
                'med_expert_001': {
                    'medical_advice': 'Patient shows symptoms consistent with bacterial infection. Recommend antibiotics and rest.',
                    'confidence_score': 0.88,
                    'vni_metadata': {'vni_id': 'med_expert_001', 'success': True, 'domain': 'medical'}
                },
                'legal_expert_001': {
                    'legal_advice': 'Ensure proper documentation for medical treatment and HIPAA compliance.',
                    'confidence_score': 0.82,
                    'vni_metadata': {'vni_id': 'legal_expert_001', 'success': True, 'domain': 'legal'}
                }
            },
            'context': {'query_complexity': 0.7, 'detected_domains': ['medical', 'legal']}
        },
        {
            'name': 'Technical-Analytical Synthesis',
            'execution_results': {
                'tech_expert_001': {
                    'technical_advice': 'Code optimization suggests 35% performance improvement potential.',
                    'confidence_score': 0.91,
                    'vni_metadata': {'vni_id': 'tech_expert_001', 'success': True, 'domain': 'technical'}
                },
                'analytics_expert_001': {
                    'analytical_advice': 'Statistical analysis confirms performance trends with 95% confidence.',
                    'confidence_score': 0.89,
                    'vni_metadata': {'vni_id': 'analytics_expert_001', 'success': True, 'domain': 'analytical'}
                }
            },
            'context': {'query_complexity': 0.8, 'detected_domains': ['technical', 'analytical']}
        },
        {
            'name': 'Cross-Domain Conflict',
            'execution_results': {
                'med_expert_001': {
                    'medical_advice': 'Traditional treatment approach recommended based on clinical guidelines.',
                    'confidence_score': 0.45,
                    'vni_metadata': {'vni_id': 'med_expert_001', 'success': True, 'domain': 'medical'}
                },
                'tech_expert_001': {
                    'technical_advice': 'AI analysis suggests experimental protocol with higher success probability.',
                    'confidence_score': 0.42,
                    'vni_metadata': {'vni_id': 'tech_expert_001', 'success': True, 'domain': 'technical'}
                }
            },
            'context': {'query_complexity': 0.9, 'detected_domains': ['medical', 'technical']}
        }
    ]
    
    # Process scenarios
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📈 Scenario {i}: {scenario['name']}")
        
        router_results = {
            'execution_results': scenario['execution_results'],
            'query_context': scenario['context']
        }
        
        result = aggregator(router_results)
        
        print(f"   Response: {result['final_response'][:120]}...")
        print(f"   Confidence: {result['confidence_metrics']['overall_confidence']:.2%}")
        print(f"   Consensus: {result['aggregation_analysis']['consensus_analysis']['consensus_level']}")
        print(f"   Conflicts: {len(result['aggregation_analysis']['conflict_analysis'])}")
        
        if result['synaptic_learning']['available']:
            print(f"   🧠 Active Connections: {len(result['synaptic_learning']['active_connections'])}")
            print(f"   📚 Learning Applied: Yes")
    
    # Show synaptic network status
    print("\n" + "=" * 80)
    print("🕸️  SYNAPTIC NETWORK STATUS AFTER LEARNING")
    print("=" * 80)
    
    network_status = aggregator.get_synaptic_network_status()
    stats = network_status['learning_statistics']
    
    print(f"\n📊 Network Statistics:")
    print(f"   • Total Connections: {stats['total_connections']}")
    print(f"   • Average Strength: {stats['average_strength']:.3f}")
    print(f"   • Successful Patterns: {stats['successful_patterns']}")
    print(f"   • Context Patterns: {network_status['context_patterns_count']}")
    
    print(f"\n📈 Performance Trends:")
    print(f"   • Connection Strength: {network_status['connection_strength_trend']['trend']}")
    print(f"   • Response Quality: {network_status['response_quality_trend']['trend']}")
    
    print(f"\n💪 Top Learned Connections:")
    for i, conn in enumerate(network_status['strongest_connections'][:3], 1):
        print(f"   {i}. {conn['source']} ↔ {conn['target']}")
        print(f"      Strength: {conn['strength']:.3f} | Type: {conn['type']}")
        print(f"      Success Rate: {conn['success_rate']:.1%} | Activations: {conn['activations']}")
    
    # Test learned routing
    print("\n" + "=" * 80)
    print("🎯 TESTING LEARNED ROUTING CAPABILITIES")
    print("=" * 80)
    
    test_context = {
        'query_complexity': 0.7,
        'detected_domains': ['medical', 'legal'],
        'session_id': 'demo_session'
    }
    
    available_vnis = ['med_expert_001', 'legal_expert_001', 'tech_expert_001', 'analytics_expert_001']
    
    if aggregator.config.enable_hebbian_learning:
        suggestions = aggregator.hebbian_engine.suggest_vni_routing(test_context, available_vnis)
        
        print(f"\n📋 Context: Medical + Legal query (complexity: {test_context['query_complexity']})")
        print(f"Suggested VNI Routing (based on learned patterns):")
        for i, (vni, confidence) in enumerate(suggestions[:4], 1):
            print(f"   {i}. {vni}: {confidence:.3f}")
    
    # Get capabilities
    print("\n" + "=" * 80)
    print("🔧 AGGREGATOR CAPABILITIES")
    print("=" * 80)
    
    capabilities = aggregator.get_aggregator_capabilities()
    print(f"\nVersion: {capabilities['version']}")
    print(f"Description: {capabilities['description']}")
    
    print(f"\nKey Features ({len(capabilities['capabilities'])} total):")
    for i, feature in enumerate(capabilities['capabilities'][:8], 1):
        print(f"  {i}. {feature}")
    
    # Save state
    print("\n" + "=" * 80)
    state_file = aggregator.save_state("demo_aggregator_state.pkl")
    print(f"💾 Saved complete state to: {state_file}")
    
    # Performance report
    print("\n" + "=" * 80)
    print("📈 PERFORMANCE REPORT")
    print("=" * 80)
    
    performance = aggregator.get_performance_report()
    summary = performance['performance_summary']
    
    print(f"\n📊 Summary:")
    print(f"   • Total Queries: {summary['total_queries']}")
    print(f"   • Avg Response Quality: {summary['avg_response_quality']:.3f}")
    print(f"   • Learning Cycles: {summary['learning_cycles']}")
    print(f"   • Active Sessions: {summary['active_sessions']}")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(performance['recommendations'][:3], 1):
        print(f"   {i}. {rec}")
    
    print("\n✅ Enhanced aggregator demo completed successfully!")
    print("=" * 80)
    
    return aggregator

if __name__ == "__main__":
    # Run the enhanced demo
    aggregator = asyncio.run(demo_enhanced_aggregator())
    
    print("\n" + "=" * 80)
    print("🎓 KEY ENHANCEMENTS DEMONSTRATED:")
    print("=" * 80)
    print("""
    ✅ Enhanced Hebbian Learning:
       • Context-aware connection updates
       • Connection type classification (collaborative, competitive, sequential, complementary)
       • STDP with actual timing considerations
       • Context pattern storage and retrieval
    
    ✅ Advanced Connection Management:
       • Context-specific success tracking
       • Performance history with trend analysis
       • Enhanced pruning with usage patterns
       • Connection strength trend monitoring
    
    ✅ Improved Routing Intelligence:
       • Context-aware VNI suggestions
       • Pattern-based routing from successful combinations
       • Domain-aware cluster selection
       • Performance-based auto-spawning
    
    ✅ Enhanced Neural Components:
       • Better conflict detection with semantic analysis
       • Improved consensus calculation with neural networks
       • Domain-aware response synthesis
       • Comprehensive confidence metrics
    
    ✅ System Monitoring & Optimization:
       • Response quality tracking
       • Connection strength trends
       • Performance recommendations
       • Comprehensive state persistence
    
    ✅ Extended Capabilities:
       • Support for more domains (financial, creative, research)
       • Enhanced session management
       • Better error handling
       • Comprehensive visualization
    """) 
