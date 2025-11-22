# main.py - FINAL CLEAN INTEGRATED VERSION
#!/usr/bin/env python3
"""
BabyBIONN Main Bridge Module - FINAL CLEAN INTEGRATED VERSION
"""
import os
import cv2
import sys
import json
import torch
import asyncio
import logging
import logging
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime
from safety_check import SafetyManager
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from learning_analytics import LearningAnalytics
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any, Optional, Tuple
from synaptic_visualization import SynapticVisualizer
from fastapi.responses import FileResponse, HTMLResponse
from neuron.demoHybridAttention import DemoHybridAttention
#from neuron.smart_activation_router import SmartActivationRouter
from neuron.aggregator import ResponseAggregator, AggregatorConfig
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Form, UploadFile, File
from neuron.reinforcement_learning.reinforcement_learning import RLConfig, VNIReinforcementEngine
from neuron.reinforcement_learning.vni_rl_integration import VNILearningOrchestrator, integrate_with_existing_vnis

try:
    from neuron.smart_activation_router import SmartActivationRouter
except ImportError as e:
    logger = logging.getLogger("BabyBIONN-Main")
    logger.warning(f"Failed to import SmartActivationRouter: {e}")
    
    # Simple fallback implementation
    class SmartActivationRouter:
        def __init__(self):
            self.activation_threshold = 0.3

        def analyze_query(self, query_text: str):
            """Analyze query text and return domain scores."""
            # Use the same logic as SynapticAttentionBridge._compute_domain_scores
            words = query_text.lower().split()
            medical_keywords = {
                'medical', 'health', 'symptom', 'treatment', 'medicine', 'doctor', 'patient',
                'hospital', 'disease', 'diagnosis', 'pain', 'therapy', 'clinical', 'physical'
            }
            legal_keywords = {
                'legal', 'law', 'contract', 'rights', 'agreement', 'lawyer', 'court',
                'case', 'judge', 'legal', 'regulation', 'compliance', 'liability'
            }
            technical_keywords = {
                'code', 'programming', 'technical', 'system', 'algorithm', 'software',
                'python', 'java', 'database', 'api', 'framework', 'development', 'debug'
            }

            medical_score = sum(1 for word in words if any(med_word in word or word in med_word 
                                                          for med_word in medical_keywords))
            legal_score = sum(1 for word in words if any(leg_word in word or word in leg_word 
                                                        for leg_word in legal_keywords))
            technical_score = sum(1 for word in words if any(tech_word in word or word in tech_word 
                                                            for tech_word in technical_keywords))

            total = medical_score + legal_score + technical_score + 0.001  # Avoid division by zero

            return {
                'medical': medical_score / total,
                'legal': legal_score / total,
                'technical': technical_score / total
            }

        def select_vnis(self, attention_scores):
            """Simple fallback VNI selection"""
            if not attention_scores:
                return []
            
            # Return VNIs with scores above threshold
            selected = [vni_id for vni_id, score in attention_scores.items() 
                       if score >= self.activation_threshold]
            
            # If none above threshold, return top 2
            if not selected:
                sorted_scores = sorted(attention_scores.items(), key=lambda x: x[1], reverse=True)
                selected = [vni_id for vni_id, score in sorted_scores[:2]]
            
            return selected
        
RL_Engine = VNIReinforcementEngine
from enhanced_vni_classes import EnhancedMedicalVNI, EnhancedLegalVNI, EnhancedTechnicalVNI, NeuralPathway

# Import enhanced modules
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# try:
#    from enhanced_vni_classes import EnhancedMedicalVNI, EnhancedLegalVNI, EnhancedTechnicalVNI, NeuralPathway
#    from learning_analytics import LearningAnalytics
#    from synaptic_visualization import SynapticVisualizer
#except ImportError as e:
#    print(f"Import warning: {e}")
    # Fallback implementations would go here

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BabyBIONN-Main")

class SynapticAttentionBridge:
    """Converts BabyBIONN's synaptic connections into Q, K, V tensors using time-series analysis"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.dim = 512  # Match hybrid attention dimension
        self.activation_history = {}  # Track temporal activation patterns
        self.sliding_window_size = 10  # Time steps to consider
        self.connection_weights = {}  # Learned weights for different connection types
        
        # Initialize connection type weights
        self._initialize_connection_weights()
    
    def _initialize_connection_weights(self):
        """Initialize weights for different types of synaptic connections"""
        self.connection_weights = {
            'medical_medical': 1.2,
            'legal_legal': 1.2,
            'technical_technical': 1.2,
            'medical_technical': 0.8,
            'technical_medical': 0.8,
            'medical_legal': 0.6,
            'legal_medical': 0.6,
            'legal_technical': 0.7,
            'technical_legal': 0.7
        }
    
    def synaptic_connections_to_tensors(self, query: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert current synaptic state to Q, K, V tensors using time-series analysis
        """
        # Update activation history with current query context
        self._update_activation_history(query)
        
        # Q: Current query + recent temporal context
        query_tensor = self._query_to_tensor_with_context(query)
        
        # K: VNI expertise + incoming connection patterns with temporal decay
        key_tensor = self._synaptic_strengths_to_keys_with_temporal_context()
        
        # V: VNI response capabilities + outgoing influence with success history
        value_tensor = self._connection_patterns_to_values_with_learning()
        
        return query_tensor, key_tensor, value_tensor
    
    def _update_activation_history(self, query: str):
        """Update temporal activation history for time-series analysis"""
        current_time = datetime.now()
        
        # Add current query to history
        if 'queries' not in self.activation_history:
            self.activation_history['queries'] = []
        
        self.activation_history['queries'].append({
            'timestamp': current_time,
            'query': query,
            'embedding': self._text_to_advanced_embedding(query)
        })
        
        # Trim history to sliding window size
        if len(self.activation_history['queries']) > self.sliding_window_size:
            self.activation_history['queries'] = self.activation_history['queries'][-self.sliding_window_size:]
    
    def _query_to_tensor_with_context(self, query: str) -> torch.Tensor:
        """Convert text query to tensor with temporal context from synaptic activations"""
        # Base query embedding
        base_embedding = self._text_to_advanced_embedding(query)
        
        # Get temporal context from recent activations
        temporal_context = self._get_temporal_context_embedding()
        
        # Get spatial context from current synaptic state
        spatial_context = self._get_spatial_context_embedding()
        
        # Combine: query + temporal + spatial context
        contextual_embedding = (
            base_embedding * 0.6 + 
            temporal_context * 0.25 + 
            spatial_context * 0.15
        )
        
        return contextual_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
    
    def _synaptic_strengths_to_keys_with_temporal_context(self) -> torch.Tensor:
        """Convert synaptic connection strengths to key tensor with temporal decay"""
        keys = []
        current_time = datetime.now()
        
        for vni_id, vni_instance in self.orchestrator.vni_instances.items():
            # Calculate incoming strength with temporal decay
            incoming_strength = self._get_temporal_incoming_strength(vni_id, current_time)
            
            # Get expertise vector with learning adaptations
            expertise_vector = self._get_adapted_expertise_vector(vni_id, vni_instance)
            
            # Apply connection type weighting
            connection_weight = self._get_connection_type_weight(vni_id, "incoming")
            
            # Key = adapted expertise modulated by temporally-weighted incoming strength
            key = expertise_vector * incoming_strength * connection_weight
            keys.append(key)
        
        return torch.stack(keys).unsqueeze(0)  # [1, num_vnis, dim]
    
    def _connection_patterns_to_values_with_learning(self) -> torch.Tensor:
        """Convert synaptic connection patterns to value tensor with learning history"""
        values = []
        
        for vni_id, vni_instance in self.orchestrator.vni_instances.items():
            # Calculate outgoing influence with success-based weighting
            outgoing_influence = self._get_learning_weighted_outgoing_influence(vni_id)
            
            # Get response capability with performance adaptation
            response_capability = self._get_performance_adapted_capability(vni_id, vni_instance)
            
            # Apply success-based modulation
            success_modulation = self._get_success_modulation(vni_id)
            
            # Value = performance-adapted capability modulated by learning-weighted influence
            value = response_capability * outgoing_influence * success_modulation
            values.append(value)
        
        return torch.stack(values).unsqueeze(0)  # [1, num_vnis, dim]
    
    def _text_to_advanced_embedding(self, text: str) -> torch.Tensor:
        """Advanced text embedding with semantic understanding"""
        words = text.lower().split()
        embedding = torch.zeros(self.dim)
        
        # Semantic domain detection with fuzzy matching
        domain_scores = self._compute_domain_scores(words)
        
        # Set embedding based on domain scores
        embedding[0:128] = domain_scores['medical']  # Medical domain features
        embedding[128:256] = domain_scores['legal']   # Legal domain features  
        embedding[256:384] = domain_scores['technical']  # Technical domain features
        
        # Add query complexity features
        embedding[384:448] = self._compute_complexity_features(text)
        
        # Add temporal relevance features (recent topics)
        embedding[448:512] = self._compute_temporal_relevance_features(text)
        
        return embedding
    
    def _compute_domain_scores(self, words: List[str]) -> Dict[str, float]:
        """Compute domain relevance scores using keyword expansion and semantic similarity"""
        medical_keywords = {
            'medical', 'health', 'symptom', 'treatment', 'medicine', 'doctor', 'patient',
            'hospital', 'disease', 'diagnosis', 'pain', 'therapy', 'clinical', 'physical'
        }
        legal_keywords = {
            'legal', 'law', 'contract', 'rights', 'agreement', 'lawyer', 'court',
            'case', 'judge', 'legal', 'regulation', 'compliance', 'liability'
        }
        technical_keywords = {
            'code', 'programming', 'technical', 'system', 'algorithm', 'software',
            'python', 'java', 'database', 'api', 'framework', 'development', 'debug'
        }
        
        # Compute fuzzy matches
        medical_score = sum(1 for word in words if any(med_word in word or word in med_word 
                                                     for med_word in medical_keywords))
        legal_score = sum(1 for word in words if any(leg_word in word or word in leg_word 
                                                   for leg_word in legal_keywords))
        technical_score = sum(1 for word in words if any(tech_word in word or word in tech_word 
                                                       for tech_word in technical_keywords))
        
        total = medical_score + legal_score + technical_score + 0.001  # Avoid division by zero
        
        return {
            'medical': medical_score / total,
            'legal': legal_score / total,
            'technical': technical_score / total
        }
    
    def _compute_complexity_features(self, text: str) -> torch.Tensor:
        """Compute text complexity features"""
        words = text.split()
        features = torch.zeros(64)
        
        # Basic complexity metrics
        features[0] = len(words) / 50.0  # Normalized length
        features[1] = len([w for w in words if len(w) > 6]) / len(words) if words else 0  # Long words ratio
        features[2] = len([w for w in words if w.istitle()]) / len(words) if words else 0  # Proper nouns ratio
        
        # Question features
        features[3] = 1.0 if text.strip().endswith('?') else 0.0
        
        return features
    
    def _compute_temporal_relevance_features(self, text: str) -> torch.Tensor:
        """Compute temporal relevance based on recent conversation history"""
        features = torch.zeros(64)
        
        if 'queries' not in self.activation_history or len(self.activation_history['queries']) < 2:
            return features
        
        # Compare with recent queries for topic continuity
        recent_queries = self.activation_history['queries'][-3:]  # Last 3 queries
        current_embedding = self._text_to_advanced_embedding(text)
        
        similarities = []
        for past_query in recent_queries:
            past_embedding = past_query['embedding']
            similarity = torch.cosine_similarity(current_embedding, past_embedding, dim=0)
            similarities.append(similarity.item())
        
        if similarities:
            features[0] = max(similarities)  # Maximum similarity to recent context
            features[1] = sum(similarities) / len(similarities)  # Average similarity
        
        return features
    
    def _get_temporal_context_embedding(self) -> torch.Tensor:
        """Get temporal context from recent activation patterns"""
        if 'queries' not in self.activation_history or not self.activation_history['queries']:
            return torch.zeros(self.dim)
        
        # Weight recent queries by recency (exponential decay)
        recent_queries = self.activation_history['queries']
        total_weight = 0.0
        weighted_sum = torch.zeros(self.dim)
        
        for i, query_data in enumerate(recent_queries):
            # Exponential decay: more recent = higher weight
            weight = 0.9 ** (len(recent_queries) - i - 1)
            weighted_sum += query_data['embedding'] * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else torch.zeros(self.dim)
    
    def _get_spatial_context_embedding(self) -> torch.Tensor:
        """Get spatial context from current synaptic connection patterns"""
        if not self.orchestrator.synaptic_connections:
            return torch.zeros(self.dim)
        
        # Analyze connection patterns for spatial context
        connection_strengths = torch.zeros(3)  # medical, legal, technical
        
        for pathway in self.orchestrator.synaptic_connections.values():
            source_type = pathway.source.split('_')[0]
            target_type = pathway.target.split('_')[0]
            
            if source_type == 'medical' or target_type == 'medical':
                connection_strengths[0] += pathway.strength
            if source_type == 'legal' or target_type == 'legal':
                connection_strengths[1] += pathway.strength
            if source_type == 'technical' or target_type == 'technical':
                connection_strengths[2] += pathway.strength
        
        # Normalize and expand to full dimension
        if connection_strengths.sum() > 0:
            connection_strengths = connection_strengths / connection_strengths.sum()
        
        spatial_embedding = torch.zeros(self.dim)
        spatial_embedding[0:128] = connection_strengths[0]  # Medical context
        spatial_embedding[128:256] = connection_strengths[1]  # Legal context
        spatial_embedding[256:384] = connection_strengths[2]  # Technical context
        
        return spatial_embedding
    
    def _get_temporal_incoming_strength(self, vni_id: str, current_time: datetime) -> float:
        """Calculate incoming synaptic strength with temporal decay"""
        total_strength = 0.0
        count = 0
        
        for conn_id, pathway in self.orchestrator.synaptic_connections.items():
            if pathway.target == vni_id:
                # Apply temporal decay based on last activation
                decay_factor = self._compute_temporal_decay(pathway, current_time)
                total_strength += pathway.strength * decay_factor
                count += 1
        
        return total_strength / max(count, 1)
    
    def _get_learning_weighted_outgoing_influence(self, vni_id: str) -> float:
        """Calculate outgoing influence weighted by learning success"""
        total_influence = 0.0
        count = 0
        
        for conn_id, pathway in self.orchestrator.synaptic_connections.items():
            if pathway.source == vni_id:
                # Weight by learning success rate if available
                success_weight = 1.0
                if hasattr(pathway, 'success_rate'):
                    success_weight = pathway.success_rate
                elif hasattr(pathway, 'activation_count') and pathway.activation_count > 0:
                    # Estimate success rate from activation patterns
                    success_weight = min(1.0, pathway.strength * 1.5)
                
                total_influence += pathway.strength * success_weight
                count += 1
        
        return total_influence / max(count, 1)
    
    def _get_adapted_expertise_vector(self, vni_id: str, vni_instance) -> torch.Tensor:
        """Get expertise vector adapted through learning"""
        base_vector = torch.zeros(self.dim)
        vni_type = vni_id.split('_')[0]
        
        # Set base domain expertise
        if vni_type == 'medical':
            base_vector[0:128] = 1.0
        elif vni_type == 'legal':
            base_vector[128:256] = 1.0
        elif vni_type == 'technical':
            base_vector[256:384] = 1.0
        
        # Apply learning adaptations if available
        if hasattr(vni_instance, 'get_learning_adaptation'):
            adaptation = vni_instance.get_learning_adaptation()
            if adaptation is not None:
                # Blend base expertise with learned adaptations
                base_vector = base_vector * 0.7 + adaptation * 0.3
        
        return base_vector
    
    def _get_performance_adapted_capability(self, vni_id: str, vni_instance) -> torch.Tensor:
        """Get response capability adapted by performance history"""
        base_capability = self._get_adapted_expertise_vector(vni_id, vni_instance)
        
        # Apply performance-based scaling
        performance_factor = self._get_performance_factor(vni_id)
        
        return base_capability * performance_factor
    
    def _get_connection_type_weight(self, vni_id: str, direction: str) -> float:
        """Get weight based on connection type and direction"""
        vni_type = vni_id.split('_')[0]
        
        if direction == "incoming":
            # For incoming, look at sources connecting to this VNI
            connection_types = set()
            for pathway in self.orchestrator.synaptic_connections.values():
                if pathway.target == vni_id:
                    source_type = pathway.source.split('_')[0]
                    connection_types.add(f"{source_type}_{vni_type}")
        else:  # outgoing
            # For outgoing, look at targets from this VNI
            connection_types = set()
            for pathway in self.orchestrator.synaptic_connections.values():
                if pathway.source == vni_id:
                    target_type = pathway.target.split('_')[0]
                    connection_types.add(f"{vni_type}_{target_type}")
        
        # Return average weight of all connection types
        if not connection_types:
            return 1.0
        
        total_weight = sum(self.connection_weights.get(conn_type, 1.0) for conn_type in connection_types)
        return total_weight / len(connection_types)
    
    def _compute_temporal_decay(self, pathway, current_time: datetime) -> float:
        """Compute temporal decay factor for synaptic connections"""
        if not hasattr(pathway, 'last_activated') or pathway.last_activated is None:
            return 1.0  # No decay if no activation history
        
        time_diff = (current_time - pathway.last_activated).total_seconds()
        
        # Exponential decay: half-life of 1 hour
        decay_rate = 0.5 ** (time_diff / 3600)
        
        return max(0.1, decay_rate)  # Minimum 10% strength
    
    def _get_success_modulation(self, vni_id: str) -> float:
        """Get success-based modulation factor for VNI"""
        # This would ideally come from the VNI's performance history
        # For now, use a simple heuristic based on connection strengths
        total_strength = 0.0
        count = 0
        
        for pathway in self.orchestrator.synaptic_connections.values():
            if pathway.source == vni_id or pathway.target == vni_id:
                total_strength += pathway.strength
                count += 1
        
        if count == 0:
            return 1.0
        
        avg_strength = total_strength / count
        # Map average strength to success modulation (0.5 to 1.5 range)
        return 0.5 + avg_strength
    
    def _get_performance_factor(self, vni_id: str) -> float:
        """Get performance factor based on historical success"""
        # Simple implementation - could be enhanced with actual performance tracking
        return 1.0  # Default neutral performance

class EnhancedBabyBIONNOrchestrator:
    """FINAL orchestrator with real learning and dynamic VNI networking"""
    def __init__(self):
        self.initialized = False
        self.vni_instances = {}
        self.hybrid_attention = DemoHybridAttention(
            dim=512,
            num_heads=8,
            window_size=256,
            use_sliding=True,
            use_global=True, 
            use_hierarchical=True,
            global_token_ratio=0.05,
            memory_tokens=16,
            multi_modal=True
        )
        self.learning_orchestrator = None # integrate_with_existing_vnis(self)
        self.rl_engine = VNIReinforcementEngine(RLConfig()) 
        self.smart_router = SmartActivationRouter()        
        self.synaptic_bridge = SynapticAttentionBridge(self) 
        self.synaptic_connections = {}
        self.conversation_history = []
        self.max_history_length = 100
        self.analytics = LearningAnalytics()
        self.visualizer = SynapticVisualizer()
        self.safety_manager = SafetyManager()
        aggregator_config = AggregatorConfig(
            aggregator_id="babybionn_main_aggregator",
            consensus_threshold=0.7,
            conflict_resolution_strategy="confidence_weighted"
        )
        self.response_aggregator = ResponseAggregator(aggregator_config)
        
    async def initialize(self):
        """Initialize enhanced BabyBIONN system"""
        try:
            logger.info("🚀 Initializing Enhanced BabyBIONN Orchestrator...")
            
            # Spawn multiple VNI instances
            self.spawn_vni_instances("medical", 2)
            self.spawn_vni_instances("legal", 2) 
            self.spawn_vni_instances("technical", 2)
            
            # Initialize synaptic connections
            self.initialize_synaptic_connections()

            # Initialize learning orchestrator
            try:
                self.learning_orchestrator = integrate_with_existing_vnis(self)
                logger.info("✅ Learning orchestrator integrated")

            except Exception as e:
                logger.warning(f"Learning orchestrator initialization failed: {e}")
                self.learning_orchestrator = None

            self.initialized = True
            logger.info("✅ Enhanced BabyBIONN Orchestrator initialized successfully!")
            logger.info(f"📊 Spawned {len(self.vni_instances)} VNI instances")
            logger.info(f"🔗 Created {len(self.synaptic_connections)} synaptic connections")
            
        except Exception as e:
            logger.error(f"❌ Enhanced initialization failed: {e}")
            raise
    
    def spawn_vni_instances(self, vni_type: str, count: int):
        """Spawn multiple VNI instances of a type"""
        for i in range(count):
            instance_id = f"{vni_type}_{i}"
            
            if vni_type == "medical":
                instance = EnhancedMedicalVNI(instance_id)
            elif vni_type == "legal":
                instance = EnhancedLegalVNI(instance_id)
            elif vni_type == "technical":
                instance = EnhancedTechnicalVNI(instance_id)
            else:
                continue
                
            self.vni_instances[instance_id] = instance
            logger.info(f"  ➕ Spawned {instance_id}")
    
    def initialize_synaptic_connections(self):
        """Initialize synaptic connections between VNI instances"""
        instance_ids = list(self.vni_instances.keys())
        
        # Create connections between different VNI types
        for i, source_id in enumerate(instance_ids):
            for j, target_id in enumerate(instance_ids):
                if i != j:  # No self-connections
                    source_type = source_id.split('_')[0]
                    target_type = target_id.split('_')[0]
                    
                    # Different connection strengths based on type compatibility
                    if source_type == target_type:
                        strength = 0.8  # Stronger within same type
                    elif (source_type, target_type) in [('medical', 'technical'), ('technical', 'medical')]:
                        strength = 0.6  # Medium strength
                    else:
                        strength = 0.4  # Weaker connections
                    
                    connection_id = f"{source_id}→{target_id}"
                    self.synaptic_connections[connection_id] = NeuralPathway(
                        source_id, target_id, strength
                    )

    async def route_through_vni_network(self, input_text: str, context: str = "general") -> List[Dict]:
        """Route input through the VNI network for processing"""
        logger.info(f"🧠 Processing: {input_text}")
        try:
            # TRY USING LEARNING ORCHESTRATOR FIRST
            if self.learning_orchestrator:
                from neuron.reinforcement_learning.vni_rl_integration import VNIStimulus
            
                # Use learning orchestrator for intelligent routing
                stimulus = VNIStimulus(
                    content=input_text,
                    stimulus_type="chat_query",
                    metadata={"context": context}
                )
            
                learning_response = self.learning_orchestrator.process_stimulus_with_learning(stimulus)
            
                # Convert learning response to expected format
                vni_responses = []
                for vni_id, vni_response in learning_response['vni_responses'].items():
                    vni_responses.append({
                        'response': vni_response.content,
                        'confidence': vni_response.confidence,
                        'vni_instance': vni_id,
                        'response_type': getattr(vni_response, 'response_type', 'general'),
                        'concepts_used': [],
                        'patterns_matched': []
                    })
            
                logger.info(f"🎯 Learning-based routing activated: {len(vni_responses)} responses")
                return vni_responses
            
        except Exception as e:
            logger.warning(f"Learning orchestrator failed, falling back to basic routing: {e}")

        # First, try to use the smart router to determine the best domain
        # Use the global instance of SmartActivationRouter (either real or fallback)
        # Do NOT import it locally again here. Like this : 'router = SmartActivationRouter()' - This line was problematic

        # Use the instance created during __init__
        # This instance is self.smart_router, but select_vnis needs attention_scores,
        # not a raw query text for domain analysis.

        # Use the same logic as SynapticAttentionBridge._compute_domain_scores
        # (or define a simple helper function for this specific task)
        words = input_text.lower().split()
        medical_keywords = {
            'medical', 'health', 'symptom', 'treatment', 'medicine', 'doctor', 'patient',
            'hospital', 'disease', 'diagnosis', 'pain', 'therapy', 'clinical', 'physical'
        }
        legal_keywords = {
            'legal', 'law', 'contract', 'rights', 'agreement', 'lawyer', 'court',
            'case', 'judge', 'legal', 'regulation', 'compliance', 'liability'
        }
        technical_keywords = {
            'code', 'programming', 'technical', 'system', 'algorithm', 'software',
            'python', 'java', 'database', 'api', 'framework', 'development', 'debug'
        }

        medical_score = sum(1 for word in words if any(med_word in word or word in med_word
                                                      for med_word in medical_keywords))
        legal_score = sum(1 for word in words if any(leg_word in word or word in leg_word
                                                    for leg_word in legal_keywords))
        technical_score = sum(1 for word in words if any(tech_word in word or word in tech_word
                                                        for tech_word in technical_keywords))

        total = medical_score + legal_score + technical_score + 0.001  # Avoid division by zero

        domain_scores = {
            'medical': medical_score / total,
            'legal': legal_score / total,
            'technical': technical_score / total
        }

        # Get the top domain
        top_domain = max(domain_scores, key=domain_scores.get)
        logger.info(f"🎯 Top domain: {top_domain} (score: {domain_scores[top_domain]:.2f})")

        # Route to the best VNI instance in that domain
        # Find the best available VNI instance for the top domain
        available_vnis = [vni_id for vni_id in self.vni_instances.keys() if vni_id.startswith(top_domain)]
        if available_vnis:
            # For now, just use the first instance found for the domain
            vni_id = available_vnis[0]
            vni_instance = self.vni_instances[vni_id]
            vni_response = vni_instance.process_query(input_text, context)
            logger.info(f"📝 Response from {vni_id}: {vni_response}")
            # Return a list containing the VNI's response dict
            # Ensure the response dict has the expected keys for aggregate_vni_responses
            if isinstance(vni_response, dict) and 'response' in vni_response:
                # Assume it has other keys like 'vni_instance', 'confidence', etc., as expected by aggregator
                return [vni_response]
            else:
                # Fallback if response format is unexpected, wrap the raw result
                logger.warning(f"Response from {vni_id} might be in unexpected format: {type(vni_response)}")
                return [{
                    'response': str(vni_response), # Convert to string just in case
                    'confidence': 0.5, # Default confidence
                    'vni_instance': vni_id,
                    'success': True # Assume success if we got a response
                }]

        else:
            logger.warning(f"⚠️  No VNI instances available for domain: {top_domain}")
            # Fallback: try another top domain or use a general response
            sorted_domains = sorted(domain_scores.items(), key=lambda item: item[1], reverse=True)
            for domain, _ in sorted_domains:
                available_vnis = [vni_id for vni_id in self.vni_instances.keys() if vni_id.startswith(domain)]
                if available_vnis:
                    vni_id = available_vnis[0]
                    vni_instance = self.vni_instances[vni_id]
                    vni_response = vni_instance.process_query(input_text, context)
                    logger.info(f"📝 Fallback response from {vni_id}: {vni_response}")
                    # Return a list containing the VNI's response dict
                    if isinstance(vni_response, dict) and 'response' in vni_response:
                        return [vni_response]
                    else:
                        logger.warning(f"Fallback response from {vni_id} might be in unexpected format: {type(vni_response)}")
                        return [{
                            'response': str(vni_response),
                            'confidence': 0.5,
                            'vni_instance': vni_id,
                            'success': True
                        }]

            # If no VNIs were found even after fallback domains
            return [{
                'response': "I'm not sure how to help with that. Please try rephrasing your question.",
                'confidence': 0.3, # Low confidence for fallback
                'vni_instance': 'fallback',
                'success': False # Indicate fallback failure
            }]

    async def process_message(self, user_message: str, session_id: str = "default_user") -> Dict[str, Any]:
        """Enhanced message processing with dynamic VNI networking"""
        if not self.initialized:
            await self.initialize()
        
        try:
            logger.info(f"🧠 Processing: {user_message}")
            
            # Store in conversation history
            self._add_to_history("user", user_message, session_id)
            
            # Route through VNI network
            context = self._build_context(user_message)
            vni_responses = await self.route_through_vni_network(user_message, context)
            
            # Aggregate responses
            final_response = self.aggregate_vni_responses(vni_responses, user_message)
            
            # Store bot response
            bot_response = self._format_response(final_response)
            self._add_to_history("assistant", bot_response, session_id)
            
            # Update analytics
            for response in vni_responses:
                self.analytics.record_interaction(
                    session_id, 
                    response['vni_instance'].split('_')[0],
                    response.get('confidence', 0.5)
                )
            
            # Update synaptic connections based on success
            self.update_synaptic_connections(vni_responses, success=True)
            
            result = {
                "response": bot_response,
                "session_id": session_id,
                "activated_vnis": [r['vni_instance'] for r in vni_responses],
                "average_confidence": final_response.get('confidence', 0.5),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Response ready from {len(vni_responses)} VNIs")
            return result
            
        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
            error_response = "I apologize, but I encountered an error. Please try again."
            self._add_to_history("assistant", error_response, session_id)
            return {
                "response": error_response,
                "session_id": session_id,
                "error": str(e)
            }
    
    async def _vni_network(self, query: str, context: Dict = None) -> List[Dict]:
        """Intelligent query routing using synaptic-based attention"""
        try:
            # Convert synaptic state to attention tensors
            query_tensor, key_tensor, value_tensor = self.synaptic_bridge.synaptic_connections_to_tensors(query)
    
            # Pass through hybrid attention
            with torch.no_grad():
                attention_output = self.hybrid_attention(
                    query=query_tensor,
                    key=key_tensor, 
                    value=value_tensor
                )
        
            # Extract attention weights safely
            if isinstance(attention_output, tuple) and len(attention_output) >= 2:
                attention_weights = attention_output[1]
            else:
                attention_weights = attention_output
            
            # Process attention weights safely
            vni_ids = list(self.vni_instances.keys())
            num_vnis = len(vni_ids)
        
            if attention_weights.dim() == 4:  # [batch_size, num_heads, target_len, source_len]
                try:
                    # Average over heads and squeeze batch dimension
                    vni_scores = attention_weights.mean(dim=1).squeeze(0)  # [target_len, source_len]
                
                    # FIX: Ensure we have the right dimensions
                    if vni_scores.dim() == 2 and vni_scores.size(0) >= num_vnis:
                        # Take attention from query (first token) to each VNI
                        attention_from_query = vni_scores[0, :]  # Attention from query to all sources
                    
                        # Map to VNIs
                        if len(attention_from_query) >= num_vnis:
                            attention_scores = {vni_id: attention_from_query[i].item() for i, vni_id in enumerate(vni_ids)}
                        else:
                            attention_scores = {vni_id: 0.5 for vni_id in vni_ids}
                    else:
                        attention_scores = {vni_id: 0.5 for vni_id in vni_ids}
                    
                except Exception as e:
                    logger.warning(f"Attention score processing failed: {e}")
                    attention_scores = {vni_id: 0.5 for vni_id in vni_ids}
            else:
                attention_scores = {vni_id: 0.5 for vni_id in vni_ids}
            
        except Exception as e:
            logger.warning(f"Synaptic attention failed: {e}")
            attention_scores = self._fallback_keyword_routing(query)
    
        # SMART ACTIVATION ROUTING - only activate relevant VNIs
        activated_vnis = self.smart_router.select_vnis(attention_scores)

        responses = []
        for vni_id in activated_vnis:
            if vni_id in self.vni_instances:
                vni_instance = self.vni_instances[vni_id]
                response = vni_instance.process_query(query, context)
                responses.append(response)
        
                # REINFORCEMENT LEARNING - strengthen used pathways
                self.rl_engine.record_activation(vni_id, query, response)

                # Activate outgoing synaptic connections
                self.activate_synaptic_connections(vni_id, responses)

        return responses

    def _fallback_keyword_routing(self, query: str) -> Dict[str, float]:
        """Fallback routing when attention fails"""
        attention_scores = {}
        query_lower = query.lower()
    
        for vni_id in self.vni_instances.keys():
            # Simple domain matching
            if 'medical' in vni_id and any(word in query_lower for word in 
                                     ['health', 'symptom', 'treatment', 'medicine', 'doctor', 'pain']):
                attention_scores[vni_id] = 0.8
            elif 'legal' in vni_id and any(word in query_lower for word in 
                                     ['legal', 'law', 'contract', 'rights', 'lawyer']):
                attention_scores[vni_id] = 0.8
            elif 'technical' in vni_id and any(word in query_lower for word in 
                                         ['code', 'programming', 'technical', 'system', 'software']):
                attention_scores[vni_id] = 0.8
            else:
                attention_scores[vni_id] = 0.3
    
        return attention_scores    
    def identify_relevant_vnis(self, user_message: str) -> List[str]:
        """Identify which VNIs are relevant for this query"""
        message_lower = user_message.lower()
        relevant = []
        
        # Simple keyword-based routing - could be enhanced with ML
        medical_keywords = ['health', 'symptom', 'treatment', 'medicine', 'medical', 'doctor']
        legal_keywords = ['legal', 'law', 'contract', 'rights', 'agreement', 'lawyer']
        technical_keywords = ['code', 'programming', 'technical', 'system', 'algorithm', 'software']
        
        if any(keyword in message_lower for keyword in medical_keywords):
            relevant.extend([vni_id for vni_id in self.vni_instances if vni_id.startswith('medical')])
        
        if any(keyword in message_lower for keyword in legal_keywords):
            relevant.extend([vni_id for vni_id in self.vni_instances if vni_id.startswith('legal')])
            
        if any(keyword in message_lower for keyword in technical_keywords):
            relevant.extend([vni_id for vni_id in self.vni_instances if vni_id.startswith('technical')])
        
        # If no specific keywords, use all VNIs
        if not relevant:
            relevant = list(self.vni_instances.keys())
            
        return relevant[:3]  # Limit to 3 most relevant
    
    def activate_synaptic_connections(self, source_vni: str, responses: List[Dict]):
        """Activate synaptic connections from source VNI"""
        for connection_id, pathway in self.synaptic_connections.items():
            if pathway.source == source_vni:
                # Determine success based on response confidence
                success = any(r.get('confidence', 0) > 0.7 for r in responses)
                pathway.activate(success)
    
    def aggregate_vni_responses(self, responses: List[Dict], user_message: str) -> Dict:
        """Use the ResponseAggregator to properly combine VNI outputs"""
        if not responses:
            return {"response": "I'm not sure how to help with that.", "confidence": 0.3}
    
        # Convert responses to the format expected by ResponseAggregator
        execution_results = {}
        for i, response in enumerate(responses):
            vni_id = response.get('vni_instance', f'vni_{i}')
            # Convert the Python float to a torch.Tensor with dtype=torch.float32
            confidence_float = response.get('confidence', 0.5)
            confidence_tensor = torch.tensor(confidence_float, dtype=torch.float32) # <-- FIX: Convert to tensor
            execution_results[vni_id] = {
                'response': response.get('response', ''),
                'confidence_score': confidence_tensor, # <-- Use the tensor, not the float
                'vni_metadata': {
                    'vni_id': vni_id,
                    'success': True,
                    'domain': vni_id.split('_')[0] if '_' in vni_id else 'general' 
                }
            }    
            # Create router results format expected by aggregator
            router_results = {
                'execution_results': execution_results,
                'activation_plan': {
                'activated_vnis': [{'vni_id': vni_id} for vni_id in execution_results.keys()]
            }
        }
    
        # Use the neural aggregator to combine responses
        with torch.no_grad():
            aggregation_result = self.response_aggregator(router_results)
    
        # Extract the final synthesized response
        final_response = aggregation_result.get('final_response', '')
        overall_confidence = aggregation_result.get('confidence_metrics', {}).get('overall_confidence', 0.5)
    
        return {
            "response": final_response,
            "confidence": overall_confidence,
            "aggregation_analysis": aggregation_result.get('aggregation_analysis', {}),
            "vni_contributions": len(responses)
        
        }    
    def update_synaptic_connections(self, responses: List[Dict], success: bool):
        """Enhanced synaptic learning with Hebbian rules"""
        for response in responses:
            vni_id = response['vni_instance']
            confidence = response.get('confidence', 0.5)
        
            for connection_id, pathway in self.synaptic_connections.items():
                if pathway.source == vni_id:
                    # Hebbian learning: "Neurons that fire together, wire together"
                    if success and confidence > 0.7:
                        pathway.strength = min(1.0, pathway.strength + 0.1)
                    elif not success:
                        pathway.strength = max(0.1, pathway.strength - 0.05)
                
                    # Spike-timing dependent plasticity (simplified)
                    recent_activation = self._check_recent_activation(pathway.target)
                    if recent_activation:
                        pathway.strength = min(1.0, pathway.strength + 0.05)

    def _check_recent_activation(self, vni_id: str) -> bool:
        """Check if VNI was recently activated"""
        # Simple implementation - check last 5 conversation entries
        recent_messages = self.conversation_history[-5:] if self.conversation_history else []
        for msg in recent_messages:
            if msg.get('role') == 'assistant' and vni_id in msg.get('message', ''):
                return True
        return False
    async def learn_from_feedback(self, feedback_data: Dict):
        """Enhanced learning from user feedback"""
        try:
            message_id = feedback_data.get('message_id')
            feedback_type = feedback_data.get('feedback_type')
            correction = feedback_data.get('correction')
            session_id = feedback_data.get('session_id')
            
            # Find the original message
            original_message = await self.find_original_message(message_id)
            if original_message:
                query = original_message.get('message', '')
                
                # Update relevant VNI instances
                activated_vnis = original_message.get('activated_vnis', [])
                for vni_id in activated_vnis:
                    if vni_id in self.vni_instances:
                        learning_data = {
                            'feedback_type': feedback_type,
                            'query': query,
                            'correction': correction
                        }
                        self.vni_instances[vni_id].learn_from_feedback(learning_data)
                
                # Update analytics
                for vni_id in activated_vnis:
                    vni_type = vni_id.split('_')[0]
                    self.analytics.record_interaction(session_id, vni_type, 0.5, feedback_type)
                
                # Update synaptic connections
                self.update_synaptic_connections_based_on_feedback(activated_vnis, feedback_type)
                
                logger.info(f"📚 Learned from {feedback_type} feedback for {len(activated_vnis)} VNIs")
            
        except Exception as e:
            logger.error(f"❌ Learning from feedback failed: {e}")
    
    def update_synaptic_connections_based_on_feedback(self, vni_ids: List[str], feedback_type: str):
        """Update synaptic connections based on feedback"""
        for vni_id in vni_ids:
            for connection_id, pathway in self.synaptic_connections.items():
                if pathway.source == vni_id:
                    if feedback_type == 'positive':
                        pathway.activate(success=True)
                    elif feedback_type == 'negative':
                        pathway.activate(success=False)
    
    async def find_original_message(self, message_id: str) -> Optional[Dict]:
        """Find original message by ID"""
        for message in self.conversation_history:
            # Simple ID matching - in production, use proper message IDs
            if str(hash(message.get('message', '')))[:8] == message_id:
                return message
        return None
    
    def _add_to_history(self, role: str, message: str, session_id: str = "default_user"):
        """Compatibility method for predictive response system"""
        # Generate a simple session ID for compatibility
        # session_id = user_id
    
        # Create history entry directly
        history_entry = {
            "role": role,
            "message": message,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "message_id": str(hash(message))[:8]
        }
    
        #if not hasattr(self, 'conversation_history'):
        #    self.conversation_history = []
        
        self.conversation_history.append(history_entry)
    
        # Trim history if too long
        # max_length = getattr(self, 'max_history_length', 50)
        if len(self.conversation_history) > self.max_history_length: #max_length:
            self.conversation_history = self.conversation_history[-self.max_history_length: ]  #max_length:]

    def _format_response(self, response_data: Dict) -> str:
        """Format the final response for the user"""
        if isinstance(response_data, str):
            return response_data
        
        return response_data.get('response', 'I need more information to help with that.')
    
    def _build_context(self, user_message: str) -> Dict:
        """Build context for VNI processing"""
        return {
            "user_message": user_message,
            "conversation_history": self.conversation_history[-5:],  # Last 5 messages
            "timestamp": datetime.now().isoformat()
        }

    # ==================== TRANSFER LEARNING METHODS ====================
    
    def export_learning_patterns(self, filename: str = "babybionn_patterns.json"):
        """Export complete learning state for transfer"""
        import json
        import hashlib
        from datetime import datetime
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'source_instance': 'BabyBIONN_Demo_Unit',
            'patterns': {
                'synaptic_connections': {
                    conn_id: {
                        'source': pathway.source,
                        'target': pathway.target, 
                        'strength': pathway.strength,
                        'activation_count': pathway.activation_count,
                        'success_rate': pathway.get_success_rate() if hasattr(pathway, 'get_success_rate') else 0.5
                    }
                    for conn_id, pathway in self.synaptic_connections.items()
                },
                'vni_knowledge': {},
                'conversation_patterns': [
                    {
                        'user_message': msg.get('message', ''),
                        'role': msg.get('role', ''),
                        'session_id': msg.get('session_id', ''),
                        'timestamp': msg.get('timestamp', '')
                    }
                    for msg in self.conversation_history[-100:]  # Last 100 conversations
                ]
            }
        }
        
        # Export VNI knowledge for each instance
        for vni_id, vni_instance in self.vni_instances.items():
            if hasattr(vni_instance, 'knowledge_base'):
                export_data['patterns']['vni_knowledge'][vni_id] = {
                    'concepts': vni_instance.knowledge_base.get('concepts', {}),
                    'patterns': vni_instance.knowledge_base.get('patterns', {}),
                    'learning_history_count': len(getattr(vni_instance, 'learning_history', []))
                }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"💾 Learning patterns exported to {filename}")
        return export_data

    def import_learning_patterns(self, import_data: Dict):
        """Import learning patterns from another BabyBIONN instance"""
        try:
            patterns = import_data.get('patterns', {})
            
            # Import synaptic connections
            synaptic_data = patterns.get('synaptic_connections', {})
            for conn_id, conn_data in synaptic_data.items():
                source = conn_data['source']
                target = conn_data['target']
                strength = conn_data['strength']
                
                # Create or update connection
                if conn_id not in self.synaptic_connections:
                    self.synaptic_connections[conn_id] = NeuralPathway(source, target, strength)
                else:
                    # Average the strengths (knowledge fusion)
                    existing = self.synaptic_connections[conn_id]
                    existing.strength = (existing.strength + strength) / 2
            
            # Import VNI knowledge
            vni_knowledge = patterns.get('vni_knowledge', {})
            for vni_id, knowledge_data in vni_knowledge.items():
                if vni_id in self.vni_instances:
                    vni = self.vni_instances[vni_id]
                    self._fuse_vni_knowledge(vni, knowledge_data)
            
            # Import conversation patterns for routing intelligence
            conversation_patterns = patterns.get('conversation_patterns', [])
            for pattern in conversation_patterns:
                self._learn_from_conversation_pattern(pattern)
            
            logger.info(f"✅ Successfully imported patterns from {import_data.get('source_instance', 'unknown')}")
            logger.info(f"📊 Updated {len(synaptic_data)} synaptic connections")
            logger.info(f"🧠 Enhanced {len(vni_knowledge)} VNI knowledge bases")
            
        except Exception as e:
            logger.error(f"❌ Pattern import failed: {e}")
            raise

    def _fuse_vni_knowledge(self, vni, imported_knowledge: Dict):
        """Fuse imported knowledge with existing VNI knowledge"""
        try:
            # Fuse concepts
            imported_concepts = imported_knowledge.get('concepts', {})
            for concept, imported_data in imported_concepts.items():
                if concept in vni.knowledge_base.get('concepts', {}):
                    # Average the strengths
                    existing = vni.knowledge_base['concepts'][concept]
                    existing_strength = existing.get('strength', 0.5)
                    imported_strength = imported_data.get('strength', 0.5)
                    existing['strength'] = (existing_strength + imported_strength) / 2
                    # Sum usage counts
                    existing['usage_count'] = existing.get('usage_count', 0) + imported_data.get('usage_count', 0)
                else:
                    # Add new concept
                    vni.knowledge_base.setdefault('concepts', {})[concept] = imported_data
            
            # Fuse patterns
            imported_patterns = imported_knowledge.get('patterns', {})
            for pattern_id, imported_pattern in imported_patterns.items():
                if pattern_id in vni.knowledge_base.get('patterns', {}):
                    existing = vni.knowledge_base['patterns'][pattern_id]
                    existing_strength = existing.get('strength', 0.5)
                    imported_strength = imported_pattern.get('strength', 0.5)
                    existing['strength'] = (existing_strength + imported_strength) / 2
                    
                    # Merge responses (avoid duplicates)
                    existing_responses = existing.get('responses', [])
                    imported_responses = imported_pattern.get('responses', [])
                    combined_responses = list(set(existing_responses + imported_responses))
                    existing['responses'] = combined_responses
                else:
                    vni.knowledge_base.setdefault('patterns', {})[pattern_id] = imported_pattern
            
            # Save the updated knowledge base
            if hasattr(vni, 'save_knowledge_base'):
                vni.save_knowledge_base()
                
        except Exception as e:
            logger.error(f"❌ VNI knowledge fusion failed for {vni.instance_id}: {e}")

    def _learn_from_conversation_pattern(self, pattern: Dict):
        """Learn from imported conversation patterns"""
        # This helps with routing intelligence
        user_message = pattern.get('user_message', '')
        if user_message:
            # Simulate processing to build routing intelligence
            relevant_vnis = self.identify_relevant_vnis(user_message)
            for vni_id in relevant_vnis:
                if vni_id in self.vni_instances:
                    # Strengthen this VNI for similar queries
                    pass  # This would update routing weights in a more advanced system

# Global enhanced orchestrator instance
orchestrator = EnhancedBabyBIONNOrchestrator()

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

# Lifespan context manager first
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize enhanced orchestrator on startup"""
    await orchestrator.initialize()
    yield

# FastAPI Application with lifespan
app = FastAPI(
    title="BabyBIONN API - Enhanced",
    description="Enhanced BabyBIONN with Real Learning and Dynamic VNI Networking",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = "/app/static" # os.path.join(PROJECT_ROOT, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ==================== ENHANCED API ENDPOINTS ====================
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#    """Initialize enhanced orchestrator on startup"""
#    await orchestrator.initialize()
    
#    yield  # This is where the application runs


    # Add shutdown code here if needed
    # For example: await orchestrator.cleanup()

# Create your app with the lifespan
# app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_root():
    """Root endpoint with enhanced API information"""
    return {
        "message": "Enhanced BabyBIONN API Server",
        "status": "running", 
        "version": "2.0.0",
        "features": [
            "Dynamic VNI Instance Networking",
            "Real Reinforcement Learning", 
            "Synaptic Connection Visualization",
            "Progress Tracking Dashboard",
            "Direct Knowledge Correction"
        ],
        "endpoints": {
            "chat": "/api/chat (POST)",
            "feedback": "/api/feedback (POST)", 
            "analytics": "/api/analytics",
            "visualization": "/api/synaptic-visualization",
            "learning_report": "/api/learning-report",
            "websocket": "/ws",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    return {
        "status": "healthy",
        "initialized": orchestrator.initialized,
        "vni_instances": len(orchestrator.vni_instances),
        "synaptic_connections": len(orchestrator.synaptic_connections),
        "total_interactions": len(orchestrator.conversation_history)
    }

@app.get("/api/safety-report")
async def get_safety_report():
    """Get safety monitoring report"""
    try:
        report = orchestrator.safety_manager.get_safety_report()
        return {
            "status": "success",
            "safety_report": report
        }
    except Exception as e:
        logger.error(f"Safety report error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_endpoint(request: Dict[str, Any]):
    """Enhanced chat endpoint"""
    try:
        message = request.get("message", "")
        session_id = request.get("session_id", "default")
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        response = await orchestrator.process_message(message, session_id)
        return response
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat-with-image")
async def chat_with_image(
    message: str = Form(...),
    image: UploadFile = File(...),
    session_id: str = Form("default")
):
    """Process chat messages with image attachments"""
    try:
        # Validate image file
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image file
        image_data = await image.read()
        
        # Process image with YOLO (you'll need to implement this)
        image_analysis = await process_image_with_yolo(image_data)
        
        # Combine image analysis with text query
        combined_query = f"{message} [Image analysis: {image_analysis}]"
        
        # Route through VNI network
        context = {
            "user_message": message,
            "image_analysis": image_analysis,
            "has_image": True
        }
        
        response = await orchestrator.process_message(combined_query, session_id)
        response["image_analysis"] = image_analysis
        
        return response
        
    except Exception as e:
        logger.error(f"Image chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_image_with_yolo(image_data: bytes) -> str:
    """Process image using YOLO model and return analysis"""
    try:
        # Import YOLO (make sure it's installed)
        from ultralytics import YOLO
        import cv2
        import numpy as np
        from io import BytesIO
        from PIL import Image
        # LAZY LOAD YOLO MODEL
        print("🔄 Loading YOLO model on first use...")        
        # Load model (this will use the cached model from Docker build)
        model = YOLO('yolov8n.pt')
        
        # Convert bytes to image
        image = Image.open(BytesIO(image_data))
        image_np = np.array(image)
        
        # Run YOLO inference
        results = model(image_np)
        
        if not results or len(results) == 0:
            return {
                "detected_objects": [],
                "analysis": "No objects detected in the image",
                "object_count": 0,
                "primary_objects": []
            }
        
        # Extract detection information
        result = results[0]
        detections = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                
                # Only include confident detections
                if confidence > 0.5:
                    detections.append({
                        "object": class_name,
                        "confidence": round(confidence, 2),
                        "class_id": class_id
                    })

        # Sort by confidence and get top objects
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        top_objects = [det["object"] for det in detections[:5]]  # Top 5 objects
        
        # Generate natural language analysis
        if detections:
            object_counts = {}
            for det in detections:
                obj_name = det["object"]
                object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
            
            # Create descriptive analysis
            object_descriptions = []
            for obj, count in object_counts.items():
                if count == 1:
                    object_descriptions.append(f"a {obj}")
                else:
                    object_descriptions.append(f"{count} {obj}s")
            
            analysis = f"I detected {', '.join(object_descriptions)} in the image."
            
            # Add context about primary objects
            if top_objects:
                primary_objs = list(set(top_objects[:3]))  # Unique top objects
                if len(primary_objs) == 1:
                    analysis += f" The main object appears to be a {primary_objs[0]}."
                elif len(primary_objs) > 1:
                    analysis += f" Primary objects include {', '.join(primary_objs)}."
        
        else:
            analysis = "No significant objects detected in the image."
        
        return {
            "detected_objects": detections,
            "analysis": analysis,
            "object_count": len(detections),
            "primary_objects": top_objects,
            "total_detections": len(detections)
        }
        
    except ImportError as e:
        logger.error(f"YOLO import error: {e}")
        return {
            "detected_objects": [],
            "analysis": "YOLO model not available for image processing",
            "object_count": 0,
            "primary_objects": [],
            "error": "YOLO dependencies missing"
        }
    except Exception as e:
        logger.error(f"YOLO processing error: {e}")
        return {
            "detected_objects": [],
            "analysis": "Error processing image with YOLO",
            "object_count": 0,
            "primary_objects": [],
            "error": str(e)
        }
        
@app.post("/api/feedback") 
async def submit_feedback(feedback_data: Dict[str, Any]):
    """Enhanced feedback endpoint for learning"""
    try:
        # EXISTING: Basic feedback
        await orchestrator.learn_from_feedback(feedback_data)
        
        # ADD: Learning orchestrator feedback
        if (orchestrator.learning_orchestrator and 
            'session_id' in feedback_data and 
            'quality_score' in feedback_data):
            
            orchestrator.learning_orchestrator.provide_learning_feedback(
                feedback_data['session_id'],
                feedback_data['quality_score'],
                {
                    'user_feedback': feedback_data.get('feedback_text', ''),
                    'target_vnis': feedback_data.get('vni_instances', [])
                }
            )
            logger.info(f"📚 Learning feedback applied to session {feedback_data['session_id']}")
        
        return {"status": "success", "message": "Feedback processed for learning"}
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics")
async def get_analytics():
    """Get learning analytics"""
    try:
        metrics = orchestrator.analytics.get_vni_performance_metrics()
        return {
            "vni_performance": metrics,
            "total_sessions": len(orchestrator.analytics.data["sessions"]),
            "total_interactions": sum(len(session["interactions"]) for session in orchestrator.analytics.data["sessions"].values())
        }
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/learning-report")
async def get_learning_report():
    """Get comprehensive learning report"""
    try:
        report = orchestrator.analytics.generate_learning_report()
        return {"report": report}
    except Exception as e:
        logger.error(f"Learning report error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/synaptic-visualization")
async def get_synaptic_visualization():
    """Generate synaptic network visualization"""
    try:
        orchestrator.visualizer.update_connections(orchestrator.synaptic_connections)
        orchestrator.visualizer.create_static_visualization("static/synaptic_network.png")
        
        return {
            "status": "success", 
            "image_url": "/static/synaptic_network.png",
            "connection_count": len(orchestrator.synaptic_connections),
            "average_strength": sum(p.strength for p in orchestrator.synaptic_connections.values()) / len(orchestrator.synaptic_connections) if orchestrator.synaptic_connections else 0
        }
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export-patterns")
async def export_patterns():
    """Export learning patterns for transfer"""
    try:
        export_data = orchestrator.export_learning_patterns()
        return {
            "status": "success",
            "patterns_exported": len(export_data['patterns']['synaptic_connections']),
            "vnis_enhanced": len(export_data['patterns']['vni_knowledge']),
            "export_timestamp": export_data['export_timestamp']
        }
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/import-patterns")
async def import_patterns(import_request: Dict[str, Any]):
    """Import learning patterns from another instance"""
    try:
        orchestrator.import_learning_patterns(import_request)
        return {
            "status": "success", 
            "message": "Patterns imported successfully",
            "synaptic_connections": len(orchestrator.synaptic_connections),
            "vni_instances": len(orchestrator.vni_instances)
        }
    except Exception as e:
        logger.error(f"Import error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export-patterns-file")
async def export_patterns_file():
    """Export learning patterns as downloadable file"""
    try:
        filename = f"babybionn_patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        export_data = orchestrator.export_learning_patterns(filename)
        
        return FileResponse(
            filename, 
            media_type='application/json',
            filename=filename
        )
    except Exception as e:
        logger.error(f"Export file error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time communication"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            response = await orchestrator.process_message(
                message_data.get("message", ""),
                message_data.get("session_id", "websocket")
            )
            
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

@app.get("/chat", response_class=HTMLResponse)
async def serve_chat_interface():
    """Serve the enhanced chatbot interface"""
    chatbot_html_path = os.path.join("/app", "bionn_demo_chatbot.html")
    
    if os.path.exists(chatbot_html_path):
        return FileResponse(chatbot_html_path)
    else:
        return HTMLResponse(content="<h1>Enhanced BabyBIONN Chat Interface</h1><p>Interface file not found at: " + chatbot_html_path + "</p>")

# Configuration
class Config:
    HOST = "0.0.0.0"
    PORT = 8000
    RELOAD = True

if __name__ == "__main__":
    logger.info(f"🚀 Starting ENHANCED BabyBIONN Main Bridge on {Config.HOST}:{Config.PORT}")
    logger.info(f"📁 Project root: {PROJECT_ROOT}")
    logger.info(f"📁 Static files: {STATIC_DIR}")
    logger.info(f"🌐 Chat interface: http://{Config.HOST}:{Config.PORT}/chat")
    logger.info(f"📊 Analytics: http://{Config.HOST}:{Config.PORT}/api/analytics")
    logger.info(f"🔗 Visualization: http://{Config.HOST}:{Config.PORT}/api/synaptic-visualization")
    
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.RELOAD
    ) 
