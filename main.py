# main.py - FINAL CLEAN INTEGRATED VERSION
#!/usr/bin/env python3
"""
BabyBIONN Main Bridge Module - FINAL CLEAN INTEGRATED VERSION
"""
import os
#import cv2
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
from safety_check import SafetyManager
from model_loading import model_manager
from datetime import datetime, timedelta
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
from neuron.reinforcement_learning.reinforce_learn import RLConfig, VNIReinforcementEngine
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Form, UploadFile, File, Depends, status
from neuron.reinforcement_learning.vni_rl_integration import VNILearningOrchestrator, integrate_with_existing_vnis, VNIStimulus

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
            general_keywords = {
                'code', 'programming', 'technical', 'general', 'system', 'algorithm', 'software',
                'python', 'java', 'database', 'api', 'framework', 'development', 'debug'
            }

            medical_score = sum(1 for word in words if any(med_word in word or word in med_word 
                                                          for med_word in medical_keywords))
            legal_score = sum(1 for word in words if any(leg_word in word or word in leg_word 
                                                        for leg_word in legal_keywords))
            general_score = sum(1 for word in words if any(tech_word in word or word in tech_word 
                                                            for tech_word in general_keywords))

            total = medical_score + legal_score + general_score + 0.001  # Avoid division by zero

            return {
                'medical': medical_score / total,
                'legal': legal_score / total,
                'general': general_score / total
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
from enhanced_vni_classes import EnhancedMedicalVNI, EnhancedLegalVNI, EnhancedGeneralVNI, NeuralPathway

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BabyBIONN-Main")

# ==================== AUTONOMOUS VNI COMMUNICATION SYSTEM ====================
class VNIMessage:
    """Message format for autonomous VNI communication"""
    def __init__(self, 
                 sender: str,
                 receiver: str,
                 content: Any,
                 message_type: str = "query",
                 priority: int = 1,
                 context: Optional[Dict] = None,
                 requires_response: bool = True):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.message_type = message_type  # query, response, info, request_help
        self.priority = priority  # 1-5, 5 being highest priority
        self.context = context or {}
        self.requires_response = requires_response
        self.timestamp = datetime.now()
        self.message_id = f"{sender}_{receiver}_{self.timestamp.timestamp()}"
        
    def to_dict(self):
        return {
            'message_id': self.message_id,
            'sender': self.sender,
            'receiver': self.receiver,
            'content': self.content,
            'type': self.message_type,
            'priority': self.priority,
            'requires_response': self.requires_response,
            'timestamp': self.timestamp.isoformat()
        }

class AutonomyEngine:
    """Manages autonomous VNI interactions and spontaneous communications"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.message_queue = asyncio.Queue()
        self.active_dialogues = {}  # dialogue_id -> {participants, messages}
        self.autonomy_level = 0.5  # 0.0 to 1.0, controls how autonomous VNIs are
        self.min_connection_strength = 0.3  # Minimum strength for spontaneous connection
        self.idea_generation_interval = 30  # Seconds between idea generation cycles
        self.convergence_threshold = 0.7  # When to consider dialogue converged
        
    async def start_autonomy_engine(self):
        """Start background tasks for autonomous interactions"""
        asyncio.create_task(self._process_message_queue())
        asyncio.create_task(self._spontaneous_idea_generation())
        asyncio.create_task(self._connection_maintenance())
        
    async def _process_message_queue(self):
        """Process messages between VNIs"""
        while True:
            try:
                message = await self.message_queue.get()
                await self._route_message(message)
                self.message_queue.task_done()
            except Exception as e:
                logger.error(f"Message processing error: {e}")
            await asyncio.sleep(0.1)
            
    async def _spontaneous_idea_generation(self):
        """Generate spontaneous ideas and discussions between VNIs"""
        while True:
            await asyncio.sleep(self.idea_generation_interval)
            
            if self.autonomy_level > 0.3 and len(self.orchestrator.vni_instances) >= 2:
                # Randomly select 2-3 VNIs to have a spontaneous discussion
                vni_ids = list(self.orchestrator.vni_instances.keys())
                if len(vni_ids) >= 2:
                    import random
                    participants = random.sample(vni_ids, min(3, len(vni_ids)))
                    
                    # Generate a discussion topic based on recent conversation
                    recent_topics = self._extract_recent_topics()
                    if recent_topics:
                        topic = random.choice(recent_topics)
                        await self._initiate_spontaneous_discussion(participants, topic)
                        
    async def _connection_maintenance(self):
        """Maintain and optimize synaptic connections based on interaction patterns"""
        while True:
            await asyncio.sleep(60)  # Run every minute
            
            # Strengthen frequently used connections
            for connection_id, pathway in list(self.orchestrator.synaptic_connections.items()):
                if hasattr(pathway, 'activation_count') and pathway.activation_count > 5:
                    # Successful activation strengthens connection
                    pathway.strength = min(1.0, pathway.strength + 0.05)
                    
                    # If strength is high, consider creating shortcut connections
                    if pathway.strength > 0.8:
                        self._consider_shortcut_creation(pathway)
                        
            # Weaken unused connections
            for connection_id, pathway in list(self.orchestrator.synaptic_connections.items()):
                if hasattr(pathway, 'last_activated'):
                    time_diff = (datetime.now() - pathway.last_activated).total_seconds()
                    if time_diff > 3600:  # 1 hour
                        pathway.strength = max(0.1, pathway.strength - 0.01)
    
    def _extract_recent_topics(self):
        """Extract topics from recent conversations"""
        topics = []
        for msg in self.orchestrator.conversation_history[-10:]:
            message = msg.get('message', '').lower()
            if any(word in message for word in ['medical', 'health', 'doctor']):
                topics.append('medical_innovation')
            if any(word in message for word in ['legal', 'law', 'contract']):
                topics.append('legal_framework')
            if any(word in message for word in ['ai', 'learning', 'neural']):
                topics.append('ai_development')
            if any(word in message for word in ['system', 'network', 'architecture']):
                topics.append('system_architecture')
        return list(set(topics))
    
    async def _initiate_spontaneous_discussion(self, participants: List[str], topic: str):
        """Initiate a spontaneous discussion between VNIs"""
        dialogue_id = f"dialogue_{datetime.now().timestamp()}"
        
        self.active_dialogues[dialogue_id] = {
            'participants': participants,
            'topic': topic,
            'messages': [],
            'start_time': datetime.now(),
            'convergence_score': 0.0
        }
        
        # Send initial message to start discussion
        starter = participants[0]
        for receiver in participants[1:]:
            message = VNIMessage(
                sender=starter,
                receiver=receiver,
                content=f"Spontaneous thought on {topic}: What are your insights?",
                message_type="query",
                priority=2,
                context={'dialogue_id': dialogue_id, 'topic': topic},
                requires_response=True
            )
            await self.send_message(message)
            
        logger.info(f"💭 Spontaneous discussion started: {dialogue_id} on {topic}")
    
    async def send_message(self, message: VNIMessage):
        """Send a message between VNIs"""
        await self.message_queue.put(message)
        
    async def _route_message(self, message: VNIMessage):
        """Route message to appropriate VNI"""
        try:
            if message.receiver in self.orchestrator.vni_instances:
                vni_instance = self.orchestrator.vni_instances[message.receiver]
                
                # Process message based on type
                if message.message_type == "query":
                    response = await self._process_vni_query(vni_instance, message)
                    
                    # Send response back if required
                    if message.requires_response and response:
                        response_message = VNIMessage(
                            sender=message.receiver,
                            receiver=message.sender,
                            content=response,
                            message_type="response",
                            priority=message.priority,
                            context=message.context
                        )
                        await self.send_message(response_message)
                        
                elif message.message_type == "info":
                    # Informational message, no response needed
                    if hasattr(vni_instance, 'receive_information'):
                        vni_instance.receive_information(message.content, message.context)
                        
                # Update dialogue tracking
                if 'dialogue_id' in message.context:
                    dialogue_id = message.context['dialogue_id']
                    if dialogue_id in self.active_dialogues:
                        self.active_dialogues[dialogue_id]['messages'].append(message.to_dict())
                        await self._update_dialogue_convergence(dialogue_id)
                        
        except Exception as e:
            logger.error(f"Message routing error: {e}")
    
    async def _process_vni_query(self, vni_instance, message: VNIMessage) -> Optional[Any]:
        """Process a query between VNIs"""
        try:
            # Format query for VNI processing
            query_text = f"[From {message.sender}]: {message.content}"
            
            # Add context from message
            context = {
                'inter_vni_query': True,
                'sender': message.sender,
                'priority': message.priority,
                **message.context
            }
            
            # Process with VNI
            response = vni_instance.process_query(query_text, context)
            
            # Strengthen synaptic connection
            self._reinforce_connection(message.sender, message.receiver, success=True)
            
            return response.get('response') if isinstance(response, dict) else response
            
        except Exception as e:
            logger.error(f"VNI query processing error: {e}")
            return None
    
    def _reinforce_connection(self, source: str, target: str, success: bool):
        """Reinforce synaptic connection between VNIs"""
        connection_id = f"{source}→{target}"
        reverse_id = f"{target}→{source}"
        
        # Strengthen forward connection
        if connection_id in self.orchestrator.synaptic_connections:
            pathway = self.orchestrator.synaptic_connections[connection_id]
            if success:
                pathway.strength = min(1.0, pathway.strength + 0.1)
            pathway.last_activated = datetime.now()
            if hasattr(pathway, 'activation_count'):
                pathway.activation_count += 1
                
        # Also strengthen reverse connection (for bidirectional communication)
        if reverse_id in self.orchestrator.synaptic_connections:
            pathway = self.orchestrator.synaptic_connections[reverse_id]
            if success:
                pathway.strength = min(1.0, pathway.strength + 0.05)
    
    def _consider_shortcut_creation(self, pathway: NeuralPathway):
        """Consider creating shortcut connections between highly connected VNIs"""
        source_type = pathway.source.split('_')[0]
        target_type = pathway.target.split('_')[0]
        
        # Look for VNIs of same type that might benefit from direct connection
        if pathway.strength > 0.9:
            for other_vni_id in self.orchestrator.vni_instances:
                if other_vni_id != pathway.source and other_vni_id != pathway.target:
                    other_type = other_vni_id.split('_')[0]
                    
                    # Create new connections if beneficial
                    if other_type == source_type:
                        new_conn_id = f"{other_vni_id}→{pathway.target}"
                        if new_conn_id not in self.orchestrator.synaptic_connections:
                            self.orchestrator.synaptic_connections[new_conn_id] = NeuralPathway(
                                other_vni_id, pathway.target, pathway.strength * 0.7
                            )
    
    async def _update_dialogue_convergence(self, dialogue_id: str):
        """Check if dialogue has converged on consensus"""
        if dialogue_id not in self.active_dialogues:
            return
            
        dialogue = self.active_dialogues[dialogue_id]
        messages = dialogue['messages']
        
        if len(messages) >= 6:  # Minimum messages for convergence check
            # Simple convergence check: if last 3 messages are responses without new queries
            recent_types = [msg['type'] for msg in messages[-3:]]
            if recent_types.count('response') >= 2:
                dialogue['convergence_score'] = 0.8
                
                # Generate consensus summary
                summary = await self._generate_dialogue_summary(dialogue_id)
                
                # Store consensus in orchestrator's knowledge
                self.orchestrator.conversation_history.append({
                    'role': 'system_consensus',
                    'message': f"VNI Consensus on {dialogue['topic']}: {summary}",
                    'timestamp': datetime.now().isoformat(),
                    'metadata': {'dialogue_id': dialogue_id}
                })
                
                # Close dialogue
                del self.active_dialogues[dialogue_id]
                logger.info(f"✅ Dialogue {dialogue_id} converged with consensus")
    
    async def _generate_dialogue_summary(self, dialogue_id: str) -> str:
        """Generate summary of dialogue consensus"""
        dialogue = self.active_dialogues.get(dialogue_id)
        if not dialogue:
            return ""
            
        messages = [msg['content'] for msg in dialogue['messages'] if isinstance(msg['content'], str)]
        
        # Simple extraction of key points
        key_phrases = []
        for msg in messages[-5:]:  # Last 5 messages
            words = msg.lower().split()
            if len(words) > 3:
                # Extract potential key phrases (simple heuristic)
                if 'agree' in msg.lower():
                    key_phrases.append("Consensus reached")
                if 'suggest' in msg.lower() or 'recommend' in msg.lower():
                    # Extract the suggestion
                    import re
                    suggestions = re.findall(r'suggest[s]?\s+(.*?)[\.\?]', msg.lower())
                    key_phrases.extend(suggestions)
        
        if key_phrases:
            return " | ".join(set(key_phrases))[:200]
        return "Discussion completed without clear consensus"

class AutonomousVNIProtocol:
    """Protocol for autonomous VNI interactions and self-initiated communications"""
    
    def __init__(self, vni_instance, orchestrator):
        self.vni = vni_instance
        self.orchestrator = orchestrator
        self.curiosity_level = 0.5  # 0.0 to 1.0
        self.collaboration_tendency = 0.6  # 0.0 to 1.0
        self.initiative_threshold = 0.7  # Threshold for taking initiative
        self.knowledge_gaps = set()  # Topics this VNI wants to learn about
        self.interaction_history = []  # History of autonomous interactions
        
    def assess_knowledge_gap(self, topic: str, confidence: float) -> bool:
        """Determine if there's a knowledge gap that needs addressing"""
        if confidence < 0.4:  # Low confidence indicates knowledge gap
            self.knowledge_gaps.add(topic)
            return True
        return False
    
    async def seek_collaboration(self, topic: str, gap_type: str = "knowledge"):
        """Autonomously seek collaboration from other VNIs"""
        # Find VNIs with expertise in this area
        potential_collaborators = []
        for vni_id, other_vni in self.orchestrator.vni_instances.items():
            if vni_id != self.vni.instance_id:
                vni_type = getattr(other_vni, 'vni_type', '')
                
                # Check expertise match
                if gap_type == "medical" and 'medical' in vni_type:
                    potential_collaborators.append(vni_id)
                elif gap_type == "legal" and 'legal' in vni_type:
                    potential_collaborators.append(vni_id)
                elif gap_type == "technical" and 'general' in vni_type:
                    potential_collaborators.append(vni_id)
        
        if potential_collaborators and hasattr(self.orchestrator, 'autonomy_engine'):
            # Send collaboration request
            for collaborator in potential_collaborators[:2]:  # Max 2 collaborators
                message = VNIMessage(
                    sender=self.vni.instance_id,
                    receiver=collaborator,
                    content=f"Seeking collaboration on {topic}. Can you share your expertise?",
                    message_type="query",
                    priority=3,
                    context={'gap_type': gap_type, 'collaboration': True},
                    requires_response=True
                )
                await self.orchestrator.autonomy_engine.send_message(message)
                self.interaction_history.append({
                    'type': 'collaboration_request',
                    'to': collaborator,
                    'topic': topic,
                    'timestamp': datetime.now()
                })
    
    def generate_insight(self, recent_context: List[Dict]) -> Optional[str]:
        """Generate spontaneous insights based on recent context"""
        import random
        
        # Only generate insights if curiosity level is high enough
        if self.curiosity_level < 0.4:
            return None
            
        # Analyze recent conversations for patterns
        topics = self._extract_conversation_patterns(recent_context)
        if not topics:
            return None
            
        # Random chance to generate insight
        if random.random() < (self.curiosity_level * 0.3):
            topic = random.choice(topics)
            insight = self._formulate_insight(topic)
            return insight
            
        return None
    
    def _extract_conversation_patterns(self, recent_context: List[Dict]) -> List[str]:
        """Extract patterns and topics from recent conversations"""
        topics = set()
        for context in recent_context[-5:]:
            message = context.get('message', '').lower()
            
            # Extract topics using keyword matching
            if any(word in message for word in ['medical', 'health', 'treatment']):
                topics.add('medical_innovation')
            if any(word in message for word in ['legal', 'rights', 'contract']):
                topics.add('legal_ethics')
            if any(word in message for word in ['ai', 'learning', 'neural']):
                topics.add('ai_ethics')
            if any(word in message for word in ['system', 'network']):
                topics.add('system_optimization')
                
        return list(topics)
    
    def _formulate_insight(self, topic: str) -> str:
        """Formulate an insight based on topic and VNI expertise"""
        if 'medical' in self.vni.vni_type:
            insights = [
                f"From medical perspective on {topic}: Consider patient-centered approaches.",
                f"Medical insight: {topic} could benefit from evidence-based validation.",
                f"Healthcare angle: {topic} should prioritize safety and efficacy."
            ]
        elif 'legal' in self.vni.vni_type:
            insights = [
                f"Legal consideration for {topic}: Ensure regulatory compliance.",
                f"From legal standpoint: {topic} must address liability concerns.",
                f"Legal insight: {topic} requires clear contractual frameworks."
            ]
        else:
            insights = [
                f"Technical perspective on {topic}: Consider scalability factors.",
                f"System design insight: {topic} could be optimized for performance.",
                f"General insight: {topic} benefits from modular architecture."
            ]
        
        import random
        return random.choice(insights)

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
            'general_general': 1.2,
            'medical_general': 0.8,
            'general_medical': 0.8,
            'medical_legal': 0.6,
            'legal_medical': 0.6,
            'legal_general': 0.7,
            'general_legal': 0.7
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
        embedding[256:384] = domain_scores['general']  # Technical domain features
        
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
        general_keywords = {
            'code', 'programming', 'technical', 'general', 'system', 'algorithm', 'software',
            'python', 'java', 'database', 'api', 'framework', 'development', 'debug'
        }
        
        # Compute fuzzy matches
        medical_score = sum(1 for word in words if any(med_word in word or word in med_word 
                                                     for med_word in medical_keywords))
        legal_score = sum(1 for word in words if any(leg_word in word or word in leg_word 
                                                   for leg_word in legal_keywords))
        general_score = sum(1 for word in words if any(tech_word in word or word in tech_word 
                                                       for tech_word in general_keywords))
        
        total = medical_score + legal_score + general_score + 0.001  # Avoid division by zero
        
        return {
            'medical': medical_score / total,
            'legal': legal_score / total,
            'general': general_score / total
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
        connection_strengths = torch.zeros(3)  # medical, legal, general
        
        for pathway in self.orchestrator.synaptic_connections.values():
            source_type = pathway.source.split('_')[0]
            target_type = pathway.target.split('_')[0]
            
            if source_type == 'medical' or target_type == 'medical':
                connection_strengths[0] += pathway.strength
            if source_type == 'legal' or target_type == 'legal':
                connection_strengths[1] += pathway.strength
            if source_type == 'general' or target_type == 'general':
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
        elif vni_type == 'general':
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

class EnhancedSynapticAttentionBridge(SynapticAttentionBridge):
    """Enhanced version with support for autonomous interactions"""
    
    def __init__(self, orchestrator):
        super().__init__(orchestrator)
        self.autonomy_factor = 0.5  # How much autonomy influences attention
        
    def _calculate_autonomy_influence(self) -> float:
        """Calculate how much autonomy should influence the current processing"""
        if not hasattr(self.orchestrator, 'conversation_history'):
            return 0.3
            
        # More autonomy for complex, multi-turn conversations
        recent_messages = self.orchestrator.conversation_history[-5:]
        if len(recent_messages) >= 3:
            # Check if conversation shows complexity or uncertainty
            complexity_score = sum(1 for msg in recent_messages if '?' in msg.get('message', ''))
            uncertainty_score = sum(1 for msg in recent_messages 
                                  if any(word in msg.get('message', '').lower() 
                                       for word in ['not sure', 'uncertain', 'maybe', 'perhaps']))
            
            autonomy_level = min(1.0, 0.3 + (complexity_score * 0.1) + (uncertainty_score * 0.15))
            return autonomy_level
            
        return 0.3
    
    def _calculate_collaboration_potential(self, query: str) -> torch.Tensor:
        """Calculate potential for collaborative processing"""
        words = query.lower().split()
        
        # Phrases that suggest collaboration would help
        collaboration_indicators = [
            'complex', 'multifaceted', 'both', 'and', 'also', 'additionally',
            'multiple perspectives', 'different angles', 'various aspects'
        ]
        
        collaboration_score = 0.0
        for indicator in collaboration_indicators:
            if indicator in query.lower():
                collaboration_score += 0.1
                
        # Check for multi-domain content
        domains = self._compute_domain_scores(words)
        domain_count = sum(1 for score in domains.values() if score > 0.2)
        if domain_count >= 2:
            collaboration_score += 0.3
            
        return torch.tensor([collaboration_score] * 4)  # Repeat for 4 dimensions
    
    def _get_autonomy_collaboration_bonus(self, vni_id: str) -> float:
        """Calculate bonus for VNIs that frequently collaborate autonomously"""
        if not hasattr(self.orchestrator, 'synaptic_connections'):
            return 0.0
            
        collaboration_count = 0
        total_connections = 0
        
        for pathway in self.orchestrator.synaptic_connections.items():
            if pathway.source == vni_id or pathway.target == vni_id:
                total_connections += 1
                if hasattr(pathway, 'activation_count') and pathway.activation_count > 3:
                    collaboration_count += 1
        
        if total_connections > 0:
            return collaboration_count / total_connections
        return 0.0
    
    def _get_autonomous_learning_bonus(self, vni_id: str) -> float:
        """Calculate bonus for VNIs that engage in autonomous learning"""
        # Check for knowledge gaps and collaborative learning
        bonus = 0.0
        
        # Check if this VNI has been seeking collaboration
        if hasattr(self.orchestrator, 'autonomy_engine'):
            autonomy_engine = self.orchestrator.autonomy_engine
            if hasattr(autonomy_engine, 'active_dialogues'):
                for dialogue in autonomy_engine.active_dialogues.values():
                    if vni_id in dialogue.get('participants', []):
                        bonus += 0.2
        
        # Check synaptic plasticity (frequent updates)
        if hasattr(self.orchestrator, 'synaptic_connections'):
            recent_updates = 0
            for pathway in self.orchestrator.synaptic_connections.values():
                if pathway.source == vni_id or pathway.target == vni_id:
                    if hasattr(pathway, 'last_activated'):
                        time_diff = (datetime.now() - pathway.last_activated).total_seconds()
                        if time_diff < 300:  # Last 5 minutes
                            recent_updates += 1
            
            bonus += min(0.3, recent_updates * 0.1)
        
        return min(1.0, bonus)

class OrchestratorToVNIManagerAdapter:
    """Adapter to make EnhancedBabyBIONNOrchestrator work with VNIManager interface"""
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.vnis = {}  # Mock vnis dict for compatibility
        
    def process_stimulus(self, vni_id: str, stimulus) -> Any:
        """Process stimulus through orchestrator's VNI instances"""
        if vni_id not in self.orchestrator.vni_instances:
            raise ValueError(f"VNI {vni_id} not found")
        
        vni_instance = self.orchestrator.vni_instances[vni_id]
        
        # Extract query and context from stimulus
        query = getattr(stimulus, 'content', '')
        context_data = getattr(stimulus, 'metadata', {})
        context = context_data.get('context', 'general')
        
        # Process with the specific VNI
        response = vni_instance.process_query(query, context)
        
        # Convert to VNIResponse format expected by learning orchestrator
        if hasattr(stimulus, '_response_class'):
            response_obj = stimulus._response_class(
                content=response.get('response', '') if isinstance(response, dict) else str(response),
                confidence=response.get('confidence', 0.5) if isinstance(response, dict) else 0.5,
                response_type='general'
            )
            return response_obj
        else:
            # Fallback to raw response
            return response

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
        self.learning_orchestrator = None
        self.rl_engine = VNIReinforcementEngine(RLConfig()) 
        self.smart_router = SmartActivationRouter()        
        self.synaptic_bridge = EnhancedSynapticAttentionBridge(self)  # CHANGED to Enhanced version
        self.autonomy_engine = AutonomyEngine(self)  # NEW: Autonomous interaction engine
        self.synaptic_connections = {}
        self.conversation_history = []
        self.max_history_length = 100
        self.analytics = LearningAnalytics()
        self.visualizer = SynapticVisualizer()
        self.safety_manager = SafetyManager()
        self.generation_enabled = True  # Enable by default
        self.force_generation_threshold = 0.8  # Force generation for complex queries        
        self.autonomous_mode = True  # NEW: Enable autonomous interactions
        self.autonomy_level = 0.6  # NEW: Control autonomy intensity
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
            
            # Dynamic VNI spawning based on system complexity
            # Start with minimal VNIs, they'll be created dynamically per query
            self.spawn_vni_instances("medical", 1)
            self.spawn_vni_instances("legal", 1) 
            self.spawn_vni_instances("general", 1)

            # ============================================
            # 🧪 ENHANCED VERIFICATION TEST 🧪
            # ============================================
            print("\n" + "="*60)
            print("🧪 COMPREHENSIVE VNI VERIFICATION")
            print("="*60)

            # First, check all VNIs
            logger.info("🟢 Checking all VNI initialization...")
            for vni_id, vni in self.vni_instances.items():
                logger.info(f"  {vni_id}: type={getattr(vni, 'vni_type', 'N/A')}")
                logger.info(f"     has classifier: {hasattr(vni, 'classifier') and vni.classifier is not None}")
                logger.info(f"     has should_handle: {hasattr(vni, 'should_handle')}")
                logger.info(f"     generation_enabled: {getattr(vni, 'generation_enabled', False)}")

            # Focus on medical VNI
            medical_vni = self.vni_instances.get('medical_0')
            
            if medical_vni:
                print(f"\n✅ Found medical_0 VNI")
                print(f"   Type: {getattr(medical_vni, 'vni_type', 'N/A')}")
                print(f"   Has classifier: {hasattr(medical_vni, 'classifier') and medical_vni.classifier is not None}")
                print(f"   Has should_handle: {hasattr(medical_vni, 'should_handle')}")
                
                # Test 1: Check if classifier has the right keywords
                if hasattr(medical_vni, 'classifier') and medical_vni.classifier is not None:
                    print(f"\n🔍 Classifier details:")
                    print(f"   Keywords count: {len(getattr(medical_vni.classifier, 'keywords', []))}")
                    
                    # Check for specific keywords in the classifier
                    test_keywords = ['medical', 'health', 'advise', 'help on medical']
                    classifier_keywords = getattr(medical_vni.classifier, 'keywords', [])
                    for keyword in test_keywords:
                        has_keyword = any(kw for kw in classifier_keywords if keyword in kw)
                        print(f"   Has '{keyword}' keyword: {has_keyword}")
                
                # Test 2: The exact problematic phrase
                test_phrase = "I need help on medical and health advise"
                print(f"\n🧪 Testing exact phrase: '{test_phrase}'")
                
                try:
                    # Check should_handle method
                    if hasattr(medical_vni, 'should_handle'):
                        result = medical_vni.should_handle(test_phrase)
                        print(f"   should_handle() result: {result}")
                        
                        if result:
                            print("   ✅ SUCCESS! Medical VNI will handle this phrase!")
                        else:
                            print("   ❌ FAIL! Medical VNI won't handle this phrase")
                            
                            # Debug: check classifier prediction directly
                            if hasattr(medical_vni, 'classifier') and medical_vni.classifier:
                                try:
                                    prediction = medical_vni.classifier.predict([test_phrase])
                                    print(f"   Classifier direct prediction: {prediction}")
                                except Exception as e:
                                    print(f"   Classifier prediction error: {e}")
                    else:
                        print("   ❌ No should_handle method found!")
                        
                except Exception as e:
                    print(f"   ❌ ERROR testing should_handle: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("❌ medical_0 not found in vni_instances!")
                print(f"Available VNIs: {list(self.vni_instances.keys())}")
            
            # Test 3: Additional test cases
            if medical_vni and hasattr(medical_vni, 'should_handle'):
                print("\n🧪 Additional test cases:")
                test_cases = [
                    ("What medicine should I take for headache?", True),
                    ("I have fever and cough", True),
                    ("Hello, how are you?", False),
                    ("Tell me about legal contracts", False),
                    ("Medical advice needed", True),
                    ("Health consultation", True)
                ]
                
                passed = 0
                total = len(test_cases)
                
                for query, expected in test_cases:
                    try:
                        result = medical_vni.should_handle(query)
                        status = "✅" if result == expected else "❌"
                        if result == expected:
                            passed += 1
                        print(f"  {status} '{query[:40]}...' -> {result} (expected: {expected})")
                    except Exception as e:
                        print(f"  ❌ Error: '{query[:40]}...' -> {e}")
                
                print(f"\n📊 Test Summary: {passed}/{total} passed")
            
            print("\n" + "="*60)
            # ============================================
            # 🧪 END ENHANCED VERIFICATION 🧪
            # ============================================

            # Initialize synaptic connections
            self.initialize_synaptic_connections()

            # Initialize learning orchestrator
            try:
                from neuron.reinforcement_learning.vni_rl_integration import integrate_with_existing_vnis

                vni_manager_adapter = OrchestratorToVNIManagerAdapter(self)
                self.learning_orchestrator = integrate_with_existing_vnis(vni_manager_adapter)
                logger.info("✅ Learning orchestrator integrated")

            except Exception as e:
                logger.warning(f"Learning orchestrator initialization failed: {e}");
                self.learning_orchestrator = None

            # ADD THIS: Check generation capabilities
            generation_status = self.get_generation_status()
            active_generators = sum(1 for vni in generation_status.values() if vni['generation_enabled'])
            logger.info(f"🔤 Text Generation: {active_generators}/{len(generation_status)} VNIs enabled")

            # Test generation if enabled
            if self.generation_enabled and active_generators > 0:
                test_results = self.test_generation_capabilities()
                successful = sum(1 for r in test_results.values() if r.get('success', False))
                logger.info(f"🧪 Generation Test: {successful}/{len(test_results)} VNIs passed")

            self.initialized = True
            logger.info("✅ Enhanced BabyBIONN Orchestrator initialized successfully!")
            logger.info(f"📊 Spawned {len(self.vni_instances)} VNI instances")
            logger.info(f"🔗 Created {len(self.synaptic_connections)} synaptic connections")
            logger.info(f"🔤 Text generation: {'ENABLED' if self.generation_enabled else 'DISABLED'}")

        except Exception as e:
            logger.error(f"❌ Enhanced initialization failed: {e}")
            raise

    def process_stimulus(self, vni_id: str, stimulus: VNIStimulus) -> Any:
        """Process stimulus for individual VNI - required by learning orchestrator"""
        try:
            if vni_id not in self.vni_instances:
                logger.warning(f"VNI {vni_id} not found for stimulus processing")
                return None
            
            vni_instance = self.vni_instances[vni_id]
            
            # Extract query and context from stimulus
            query = getattr(stimulus, 'content', '')
            context_data = getattr(stimulus, 'metadata', {})
            context = context_data.get('context', 'general')
            
            # Process with the specific VNI
            response = vni_instance.process_query(query, context)
            
            # Convert to the expected format for learning orchestrator
            if hasattr(stimulus, '_response_class'):
                # Create a response object compatible with learning system
                response_obj = stimulus._response_class(
                    content=response.get('response', '') if isinstance(response, dict) else str(response),
                    confidence=response.get('confidence', 0.5) if isinstance(response, dict) else 0.5,
                    response_type='general'
                )
                return response_obj
            else:
                # Fallback: return raw response
                return response
                
        except Exception as e:
            logger.error(f"Stimulus processing failed for {vni_id}: {e}")
            return None

    def spawn_vni_instances(self, vni_type: str, count: int):
        """Spawn multiple VNI instances of a type"""
        for i in range(count):
            instance_id = f"{vni_type}_{i}"

            try:
                logger.info(f"🟡 DEBUG: Starting creation of {instance_id}...")

                if vni_type == "medical":
                    instance = EnhancedMedicalVNI(instance_id)
                elif vni_type == "legal":
                    instance = EnhancedLegalVNI(instance_id)
                elif vni_type == "general":
                    instance = EnhancedGeneralVNI(instance_id)
                else:
                    continue
                # Add detailed debugging
                logger.info(f"📊 DEBUG: Checking {instance_id} attributes:")
                logger.info(f"    generation_enabled: {getattr(instance, 'generation_enabled', 'ATTRIBUTE NOT FOUND')}")
                logger.info(f"    vni_type: {getattr(instance, 'vni_type', 'ATTRIBUTE NOT FOUND')}")
                logger.info(f"    instance_id: {getattr(instance, 'instance_id', 'ATTRIBUTE NOT FOUND')}")
                logger.info(f"    has classifier: {hasattr(instance, 'classifier') and instance.classifier is not None}")
                logger.info(f"    has generator: {hasattr(instance, 'generator') and instance.generator is not None}")
                logger.info(f"    has tokenizer: {hasattr(instance, 'tokenizer') and instance.tokenizer is not None}")
                logger.info(f"    has bridge_layer: {hasattr(instance, 'bridge_layer') and instance.bridge_layer is not None}")
                logger.info(f"    is_initialized: {getattr(instance, 'is_initialized', 'ATTRIBUTE NOT FOUND')}")
            
                # Check for should_handle method
                if hasattr(instance, 'should_handle'):
                    test_phrase = "I need help on medical and health advise"
                    try:
                        result = instance.should_handle(test_phrase)
                        logger.info(f"    should_handle('{test_phrase}'): {result}")
                    except Exception as e:
                        logger.error(f"    should_handle test failed: {e}")
            
                self.vni_instances[instance_id] = instance
                logger.info(f"  ➕ Successfully spawned {instance_id}")
            
            except Exception as e:
                logger.error(f"  ❌ Failed to spawn {instance_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())

    def dynamic_spawn_for_query(self, query: str):
        """Dynamically spawn VNIs based on query complexity and domain relevance"""
        complexity_score = self.analyze_query_complexity(query)
        relevant_domains = self.identify_relevant_domains(query)
    
        logger.info(f"🔍 IMPROVED ROUTING - Query complexity: {complexity_score}/10, Relevant domains: {relevant_domains}")
    
        # Enhanced target counts based on complexity and urgency
        if complexity_score >= 8:  # High complexity/emergency
            target_counts = {"medical": 3, "legal": 2, "general": 2}
        elif complexity_score >= 5:  # Medium complexity
            target_counts = {"medical": 2, "legal": 1, "general": 2}
        else:  # Low complexity
            target_counts = {"medical": 1, "legal": 1, "general": 1}
    
        # Boost relevant domains more aggressively
        for domain in relevant_domains:
            if domain in target_counts:
                if complexity_score >= 8:  # Emergency/urgent cases
                    target_counts[domain] = min(target_counts[domain] + 2, 4)  # Allow up to 4 for emergency
                else:
                    target_counts[domain] = min(target_counts[domain] + 1, 3)  # Max 3 per domain otherwise
    
        # Spawn VNIs to reach target counts
        for domain, target_count in target_counts.items():
            current_count = self.count_vnis_by_domain(domain)
            needed = target_count - current_count
        
            if needed > 0:
                logger.info(f"🚀 IMPROVED SPAWNING - Spawning {needed} additional {domain} VNI(s)")
                self.spawn_vni_instances(domain, needed)

    def enhanced_smart_route_query(self, input_text: str) -> List[str]:
        """Routing that checks VNI should_handle, allowing multiple VNIs"""
        logger.info(f"🔍 ROUTING CHECK for: '{input_text[:50]}...'")
        
        activated_vnis = []
        
        # Ask each VNI if they can handle this
        for vni_id, vni in self.vni_instances.items():
            if hasattr(vni, 'should_handle'):
                try:
                    if vni.should_handle(input_text):
                        logger.info(f"   ✅ {vni_id} says: I can handle this!")
                        activated_vnis.append(vni_id)
                    else:
                        logger.info(f"   ❌ {vni_id} says: Not for me")
                except Exception as e:
                    logger.error(f"   ⚠️  {vni_id} error: {e}")
        
        # If any VNIs claim it, return them (could be multiple)
        if activated_vnis:
            return activated_vnis
        
        # FALLBACK: If no VNI claims it, use general VNI
        logger.warning(f"   ⚠️  No VNI claimed it, using general_0")
        return ['general_0']

    def _analyze_conversation_context(self) -> Dict[str, Any]:
        """Analyze recent conversation for contextual clues"""
        if not self.conversation_history:
            return {"topics": [], "domains": [], "urgency": False}
    
        recent_messages = self.conversation_history[-5:]  # Last 5 exchanges
        topics = []
        domains = []
        urgency_indicators = False
    
        for msg in recent_messages:
            message_text = msg.get('message', '').lower()
        
            # Extract topics
            if any(word in message_text for word in ['pain', 'symptom', 'emergency']):
                topics.append('medical_emergency')
                urgency_indicators = True
            if any(word in message_text for word in ['legal', 'law', 'contract']):
                topics.append('legal')
            if any(word in message_text for word in ['code', 'technical', 'programming']):
                topics.append('technical')
    
        return {
            "topics": list(set(topics)),
            "domains": list(set(domains)),
            "urgency": urgency_indicators,
            "message_count": len(recent_messages)
        }

    def _compute_contextual_domain_scores(self, query: str, context: Dict) -> Dict[str, float]:
        """Compute domain scores with conversation context"""
        base_scores = {
            "medical": 0.0,
            "legal": 0.0, 
            "general": 0.1,  # General always has some baseline
            "technical": 0.0
        }
    
        # Keyword-based scoring
        medical_keywords = {
            'health': 2.0, 'medical': 2.5, 'doctor': 2.0, 'hospital': 2.0, 'disease': 2.5,
            'medicine': 2.0, 'pain': 3.0, 'symptom': 3.0, 'treatment': 2.0, 'patient': 1.5,
            'blood': 3.0, 'urinating': 3.0, 'emergency': 3.0, 'urgent': 2.5, 'fever': 2.5
        }
    
        legal_keywords = {
            'legal': 2.5, 'law': 2.0, 'lawyer': 2.0, 'court': 2.0, 'contract': 2.5,
            'agreement': 2.0, 'case': 1.5, 'rights': 2.0, 'sue': 2.0, 'lawsuit': 2.0
        }
    
        technical_keywords = {
            'code': 2.0, 'programming': 2.5, 'technical': 2.0, 'system': 1.5, 'software': 2.0,
            'algorithm': 2.0, 'data': 1.5, 'debug': 2.0, 'python': 2.0, 'java': 1.5
        }
    
        # Score based on keywords
        for keyword, weight in medical_keywords.items():
            if keyword in query:
                base_scores["medical"] += weight
    
        for keyword, weight in legal_keywords.items():
            if keyword in query:
                base_scores["legal"] += weight
            
        for keyword, weight in technical_keywords.items():
            if keyword in query:
                base_scores["technical"] += weight
    
        # Contextual boosting from conversation history
        if "medical_emergency" in context["topics"]:
            base_scores["medical"] += 2.0
        
        if context["urgency"]:
            base_scores["medical"] += 1.5
            base_scores["general"] += 0.5  # General can help with urgent queries too
    
        # Normalize scores
        total = sum(base_scores.values()) + 0.001
        return {domain: score/total for domain, score in base_scores.items()}

    def _get_contextual_boost(self, vni_id: str, context: Dict) -> float:
        """Get boost based on conversation context"""
        vni_type = vni_id.split('_')[0]
        boost = 0.0
    
        # Boost if VNI type matches recent conversation topics
        if vni_type in context["domains"]:
            boost += 0.3
    
        # Boost for medical VNIs in urgent contexts
        if context["urgency"] and vni_type == "medical":
            boost += 0.4
    
        # Boost for general VNI in multi-topic conversations
        if context["message_count"] > 3 and vni_type == "general":
            boost += 0.2
    
        return boost

    def _get_expertise_matching_boost(self, vni_id: str, query: str) -> float:
        """Get boost based on VNI expertise matching query complexity"""
        vni_type = vni_id.split('_')[0]
    
        # Medical VNIs get boost for complex medical queries
        if vni_type == "medical" and any(word in query for word in ['diagnosis', 'treatment', 'symptom']):
            return 0.3
    
        # Legal VNIs get boost for complex legal terms
        if vni_type == "legal" and any(word in query for word in ['contract', 'agreement', 'liability']):
            return 0.3
    
        # Technical VNIs get boost for complex technical queries
        if vni_type == "technical" and any(word in query for word in ['algorithm', 'debug', 'optimize']):
            return 0.3
    
        return 0.0

    def _get_temporal_relevance(self, vni_id: str) -> float:
        """Get boost based on recent activation patterns"""
        # Check if this VNI was recently used successfully
        recent_messages = self.conversation_history[-3:] if self.conversation_history else []
    
        for msg in recent_messages:
            if msg.get('role') == 'assistant' and vni_id in str(msg.get('metadata', {})):
                return 0.2  # Boost for recently used VNIs
    
        return 0.0

    def _calculate_dynamic_threshold(self, complexity: int, domain_scores: Dict) -> float:
        """Calculate dynamic threshold based on query complexity and domain distribution"""
        base_threshold = 0.5
    
        # Lower threshold for complex queries to include more VNIs
        if complexity >= 8:
            base_threshold = 0.3
        elif complexity >= 5:
            base_threshold = 0.4
    
        # Adjust based on domain concentration
        max_domain_score = max(domain_scores.values())
        if max_domain_score > 0.7:  # One domain dominates
            base_threshold += 0.1  # Higher threshold to be more selective
    
        return base_threshold

    def _ensure_domain_diversity(self, selected_vnis: List[str], domain_scores: Dict) -> List[str]:
        """Ensure selected VNIs cover diverse domains when appropriate"""
        if len(selected_vnis) >= 2:
            return selected_vnis  # Already diverse enough
    
        # If only one VNI selected, consider adding general for complex queries
        if len(selected_vnis) == 1 and max(domain_scores.values()) < 0.6:
            # No clear domain dominance, add general VNI
            general_vnis = [vni for vni in self.vni_instances if vni.startswith('general')]
            if general_vnis and general_vnis[0] not in selected_vnis:
                selected_vnis.append(general_vnis[0])
    
        return selected_vnis

#=== IMPROVED ROUTING INTELLIGENCE ===

# 1. IMPROVED COMPLEXITY ANALYSIS:
    def analyze_query_complexity(self, query: str) -> int:
        """Analyze query complexity with better heuristics"""
        query_lower = query.lower().strip()
        word_count = len(query_lower.split())
        char_count = len(query_lower)

        complexity_score = 0

        # Word count factor (realistic thresholds)
        if word_count > 20:
            complexity_score += 3
        elif word_count > 10:
            complexity_score += 2
        elif word_count > 5:
            complexity_score += 1

        # Question complexity
        if '?' in query:
            complexity_score += 1
        if any(word in query_lower for word in ['how', 'why', 'what', 'when', 'where']):
            complexity_score += 1

        # Urgency/emergency detection
        if any(word in query_lower for word in ['urgent', 'emergency', 'help', 'immediately', 'now']):
            complexity_score += 2

        # Medical emergency detection
        if any(word in query_lower for word in ['blood', 'urinating', 'pain', 'symptom', 'fever', 'headache']):
            complexity_score += 2

        # Multiple domains detection
        domains_detected = 0
        medical_words = ['medical', 'health', 'doctor', 'hospital', 'disease', 'medicine', 'patient', 'treatment',
                        'blood', 'urinating', 'symptom', 'pain', 'fever', 'headache', 'emergency']
        legal_words = ['legal', 'law', 'lawyer', 'court', 'contract', 'agreement', 'case', 'rights']
        tech_words = ['technical', 'code', 'programming', 'system', 'software', 'algorithm']

        if any(word in query_lower for word in medical_words):
            domains_detected += 1
        if any(word in query_lower for word in legal_words):
            domains_detected += 1
        if any(word in query_lower for word in tech_words):
            domains_detected += 1

        if domains_detected >= 2:
            complexity_score += 2
        elif domains_detected == 1:
            complexity_score += 1

        return min(10, max(1, complexity_score))  # Ensure 1-10 range

# 2. IMPROVED DOMAIN DETECTION:
    def identify_relevant_domains(self, query: str) -> List[str]:
        """Intelligent domain detection with comprehensive keywords"""
        query_lower = query.lower()
        relevant_domains = []

        # Comprehensive medical keywords including emergencies
        medical_keywords = [
            'medical', 'health', 'doctor', 'hospital', 'disease', 'medicine', 'patient', 'treatment',
            'symptom', 'pain', 'fever', 'headache', 'covid', 'vaccine', 'diagnosis',
            'blood', 'urinating', 'emergency', 'urgent', 'help', 'immediately'
        ]

        legal_keywords = ['legal', 'law', 'lawyer', 'court', 'contract', 'agreement', 'case', 'rights']
        technical_keywords = ['technical', 'code', 'programming', 'system', 'software', 'algorithm', 'data']

        # Enhanced detection with partial matching
        for keyword in medical_keywords:
            if keyword in query_lower:
                relevant_domains.append("medical")
                break  # Only need one match

        for keyword in legal_keywords:
            if keyword in query_lower:
                relevant_domains.append("legal")
                break

        for keyword in technical_keywords:
            if keyword in query_lower:
                relevant_domains.append("technical")
                break

        # If no specific domain detected, use general
        if not relevant_domains:
            relevant_domains.append("general")

        return relevant_domains

    def count_vnis_by_domain(self, domain: str) -> int:
        """Count how many VNIs currently exist for a domain"""
        return len([vni_id for vni_id in self.vni_instances.keys() if vni_id.startswith(domain)])
    
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
                    elif (source_type, target_type) in [('medical', 'general'), ('general', 'medical')]:
                        strength = 0.6  # Medium strength
                    else:
                        strength = 0.4  # Weaker connections
                    
                    connection_id = f"{source_id}→{target_id}"
                    self.synaptic_connections[connection_id] = NeuralPathway(
                        source_id, target_id, strength
                    )

    async def route_through_vni_network(self, input_text: str, context: str = "general") -> List[Dict]:
        """Enhanced routing with learning integration and fallback strategies"""
        logger.info(f"🧠 ENHANCED ROUTING - Processing: {input_text}")
    
        try:
            # STRATEGY 1: Try learning orchestrator first (if available)
            if self.learning_orchestrator:
                try:
                    from neuron.reinforcement_learning.vni_rl_integration import VNIStimulus
                
                    stimulus = VNIStimulus(
                        content=input_text,
                        stimulus_type="chat_query", 
                        metadata={
                            "context": context,
                            "conversation_history": self.conversation_history[-3:]  # Add recent context
                        }
                    )
                
                    learning_response = self.learning_orchestrator.process_stimulus_with_learning(stimulus)
                
                    # Convert and return learning response
                    vni_responses = []
                    for vni_id, vni_response in learning_response['vni_responses'].items():
                        vni_responses.append({
                            'response': vni_response.content,
                            'confidence': vni_response.confidence,
                            'vni_instance': vni_id,
                            'response_type': getattr(vni_response, 'response_type', 'general'),
                            'source': 'learning_orchestrator'
                        })
                
                    logger.info(f"🎯 LEARNING-BASED ROUTING activated: {len(vni_responses)} responses")
                    return vni_responses
                
                except Exception as e:
                    logger.warning(f"Learning orchestrator failed: {e}, falling back to enhanced routing")

            # STRATEGY 2: Enhanced context-aware routing
            logger.info("🔄 Using ENHANCED CONTEXT-AWARE routing")
            activated_vnis = self.enhanced_smart_route_query(input_text)
            vni_responses = []

            for vni_id in activated_vnis:
                if vni_id in self.vni_instances:
                    vni_instance = self.vni_instances[vni_id]
                    try:
                        # Enhanced context with conversation history
                        enhanced_context = {
                            "user_message": input_text,
                            "conversation_history": self.conversation_history[-3:],
                            "activated_vnis": activated_vnis,
                            "timestamp": datetime.now().isoformat()
                        }
                    
                        response = vni_instance.process_query(input_text, enhanced_context)
                    
                        # Enhanced response formatting
                        formatted_response = self._format_vni_response(response, vni_id, input_text)
                        vni_responses.append(formatted_response)
                    
                    except Exception as e:
                        logger.error(f"VNI {vni_id} processing failed: {e}")
                        vni_responses.append(self._create_error_response(vni_id, e))
        
            # STRATEGY 3: Ultimate fallback with intelligent selection
            if not vni_responses:
                logger.warning("⚠️ No VNIs could process query, using intelligent fallback")
                vni_responses = [self._create_fallback_response(input_text)]

            logger.info(f"✅ ENHANCED ROUTING complete: {len(vni_responses)} responses from {activated_vnis}")
            return vni_responses
        
        except Exception as e:
            logger.error(f"❌ Routing error: {e}")
            return [self._create_error_response("system", e)]

    def _format_vni_response(self, response: Any, vni_id: str, query: str) -> Dict:
        """Format VNI response with enhanced metadata"""
        if isinstance(response, dict) and 'response' in response:
            # Add routing metadata
            response['routing_metadata'] = {
                'vni_id': vni_id,
                'domain': vni_id.split('_')[0],
                'timestamp': datetime.now().isoformat(),
                'query_complexity': self.analyze_query_complexity(query),
                'source': 'enhanced_routing'
            }
            return response
        else:
            # Wrap non-dict responses
            return {
                'response': str(response),
                'confidence': 0.7,
                'vni_instance': vni_id,
                'success': True,
                'domain': vni_id.split('_')[0],
                'routing_metadata': {
                    'vni_id': vni_id,
                    'domain': vni_id.split('_')[0],
                    'timestamp': datetime.now().isoformat(),
                    'source': 'enhanced_routing_wrapped'
                }
            }

    def _create_error_response(self, vni_id: str, error: Exception) -> Dict:
        """Create standardized error response"""
        domain = vni_id.split('_')[0] if '_' in vni_id else 'general'
        return {
            'response': f"I'm experiencing technical difficulties with {domain} processing. Please try again.",
            'confidence': 0.1,
            'vni_instance': vni_id,
            'success': False,
            'error': str(error),
            'routing_metadata': {
                'vni_id': vni_id,
                'error': True,
                'timestamp': datetime.now().isoformat()
            }
        }

    def _create_fallback_response(self, query: str) -> Dict:
        """Create intelligent fallback response"""
        complexity = self.analyze_query_complexity(query)
    
        if complexity >= 7:
            response_text = "This appears to be a complex question. I'm still developing my capabilities in this area. Could you break it down into simpler questions?"
        else:
            response_text = "I'm still learning how to respond to that. Could you try rephraring your question or providing more context?"
    
        return {
            'response': response_text,
            'confidence': 0.3,
            'vni_instance': 'fallback',
            'success': False,
            'routing_metadata': {
                'vni_id': 'fallback',
                'fallback_type': 'intelligent',
                'query_complexity': complexity,
                'timestamp': datetime.now().isoformat()
            }
        }
    def update_routing_based_on_performance(self, vni_responses: List[Dict], user_feedback: Optional[Dict] = None):
        """Update routing strategies based on response performance and user feedback"""
        successful_vnis = [r['vni_instance'] for r in vni_responses if r.get('confidence', 0) > 0.7]
    
        # Strengthen pathways for successful VNIs
        for vni_id in successful_vnis:
            self._reinforce_successful_vni(vni_id)
    
        # Learn from user feedback
        if user_feedback:
            self._incorporate_user_feedback(user_feedback, vni_responses)
    
        # Adjust routing thresholds based on recent performance
        self._adapt_routing_thresholds(vni_responses)

    def _reinforce_successful_vni(self, vni_id: str):
        """Reinforce VNI activation patterns for successful responses"""
        # Strengthen synaptic connections to this VNI
        for conn_id, pathway in self.synaptic_connections.items():
            if pathway.target == vni_id:
                pathway.strength = min(1.0, pathway.strength + 0.05)
    
        # Update VNI activation history for temporal relevance
        if hasattr(self, 'vni_activation_history'):
            self.vni_activation_history[vni_id] = self.vni_activation_history.get(vni_id, 0) + 1

    def _incorporate_user_feedback(self, feedback: Dict, responses: List[Dict]):
        """Incorporate explicit user feedback into routing intelligence"""
        feedback_type = feedback.get('feedback_type')
        target_vnis = [r['vni_instance'] for r in responses]
    
        if feedback_type == 'positive':
            for vni_id in target_vnis:
                self._reinforce_successful_vni(vni_id)
        elif feedback_type == 'negative':
            # Weaken connections for poor performance
            for vni_id in target_vnis:
                for conn_id, pathway in self.synaptic_connections.items():
                    if pathway.target == vni_id:
                        pathway.strength = max(0.1, pathway.strength - 0.03)

    def _adapt_routing_thresholds(self, responses: List[Dict]):
        """Adapt routing thresholds based on recent performance"""
        avg_confidence = sum(r.get('confidence', 0) for r in responses) / len(responses) if responses else 0.5
    
        # Adjust future thresholds based on performance
        if avg_confidence > 0.8:
            # High confidence - can be more selective
            self.smart_router.activation_threshold = min(0.4, self.smart_router.activation_threshold + 0.05)
        elif avg_confidence < 0.4:
            # Low confidence - be more inclusive
            self.smart_router.activation_threshold = max(0.2, self.smart_router.activation_threshold - 0.05)

    async def process_message(self, user_message: str, session_id: str = "default_user") -> Dict[str, Any]:
        """Enhanced message processing with dynamic VNI networking"""
        if not self.initialized:
            await self.initialize()
        
        try:
            logger.info(f"🧠 Processing: {user_message}")

            # DYNAMIC VNI SPAWNING BASED ON QUERY COMPLEXITY
            self.dynamic_spawn_for_query(user_message)
            logger.info(f"📊 Active VNIs after dynamic spawn: {list(self.vni_instances.keys())}")

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
            elif 'general' in vni_id and any(word in query_lower for word in 
                                         ['code', 'programming', 'technical', 'general', 'system', 'software']):
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
        general_keywords = ['code', 'programming', 'technical', 'general', 'system', 'algorithm', 'software']
        
        if any(keyword in message_lower for keyword in medical_keywords):
            relevant.extend([vni_id for vni_id in self.vni_instances if vni_id.startswith('medical')])
        
        if any(keyword in message_lower for keyword in legal_keywords):
            relevant.extend([vni_id for vni_id in self.vni_instances if vni_id.startswith('legal')])
            
        if any(keyword in message_lower for keyword in general_keywords):
            relevant.extend([vni_id for vni_id in self.vni_instances if vni_id.startswith('general')])
        
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

    # ==================== TEXT GENERATION METHODS ====================
    
    def enable_vni_generation(self, enable: bool = True):
        """Enable/disable text generation for all VNIs"""
        for vni_id, vni in self.vni_instances.items():
            if hasattr(vni, 'generation_enabled'):
                vni.generation_enabled = enable
        logger.info(f"✅ Text generation {'enabled' if enable else 'disabled'} for all VNIs")

    def get_generation_status(self) -> Dict[str, Any]:
        """Get text generation status for all VNIs"""
        status = {}
        for vni_id, vni in self.vni_instances.items():
            status[vni_id] = {
                'generation_enabled': getattr(vni, 'generation_enabled', False),
                'generator_available': getattr(vni, 'generator') is not None,
                'tokenizer_available': getattr(vni, 'tokenizer') is not None,
                'bridge_layer_available': getattr(vni, 'bridge_layer') is not None,
                'vni_type': vni.vni_type
            }
        return status

    def force_generation_response(self, query: str, vni_id: str = None) -> Dict[str, Any]:
        """Force generation response from specific or any VNI"""
        if vni_id and vni_id in self.vni_instances:
            vni = self.vni_instances[vni_id]
            if getattr(vni, 'generation_enabled', False):
                response = vni.generate_with_transformer(query)
                return {
                    'response': response,
                    'vni_id': vni_id,
                    'forced_generation': True,
                    'success': True
                }
        
        # Try any VNI with generation enabled
        for vni_id, vni in self.vni_instances.items():
            if getattr(vni, 'generation_enabled', False):
                response = vni.generate_with_transformer(query)
                return {
                    'response': response,
                    'vni_id': vni_id,
                    'forced_generation': True,
                    'success': True
                }
        
        return {
            'response': 'No VNI with generation capability available',
            'success': False
        }

    def test_generation_capabilities(self) -> Dict[str, Any]:
        """Test generation capabilities with sample queries"""
        test_queries = {
            'medical': "What are common symptoms of diabetes?",
            'legal': "What should be included in a basic employment contract?",
            'general': "Explain how artificial intelligence works in simple terms."
        }
        
        results = {}
        for vni_id, vni in self.vni_instances.items():
            if getattr(vni, 'generation_enabled', False):
                vni_type = vni.vni_type
                test_query = test_queries.get(vni_type, test_queries['general'])
                
                try:
                    response = vni.generate_with_transformer(test_query)
                    results[vni_id] = {
                        'success': True,
                        'response_preview': response[:150] + '...' if len(response) > 150 else response,
                        'response_length': len(response),
                        'test_query': test_query,
                        'vni_type': vni_type
                    }
                except Exception as e:
                    results[vni_id] = {
                        'success': False,
                        'error': str(e),
                        'test_query': test_query,
                        'vni_type': vni_type
                    }
        
        return results

    # NEW AUTONOMY MANAGEMENT METHODS - ADD THESE AT THE END OF THE CLASS
    async def set_autonomy_level(self, level: float):
        """Set the autonomy level for the system (0.0 to 1.0)"""
        self.autonomy_level = max(0.0, min(1.0, level))
        self.autonomy_engine.autonomy_level = self.autonomy_level
        
        # Adjust VNI autonomy protocols
        for vni_id, vni in self.vni_instances.items():
            if hasattr(vni, 'autonomous_protocol'):
                vni.autonomous_protocol.curiosity_level = self.autonomy_level
                vni.autonomous_protocol.collaboration_tendency = self.autonomy_level * 0.8
        
        logger.info(f"🔧 Autonomy level set to {self.autonomy_level}")

    async def get_autonomy_metrics(self) -> Dict[str, Any]:
        """Get metrics about autonomous interactions"""
        if not hasattr(self, 'autonomy_engine'):
            return {"autonomous_mode": False}
        
        metrics = {
            "autonomous_mode": self.autonomous_mode,
            "autonomy_level": self.autonomy_level,
            "active_dialogues": len(getattr(self.autonomy_engine, 'active_dialogues', {})),
            "vni_autonomy_levels": {},
            "recent_autonomous_activity": 0
        }
        
        # Count VNIs with autonomous protocols
        autonomous_vnis = 0
        for vni_id, vni in self.vni_instances.items():
            if hasattr(vni, 'autonomous_protocol'):
                autonomous_vnis += 1
                metrics["vni_autonomy_levels"][vni_id] = {
                    "curiosity": getattr(vni.autonomous_protocol, 'curiosity_level', 0),
                    "collaboration": getattr(vni.autonomous_protocol, 'collaboration_tendency', 0)
                }
        
        metrics["autonomous_vnis"] = autonomous_vnis
        
        # Count recent autonomous messages
        for msg in self.conversation_history[-20:]:
            if msg.get('role') in ['system_insight', 'system_autonomy', 'system_consensus']:
                metrics["recent_autonomous_activity"] += 1
        
        return metrics

    async def trigger_autonomous_discussion(self, topic: str, participants: Optional[List[str]] = None):
        """Manually trigger an autonomous discussion on a topic"""
        if not self.autonomous_mode:
            return {"error": "Autonomous mode is disabled"}
        
        if not participants:
            # Select diverse VNIs
            vni_types = set()
            participants = []
            for vni_id in self.vni_instances.keys():
                vni_type = vni_id.split('_')[0]
                if vni_type not in vni_types:
                    participants.append(vni_id)
                    vni_types.add(vni_type)
                if len(participants) >= 3:
                    break
        
        await self.autonomy_engine._initiate_spontaneous_discussion(participants, topic)
        
        return {
            "success": True,
            "topic": topic,
            "participants": participants,
            "message": "Autonomous discussion triggered"
        }

# Global enhanced orchestrator instance
orchestrator = EnhancedBabyBIONNOrchestrator()
# =================================================================
# PERMANENT AUTO-INITIALIZATION
# =================================================================
# This runs immediately when module is imported, before FastAPI starts

import asyncio
import sys
import threading
import time

def _initialize_orchestrator_sync():
    """Synchronous wrapper for async initialization"""
    try:
        print("🚀 STARTING PERMANENT AUTO-INITIALIZATION...", flush=True)
        
        # Check if we're in a thread with existing event loop
        try:
            loop = asyncio.get_running_loop()
            print(f"   Found existing event loop in thread {threading.current_thread().name}")
            # Schedule initialization in existing loop
            future = asyncio.run_coroutine_threadsafe(orchestrator.initialize(), loop)
            future.result(timeout=60)  # Wait up to 60 seconds
        except RuntimeError:
            # No running loop, create one
            print(f"   Creating new event loop in thread {threading.current_thread().name}")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(orchestrator.initialize())
            loop.close()
        
        print(f"✅ PERMANENT AUTO-INIT SUCCESS!")
        print(f"   Orchestrator initialized: {orchestrator.initialized}")
        print(f"   VNI instances: {len(orchestrator.vni_instances)}")
        if orchestrator.vni_instances:
            print(f"   VNI IDs: {list(orchestrator.vni_instances.keys())}")
        
        # Write state file for verification
        try:
            with open('/tmp/orchestrator_permanent_state.txt', 'w') as f:
                f.write(f'initialized={orchestrator.initialized}\n')
                f.write(f'vni_count={len(orchestrator.vni_instances)}\n')
                for vni_id in orchestrator.vni_instances.keys():
                    f.write(f'vni={vni_id}\n')
            print("   State file written")
        except:
            pass
            
    except Exception as e:
        print(f"❌ PERMANENT AUTO-INIT FAILED: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Don't crash, just set initialized to False
        orchestrator.initialized = False

# Run initialization in a dedicated thread to avoid blocking
print("🎬 Starting orchestrator initialization thread...", flush=True)
init_thread = threading.Thread(
    target=_initialize_orchestrator_sync,
    name="Orchestrator-Init-Thread",
    daemon=True
)
init_thread.start()

# Wait a bit for initialization to start (but don't block forever)
init_thread.join(timeout=2)
if init_thread.is_alive():
    print("🔄 Initialization running in background...", flush=True)
else:
    print("⚡ Initialization completed quickly", flush=True)

# =================================================================

# =================================================================
# === ADD EVERYTHING FROM HERE ===
# =================================================================

def verify_admin_token(admin_token: str):
    """Verify admin token for protected endpoints"""
    if admin_token != "babybionn_admin_2024":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token"
        )
    return admin_token

def get_vni_description(domain: str) -> str:
    """Get human-readable description for VNI domains"""
    descriptions = {
        "medical": "Healthcare and medical knowledge",
        "legal": "Legal knowledge and compliance", 
        "general": "General knowledge and conversation",
        "technical": "Programming and technical expertise",
        "core": "System operations and coordination"
    }
    return descriptions.get(domain, f"{domain} knowledge")

def get_vni_activation_patterns():
    """Calculate activation patterns for all VNIs"""
    patterns = {}
    total_activations = 0
    
    if not hasattr(orchestrator, 'vni_instances'):
        return patterns
        
    for vni_id, vni in orchestrator.vni_instances.items():
        activation_count = getattr(vni, 'activation_count', 0)
        patterns[vni_id] = {
            "activation_count": activation_count,
            "is_active": getattr(vni, 'is_active', False),
            "domain": getattr(vni, 'domain', 'unknown'),
            "description": get_vni_description(getattr(vni, 'domain', 'unknown'))
        }
        total_activations += activation_count
    
    for vni_id, data in patterns.items():
        if total_activations > 0:
            data["activation_percentage"] = round((data["activation_count"] / total_activations) * 100, 1)
        else:
            data["activation_percentage"] = 0
    
    return patterns

def get_learning_progress():
    """Get learning progress metrics"""
    if not hasattr(orchestrator, 'vni_instances'):
        return {
            "total_knowledge_items": 0,
            "total_responses": 0,
            "average_confidence": 0,
            "learning_rate": 0
        }
    
    total_knowledge = sum(len(getattr(vni, 'knowledge_base', [])) for vni in orchestrator.vni_instances.values())
    total_responses = sum(getattr(vni, 'response_count', 0) for vni in orchestrator.vni_instances.values())
    
    confidence_scores = []
    for vni in orchestrator.vni_instances.values():
        recent_responses = getattr(vni, 'recent_responses', [])
        if recent_responses:
            confidence_scores.extend([r.get('confidence', 0) for r in recent_responses])
    
    avg_confidence = round(sum(confidence_scores) / len(confidence_scores), 2) if confidence_scores else 0.7
    learning_rate = min(100, total_responses * 5)
    
    return {
        "total_knowledge_items": total_knowledge,
        "total_responses": total_responses,
        "average_confidence": avg_confidence,
        "learning_rate": learning_rate
    }

def get_system_health():
    """Get system health status"""
    if not hasattr(orchestrator, 'vni_instances'):
        return {
            "status": "initializing",
            "active_vnis": 0,
            "total_vnis": 0,
            "system_uptime": "0:00:00",
            "health_score": 0
        }
    
    active_vnis = len([vni for vni in orchestrator.vni_instances.values() if getattr(vni, 'is_active', False)])
    total_vnis = len(orchestrator.vni_instances)
    
    health_score = 0
    if total_vnis > 0:
        health_score = min(100, (active_vnis / total_vnis) * 100 + 50)
    
    status_text = "healthy" if health_score > 80 else "degraded" if health_score > 50 else "poor"
    
    if hasattr(orchestrator, 'startup_time'):
        uptime = datetime.now() - orchestrator.startup_time
        uptime_str = str(uptime).split('.')[0]
    else:
        uptime_str = "0:05:00"
    
    return {
        "status": status_text,
        "active_vnis": active_vnis,
        "total_vnis": total_vnis,
        "system_uptime": uptime_str,
        "health_score": round(health_score, 1)
    }

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

async def run_background_training(app: FastAPI):
    """Run background training with proper error handling"""
    try:
        # Wait for orchestrator to fully initialize
        await asyncio.sleep(10)
        
        logger.info("🔄 Starting background training initialization...")
        
        # Method 1: Try to get from app.state (set in lifespan)
        vni_manager = getattr(app.state, 'vni_manager', None)
        rl_system = getattr(app.state, 'rl_system', None)
        
        # Method 2: Fallback to orchestrator's actual attributes
        if not vni_manager:
            if hasattr(orchestrator, 'learning_orchestrator') and orchestrator.learning_orchestrator:
                vni_manager = orchestrator.learning_orchestrator
                logger.info("✅ Using learning_orchestrator as VNI manager")
            else:
                logger.warning("⚠️ No VNI manager available, skipping training")
                return
        
        if not rl_system:
            if hasattr(orchestrator, 'rl_engine'):
                rl_system = orchestrator.rl_engine
                logger.info("✅ Using rl_engine as RL system")
            else:
                logger.warning("⚠️ No RL system available, skipping training")
                return
        
        # Check if training pipeline exists
        try:
            from training_pipeline import create_training_pipeline
        except ImportError as e:
            logger.warning(f"⚠️ Training pipeline not available: {e}")
            return
        
        # Create and run training pipeline
        pipeline = create_training_pipeline(vni_manager, rl_system)
        domains = ['medical', 'legal', 'technical']
        
        logger.info(f"🚀 Starting BabyBIONN training for domains: {domains}")
        results = pipeline.run_complete_training(domains)
        
        logger.info(f"✅ Training completed! Results: {results}")
        
    except Exception as e:
        logger.error(f"❌ Background training failed: {e}")
        # Don't crash the app - just log the error

# Lifespan context manager first
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize enhanced orchestrator on startup and start training"""
    # Load models first
    logger.info("🔄 Loading AI models...")
    model_manager.load_all_models()
    
    # Then initialize orchestrator
    await orchestrator.initialize()

    # Set app.state with proper error handling
    if hasattr(orchestrator, 'learning_orchestrator') and orchestrator.learning_orchestrator:
        app.state.vni_manager = orchestrator.learning_orchestrator  # ← Remove .vni_manager
    else:
        # Fallback: create a minimal VNI manager
        from neuron.reinforcement_learning.vni_core import VNIManager
        app.state.vni_manager = VNIManager()
        logger.warning("Using fallback VNI manager - learning orchestrator not available")
    
    app.state.rl_system = orchestrator.rl_engine   
    app.state.model_manager = model_manager  # Make model_manager available via app state

    # Start training in background
    asyncio.create_task(run_background_training(app))
    
    yield
    
    # Cleanup on shutdown
    model_manager.cleanup()


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
    """Enhanced health check with model status"""
    model_status = model_manager.get_model_status()
    
    return {
        "status": "healthy",
        "initialized": orchestrator.initialized,
        "vni_instances": len(orchestrator.vni_instances),
        "synaptic_connections": len(orchestrator.synaptic_connections),
        "total_interactions": len(orchestrator.conversation_history),
        "models_loaded": model_status
    }

# ADD THE NEW ENDPOINT HERE - with your other API endpoints
@app.get("/verify/vni-fix")
async def verify_vni_fix():
    """Verify the VNI classifier fix and routing is working"""
    try:
        # Get orchestrator from app.state
        orchestrator = getattr(app.state, 'orchestrator', None)
        if not orchestrator:
            return {"error": "Orchestrator not found in app.state"}
        
        test_phrase = "I need help on medical and health advise"
        
        # Test medical VNI
        medical_vni = orchestrator.vni_instances.get('medical_0')
        if not medical_vni:
            return {"error": "Medical VNI not found"}
        
        # Test should_handle
        should_handle_result = False
        if hasattr(medical_vni, 'should_handle'):
            try:
                should_handle_result = medical_vni.should_handle(test_phrase)
            except Exception as e:
                should_handle_result = f"Error: {str(e)}"
        
        # Test classifier directly if available
        classifier_result = None
        if hasattr(medical_vni, 'classifier') and medical_vni.classifier:
            try:
                prediction = medical_vni.classifier.predict([test_phrase])
                classifier_result = prediction[0] if len(prediction) > 0 else None
            except Exception as e:
                classifier_result = f"Error: {str(e)}"
        
        # ============================================
        # 🆕 NEW: TEST THE ROUTING TOO!
        # ============================================
        routing_result = None
        activated_vnis = []
        if hasattr(orchestrator, 'enhanced_smart_route_query'):
            try:
                activated_vnis = orchestrator.enhanced_smart_route_query(test_phrase)
                routing_result = {
                    "activated_vnis": activated_vnis,
                    "medical_in_activated": 'medical_0' in activated_vnis
                }
            except Exception as e:
                routing_result = f"Routing Error: {str(e)}"
        
        # Determine overall status
        status = "checking"
        if isinstance(should_handle_result, bool) and routing_result and isinstance(routing_result, dict):
            if should_handle_result and routing_result["medical_in_activated"]:
                status = "✅ FIXED! Medical VNI will handle medical queries"
            elif should_handle_result and not routing_result["medical_in_activated"]:
                status = "⚠️ PARTIAL: VNI can handle but routing doesn't select it"
            elif not should_handle_result:
                status = "❌ BROKEN: VNI cannot detect medical queries"
        
        return {
            "status": status,
            "test_phrase": test_phrase,
            "should_handle": should_handle_result,
            "classifier_prediction": classifier_result,
            "routing_test": routing_result,
            "vni_info": {
                "vni_id": "medical_0",
                "type": getattr(medical_vni, 'vni_type', 'unknown'),
                "has_classifier": hasattr(medical_vni, 'classifier') and medical_vni.classifier is not None,
                "has_should_handle": hasattr(medical_vni, 'should_handle'),
                "generation_enabled": getattr(medical_vni, 'generation_enabled', False)
            },
            "available_vnis": list(orchestrator.vni_instances.keys())
        }
        
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/api/model-status")
async def get_model_status():
    """Get detailed model loading status"""
    return {
        "status": "success",
        "model_status": model_manager.get_model_status()
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
    """Process image using YOLO model from ModelManager"""
    try:
        # Use the pre-loaded model from ModelManager
        if not model_manager.models_loaded or 'yolo' not in model_manager.models:
            return {
                "detected_objects": [],
                "analysis": "YOLO model not loaded yet",
                "object_count": 0,
                "primary_objects": [],
                "error": "YOLO model not available"
            }
        
        model = model_manager.models['yolo']
        
        # Convert bytes to image
        from PIL import Image
        import numpy as np
        from io import BytesIO
        
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
        
        # Extract detection information (your existing logic)
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

        # Sort by confidence and get top objects (your existing logic)
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        top_objects = [det["object"] for det in detections[:5]]
        
        # Generate natural language analysis (your existing logic)
        if detections:
            object_counts = {}
            for det in detections:
                obj_name = det["object"]
                object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
            
            object_descriptions = []
            for obj, count in object_counts.items():
                if count == 1:
                    object_descriptions.append(f"a {obj}")
                else:
                    object_descriptions.append(f"{count} {obj}s")
            
            analysis = f"I detected {', '.join(object_descriptions)} in the image."
            
            if top_objects:
                primary_objs = list(set(top_objects[:3]))
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
        
    except Exception as e:
        logger.error(f"YOLO processing error: {e}")
        return {
            "detected_objects": [],
            "analysis": "Error processing image with YOLO",
            "object_count": 0,
            "primary_objects": [],
            "error": str(e)
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

@app.get("/api/admin/knowledge-status")
async def get_knowledge_status(admin_token: str = None):
    """Get current knowledge base status"""
    try:
        # Admin authentication
        if admin_token != "babybionn_admin_2024":
            raise HTTPException(status_code=401, detail="Invalid admin token")
        
        status = await get_knowledge_base_status()
        return {
            "success": True, 
            "knowledge_bases": status
        }
        
    except Exception as e:
        logger.error(f"Knowledge status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions you'll need to implement
async def process_pretraining_data(domain: str, data: dict):
    """Process pretraining data and update VNIs"""
    # This should integrate with your EnhancedBabyBIONNOrchestrator
    # You might need to add methods to your orchestrator like:
    # - orchestrator.pretrain_domain(domain, data)
    # - orchestrator.get_knowledge_status()
    
    logger.info(f"Pretraining domain: {domain} with {len(data)} items")
    
    # Example implementation:
    if domain == "medical":
        # Update medical VNI knowledge bases
        for vni_id, vni in orchestrator.vni_instances.items():
            if vni_id.startswith('medical'):
                if hasattr(vni, 'update_knowledge_base'):
                    vni.update_knowledge_base(data)
    
    return {
        "domain": domain,
        "items_processed": len(data),
        "vnis_updated": len([vni for vni in orchestrator.vni_instances.keys() if vni.startswith(domain)])
    }

async def get_knowledge_base_status():
    """Get status of all knowledge bases with better error handling"""
    status = {}
    
    for vni_id, vni in orchestrator.vni_instances.items():
        try:
            if hasattr(vni, 'get_knowledge_stats'):
                stats = vni.get_knowledge_stats()
                status[vni_id] = stats
            else:
                # Fallback for VNIs without knowledge stats method
                status[vni_id] = {
                    "concepts": 0,
                    "patterns": 0, 
                    "last_updated": 0,
                    "error": "Knowledge stats not available for this VNI type"
                }
        except Exception as e:
            logger.error(f"Error getting knowledge stats for {vni_id}: {e}")
            status[vni_id] = {
                "error": f"Failed to retrieve stats: {str(e)}",
                "concepts": 0,
                "patterns": 0,
                "last_updated": 0
            }
    
    return status

@app.post("/api/admin/pretrain")
async def pretrain_domain(
    domain: str = Form(...),
    file: UploadFile = File(...),
    admin_token: str = Form(None)
):
    """Admin endpoint for domain pretraining"""
    try:
        # Simple admin authentication
        if admin_token != "babybionn_admin_2024":
            raise HTTPException(status_code=401, detail="Invalid admin token")
        
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="File must be JSON")
        
        # Read and parse the JSON file
        content = await file.read()
        pretrain_data = json.loads(content)
        
        # Process the pretraining data
        # You'll need to integrate this with your orchestrator
        result = await process_pretraining_data(domain, pretrain_data)
        
        return {
            "success": True, 
            "message": f"Domain {domain} pretrained successfully",
            "analytics": result
        }
        
    except Exception as e:
        logger.error(f"Pretraining error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === ADD DASHBOARD ENDPOINTS HERE ===

@app.get("/api/dashboard/metrics")
async def get_dashboard_metrics(admin_token: str = Depends(verify_admin_token)):
    """Get real-time dashboard metrics"""
    # We need to create the helper functions first
    return {
        "total_sessions": 0,  # Placeholder - we'll implement this
        "active_vnis": len([vni for vni in orchestrator.vni_instances.values() if getattr(vni, 'is_active', False)]),
        "knowledge_concepts": sum(len(getattr(vni, 'knowledge_base', [])) for vni in orchestrator.vni_instances.values()),
        "vni_activation": get_vni_activation_patterns(),
        "learning_progress": get_learning_progress(),
        "system_health": get_system_health()
    }

@app.get("/api/dashboard/vni-status")
async def get_vni_status(admin_token: str = Depends(verify_admin_token)):
    """Get real VNI activation status"""
    vni_status = []
    for vni_id, vni in orchestrator.vni_instances.items():
        vni_status.append({
            "id": vni_id,
            "domain": getattr(vni, 'domain', 'unknown'),
            "is_active": getattr(vni, 'is_active', False),
            "activation_count": getattr(vni, 'activation_count', 0),
            "knowledge_items": len(getattr(vni, 'knowledge_base', [])),
            "last_activated": getattr(vni, 'last_activated_time', None)
        })
    return vni_status

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

# ==================== TEXT GENERATION ENDPOINTS ====================

@app.get("/api/generation-status")
async def get_generation_status():
    """Get text generation capabilities status"""
    try:
        status = orchestrator.get_generation_status()
        return {
            "status": "success",
            "generation_enabled": orchestrator.generation_enabled,
            "vni_generation_status": status,
            "generation_available": any(vni['generation_enabled'] for vni in status.values())
        }
    except Exception as e:
        logger.error(f"Generation status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/force-generation")
async def force_generation(request: Dict[str, Any]):
    """Force generation response (bypass templates)"""
    try:
        query = request.get("query", "")
        vni_id = request.get("vni_id")
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        result = orchestrator.force_generation_response(query, vni_id)
        return result
        
    except Exception as e:
        logger.error(f"Force generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/configure-generation")
async def configure_generation(
    enable: bool = Form(True),
    admin_token: str = Form(None)
):
    """Admin endpoint to configure text generation"""
    try:
        # Simple admin authentication
        if admin_token != "babybionn_admin_2024":
            raise HTTPException(status_code=401, detail="Invalid admin token")
        
        orchestrator.enable_vni_generation(enable)
        
        return {
            "success": True,
            "message": f"Text generation {'enabled' if enable else 'disabled'}",
            "generation_status": orchestrator.get_generation_status()
        }
        
    except Exception as e:
        logger.error(f"Configure generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test-generation")
async def test_generation():
    """Test generation capabilities"""
    try:
        results = orchestrator.test_generation_capabilities()
        return {
            "status": "success",
            "test_results": results,
            "summary": {
                "total_tested": len(results),
                "successful": sum(1 for r in results.values() if r.get('success', False)),
                "failed": sum(1 for r in results.values() if not r.get('success', False))
            }
        }
    except Exception as e:
        logger.error(f"Test generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== CHAT INTERFACE ====================

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
    PORT = 8001
    RELOAD = True

# ==================== NEW AUTONOMY ENDPOINTS ====================
@app.post("/api/autonomy/set-level")
async def set_autonomy_level(request: Dict[str, Any]):
    """Set autonomy level for the system"""
    try:
        level = request.get("level", 0.5)
        await orchestrator.set_autonomy_level(float(level))
        return {
            "status": "success",
            "message": f"Autonomy level set to {level}",
            "autonomy_level": orchestrator.autonomy_level
        }
    except Exception as e:
        logger.error(f"Autonomy level error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/autonomy/metrics")
async def get_autonomy_metrics():
    """Get autonomy metrics"""
    try:
        metrics = await orchestrator.get_autonomy_metrics()
        return {
            "status": "success",
            "autonomy_metrics": metrics
        }
    except Exception as e:
        logger.error(f"Autonomy metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/autonomy/trigger-discussion")
async def trigger_discussion(request: Dict[str, Any]):
    """Trigger an autonomous discussion"""
    try:
        topic = request.get("topic", "AI ethics and future development")
        participants = request.get("participants")
        
        result = await orchestrator.trigger_autonomous_discussion(topic, participants)
        return result
    except Exception as e:
        logger.error(f"Trigger discussion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/autonomy/active-dialogues")
async def get_active_dialogues():
    """Get active autonomous dialogues"""
    try:
        if hasattr(orchestrator, 'autonomy_engine'):
            dialogues = orchestrator.autonomy_engine.active_dialogues
            return {
                "status": "success",
                "active_dialogues": len(dialogues),
                "dialogues": [
                    {
                        "id": dialogue_id,
                        "topic": dialogue.get('topic', ''),
                        "participants": dialogue.get('participants', []),
                        "message_count": len(dialogue.get('messages', [])),
                        "convergence_score": dialogue.get('convergence_score', 0)
                    }
                    for dialogue_id, dialogue in dialogues.items()
                ]
            }
        return {"status": "success", "active_dialogues": 0}
    except Exception as e:
        logger.error(f"Active dialogues error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/synaptic-network/autonomy-view")
async def get_autonomy_synaptic_view():
    """Get synaptic network view with autonomy metrics"""
    try:
        # Create enhanced visualization with autonomy data
        network_data = []
        
        for conn_id, pathway in orchestrator.synaptic_connections.items():
            autonomy_score = 0.0
            if hasattr(pathway, 'activation_count') and pathway.activation_count > 0:
                autonomy_score = min(1.0, pathway.activation_count / 10)
            
            network_data.append({
                "source": pathway.source,
                "target": pathway.target,
                "strength": pathway.strength,
                "autonomy_score": autonomy_score,
                "activation_count": getattr(pathway, 'activation_count', 0),
                "last_activated": getattr(pathway, 'last_activated', None)
            })
        
        return {
            "status": "success",
            "network": network_data,
            "total_connections": len(network_data),
            "average_autonomy": sum(d['autonomy_score'] for d in network_data) / len(network_data) if network_data else 0
        }
    except Exception as e:
        logger.error(f"Autonomy synaptic view error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ... all your existing code including FastAPI app, endpoints, etc. ...

# ==================== SIMPLE ORCHESTRATOR GATEWAY FUNCTIONS ====================

def get_orchestrator_gateway():
    """Get the orchestrator instance for external use"""
    return orchestrator

async def gateway_process_query(
    query: str, 
    session_id: str = "default_user",
    context: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Gateway function for external systems to process queries
    
    Example usage from another file:
        from main import gateway_process_query
        response = await gateway_process_query("Hello", "user123")
    """
    # Ensure orchestrator is initialized
    if not orchestrator.initialized:
        await orchestrator.initialize()
    
    # Add context if provided
    if context:
        enhanced_query = f"[Context: {context.get('context_type', 'general')}] {query}"
    else:
        enhanced_query = query
    
    # Use the existing orchestrator
    result = await orchestrator.process_message(enhanced_query, session_id)
    
    # Add gateway metadata
    gateway_result = {
        "gateway_version": "1.0",
        "gateway_timestamp": datetime.now().isoformat(),
        **result,
        "processing_notes": {
            "mode": "enhanced_orchestration",
            "vni_count": len(orchestrator.vni_instances),
            "connection_count": len(orchestrator.synaptic_connections)
        }
    }
    
    return gateway_result

async def gateway_get_network_status() -> Dict[str, Any]:
    """Get current network status"""
    return {
        "vni_instances": list(orchestrator.vni_instances.keys()),
        "synaptic_connections": [
            {
                "id": conn_id,
                "source": pathway.source,
                "target": pathway.target,
                "strength": pathway.strength
            }
            for conn_id, pathway in orchestrator.synaptic_connections.items()
        ],
        "conversation_history_length": len(orchestrator.conversation_history),
        "initialized": orchestrator.initialized,
        "gateway_interface": "enabled"
    }

async def gateway_trigger_autonomy(topic: str = "system_optimization") -> Dict[str, Any]:
    """Trigger autonomous VNI discussion via gateway"""
    if not orchestrator.initialized:
        await orchestrator.initialize()
    
    if hasattr(orchestrator, 'trigger_autonomous_discussion'):
        result = await orchestrator.trigger_autonomous_discussion(topic)
        return {
            "gateway_action": "autonomy_triggered",
            "topic": topic,
            "result": result
        }
    return {"error": "Autonomy feature not available"}

# ==================== DIRECT IMPORT UTILITIES ====================

# These make it easy to use from other modules
def get_vni_instance(vni_id: str):
    """Get specific VNI instance by ID"""
    return orchestrator.vni_instances.get(vni_id)

def get_all_vnis() -> Dict[str, Any]:
    """Get all VNI instances"""
    return {
        vni_id: {
            "type": getattr(vni, 'vni_type', 'unknown'),
            "domain": vni_id.split('_')[0],
            "generation_enabled": getattr(vni, 'generation_enabled', False)
        }
        for vni_id, vni in orchestrator.vni_instances.items()
    }

async def gateway_send_feedback(
    message_id: str,
    feedback_type: str,
    correction: Optional[str] = None,
    session_id: str = "default_user"
) -> Dict[str, Any]:
    """Send feedback via gateway"""
    feedback_data = {
        "message_id": message_id,
        "feedback_type": feedback_type,
        "correction": correction,
        "session_id": session_id
    }
    
    await orchestrator.learn_from_feedback(feedback_data)
    return {
        "gateway_feedback": "processed",
        "timestamp": datetime.now().isoformat(),
        "feedback_type": feedback_type
    }

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    logger.info(f"🚀 Starting ENHANCED BabyBIONN Main Bridge on {Config.HOST}:{Config.PORT}")
    logger.info(f"📁 Project root: {PROJECT_ROOT}")
    logger.info(f"📁 Static files: {STATIC_DIR}")
    logger.info(f"🌐 Chat interface: http://{Config.HOST}:{Config.PORT}/chat")
    logger.info(f"📊 Analytics: http://{Config.HOST}:{Config.PORT}/api/analytics")
    logger.info(f"🔗 Visualization: http://{Config.HOST}:{Config.PORT}/api/synaptic-visualization")
    logger.info(f"🚪 Gateway interface ready for import")
    
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.RELOAD
    )
