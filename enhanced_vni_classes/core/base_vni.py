# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

"""Base VNI class with core functionality. Extracted from enhanced_vni_classes.py"""
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod
from ..utils.logger import get_logger
from ..core.capabilities import VNICapabilities, VNIType
from ..core.neural_pathway import NeuralPathway
from ..core.collaboration import CollaborationRequest
from ..modules.knowledge_base import KnowledgeBase
from ..modules.classifier import DomainClassifier
from ..core.pipeline_steps import (
    PipelineStep, 
    ClassifyStep, 
    SafetyCheckStep, 
    KnowledgeLookupStep, 
    GenerateResponseStep
)
from .biological_mixin import BiologicalSystemsMixin

# ============ BIOLOGICAL SYSTEMS (BIONN-ATTN-ACT) ============
# Biological attention and activation systems
from neuron.demoHybridAttention import DemoHybridAttention
from neuron.smart_activation_router import SmartActivationRouter
from neuron.vni_memory import VniMemory
# ===========================================================

logger = get_logger(__name__)

class EnhancedBaseVNI(BiologicalSystemsMixin, ABC):  # Added BiologicalSystemsMixin
    """Base class for all VNIs with shared functionality including biological systems"""
    def __init__(self, 
                 instance_id: str,
                 domain: str = "general",
                 capabilities: VNICapabilities = None,
                 vni_config: Dict[str, Any] = None,
                 name: str = None,
                 description: str = None,
                 vni_type: str = None,
                 # Biological system parameters - will be handled by mixin
                 enable_biological_systems: bool = True,  # Default to True for all VNIs
                 attention_config: Dict[str, Any] = None,
                 memory_config: Dict[str, Any] = None,
                 # NEW: Validation parameters
                 auto_validate: bool = True,
                 strict_validation: bool = False,
                 validation_timeout: float = 5.0):
        """Initialize EnhancedBaseVNI with biological systems integration
        Args:
            auto_validate: Automatically validate VNI architecture after initialization
            strict_validation: If True, raise error on validation failure instead of warning
            validation_timeout: Maximum time allowed for validation (seconds)"""        
        self.instance_id = instance_id
        self.config = vni_config or {}
        self.auto_validate = auto_validate
        self.strict_validation = strict_validation
        self.validation_timeout = validation_timeout

        # Store biological config for mixin
        self.attention_config = attention_config
        self.memory_config = memory_config

        super().__init__(
            instance_id=instance_id,
            domain=domain,
            capabilities=capabilities,
            vni_config=vni_config,
            name=name,
            description=description,
            vni_type=vni_type,
            enable_biological_systems=enable_biological_systems,
            attention_config=attention_config,
            memory_config=memory_config,
            auto_validate=auto_validate,
            strict_validation=strict_validation,
            validation_timeout=validation_timeout
        )

        # Handle capabilities
        if capabilities:
            self.capabilities = capabilities
        else:
            self.capabilities = VNICapabilities(
                domains=[domain],
                can_search=True,
                can_learn=True,
                can_collaborate=True,
                max_context_length=2000,
                special_abilities=["base_functionality"],
                vni_type=vni_type or domain                
            )

        self._init_classifier()
        self._init_generation()
        self._init_collaboration()
        self._init_learning()
        self._init_attention()
                
        # Set additional properties if provided
        self.name = name or instance_id
        self.description = description or f"VNI instance for {domain} domain"
        self.created_at = datetime.now()
        
        if vni_type and hasattr(self.capabilities, 'vni_type'):
            self.capabilities.vni_type = vni_type
        
        # Determine domain
        if hasattr(self.capabilities, 'domains') and self.capabilities.domains:
            self.domain = self.capabilities.domains[0]
        elif hasattr(self.capabilities, 'domain') and self.capabilities.domain:
            self.domain = self.capabilities.domain
        else:
            self.domain = domain

        # Core components - use domain from capabilities
        self.knowledge_base = KnowledgeBase(domain=self.domain)
        
        # Try to import DynamicDomainClassifier, fall back to DomainClassifier
        try:
            from enhanced_vni_classes.modules.classifier import DynamicDomainClassifier
            classifier_class = DynamicDomainClassifier
        except ImportError:
            logger.warning("DynamicDomainClassifier not found, falling back to DomainClassifier")
            from ..modules.classifier import DomainClassifier
            classifier_class = DomainClassifier
        
        # Get domain-specific configuration
        domain_config = self._get_domain_config()
        self.classifier = classifier_class(
            domain_name=self.domain,
            keywords=domain_config["keywords"],
            priority_keywords=domain_config["priority_keywords"],
            regex_patterns=domain_config.get("regex_patterns", []),
            confidence_threshold=0.25
        )
        
        self.neural_pathways: Dict[str, NeuralPathway] = {}    
        
        # State
        self.generation_enabled = False
        self.model = None
        self.tokenizer = None
        self.bridge_layer = None
        self.generation_config = {}
        
        # Collaboration
        self.active_collaborations: List[CollaborationRequest] = []
        self.connection_strengths: Dict[str, float] = {}
        
        # Initialize via mixin (BiologicalSystemsMixin.__init__ will be called)
#        super().__init__()  # This calls BiologicalSystemsMixin.__init__
        
        logger.info(f"🧬 Initialized EnhancedBaseVNI: {self.instance_id} "
                   f"(domain: {self.domain}, bio: {self.enable_biological_systems})")

    def _init_classifier(self):
        """Initialize classifier components."""
        try:
            # Try to import DynamicDomainClassifier, fall back to DomainClassifier
            try:
                from enhanced_vni_classes.modules.classifier import DynamicDomainClassifier
                classifier_class = DynamicDomainClassifier
            except ImportError:
                logger.warning("DynamicDomainClassifier not found, falling back to DomainClassifier")
                from ..modules.classifier import DomainClassifier
                classifier_class = DomainClassifier
            
            # Get domain-specific configuration
            domain_config = self._get_domain_config()
            
            self.classifier = classifier_class(
                domain_name=self.domain,
                keywords=domain_config["keywords"],
                priority_keywords=domain_config["priority_keywords"],
                regex_patterns=domain_config.get("regex_patterns", []),
                confidence_threshold=0.25
            )
            
            logger.debug(f"Initialized classifier for domain: {self.domain}")
        except Exception as e:
            logger.error(f"Failed to initialize classifier: {e}")
            # Create a basic classifier as fallback
            from ..modules.classifier import DomainClassifier
            self.classifier = DomainClassifier(domain_name=self.domain)

    def _init_generation(self):
        """Initialize text generation components."""
        try:
            # Check if generation is enabled in capabilities
            if hasattr(self.capabilities, 'can_generate') and self.capabilities.can_generate:
                logger.info(f"Initializing generation for {self.instance_id}")
                # Setup basic generation configuration
                self.generation_enabled = True
                self.generation_config = {
                    "enabled": True,
                    "model_loaded": False,  # Will be loaded on-demand
                    "max_length": 200,
                    "temperature": 0.7
                }
            else:
                self.generation_enabled = False
                self.generation_config = {"enabled": False}
                
        except Exception as e:
            logger.error(f"Failed to initialize generation: {e}")
            self.generation_enabled = False
            self.generation_config = {"enabled": False}

    def _init_collaboration(self):
        """Initialize collaboration components."""
        try:
            # Check if collaboration is enabled in capabilities
            if hasattr(self.capabilities, 'can_collaborate') and self.capabilities.can_collaborate:
                logger.debug(f"Initializing collaboration for {self.instance_id}")
                # Initialize empty collaboration structures
                self.active_collaborations = []
                self.connection_strengths = {}
                self.neural_pathways = {}
            else:
                logger.debug(f"Collaboration disabled for {self.instance_id}")
                self.active_collaborations = []
                self.connection_strengths = {}
                self.neural_pathways = {}
                
        except Exception as e:
            logger.error(f"Failed to initialize collaboration: {e}")
            self.active_collaborations = []
            self.connection_strengths = {}
            self.neural_pathways = {}

    def _init_learning(self):
        """Initialize learning components."""
        try:
            # Check if learning is enabled in capabilities
            if hasattr(self.capabilities, 'can_learn') and self.capabilities.can_learn:
                logger.debug(f"Initializing learning for {self.instance_id}")
                # Learning will be initialized on-demand
                self.learning_enabled = True
            else:
                self.learning_enabled = False
                
        except Exception as e:
            logger.error(f"Failed to initialize learning: {e}")
            self.learning_enabled = False

    def _init_attention(self):
        """Initialize attention system components."""
        try:
            # This is handled by BiologicalSystemsMixin
            # Just log that attention initialization is managed by mixin
            if self.enable_biological_systems:
                logger.debug(f"Attention system will be initialized by BiologicalSystemsMixin for {self.instance_id}")
            else:
                logger.debug(f"Biological systems disabled for {self.instance_id}")
                
        except Exception as e:
            logger.error(f"Error in attention initialization: {e}")
    
    def _get_domain_config(self) -> Dict[str, Any]:
        """Get domain-specific configuration for the classifier."""
        # Domain-specific keyword configurations
        domain_configs = {
            "medical": {
                "keywords": [
                    "health", "medicine", "treatment", "diagnosis", "symptoms",
                    "patient", "doctor", "hospital", "clinical", "pharmaceutical",
                    "pain", "fever", "cough", "headache", "infection", "blood"
                ],
                "priority_keywords": [
                    "emergency", "urgent", "911", "heart attack", "stroke", 
                    "bleeding", "chest pain", "difficulty breathing"
                ],
                "regex_patterns": [
                    r"(?i)\b(doctor|patient|symptom|disease|treatment|medicine)\b",
                    r"(?i)\b(covid|vaccine|prescription|surgery|therapy)\b"
                ]
            },
            "legal": {
                "keywords": [
                    "legal", "law", "court", "contract", "agreement",
                    "rights", "case", "judge", "attorney", "jurisdiction",
                    "evidence", "testimony", "verdict", "settlement"
                ],
                "priority_keywords": [
                    "arrest", "lawsuit", "eviction", "legal emergency", 
                    "court date", "summons", "warrant"
                ],
                "regex_patterns": [
                    r"(?i)\b(law|legal|attorney|lawyer|court|case)\b",
                    r"(?i)\b(rights|lawsuit|settlement|jurisdiction)\b"
                ]
            },
            "general": {
                "keywords": [
                    "information", "knowledge", "explain", "describe", "tell",
                    "what", "how", "why", "when", "where", "who",
                    "help", "assist", "support", "question", "answer"
                ],
                "priority_keywords": [
                    "help", "urgent", "emergency", "quick question",
                    "need help", "assistance needed"
                ],
                "regex_patterns": [
                    r"(?i)\b(hello|hi|hey|how are you|what is|who is)\b",
                    r"(?i)\b(help|assist|support|question|answer|explain)\b"
                ]
            },
            "technical": {
                "keywords": [
                    "technical", "technology", "software", "hardware", "system",
                    "code", "programming", "algorithm", "database", "network",
                    "debug", "optimize", "deploy", "configure", "install"
                ],
                "priority_keywords": [
                    "error", "bug", "crash", "broken", "not working",
                    "system down", "server offline"
                ],
                "regex_patterns": [
                    r"(?i)\b(technical|technology|software|hardware|system)\b",
                    r"(?i)\b(code|programming|algorithm|database|network)\b"
                ]
            },
            "academic": {
                "keywords": [
                    "research", "study", "academic", "paper", "thesis",
                    "publication", "journal", "scholar", "university", "campus"
                ],
                "priority_keywords": [
                    "deadline", "due date", "submit", "paper due"
                ],
                "regex_patterns": [
                    r"(?i)\b(research|study|academic|paper|thesis)\b",
                    r"(?i)\b(publication|journal|university|campus)\b"
                ]
            },
            "financial": {
                "keywords": [
                    "finance", "money", "investment", "bank", "stock",
                    "market", "trading", "portfolio", "budget", "savings",
                    "loan", "credit", "debit", "interest", "tax"
                ],
                "priority_keywords": [
                    "fraud", "scam", "emergency", "lost card", "stolen",
                    "unauthorized transaction"
                ],
                "regex_patterns": [
                    r"(?i)\b(finance|money|investment|bank|stock)\b",
                    r"(?i)\b(market|trading|portfolio|budget|savings)\b"
                ]
            },
            "customer_service": {
                "keywords": [
                    "customer", "service", "support", "help", "issue",
                    "problem", "complaint", "order", "refund", "return",
                    "shipping", "delivery", "product", "account", "payment"
                ],
                "priority_keywords": [
                    "urgent", "emergency", "broken", "not working",
                    "immediate assistance", "critical issue"
                ],
                "regex_patterns": [
                    r"(?i)\b(customer|service|support|help|issue)\b",
                    r"(?i)\b(problem|complaint|order|refund|return)\b"
                ]
            }
        }
        
        # Return config for this domain, or default general config
        config = domain_configs.get(self.domain, domain_configs["general"])
        logger.debug(f"Loaded domain config for {self.domain}: {len(config['keywords'])} keywords")
        return config

    def process_with_biological_systems(self, 
                                      query: str, 
                                      context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process query using biological attention and activation systems.
        
        Args:
            query: The query to process
            context: Additional context information
            
        Returns:
            Dict with response and biological processing metadata
        """
        if not self.enable_biological_systems:
            return {
                "response": f"Biological systems not enabled: {query}",
                "biological_processing": False,
                "error": "Biological systems disabled"
            }
        
        try:
            # 1. Apply attention to the query
            attention_result = self.attention_system.focus(
                input_data=query,
                context=context
            )
            
            # 2. Route through activation system
            activation_result = self.activation_router.route(
                input_text=query,
                attention_weights=attention_result.get('attention_weights', {}),
                domain=self.domain
            )
            
            # 3. Store in memory (both short-term and potentially long-term)
            memory_result = self.memory_system.store(
                query=query,
                context=context,
                activation_level=activation_result.get('activation_level', 0.5),
                domain=self.domain
            )
            
            # 4. Retrieve relevant memories
            relevant_memories = self.memory_system.retrieve(
                query=query,
                context=context,
                top_k=3
            )
            
            # 5. Generate response (basic implementation)
            response = self._generate_biological_response(
                query=query,
                attention_result=attention_result,
                activation_result=activation_result,
                memories=relevant_memories,
                context=context
            )
            
            return {
                "response": response,
                "biological_processing": True,
                "attention_result": attention_result,
                "activation_result": activation_result,
                "memory_result": memory_result,
                "relevant_memories": relevant_memories,
                "domain": self.domain,
                "vni_id": self.instance_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Biological processing failed: {e}")
            return {
                "response": f"Error in biological processing: {query}",
                "biological_processing": False,
                "error": str(e)
            }
    
    def _generate_biological_response(self,
                                    query: str,
                                    attention_result: Dict[str, Any],
                                    activation_result: Dict[str, Any],
                                    memories: List[Dict[str, Any]],
                                    context: Dict[str, Any] = None) -> str:
        """
        Generate response based on biological processing results.
        
        Args:
            query: Original query
            attention_result: Results from attention system
            activation_result: Results from activation router
            memories: Retrieved relevant memories
            context: Additional context
            
        Returns:
            Generated response string
        """
        # Simple response generation based on activation level
        activation_level = activation_result.get('activation_level', 0.5)
        
        if activation_level > 0.8:
            # High activation - detailed response
            response = f"[HIGH ACTIVATION: {activation_level:.2f}] I've processed your query about '{query}' with high attention. "
            if memories:
                response += f"I recall: {memories[0].get('content', '')[:100]}... "
            response += "This is an important topic that requires careful consideration."
            
        elif activation_level > 0.5:
            # Medium activation - standard response
            response = f"[MEDIUM ACTIVATION: {activation_level:.2f}] Regarding '{query}', "
            attention_type = attention_result.get('attention_type', 'balanced')
            response += f"I'm applying {attention_type} attention to this. "
            response += "I can provide you with information on this topic."
            
        else:
            # Low activation - brief response
            response = f"[LOW ACTIVATION: {activation_level:.2f}] I've noted your query about '{query}'. "
            response += "This appears to be a routine inquiry that I can address."
        
        return response

    def load_knowledge(self, knowledge_paths: List[str]) -> Dict[str, Any]:
        """
        Load knowledge from files
        
        Args:
            knowledge_paths: List of file paths to load knowledge from
            
        Returns:
            Dict with loading results and statistics
        """
        try:
            return self.knowledge_base.load_multiple(knowledge_paths)
        except Exception as e:
            logger.error(f"Failed to load knowledge: {e}")
            return {"success": False, "error": str(e), "loaded_files": 0}

    def save_knowledge(self, filepath: str) -> bool:
        """
        Save knowledge to file
        
        Args:
            filepath: Path to save knowledge to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = self.knowledge_base.save(filepath)
            if success:
                logger.info(f"Knowledge saved successfully to {filepath}")
            else:
                logger.warning(f"Failed to save knowledge to {filepath}")
            return success
        except Exception as e:
            logger.error(f"Error saving knowledge: {e}")
            return False

    def add_concept(self, concept: str, data: Dict[str, Any]) -> bool:
        """
        Add new concept to knowledge base
        
        Args:
            concept: Concept name or identifier
            data: Concept data and attributes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = self.knowledge_base.add_concept(concept, data)
            if success:
                logger.debug(f"Added concept '{concept}' to knowledge base")
            return success
        except Exception as e:
            logger.error(f"Failed to add concept '{concept}': {e}")
            return False

    def query_knowledge(self, query: str, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Query knowledge base
        
        Args:
            query: Query string
            threshold: Similarity threshold for results
            
        Returns:
            List of matching knowledge entries
        """
        try:
            results = self.knowledge_base.query(query, threshold)
            logger.debug(f"Knowledge query returned {len(results)} results for '{query}'")
            return results
        except Exception as e:
            logger.error(f"Knowledge query failed: {e}")
            return []

    def setup_generation(self, 
                        model_name: str = "microsoft/DialoGPT-medium",
                        bridge_dim: int = 512) -> bool:
        """
        Setup text generation capability
        
        Args:
            model_name: Name of the model to use for generation
            bridge_dim: Dimension of the bridge layer
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Setting up generation with model: {model_name}")
            
            # Simulate model loading
            self.generation_enabled = True
            self.generation_config = {
                "model_name": model_name,
                "bridge_dim": bridge_dim,
                "max_length": 100,
                "temperature": 0.7
            }
            
            logger.info(f"Generation setup complete for {self.instance_id}")
            return True
        except Exception as e:
            logger.error(f"Generation setup failed: {e}")
            self.generation_enabled = False
            return False

    @abstractmethod
    def process(self, query: str, pipeline: Optional[List[str]] = None, 
                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a query using an optional pipeline specification
        
        Args:
            query: The query to process
            pipeline: List of processing step names (e.g., ["classify", "safety_check", "generate"])
                     If None, use the VNI's default pipeline
            context: Additional context information (user data, location, etc.)
            
        Returns:
            Dict with response and metadata
        """
        pass

    def get_default_pipeline(self) -> List[str]:
        """Get the default pipeline for this VNI type
        Returns:
            List of pipeline step names"""
        # Include biological processing if enabled
        if self.enable_biological_systems:
            return ["classify", "biological_attention", "knowledge_lookup", "generate_response"]
        else:
            return ["classify", "knowledge_lookup", "generate_response"]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of VNI including biological systems"""
        try:
            knowledge_stats = self.knowledge_base.get_stats() if hasattr(self.knowledge_base, 'get_stats') else {}
            
            # Get biological status from mixin
            biological_status = self.get_biological_status()
            
            return {
                "instance_id": self.instance_id,
                "name": self.name,
                "domain": self.domain,
                "vni_type": self.vni_type,
                "created_at": self.created_at.isoformat(),
                "capabilities": self.capabilities.to_dict() if hasattr(self.capabilities, 'to_dict') else str(self.capabilities),
                "knowledge_stats": knowledge_stats,
                "generation_enabled": self.generation_enabled,
                "active_collaborations": len(self.active_collaborations),
                "neural_pathways": len(self.neural_pathways),
                "connection_strengths": self.connection_strengths,
                "biological_systems": biological_status,
                "health": "healthy"
            }
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {
                "instance_id": self.instance_id,
                "error": str(e),
                "health": "error"
            }
        
    def get_available_steps(self) -> Dict[str, PipelineStep]:
        """
        Get available pipeline steps for this VNI
        
        Returns:
            Dict mapping step names to PipelineStep instances
        """
        steps = {
            "classify": ClassifyStep(),
            "safety_check": SafetyCheckStep(),
            "knowledge_lookup": KnowledgeLookupStep(),
            "generate_response": GenerateResponseStep()
        }
        
        # Add biological processing step if systems are enabled
        if self.enable_biological_systems:
            from ..core.pipeline_steps import BiologicalProcessingStep
            steps["biological_attention"] = BiologicalProcessingStep(vni_instance=self)
        
        return steps

    def _execute_pipeline(self, query: str, pipeline: List[str], 
                         context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a pipeline of steps
        
        Args:
            query: Query to process
            pipeline: List of step names to execute
            context: Additional context
            
        Returns:
            Dict with processing results
        """
        data = {
            "query": query, 
            "context": context or {},
            "vni_id": self.instance_id,
            "domain": self.domain,
            "timestamp": datetime.now().isoformat(),
            "biological_processing_enabled": self.enable_biological_systems
        }
        
        available_steps = self.get_available_steps()
        executed_steps = []
        
        for step_name in pipeline:
            try:
                if step_name in available_steps:
                    step = available_steps[step_name]
                    logger.debug(f"Executing pipeline step: {step_name}")
                    data = step.execute(query, data, context)
                    executed_steps.append(step_name)
                else:
                    logger.warning(f"Unknown pipeline step: {step_name}")
                    data[f"step_{step_name}_error"] = "Unknown step"
            except Exception as e:
                logger.error(f"Error in pipeline step {step_name}: {e}")
                data[f"step_{step_name}_error"] = str(e)
        
        data['executed_steps'] = executed_steps
        
        # Ensure response exists
        if 'response' not in data:
            data['response'] = f"Processed query: {query}"
            data['confidence'] = 0.5
            data['source'] = 'fallback'
        
        return data

    def collaborate(self, 
                   target_vni_id: str,
                   query: str,
                   collaboration_type: str = "knowledge_share") -> Dict[str, Any]:
        """
        Initiate collaboration with another VNI
        
        Args:
            target_vni_id: ID of the VNI to collaborate with
            query: Query or information to share
            collaboration_type: Type of collaboration
            
        Returns:
            Dict with collaboration request details
        """
        try:
            request = CollaborationRequest(
                source_id=self.instance_id,
                target_id=target_vni_id,
                query=query,
                collaboration_type=collaboration_type,
                timestamp=datetime.now()
            )
            
            self.active_collaborations.append(request)
            logger.info(f"Created collaboration request to {target_vni_id}")
            
            return request.to_dict()
        except Exception as e:
            logger.error(f"Failed to create collaboration request: {e}")
            return {
                "success": False,
                "error": str(e),
                "source_id": self.instance_id,
                "target_id": target_vni_id
            }

    def add_neural_pathway(self, 
                          target_vni_id: str,
                          pathway_type: str = "bidirectional",
                          initial_strength: float = 0.1) -> Optional[NeuralPathway]:
        """
        Add neural pathway to another VNI
        
        Args:
            target_vni_id: ID of the target VNI
            pathway_type: Type of pathway (unidirectional, bidirectional, etc.)
            initial_strength: Initial connection strength
            
        Returns:
            NeuralPathway object if successful, None otherwise
        """
        try:
            pathway = NeuralPathway(
                source_id=self.instance_id,
                target_id=target_vni_id,
                pathway_type=pathway_type,
                strength=initial_strength
            )
            
            self.neural_pathways[target_vni_id] = pathway
            logger.info(f"Added neural pathway to {target_vni_id}")
            
            return pathway
        except Exception as e:
            logger.error(f"Failed to add neural pathway: {e}")
            return None

    def update_connection_strength(self, 
                                  vni_id: str,
                                  delta: float) -> float:
        """
        Update connection strength with another VNI
        
        Args:
            vni_id: ID of the VNI to update connection with
            delta: Change in connection strength (can be positive or negative)
            
        Returns:
            New connection strength
        """
        try:
            current = self.connection_strengths.get(vni_id, 0.1)
            new_strength = max(0.0, min(1.0, current + delta))
            self.connection_strengths[vni_id] = new_strength
            
            # Update corresponding neural pathway if exists
            if vni_id in self.neural_pathways:
                self.neural_pathways[vni_id].update_strength(delta)
            
            logger.debug(f"Updated connection strength with {vni_id}: {current:.3f} -> {new_strength:.3f}")
            return new_strength
        except Exception as e:
            logger.error(f"Failed to update connection strength: {e}")
            return self.connection_strengths.get(vni_id, 0.1)

    def __str__(self) -> str:
        """String representation of the VNI"""
        bio_status = "bio+" if self.enable_biological_systems else ""
        return f"EnhancedBaseVNI({bio_status}{self.instance_id}, domain='{self.domain}')"

    def __repr__(self) -> str:
        """Detailed representation of the VNI"""
        return f"<EnhancedBaseVNI: {self.instance_id} (bio={self.enable_biological_systems}) at {hex(id(self))}>"
