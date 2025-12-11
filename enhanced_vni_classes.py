# enhanced_vni_classes_with_generation.py - COMPLETE Implementation
import os
import re
import json
import random
import asyncio
import hashlib
import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Set, Dict, List, Any, Optional, Tuple

# ==================== AUTONOMOUS CAPABILITIES IMPORTS ====================
# Sentence Transformers for topic embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    logging.warning("Sentence Transformers not available - topic embeddings disabled")

# GPT-2 for NLP processing
try:
    from transformers import GPT2Tokenizer, GPT2Model
    GPT2_AVAILABLE = True
except ImportError:
    GPT2_AVAILABLE = False
    logging.warning("GPT-2 not available - NLP processing disabled")

# ========================================================================
# Optional PyTorch support for attention mechanisms
try:
    import torch
    import torch.nn as nn
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available - attention mechanisms disabled")

# Text Generation Support
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    GENERATION_AVAILABLE = True
    logging.info("✅ Transformers available for text generation")
except ImportError:
    GENERATION_AVAILABLE = False
    logging.warning("Transformers not available - text generation disabled")

# Predictive modules with fallback
try:
    from predictive_vocabulary import PredictiveVocabulary
    from predictive_response import PredictiveResponseGenerator
    PREDICTIVE_AVAILABLE = True
except ImportError:
    PREDICTIVE_AVAILABLE = False
    logging.warning("Predictive modules not available - using fallback implementations")

# Web search capabilities
try:
    from duckduckgo_search import DDGS
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    logging.warning("DuckDuckGo search not available - install with: pip install duckduckgo-search")

logger = logging.getLogger("enhanced_vni_classes")

# Add new VNI Types
class VNIType(str, Enum):
    MEDICAL = "medical"
    LEGAL = "legal"
    TECHNICAL = "technical"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    GENERAL = "general"
    CROSS_DOMAIN = "cross_domain"

# Add new classes at the top or integrate into existing ones
@dataclass
class VNICapabilities:
    """Capabilities required/available for a VNI"""
    domain_knowledge: List[str] = field(default_factory=list)
    abstraction_levels: List[str] = field(default_factory=lambda: ["semantic", "structural"])
    processing_speed: float = 1.0  # 0.0 to 1.0
    collaboration_score: float = 0.5  # How well this VNI collaborates
    specializations: Set[str] = field(default_factory=set)

@dataclass
class CollaborationRequest:
    """Protocol for VNI-to-VNI collaboration"""
    request_id: str
    source_vni_id: str
    target_vni_ids: List[str]
    task_description: str
    data_payload: Dict[str, Any]
    required_capabilities: VNICapabilities
    priority: str = "normal"  # low, normal, high, critical
    timeout_seconds: int = 30
    response_format: str = "abstraction"  # abstraction, raw, summary
    created_at: datetime = field(default_factory=datetime.now)

# ==================== SHARED NLP COMPONENTS ====================
class SharedNLPComponents:
    """
    Singleton class to share NLP components across all VNIs
    Avoids loading multiple copies of large models
    """
    _instance = None
    
    def __init__(self):
        if SENTENCE_TRANSFORMER_AVAILABLE:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info("✅ Sentence Transformer loaded for topic embeddings")
        else:
            self.sentence_transformer = None
            
        if GPT2_AVAILABLE:
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.gpt2_model = GPT2Model.from_pretrained('gpt2')
            logging.info("✅ GPT-2 loaded for NLP processing")
        else:
            self.gpt2_tokenizer = None
            self.gpt2_model = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = SharedNLPComponents()
        return cls._instance
    
    def is_available(self) -> bool:
        """Check if NLP components are available"""
        return (self.sentence_transformer is not None and 
                self.gpt2_tokenizer is not None and 
                self.gpt2_model is not None)

# ==================== SSP CONVERTER ====================
class SSPConverter:
    """Convert between Synthetic Synaptic Patterns and tensors"""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
    
    def ssp_to_tensors(self, ssp_pattern: Dict) -> Dict[str, Any]:
        """Convert SSP pattern to Q, K, V tensors"""
        # For now, return placeholder - will integrate with smart_activation_router
        return {
            'Q': None,
            'K': None,
            'V': None,
            'metadata': {'converted': True, 'timestamp': datetime.now().isoformat()}
        }
    
    def tensors_to_ssp(self, tensors: Dict, original_pattern: Dict) -> Dict:
        """Convert Q,K,V tensors back to SSP pattern"""
        # For now, return original
        return original_pattern

# ==================== END OF PRELIMINARY CLASSES ====================

class NeuralPathway:
    """Represents a dynamic synaptic connection between VNIs"""
    
    def __init__(self, source_vni: str, target_vni: str, initial_strength: float = 0.5):
        self.source = source_vni
        self.target = target_vni
        self.strength = initial_strength
        self.activation_count = 0
        self.success_count = 0
        self.last_activated = None
        self.learning_rate = 0.1
        self.decay_rate = 0.01

    def activate(self, success: bool = True):
        """Activate pathway and update strength based on success"""
        self.activation_count += 1
        self.last_activated = datetime.now()

        if success:
            self.success_count += 1
            self.strength = min(1.0, self.strength + self.learning_rate)
        else:
            self.strength = max(0.1, self.strength - (self.learning_rate * 0.5))

    def decay(self):
        """Apply temporal decay to pathway strength"""
        if self.last_activated:
            time_diff = datetime.now() - self.last_activated
            if time_diff.days > 7:
                self.strength = max(0.1, self.strength - self.decay_rate)

    def get_success_rate(self) -> float:
        return self.success_count / self.activation_count if self.activation_count > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'target': self.target,
            'strength': self.strength,
            'activation_count': self.activation_count,
            'success_count': self.success_count,
            'last_activated': self.last_activated.isoformat() if self.last_activated else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuralPathway':
        pathway = cls(data['source'], data['target'], data['strength'])
        pathway.activation_count = data.get('activation_count', 0)
        pathway.success_count = data.get('success_count', 0)
        if data.get('last_activated'):
            pathway.last_activated = datetime.fromisoformat(data['last_activated'])
        return pathway


class EnhancedBaseVNI:
    """Base VNI with self-learning, attention mechanisms, predictive capabilities, AND TEXT GENERATION"""
    
    def __init__(self, vni_type: str, instance_id: str):
        self.vni_type = vni_type
        self.instance_id = instance_id
        self.knowledge_base = self._load_knowledge_base()
        self.connection_weights = {}
        self.learning_history = []
        
        # Self-learning attributes
        self.conversation_memory = deque(maxlen=1000)
        self.learned_responses = {}
        self.context_memory = {}
        self.adaptation_rate = 0.3
        self.usage_threshold = 2
        self.memory_window = 50
        
        self.confidence_threshold = 0.7
        self.adaptive_learning_rate = 0.1

        # Attention mechanism
        self.attention_weights = None
        self.attention_enabled = TORCH_AVAILABLE

        # Web search
        self.web_search_enabled = WEB_SEARCH_AVAILABLE
        self.search_cache = {}
        self.max_search_cache_size = 100

        # ==================== NEW: TEXT GENERATION ====================
        self.generation_enabled = GENERATION_AVAILABLE
        self.generator = None
        self.tokenizer = None
        self.bridge_layer = None
        self.generation_config = {
            'max_new_tokens': 512,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'do_sample': True,
            'num_return_sequences': 1
        }
        self._init_generation_capability()

        # ==================== NEW: DOMAIN CLASSIFIER ====================
        self.classifier = None
        self._init_domain_classifier()
        # ===============================================================

        # Predictive systems
        if PREDICTIVE_AVAILABLE:
            self.predictive_vocab = PredictiveVocabulary()
            self.response_generator = PredictiveResponseGenerator(self.predictive_vocab)
        else:
            self.predictive_vocab = self._create_fallback_predictive_vocab()
            self.response_generator = self._create_fallback_response_generator()
            
        self.conversation_context = []
        self.neural_pathways = {}

        # ==================== COLLABORATION ATTRIBUTES ====================
        # Add collaboration attributes
        self.collaboration_queue = asyncio.Queue(maxsize=100)
        self.active_collaborations: Dict[str, CollaborationRequest] = {}
        self.collaboration_partners: Dict[str, float] = {}  # partner_id -> success_score
        self.available_capabilities = VNICapabilities()
        self._init_capabilities()
        # =================================================================

        self._initialize_default_knowledge()

    def _init_capabilities(self):
        """Initialize VNI capabilities based on domain"""
        # Set default capabilities
        self.available_capabilities.domain_knowledge = [self.vni_type]
        self.available_capabilities.abstraction_levels = ["semantic", "structural"]
        self.available_capabilities.processing_speed = 1.0
        self.available_capabilities.collaboration_score = 0.5
        self.available_capabilities.specializations = {self.vni_type}

        # Domain-specific specializations
        if self.vni_type == 'medical':
            self.available_capabilities.specializations.update({'diagnosis', 'treatment', 'symptom_analysis'})
        elif self.vni_type == 'legal':
            self.available_capabilities.specializations.update({'contract_review', 'legal_analysis', 'rights_advice'})
        elif self.vni_type == 'general':
            self.available_capabilities.specializations.update({'multi_domain', 'analysis', 'problem_solving'})

    # ==================== TEXT GENERATION INITIALIZATION ====================
    
    def _init_generation_capability(self):
        """Initialize text generation using transformers"""
        if not self.generation_enabled:
            logger.warning(f"⚠️  Generation not available for {self.instance_id}")
            return
        
        try:
            # Use DialoGPT for conversational generation
            model_name = "microsoft/DialoGPT-medium"
            logger.info(f"🔄 Loading generation model: {model_name}")
        
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.generator = AutoModelForCausalLM.from_pretrained(model_name)
        
            # DEBUG: Check model configuration
            logger.info(f"DEBUG: Model hidden size: {self.generator.config.hidden_size}")
            logger.info(f"DEBUG: Model vocab size: {self.generator.config.vocab_size}")
            logger.info(f"DEBUG: Model device: {self.generator.device}")
        
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
            # Bridge layer to connect knowledge abstractions to generator
            if TORCH_AVAILABLE:
                self.bridge_layer = nn.Sequential(
                    nn.Linear(512, 1024),  # Abstraction dimension to intermediate
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(1024, self.generator.config.hidden_size)  # To model dimension
                )
                # Move bridge layer to the same device as the generator
                self.bridge_layer.to(self.generator.device)
                logger.info(f"✅ Bridge layer initialized: 512 -> {self.generator.config.hidden_size} on device {self.generator.device}")
        
            logger.info(f"✅ Generation capability initialized for {self.instance_id}")
        
        except Exception as e:
            logger.error(f"❌ Generation initialization failed: {e}")
            self.generation_enabled = False
            self.generator = None
            self.tokenizer = None
            self.bridge_layer = None

    def configure_generation(self, **kwargs):
        """Configure generation parameters"""
        self.generation_config.update(kwargs)
        logger.info(f"Generation config updated: {self.generation_config}")

    # ==================== TEXT GENERATION METHODS ====================

    def generate_with_transformer(self, query: str, context: Dict = None, 
                                  use_bridge: bool = True) -> str:
        """Generate response using transformer model with domain knowledge"""
        if not self.generation_enabled or self.generator is None:
            logger.warning(f"Generation not available, using fallback")
            return self._generate_fallback_response(query, context)
        
        try:
            # Strategy 1: Use bridge layer with knowledge abstractions
            if use_bridge and TORCH_AVAILABLE and self.bridge_layer is not None:
                return self._generate_with_knowledge_bridge(query, context)
            
            # Strategy 2: Direct generation with domain-specific prompt
            else:
                return self._generate_with_domain_prompt(query, context)
                
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return self._generate_fallback_response(query, context)

    def _generate_with_knowledge_bridge(self, query: str, context: Dict) -> str:
        """Generate using bridge layer connecting knowledge to transformer"""
        # Create knowledge abstraction
        abstraction = self._create_knowledge_abstraction(query, context)
        abstraction_tensor = self._extract_abstraction_tensor(abstraction)

        # DEBUG: Check tensor device
        logger.info(f"DEBUG: Abstraction tensor shape: {abstraction_tensor.shape}")
    
        # Move abstraction tensor to the same device as the generator
        if self.generator is not None:
            device = self.generator.device
            abstraction_tensor = abstraction_tensor.to(device)

        # Convert abstraction to generator input via bridge
        with torch.no_grad():
            bridge_output = self.bridge_layer(abstraction_tensor)

            # DEBUG: Check shapes
            logger.info(f"DEBUG: Bridge output shape: {bridge_output.shape}")
        
            # The bridge output should be (hidden_size), we need to add batch dimension
            # and sequence dimension: (batch_size=1, seq_len=1, hidden_size)
            if len(bridge_output.shape) == 1:
                bridge_output = bridge_output.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size)
            elif len(bridge_output.shape) == 2:
                bridge_output = bridge_output.unsqueeze(0)  # (batch, seq, hidden) -> (1, batch, hidden)?
        
            logger.info(f"DEBUG: Bridge output after reshape: {bridge_output.shape}")            

            # Generate text using transformer
            generated = self.generator.generate(
                inputs_embeds=bridge_output,
                max_new_tokens=self.generation_config['max_new_tokens'],
                temperature=self.generation_config['temperature'],
                top_p=self.generation_config['top_p'],
                top_k=self.generation_config['top_k'],
                do_sample=self.generation_config['do_sample'],
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            
            # Post-process response
            response = self._post_process_generated_response(response, query, context)
            
            return response if response else self._generate_fallback_response(query, context)

    def _generate_with_domain_prompt(self, query: str, context: Dict) -> str:
        """Generate with domain-specific prompting"""
        # Build domain-aware prompt
        domain_prompt = self._build_domain_prompt(query, context)
        
        # Encode and generate - FIXED VERSION
        inputs = self.tokenizer(
            domain_prompt, 
            return_tensors="pt"
        )
        # Check for empty input
        if inputs.input_ids.shape[1] == 0:
            logger.error("Empty input from tokenizer!")
            return self._generate_fallback_response(query, context)
        
        with torch.no_grad():
            generated = self.generator.generate(
                inputs,
                max_new_tokens=self.generation_config['max_new_tokens'],
                temperature=self.generation_config['temperature'],
                top_p=self.generation_config['top_p'],
                top_k=self.generation_config['top_k'],
                do_sample=self.generation_config['do_sample'],
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Remove the prompt from response
        if domain_prompt in response:
            response = response.replace(domain_prompt, "").strip()
        
        return self._post_process_generated_response(response, query, context)

    def _build_domain_prompt(self, query: str, context: Dict) -> str:
        """Build domain-specific prompt for generation"""
        # Get domain knowledge
        concepts, patterns = self.extract_concepts_and_patterns(query)
        
        # Build context-aware prompt
        prompt_parts = []
        
        # Add domain context
        if self.vni_type == 'medical':
            prompt_parts.append("As a medical AI assistant:")
        elif self.vni_type == 'legal':
            prompt_parts.append("As a legal AI assistant:")
        else:
            prompt_parts.append("As a helpful AI assistant:")
        
        # Add relevant knowledge
        if concepts:
            prompt_parts.append(f"Regarding {', '.join(concepts[:3])}:")
        
        # Add query
        prompt_parts.append(query)
        
        return " ".join(prompt_parts)

    def _create_knowledge_abstraction(self, query: str, context: Dict) -> Dict[str, Any]:
        """Create knowledge abstraction for bridge layer"""
        concepts, patterns = self.extract_concepts_and_patterns(query)
        current_context = self.get_current_context(context.get('session_id', 'default') if context else 'default')
        
        return {
            'modality': 'text',
            'abstraction_levels': {
                'semantic': {
                    'tensor': self._create_semantic_tensor(query, concepts),
                    'concepts': concepts,
                    'intent': self._predict_user_intent(query, []),
                    'domain': self.vni_type,
                    'sentiment': self.detect_emotional_tone(query)
                },
                'structural': {
                    'tensor': self._create_structural_tensor(query, patterns),
                    'patterns': patterns,
                    'complexity': self.assess_complexity(query),
                    'query_type': self._classify_query_type(query)
                }
            },
            'context': current_context,
            'confidence': self.calculate_confidence(concepts, patterns)
        }

    def _create_semantic_tensor(self, query: str, concepts: List[str]) -> torch.Tensor:
        """Create semantic representation tensor"""
        if not TORCH_AVAILABLE:
            return torch.randn(256)
        
        # Simple embedding: word count, concept presence, domain relevance
        features = []
        
        # Basic features
        features.append(len(query.split()) / 100.0)  # Normalized word count
        features.append(len(concepts) / 10.0)  # Normalized concept count
        
        # Domain-specific features
        domain_keywords = self.get_default_concepts().keys()
        domain_match = sum(1 for kw in domain_keywords if kw in query.lower()) / max(len(domain_keywords), 1)
        features.append(domain_match)
        
        # Pad to 256 dimensions
        while len(features) < 256:
            features.append(0.0)
        
        return torch.tensor(features[:256], dtype=torch.float32)

    def _create_structural_tensor(self, query: str, patterns: List[str]) -> torch.Tensor:
        """Create structural representation tensor"""
        if not TORCH_AVAILABLE:
            return torch.randn(256)
        
        features = []
        
        # Structural features
        features.append(1.0 if '?' in query else 0.0)  # Is question
        features.append(len(patterns) / 5.0)  # Normalized pattern count
        features.append(self.assess_complexity(query))  # Complexity score
        
        # Sentiment features
        features.append(1.0 if self.detect_urgency(query) == 'high' else 0.0)
        
        # Pad to 256 dimensions
        while len(features) < 256:
            features.append(0.0)
        
        return torch.tensor(features[:256], dtype=torch.float32)

    def _extract_abstraction_tensor(self, abstraction: Dict) -> torch.Tensor:
        """Extract and combine abstraction tensors"""
        if not TORCH_AVAILABLE:
            return torch.randn(512)
        
        semantic = abstraction['abstraction_levels']['semantic']['tensor']
        structural = abstraction['abstraction_levels']['structural']['tensor']
        
        # Combine semantic and structural representations
        return torch.cat([semantic, structural], dim=0)

    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query"""
        q = query.lower()
        
        if any(w in q for w in ['what', 'which', 'who']):
            return 'factual'
        elif any(w in q for w in ['how', 'can', 'could']):
            return 'procedural'
        elif any(w in q for w in ['why', 'explain']):
            return 'explanatory'
        elif any(w in q for w in ['should', 'recommend', 'advice']):
            return 'advisory'
        else:
            return 'general'

    def _post_process_generated_response(self, response: str, query: str, context: Dict) -> str:
        """Post-process generated response"""
        if not response or not response.strip():
            return self._generate_fallback_response(query, context)
        
        # Remove query repetition
        if query.lower() in response.lower():
            response = response.replace(query, "").strip()
        
        # Clean up artifacts
        response = response.strip()
        response = re.sub(r'\s+', ' ', response)  # Normalize whitespace
        
        # Add domain-specific disclaimers if needed
        if self.vni_type == 'medical' and len(response) > 50:
            if 'consult' not in response.lower() and random.random() > 0.7:
                response += " Please consult a healthcare professional for personalized advice."
        elif self.vni_type == 'legal' and len(response) > 50:
            if 'attorney' not in response.lower() and random.random() > 0.7:
                response += " Consider consulting with a qualified attorney for your specific situation."
        
        return response

    # ==================== PRETRAINING METHODS ====================
    
    def update_knowledge_base(self, data: dict):
        """Update knowledge base with pretraining data"""
        logger.info(f"📄 Updating knowledge base for {self.instance_id} with {len(data)} items")
        
        # Handle different data structures
        if isinstance(data, dict):
            # Update concepts
            if 'concepts' in data:
                for concept, concept_data in data['concepts'].items():
                    if concept not in self.knowledge_base['concepts']:
                        self.knowledge_base['concepts'][concept] = {
                            'strength': 0.8,
                            'usage_count': 0,
                            'pretrained': True,
                            'first_seen': datetime.now().isoformat(),
                            'last_used': datetime.now().isoformat()
                        }
                    else:
                        self.knowledge_base['concepts'][concept]['strength'] = min(
                            1.0, self.knowledge_base['concepts'][concept]['strength'] + 0.1
                        )
                        self.knowledge_base['concepts'][concept]['pretrained'] = True
            
            # Update patterns
            if 'patterns' in data:
                for pattern_id, pattern_data in data['patterns'].items():
                    if pattern_id not in self.knowledge_base['patterns']:
                        self.knowledge_base['patterns'][pattern_id] = pattern_data
                        self.knowledge_base['patterns'][pattern_id]['pretrained'] = True
                    else:
                        existing = self.knowledge_base['patterns'][pattern_id]
                        existing['strength'] = min(1.0, existing['strength'] + 0.1)
                        existing['pretrained'] = True
            
            # Update response templates
            if 'response_templates' in data:
                for template_type, templates in data['response_templates'].items():
                    if template_type not in self.knowledge_base['response_templates']:
                        self.knowledge_base['response_templates'][template_type] = templates
                    else:
                        self.knowledge_base['response_templates'][template_type].extend(templates)
        
        # Mark as updated and save
        self.knowledge_base['metadata']['last_updated'] = datetime.now().isoformat()
        self.knowledge_base['metadata']['pretrained'] = True
        self.save_knowledge_base()
        
        logger.info(f"✅ Knowledge base updated for {self.instance_id}")

    def get_knowledge_stats(self):
        """Get knowledge base statistics"""
        stats = {
            "concepts_count": len(self.knowledge_base.get("concepts", {})),
            "patterns_count": len(self.knowledge_base.get("patterns", {})),
            "response_templates_count": sum(len(templates) for templates in self.knowledge_base.get("response_templates", {}).values()),
            "learned_responses_count": len(self.learned_responses),
            "last_updated": self.knowledge_base.get('metadata', {}).get('last_updated', 'unknown'),
            "pretrained": self.knowledge_base.get('metadata', {}).get('pretrained', False),
            "instance_id": self.instance_id,
            "vni_type": self.vni_type,
            "generation_enabled": self.generation_enabled
        }
        
        # Count pretrained vs learned concepts
        concepts = self.knowledge_base.get("concepts", {})
        pretrained_count = sum(1 for c in concepts.values() if c.get('pretrained', False))
        stats["pretrained_concepts_count"] = pretrained_count
        stats["learned_concepts_count"] = stats["concepts_count"] - pretrained_count
        
        return stats

    # ==================== FALLBACK IMPLEMENTATIONS ====================

    def _create_fallback_predictive_vocab(self):
        """Fallback predictive vocabulary when module unavailable"""
        class FallbackPredictiveVocabulary:
            def __init__(self):
                self.vocabulary = defaultdict(int)
                self.context_patterns = defaultdict(list)
                self.domain_patterns = defaultdict(list)
                
            def update_vocabulary(self, text: str, domain: str):
                words = text.lower().split()
                for word in words:
                    if len(word) > 3:
                        self.vocabulary[word] += 1
                        self.domain_patterns[domain].append(word)
                        
            def get_predictive_suggestions(self, text: str, max_suggestions: int = 3) -> List[str]:
                words = text.lower().split()
                if not words:
                    return []
                last_word = words[-1]
                suggestions = [w for w in self.vocabulary if w.startswith(last_word) and w != last_word]
                return suggestions[:max_suggestions]
                
            def get_predictive_completions(self, text: str, max_completions: int = 3) -> List[str]:
                return self.get_predictive_suggestions(text, max_completions)
                
            def get_expansions(self, topic: str, domain: str) -> List[str]:
                related = [(w, c) for w, c in self.vocabulary.items() if w != topic and len(w) > 4]
                related.sort(key=lambda x: x[1], reverse=True)
                return [w for w, c in related[:5]]
                
            def get_guidance_patterns(self, domain: str) -> List[str]:
                patterns = {
                    'medical': ["assess symptoms carefully", "consider medical history", "consult healthcare provider"],
                    'legal': ["review relevant statutes", "document all evidence", "seek legal counsel"],
                    'general': ["analyze the situation", "consider all factors", "break down systematically"]
                }
                return patterns.get(domain, patterns['general'])
        
        return FallbackPredictiveVocabulary()

    def _create_fallback_response_generator(self):
        """Fallback response generator when module unavailable"""
        parent = self
        
        class FallbackResponseGenerator:
            def generate_response(self, query: str, vni_type: str, context=None, predictive_suggestions=None):
                base = {
                    "medical": "From a medical perspective, this requires careful symptom evaluation.",
                    "legal": "Legal matters should be reviewed by qualified professionals.",
                    "general": "I can help analyze this from multiple perspectives."
                }
                response = base.get(vni_type, "I'll help you work through this.")
                if predictive_suggestions:
                    response += f" Related: {', '.join(predictive_suggestions[:2])}."
                return {"response": response, "confidence": 0.6}
        
        return FallbackResponseGenerator()

    # ==================== ATTENTION MECHANISM ====================

    def integrate_attention_weights(self, attention_weights):
        """Integrate attention weights for improved response generation"""
        if TORCH_AVAILABLE and attention_weights is not None:
            self.attention_weights = attention_weights
            logger.debug(f"{self.instance_id} integrated attention weights")

    # ==================== DOMAIN CLASSIFIER METHODS ====================

    def _init_domain_classifier(self):
        """Initialize domain classifier - override in subclasses"""
        # Default empty classifier
        class DefaultClassifier:
            def __init__(self):
                self.keywords = []
            
            def predict(self, texts):
                """Return 0 for all texts (not this domain)"""
                return [0] * len(texts)
            
            def predict_proba(self, texts):
                """Return [1.0, 0.0] for all texts (100% not this domain)"""
                return [[1.0, 0.0]] * len(texts)
        
        self.classifier = DefaultClassifier()
        logger.info(f"Default classifier initialized for {self.instance_id}")
    
    def should_handle(self, query: str) -> bool:
        """Determine if this VNI should handle the query"""
        if self.classifier is not None:
            try:
                prediction = self.classifier.predict([query])
                return prediction[0] == 1 if len(prediction) > 0 else False
            except Exception as e:
                logger.error(f"Classifier error in {self.instance_id}: {e}")
        
        # Fallback: check if query contains domain-specific keywords
        return self._fallback_domain_check(query)
    
    def _fallback_domain_check(self, query: str) -> bool:
        """Fallback domain check - override in subclasses"""
        return False
    
    def is_initialized(self) -> bool:
        """Check if VNI is fully initialized"""
        return hasattr(self, 'generation_enabled') and self.generation_enabled


    def process_query_with_attention(self, query: str, context: Dict, attention_scores: Dict) -> Dict[str, Any]:
        """Process query with attention-guided context"""
        self_attention_score = attention_scores.get(self.instance_id, 0.5)
        
        # Low attention = minimal participation
        if self_attention_score < 0.3:
            return {
                "response": "",
                "confidence": 0.1,
                "vni_instance": self.instance_id,
                "attention_score": self_attention_score,
                "active": False
            }
        
        # Process with attention weighting
        base_response = self.process_query(query, context)
        base_response["attention_score"] = self_attention_score
        base_response["confidence"] = base_response.get("confidence", 0.5) * self_attention_score
        base_response["active"] = True
        
        return base_response

    # ==================== CORE QUERY PROCESSING ====================

    def process_query(self, query: str, context: Dict = None, session_id: str = "default",
                     use_generation: bool = None, generation_strategy: str = "auto") -> Dict[str, Any]:
        """
        Main query processing with generation support
        
        Args:
            query: User query
            context: Additional context
            session_id: Session identifier
            use_generation: Force generation on/off (None = auto-decide)
            generation_strategy: "bridge", "prompt", or "auto"
        """
        
        # Update systems
        self.update_context_memory(session_id, query, context)
        
        # Update vocabulary
        if hasattr(self.predictive_vocab, 'update_vocabulary'):
            self.predictive_vocab.update_vocabulary(query, self.vni_type)
        else:
            words = query.lower().split()
            for word in words:
                if len(word) > 2 and hasattr(self.predictive_vocab, 'learn_word'):
                    self.predictive_vocab.learn_word(word, {'domain': self.vni_type}, self.vni_type)
        
        # Get context and predictions
        current_context = self.get_current_context(session_id)
        predictive_suggestions = []
        if hasattr(self.predictive_vocab, 'get_predictive_suggestions'):
            predictive_suggestions = self.predictive_vocab.get_predictive_suggestions(query)
        
        # Extract concepts and patterns
        concepts, patterns = self.extract_concepts_and_patterns(query)
        confidence = self.calculate_confidence(concepts, patterns)
        
        # ==================== DECISION: USE GENERATION OR NOT ====================
        should_generate = self._should_use_generation(
            query, concepts, patterns, confidence, use_generation, current_context
        )
        
        response_text = ""
        generation_used = False
        
        if should_generate:
            # Try generation
            try:
                if generation_strategy == "auto":
                    # Use bridge if we have good knowledge, otherwise use prompt
                    use_bridge = confidence > 0.5 and len(concepts) > 0
                else:
                    use_bridge = generation_strategy == "bridge"
                
                response_text = self.generate_with_transformer(query, current_context, use_bridge)
                if response_text and response_text.strip():                
                    generation_used = True
                    logger.info(f"✅ Used generation ({'bridge' if use_bridge else 'prompt'}) for query")
                
                else:
                    # Generation returned empty string
                    response_text = ""
                    logger.warning("Generation returned empty string, falling back")            
                
            except Exception as e:
                logger.warning(f"Generation failed, falling back: {e}")
                response_text = ""
        
        # Fallback to template/pattern-based response
        if not response_text or not response_text.strip():
            # Try learned responses first
            learned = self.get_learned_response(query, current_context)
            if learned and learned['usage_count'] >= self.usage_threshold:
                response_text = learned['response']
                patterns_with_learned = patterns + ["learned_response"]
                patterns = patterns_with_learned
            else:
                # Generate using templates and patterns
                response_text = self.generate_truly_predictive_response(
                    query, current_context, predictive_suggestions
                )
                self.learn_from_conversation(query, response_text, current_context, session_id)
        
        # Ensure we have a response
        if not response_text or response_text.strip() == "":
            response_text = self._generate_fallback_response(query, current_context)
            if confidence < 0.5:
                confidence = 0.5

        # Web search enhancement
        web_search_used = False
        if self.web_search_enabled and self.needs_web_search(query, confidence, concepts):
            response_text = self.enhance_with_web_search(query, response_text, concepts, confidence)
            web_search_used = True

        # Remember conversation
        self.remember_conversation(session_id, query, response_text, current_context)
        
        # Build response data
        response_data = {
            'response': response_text,
            'confidence': confidence,
            'vni_instance': self.instance_id,
            'concepts_used': concepts,
            'patterns_matched': patterns,
            'response_type': f'{self.vni_type}_{"generated" if generation_used else "templated"}',
            'context_used': bool(current_context),
            'web_search_used': web_search_used,
            'generation_used': generation_used,
            'generation_available': self.generation_enabled,
            'timestamp': datetime.now().isoformat()
        }

        # Update learning history
        self.learning_history.append({
            'query': query, 
            'response': response_data,
            'timestamp': datetime.now().isoformat(),
            'context': current_context, 
            'session_id': session_id,
            'generation_used': generation_used
        })

        # Periodic save
        if len(self.learning_history) % 10 == 0:
            self.save_knowledge_base()

        return response_data

    def _should_use_generation(self, query: str, concepts: List[str], patterns: List[str],
                               confidence: float, force_generation: Optional[bool],
                               context: Dict) -> bool:
        """Decide whether to use generation or template-based response"""
        
        # Forced decision
        if force_generation is not None:
            return force_generation and self.generation_enabled
        
        # Generation not available
        if not self.generation_enabled:
            return False
        
        # Use generation for:
        # 1. Novel queries (low confidence, few patterns)
        # 2. Complex queries
        # 3. Queries requiring creative responses
        
        novelty_score = 1.0 - confidence
        complexity = self.assess_complexity(query)
        
        # Check if we have good template matches
        has_good_templates = len(patterns) >= 2 or confidence > 0.7
        
        # Generation decision criteria
        use_generation = (
            novelty_score > 0.6 or  # Novel query
            complexity > 0.7 or      # Complex query
            (novelty_score > 0.4 and not has_good_templates)  # Moderate novelty, no templates
        )
        
        return use_generation

    # ==================== PREDICTIVE RESPONSE GENERATION ====================

    def generate_truly_predictive_response(self, query: str, context: Dict, predictive_suggestions: List[str] = None) -> str:
        """Generate response using learned patterns and predictions"""
        
        # Get completions and analyze trend
        completions = []
        if hasattr(self.predictive_vocab, 'get_predictive_completions'):
            completions = self.predictive_vocab.get_predictive_completions(query)
        
        conversation_trend = self._analyze_conversation_trend(context)
        
        # Predict intent and generate
        response = ""
        if completions:
            predicted_intent = self._predict_user_intent(query, completions)
            response = self._generate_from_predicted_intent(predicted_intent, context)
        else:
            concepts, patterns = self.extract_concepts_and_patterns(query)
            response = self.generate_adaptive_response(query, concepts, patterns, context)

        # Ensure we always return a response
        if not response or response.strip() == "":
            response = self._generate_fallback_response(query, context)
        return response
    
    def _generate_fallback_response(self, query: str, context: Dict) -> str:
        """Generate a fallback response when no other response is generated"""
        input_lower = query.lower()
    
        if any(greet in input_lower for greet in ["hello", "hi", "hey", "greetings"]):
            return f"Hello! I'm {self.instance_id}, a specialized {self.vni_type} VNI. How can I help you today?"
    
        if "your name" in input_lower or "who are you" in input_lower:
            return f"My name is {self.instance_id}. I'm a specialized VNI for {self.vni_type} domain."
    
        # Domain-specific fallbacks
        fallbacks = {
            "medical": "From a medical perspective, I'd be happy to help. Could you provide more details about your health concern?",
            "legal": "From a legal standpoint, I can help analyze your situation. Please share more context.",
            "general": "I'd be happy to help analyze this. Could you provide more specific details?"
        }
    
        return fallbacks.get(self.vni_type, "I'm here to help. Could you provide more information?")

    def _analyze_conversation_trend(self, context: Dict) -> Dict[str, Any]:
        """Analyze conversation patterns"""
        recent = context.get('recent_conversations', [])
        if not recent:
            return {'trend': 'new_conversation', 'topic_consistency': 0.5}
        
        topics = [c.get('context_notes', {}).get('query_complexity', 'medium') for c in recent[-3:]]
        user_style = context.get('user_style', {})
        
        return {
            'trend': 'stable',
            'topic_consistency': len(set(topics)) / len(topics) if topics else 0.5,
            'conversation_depth': len(recent),
            'user_engagement': user_style.get('detail_level', 'medium')
        }

    def _predict_user_intent(self, query: str, completions: List[str]) -> str:
        """Predict user's underlying intent"""
        q = query.lower()
        
        # Check completions
        for comp in completions:
            if comp in q:
                return f"information_request_{comp}"
        
        # Intent classification
        if any(w in q for w in ['how', 'can i', 'what is']):
            return "how_to_question"
        elif any(w in q for w in ['why', 'explain']):
            return "explanation_request"
        elif any(w in q for w in ['help', 'problem', 'issue']):
            return "help_request"
        return "general_inquiry"

    def _generate_from_predicted_intent(self, intent: str, context: Dict) -> str:
        """Generate response based on predicted intent"""
        
        # Check learned responses
        intent_pattern = f"intent_{intent}"
        if intent_pattern in self.learned_responses:
            learned = self.learned_responses[intent_pattern]
            if learned['usage_count'] >= self.usage_threshold:
                return learned['response']
        
        # Generate by intent type
        if intent.startswith("information_request_"):
            topic = intent.replace("information_request_", "")
            return self._generate_informative_response(topic, context)
        elif intent == "how_to_question":
            return self._generate_guidance_response(context)
        elif intent == "explanation_request":
            return self._generate_explanation_response(context)
        elif intent == "help_request":
            return self._generate_help_response(context)
        else:
            return self._generate_general_response(context)

    def _generate_informative_response(self, topic: str, context: Dict) -> str:
        """Generate informative response using knowledge base"""
        expansions = self.predictive_vocab.get_expansions(topic, self.vni_type)
        
        if topic in self.knowledge_base.get('concepts', {}):
            concept_data = self.knowledge_base['concepts'][topic]
            strength = concept_data.get('strength', 0.5)
            
            if strength > 0.8:
                return f"I have strong knowledge about {topic}. {self._get_detailed_explanation(topic, expansions)}"
            elif strength > 0.6:
                return f"Regarding {topic}, {self._get_general_explanation(topic, expansions)}"
            else:
                return f"I'm still learning about {topic}, but: {self._get_basic_explanation(topic)}"
        else:
            if expansions:
                return f"Based on patterns, {topic} typically involves: {', '.join(expansions[:3])}."
            return f"{topic} requires specific expertise. Could you provide more context?"

    def _generate_guidance_response(self, context: Dict) -> str:
        """Generate step-by-step guidance"""
        recent_topics = context.get('topics_discussed', [])
        
        if 'help_seeking' in recent_topics:
            return "I notice you've been seeking help. Let me provide clear, step-by-step guidance."
        
        patterns = self.predictive_vocab.get_guidance_patterns(self.vni_type)
        if patterns:
            return f"Let me guide you: {patterns[0]}"
        return "I'll help you work through this systematically with careful planning."

    def _generate_explanation_response(self, context: Dict) -> str:
        """Generate explanation based on user style"""
        user_style = context.get('user_style', {})
        detail_level = user_style.get('detail_level', 'medium')
        
        if detail_level == 'high':
            return self._get_comprehensive_explanation()
        elif detail_level == 'low':
            return self._get_concise_explanation()
        return self._get_balanced_explanation()

    def _generate_help_response(self, context: Dict) -> str:
        """Generate helpful response based on context"""
        conv_length = context.get('conversation_length', 0)
        
        if conv_length > 10:
            return "Thank you for continuing our discussion. Let me provide focused assistance."
        return "I'm here to help! Please share more about your specific situation."

    def _generate_general_response(self, context: Dict) -> str:
        """Generate general response using successful patterns"""
        successful = [p for p in self.learned_responses.values() if p.get('success_rate', 0) > 0.7]
        
        if successful:
            best = max(successful, key=lambda x: x.get('success_rate', 0))
            return best['response']
        return "I'd be happy to help. Could you provide a bit more context?"

    # ==================== EXPLANATION HELPERS ====================

    def _get_detailed_explanation(self, topic: str, expansions: List[str]) -> str:
        if expansions:
            return f"Key aspects include {', '.join(expansions[:4])}. Want me to elaborate on any?"
        return "This is well-established with clear principles tailored to specific needs."

    def _get_general_explanation(self, topic: str, expansions: List[str]) -> str:
        if expansions:
            return f"this typically involves {expansions[0]} and requires context assessment."
        return "understanding fundamental principles and their practical applications is key."

    def _get_basic_explanation(self, topic: str) -> str:
        return "this involves important considerations that should be approached with proper understanding."

    def _get_comprehensive_explanation(self) -> str:
        return "Let me provide a comprehensive explanation covering fundamentals, applications, considerations, and variations."

    def _get_concise_explanation(self) -> str:
        return "Key points: core principles, practical application, and context-specific considerations."

    def _get_balanced_explanation(self) -> str:
        return "This involves understanding core concepts, implementation, and scenario-specific adjustments."

    # ==================== ADAPTIVE RESPONSE GENERATION ====================

    def generate_adaptive_response(self, query: str, concepts: List[str], patterns: List[str], context: Dict) -> str:
        """Generate responses that adapt based on context"""
        response = ""
        if patterns and random.random() > 0.3:
            pattern_id = patterns[0]
            if pattern_id in self.knowledge_base.get("patterns", {}):
                pattern_data = self.knowledge_base["patterns"][pattern_id]
                base_responses = pattern_data.get("responses", [])
                if base_responses:
                    base = random.choice(base_responses)
                    response = self.personalize_response(base, query, context)
    
        # If no pattern response was generated, try contextual response
        if not response or response.strip() == "":
            response = self.create_contextual_response(query, concepts, context)
    
        return response

    def personalize_response(self, base: str, query: str, context: Dict) -> str:
        """Personalize response based on context"""
        personalized = base
        
        # Add contextual references
        if 'help_seeking' in context.get('topics_discussed', []):
            personalized += " I'm here to help with any other questions."
        
        # Check for follow-up
        recent = context.get('recent_conversations', [])
        if len(recent) > 1 and any(w in query.lower() for w in ['more', 'else', 'another', 'also']):
            personalized = "Additionally, " + personalized.lower()
        
        # Adjust for user style
        user_style = context.get('user_style', {})
        if user_style.get('detail_level') == 'low' and len(personalized.split()) > 25:
            sentences = personalized.split('.')
            personalized = sentences[0] + '.' if sentences else personalized
        
        return personalized

    def create_contextual_response(self, query: str, concepts: List[str], context: Dict) -> str:
        """Create dynamic response from concepts and context"""
        if concepts:
            concept = concepts[0]
            if concept in self.knowledge_base["concepts"]:
                strength = self.knowledge_base["concepts"][concept].get('strength', 0.5)
                if strength > 0.7:
                    return self.create_confident_response(concept, query, context)
                return self.create_learning_response(concept, query, context)
        
        return self.create_contextual_fallback(query, context)

    def create_confident_response(self, concept: str, query: str, context: Dict) -> str:
        base = f"Based on established {self.vni_type} knowledge about {concept}, "
        if context.get('user_style', {}).get('detail_level') == 'high':
            base += f"this involves comprehensive understanding. Want more details?"
        else:
            base += "the approach focuses on practical application and key considerations."
        return base

    def create_learning_response(self, concept: str, query: str, context: Dict) -> str:
        return f"I'm developing understanding of {concept} in {self.vni_type} contexts. Based on what I've learned, this involves important considerations."

    def create_contextual_fallback(self, query: str, context: Dict) -> str:
        conv_length = context.get('conversation_length', 0)
        if conv_length > 5:
            return f"As we continue discussing {self.vni_type} topics, could you provide more specific details?"
        return f"As a {self.vni_type} AI, I'm here to help. Could you provide more details?"

    # ==================== KNOWLEDGE BASE MANAGEMENT ====================

    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load knowledge base from file"""
        import os
        
        knowledge_file = f"knowledge_{self.vni_type}_{self.instance_id}.json"
        default_knowledge = {
            "concepts": {},
            "patterns": {},
            "corrections": {},
            "response_templates": self.get_default_response_templates(),
            "learned_responses": {},
            "metadata": {
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "3.0",
                "generation_enabled": GENERATION_AVAILABLE
            }
        }
        
        try:
            if os.path.exists(knowledge_file):
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    for key, value in default_knowledge.items():
                        if key not in loaded:
                            loaded[key] = value
                    self.learned_responses = loaded.get('learned_responses', {})
                    return loaded
            else:
                with open(knowledge_file, 'w', encoding='utf-8') as f:
                    json.dump(default_knowledge, f, indent=2, ensure_ascii=False)
                logger.info(f"Created knowledge base: {knowledge_file}")
                return default_knowledge
        except Exception as e:
            logger.warning(f"Failed to load knowledge base: {e}")
            return default_knowledge

    def _initialize_default_knowledge(self):
        """Initialize with domain-specific defaults"""
        if not self.knowledge_base["concepts"]:
            self.knowledge_base["concepts"].update(self.get_default_concepts())
        if not self.knowledge_base["patterns"]:
            self.knowledge_base["patterns"].update(self.get_default_patterns())
        self.save_knowledge_base()

    def save_knowledge_base(self):
        """Save knowledge base to file"""
        filename = f"knowledge_{self.vni_type}_{self.instance_id}.json"
        self.knowledge_base['learned_responses'] = self.learned_responses
        self.knowledge_base['metadata']['last_updated'] = datetime.now().isoformat()
        self.knowledge_base['metadata']['generation_enabled'] = self.generation_enabled
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")

    def get_default_concepts(self) -> Dict[str, Any]:
        """Override in subclasses"""
        return {"help": {"strength": 0.8, "usage_count": 0}, "information": {"strength": 0.7, "usage_count": 0}}

    def get_default_patterns(self) -> Dict[str, Any]:
        """Override in subclasses"""
        return {"general_help": {"triggers": ["help", "assist"], "responses": ["I'm here to help."], "strength": 0.7, "usage_count": 0}}

    def get_default_response_templates(self) -> Dict[str, List[str]]:
        """Override in subclasses"""
        return {"general": ["I understand this is a {vni_type} question. Could you provide more details?"]}

    # ==================== CONTEXT MEMORY SYSTEM ====================

    def update_context_memory(self, session_id: str, query: str, context: Dict = None):
        if session_id not in self.context_memory:
            self.context_memory[session_id] = {
                'conversation_history': deque(maxlen=self.memory_window),
                'user_preferences': {},
                'topics_discussed': [],
                'last_updated': datetime.now().isoformat(),
                'session_start': datetime.now().isoformat()
            }
        
        context_notes = self.extract_context_notes(query, context)
        self.context_memory[session_id]['conversation_history'].append({
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'context_notes': context_notes
        })
        self._update_session_topics(session_id, query, context_notes)
        self.context_memory[session_id]['last_updated'] = datetime.now().isoformat()

    def get_current_context(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.context_memory:
            return {}
        
        ctx = self.context_memory[session_id]
        recent = list(ctx['conversation_history'])[-5:]
        
        return {
            'session_id': session_id,
            'recent_conversations': recent,
            'topics_discussed': ctx['topics_discussed'],
            'user_style': self.analyze_user_style(recent),
            'conversation_length': len(ctx['conversation_history']),
            'session_duration': self._get_session_duration(session_id)
        }

    def _update_session_topics(self, session_id: str, query: str, context_notes: Dict):
        new_topics = []
        if context_notes.get('query_complexity') == 'high':
            new_topics.append('complex_queries')
        if context_notes.get('emotional_tone') == 'urgent':
            new_topics.append('urgent_requests')
        if 'help' in query.lower():
            new_topics.append('help_seeking')
        
        current = set(self.context_memory[session_id]['topics_discussed'])
        current.update(new_topics)
        self.context_memory[session_id]['topics_discussed'] = list(current)

    def _get_session_duration(self, session_id: str) -> float:
        if session_id not in self.context_memory:
            return 0.0
        start = datetime.fromisoformat(self.context_memory[session_id]['session_start'])
        return (datetime.now() - start).total_seconds() / 60

    # ==================== SELF-LEARNING SYSTEM ====================

    def get_learned_response(self, query: str, context: Dict) -> Optional[Dict[str, Any]]:
        pattern = self.extract_query_pattern(query)
        
        if pattern in self.learned_responses:
            learned = self.learned_responses[pattern]
            if self.is_context_similar(learned.get('context', {}), context):
                return learned
        
        for p, learned in self.learned_responses.items():
            if self.are_patterns_similar(pattern, p) and self.is_context_similar(learned.get('context', {}), context):
                return learned
        return None

    def learn_from_conversation(self, query: str, response: str, context: Dict, session_id: str):
        pattern = self.extract_query_pattern(query)
        
        if pattern in self.learned_responses:
            self.learned_responses[pattern]['usage_count'] += 1
            self.learned_responses[pattern]['last_used'] = datetime.now().isoformat()
            self.learned_responses[pattern]['success_rate'] = min(1.0, 
                self.learned_responses[pattern].get('success_rate', 0.7) + 0.05)
            if random.random() < self.adaptation_rate:
                self.refine_response(pattern, response, context)
        else:
            self.learned_responses[pattern] = {
                'response': response,
                'usage_count': 1,
                'success_rate': 0.7,
                'last_used': datetime.now().isoformat(),
                'created': datetime.now().isoformat(),
                'context': context,
                'session_id': session_id,
                'source_type': 'conversation_learning'
            }
        
        self.learn_concepts_from_text(query + " " + response)

    def extract_query_pattern(self, query: str) -> str:
        words = query.lower().split()
        stop_words = {'i', 'you', 'the', 'a', 'an', 'is', 'are', 'what', 'how', 'my', 'me', 'can'}
        meaningful = [w for w in words if w not in stop_words and len(w) > 2]
        return " ".join(meaningful[:4]) if meaningful else query.lower()[:20]

    def are_patterns_similar(self, p1: str, p2: str, threshold: float = 0.6) -> bool:
        w1, w2 = set(p1.split()), set(p2.split())
        if not w1 or not w2:
            return False
        return len(w1 & w2) / max(len(w1), len(w2)) >= threshold

    def is_context_similar(self, c1: Dict, c2: Dict) -> bool:
        if not c1 or not c2:
            return True
        t1 = set(c1.get('topics_discussed', []))
        t2 = set(c2.get('topics_discussed', []))
        if t1 and t2 and len(t1 & t2) / len(t1 | t2) < 0.3:
            return False
        return c1.get('user_style', {}).get('style') == c2.get('user_style', {}).get('style', 'unknown')

    def refine_response(self, pattern: str, response: str, context: Dict):
        learned = self.learned_responses[pattern]
        style = context.get('user_style', {})
        if style.get('detail_level') == 'low' and len(response.split()) > 20:
            learned['response'] = self.make_concise(response)
        elif style.get('detail_level') == 'high':
            learned['response'] = self.add_detail(response, context)

    def learn_concepts_from_text(self, text: str):
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        for word in words:
            if word not in self.knowledge_base['concepts']:
                self.knowledge_base['concepts'][word] = {
                    'strength': 0.3,
                    'usage_count': 1,
                    'discovered_in_conversation': True,
                    'first_seen': datetime.now().isoformat(),
                    'last_used': datetime.now().isoformat()
                }

    def remember_conversation(self, session_id: str, query: str, response: str, context: Dict):
        self.conversation_memory.append({
            'session_id': session_id,
            'query': query,
            'response': response,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'vni_instance': self.instance_id
        })

    # ==================== WEB SEARCH ====================

    def needs_web_search(self, query: str, confidence: float, concepts: List[str]) -> bool:
        q = query.lower()
        keywords = ['current', 'recent', 'latest', 'today', 'now', 'new', 'update', 'news']
        return (confidence < 0.4 or any(k in q for k in keywords) or len(concepts) == 0) and self.web_search_enabled

    def search_web_for_information(self, query: str, max_results: int = 3) -> str:
        if not self.web_search_enabled:
            return "Web search not available"
        
        cache_key = hashlib.md5(query.lower().encode()).hexdigest()
        if cache_key in self.search_cache:
            cached_time, cached_result = self.search_cache[cache_key]
            if (datetime.now() - cached_time).seconds < 3600:
                return cached_result
        
        try:
            logger.info(f"Searching: {query}")
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                if not results:
                    return "No relevant information found."
                
                combined = []
                for i, r in enumerate(results[:max_results]):
                    combined.append(f"{i+1}. {r.get('title', '')}: {r.get('body', '')}")
                
                web_content = " ".join(combined)
                
                if len(self.search_cache) >= self.max_search_cache_size:
                    oldest = min(self.search_cache.keys(), key=lambda k: self.search_cache[k][0])
                    del self.search_cache[oldest]
                
                self.search_cache[cache_key] = (datetime.now(), web_content)
                return web_content[:1000]
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return f"Search error: {str(e)}"

    def enhance_with_web_search(self, query: str, response: str, concepts: List[str], confidence: float) -> str:
        web_info = self.search_web_for_information(query)
        if web_info and "error" not in web_info.lower() and "not available" not in web_info.lower():
            self.learn_concepts_from_text(web_info)
            return f"{response}\n\n🔍 **Latest Information:**\n{web_info}"
        return response

    # ==================== UTILITY METHODS ====================

    def extract_concepts_and_patterns(self, text: str) -> Tuple[List[str], List[str]]:
        text_lower = text.lower()
        concepts = [c for c in self.knowledge_base.get("concepts", {}) if c in text_lower]
        patterns = []
        for pid, pdata in self.knowledge_base.get("patterns", {}).items():
            if any(t in text_lower for t in pdata.get("triggers", [])):
                patterns.append(pid)
        return concepts, patterns

    def calculate_confidence(self, concepts: List[str], patterns: List[str]) -> float:
        if not concepts and not patterns:
            return 0.3
        total = len(concepts) + len(patterns)
        conf = sum(self.knowledge_base["concepts"].get(c, {}).get('strength', 0.5) for c in concepts)
        conf += sum(self.knowledge_base["patterns"].get(p, {}).get('strength', 0.7) for p in patterns)
        return min(1.0, conf / total) if total > 0 else 0.3

    def extract_context_notes(self, query: str, context: Dict = None) -> Dict[str, Any]:
        return {
            'query_complexity': self.assess_complexity(query),
            'emotional_tone': self.detect_emotional_tone(query),
            'urgency_level': self.detect_urgency(query),
            'specificity': self.assess_specificity(query),
            'user_knowledge_level': context.get('user_profile', {}).get('knowledge_level', 'unknown') if context else 'unknown'
        }

    def analyze_user_style(self, conversations: List[Dict]) -> Dict[str, str]:
        if not conversations:
            return {'style': 'unknown', 'detail_level': 'medium'}
        queries = [c.get('query', '') for c in conversations]
        avg_len = sum(len(q) for q in queries) / len(queries)
        return {
            'style': 'detailed' if avg_len > 50 else 'concise',
            'detail_level': 'high' if avg_len > 80 else 'medium' if avg_len > 30 else 'low'
        }

    def assess_complexity(self, query: str) -> float:
        wc = len(query.split())
        return min(1.0, wc / 50)

    def detect_emotional_tone(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ['urgent', 'emergency', 'help', 'asap']):
            return 'urgent'
        if any(w in q for w in ['thank', 'appreciate', 'good', 'great']):
            return 'positive'
        if any(w in q for w in ['problem', 'issue', 'wrong', 'bad']):
            return 'negative'
        return 'neutral'

    def detect_urgency(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ['emergency', 'urgent', 'asap', 'immediately']):
            return 'high'
        if any(w in q for w in ['soon', 'quick', 'fast']):
            return 'medium'
        return 'low'

    def assess_specificity(self, query: str) -> str:
        specific = len(re.findall(r'\b(\w+ing|\w+ed|\w+ly|\d+)\b', query))
        return 'high' if specific > 3 else 'medium' if specific > 1 else 'low'

    def make_concise(self, response: str) -> str:
        sentences = response.split('.')
        return sentences[0] + '.' if sentences else response

    def add_detail(self, response: str, context: Dict) -> str:
        if 'more detail' not in response.lower():
            return response + " Would you like more specific details?"
        return response

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            'vni_type': self.vni_type,
            'instance_id': self.instance_id,
            'web_search_enabled': self.web_search_enabled,
            'attention_enabled': self.attention_enabled,
            'predictive_capabilities': PREDICTIVE_AVAILABLE,
            'learning_enabled': True,
            'context_memory': True,
            'knowledge_base_size': len(self.knowledge_base.get('concepts', {})),
            'learned_responses_count': len(self.learned_responses),
            'generation_enabled': self.generation_enabled,
            'generation_model': 'DialoGPT-medium' if self.generation_enabled else 'none',
            'bridge_layer_available': self.bridge_layer is not None
        }

    def test_generation_simple(self) -> bool:
        """Simple test to verify generation works"""
        if not self.generation_enabled:
            logger.error("Generation not enabled")
            return False
    
        try:
            # Simple test prompt
            test_prompt = "Hello, how are you?"
        
            # Tokenize properly
            inputs = self.tokenizer(test_prompt, return_tensors="pt")
        
            logger.info(f"Test: Input shape: {inputs.input_ids.shape}")
            logger.info(f"Test: Tokenizer vocab size: {self.tokenizer.vocab_size}")
        
            # Generate
            generated = self.generator.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=50,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
            response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            logger.info(f"Test: Generated response: {response}")

            return len(response) > 0
        
        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

# ==================== SPECIALIZED VNI CLASSES ====================

class EnhancedMedicalVNI(EnhancedBaseVNI):
    """Medical VNI with specialized knowledge and generation"""
    
    def __init__(self, instance_id: str):
        super().__init__("medical", instance_id)
        # Override classifier with medical-specific one
        self._init_medical_classifier()
        
        # Tune generation for medical domain
        if self.generation_enabled:
            self.configure_generation(temperature=0.6, top_p=0.85)  # More conservative

    def _init_domain_classifier(self):
        """Override to initialize medical classifier"""
        self._init_medical_classifier()
    
    def _init_medical_classifier(self):
        """Initialize medical-specific classifier"""
        try:
            class MedicalClassifier:
                def __init__(self):
                    self.keywords = [
                        'medical', 'health', 'doctor', 'hospital', 'pain', 'fever',
                        'headache', 'medicine', 'treatment', 'sick', 'illness',
                        'disease', 'symptom', 'patient', 'clinic', 'advise', 'advice',
                        'emergency', 'medication', 'diagnosis', 'therapy', 'clinic',
                        'pharmacy', 'prescription', 'vaccine', 'covid', 'coronavirus',
                        'infection', 'virus', 'bacteria', 'allergy', 'asthma', 'cancer',
                        'diabetes', 'heart', 'lung', 'kidney', 'liver', 'brain',
                        'mental health', 'psychology', 'therapy', 'counseling',
                        'help on medical', 'health advise', 'medical help',
                        'what medicine', 'should i take', 'symptoms of',
                        'treatment for', 'diagnosed with'
                    ]
                    self.domain = "medical"
                
                def predict(self, texts):
                    """Return 1 if medical, 0 otherwise"""
                    results = []
                    for text in texts:
                        text_lower = text.lower()
                        is_medical = any(keyword in text_lower for keyword in self.keywords)
                        results.append(1 if is_medical else 0)
                    return results
                
                def predict_proba(self, texts):
                    """Return probability scores [not_medical, medical]"""
                    results = []
                    for text in texts:
                        text_lower = text.lower()
                        matches = sum(1 for keyword in self.keywords if keyword in text_lower)
                        prob = min(0.95, matches * 0.2)  # Scale probability
                        results.append([1 - prob, prob])
                    return results
                
                def __str__(self):
                    return f"MedicalClassifier with {len(self.keywords)} keywords"
            
            self.classifier = MedicalClassifier()
            logger.info(f"✅ Medical classifier initialized for {self.instance_id}")
            logger.info(f"   Contains keyword: 'help on medical' = {'help on medical' in self.classifier.keywords}")
            logger.info(f"   Contains keyword: 'health advise' = {'health advise' in self.classifier.keywords}")
            
        except Exception as e:
            logger.error(f"❌ Error initializing medical classifier: {e}")
            self.classifier = None
    
    def _fallback_domain_check(self, query: str) -> bool:
        """Fallback method for medical domain detection"""
        query_lower = query.lower()
        essential_keywords = ['medical', 'health', 'doctor', 'hospital', 'pain', 'fever']
        return any(keyword in query_lower for keyword in essential_keywords)
    
    def get_default_concepts(self) -> Dict[str, Any]:
        return {
            'fever': {'strength': 0.8, 'usage_count': 0},
            'headache': {'strength': 0.8, 'usage_count': 0},
            'pain': {'strength': 0.9, 'usage_count': 0},
            'symptom': {'strength': 0.8, 'usage_count': 0},
            'treatment': {'strength': 0.8, 'usage_count': 0},
            'diagnosis': {'strength': 0.8, 'usage_count': 0},
            'medicine': {'strength': 0.7, 'usage_count': 0},
            'health': {'strength': 0.9, 'usage_count': 0},
            'covid': {'strength': 0.8, 'usage_count': 0},
            'vaccine': {'strength': 0.7, 'usage_count': 0},
            'patient': {'strength': 0.8, 'usage_count': 0},
            'doctor': {'strength': 0.9, 'usage_count': 0},
            'flu': {'strength': 0.8, 'usage_count': 0},
            'cold': {'strength': 0.7, 'usage_count': 0},
            'infection': {'strength': 0.8, 'usage_count': 0},
            'medication': {'strength': 0.8, 'usage_count': 0}
        }

    def get_default_patterns(self) -> Dict[str, Any]:
        return {
            "fever_advice": {
                "triggers": ["fever", "temperature", "hot", "burning up"],
                "responses": [
                    "Fever often indicates infection. Rest, hydration, and monitoring are important. Consult a doctor if it persists.",
                    "For fever, ensure proper hydration and rest. Seek medical advice for proper diagnosis."
                ],
                "strength": 0.8, "usage_count": 0
            },
            "headache_help": {
                "triggers": ["headache", "head pain", "migraine"],
                "responses": [
                    "Headaches have various causes including stress and dehydration. Rest and hydration often help.",
                    "Consider triggers like stress, lack of sleep, or dehydration. Persistent headaches should be evaluated."
                ],
                "strength": 0.7, "usage_count": 0
            },
            "pain_assessment": {
                "triggers": ["pain", "hurts", "ache", "sore"],
                "responses": [
                    "Pain assessment requires understanding location, intensity, and duration. Professional evaluation is recommended."
                ],
                "strength": 0.8, "usage_count": 0
            }
        }

    def get_default_response_templates(self) -> Dict[str, List[str]]:
        return {
            "general": [
                "From a medical perspective, {concept} should be evaluated by a healthcare professional.",
                "Regarding {concept}, consider individual health factors and consult a doctor."
            ],
            "emergency": [
                "If this is a medical emergency, please seek immediate professional help."
            ]
        }

class EnhancedLegalVNI(EnhancedBaseVNI):
    """Legal VNI with specialized knowledge and generation"""
    
    def __init__(self, instance_id: str):
        super().__init__("legal", instance_id)
        # Tune generation for legal domain
        if self.generation_enabled:
            self.configure_generation(temperature=0.5, top_p=0.8)  # Very conservative

    def get_default_concepts(self) -> Dict[str, Any]:
        return {
            'contract': {'strength': 0.8, 'usage_count': 0},
            'law': {'strength': 0.9, 'usage_count': 0},
            'legal': {'strength': 0.9, 'usage_count': 0},
            'rights': {'strength': 0.8, 'usage_count': 0},
            'agreement': {'strength': 0.7, 'usage_count': 0},
            'court': {'strength': 0.8, 'usage_count': 0},
            'lawyer': {'strength': 0.9, 'usage_count': 0},
            'case': {'strength': 0.8, 'usage_count': 0},
            'evidence': {'strength': 0.7, 'usage_count': 0},
            'justice': {'strength': 0.8, 'usage_count': 0},
            'liability': {'strength': 0.8, 'usage_count': 0},
            'dispute': {'strength': 0.7, 'usage_count': 0},
            'compliance': {'strength': 0.8, 'usage_count': 0}
        }

    def get_default_patterns(self) -> Dict[str, Any]:
        return {
            "contract_help": {
                "triggers": ["contract", "agreement", "terms", "sign"],
                "responses": [
                    "Contracts should be reviewed carefully. Key elements include parties, terms, obligations, and termination clauses.",
                    "For contract matters, ensure all terms are clear. Legal review is recommended for important agreements."
                ],
                "strength": 0.8, "usage_count": 0
            },
            "rights_inquiry": {
                "triggers": ["rights", "entitled", "legal rights"],
                "responses": [
                    "Legal rights vary by jurisdiction and situation. Consulting a qualified attorney is recommended."
                ],
                "strength": 0.8, "usage_count": 0
            }
        }

    def get_default_response_templates(self) -> Dict[str, List[str]]:
        return {
            "general": [
                "From a legal standpoint, {concept} requires proper documentation and professional advice.",
                "Legal matters involving {concept} should be reviewed by a qualified attorney."
            ]
        }

class EnhancedGeneralVNI(EnhancedBaseVNI):
    """General VNI with multi-domain knowledge and generation"""
    def __init__(self, instance_id: str):
        super().__init__("general", instance_id)
        # Balanced generation settings
        if self.generation_enabled:
            self.configure_generation(temperature=0.7, top_p=0.9)

    def get_default_concepts(self) -> Dict[str, Any]:
        return {
            # Technical
            'code': {'strength': 0.8, 'usage_count': 0},
            'programming': {'strength': 0.8, 'usage_count': 0},
            'technical': {'strength': 0.9, 'usage_count': 0},
            'system': {'strength': 0.8, 'usage_count': 0},
            'software': {'strength': 0.9, 'usage_count': 0},
            'database': {'strength': 0.8, 'usage_count': 0},
            # Mathematical
            'calculate': {'strength': 0.8, 'usage_count': 0},
            'equation': {'strength': 0.9, 'usage_count': 0},
            'formula': {'strength': 0.8, 'usage_count': 0},
            'math': {'strength': 0.9, 'usage_count': 0},
            # Business
            'business': {'strength': 0.9, 'usage_count': 0},
            'strategy': {'strength': 0.8, 'usage_count': 0},
            'market': {'strength': 0.8, 'usage_count': 0},
            'profit': {'strength': 0.8, 'usage_count': 0},
            # Creative
            'write': {'strength': 0.8, 'usage_count': 0},
            'story': {'strength': 0.8, 'usage_count': 0},
            'creative': {'strength': 0.9, 'usage_count': 0},
            # Analytical
            'analyze': {'strength': 0.9, 'usage_count': 0},
            'compare': {'strength': 0.8, 'usage_count': 0},
            'evaluate': {'strength': 0.8, 'usage_count': 0}
        }

    def get_default_patterns(self) -> Dict[str, Any]:
        return {
            "technical_help": {
                "triggers": ["code", "programming", "debug", "error", "bug"],
                "responses": [
                    "For technical issues, systematic debugging helps. Check syntax, test components, review errors."
                ],
                "strength": 0.8, "usage_count": 0
            },
            "analysis_help": {
                "triggers": ["analyze", "compare", "evaluate", "assess"],
                "responses": [
                    "Analytical thinking requires systematic evaluation and logical reasoning."
                ],
                "strength": 0.8, "usage_count": 0
            },
            "business_analysis": {
                "triggers": ["business", "strategy", "market", "profit"],
                "responses": [
                    "Business analysis involves market research, financial modeling, and strategic planning."
                ],
                "strength": 0.8, "usage_count": 0
            }
        }

    def get_default_response_templates(self) -> Dict[str, List[str]]:
        return {
            "technical": [
                "From a technical perspective, {concept} requires understanding implementation and architecture."
            ],
            "analytical": [
                "From an analytical perspective, {concept} requires systematic evaluation."
            ],
            "general": [
                "Regarding {concept}, a comprehensive approach considers multiple perspectives."
            ]
        }

    def process_query(self, query: str, context: Dict = None, session_id: str = "default",
                     use_generation: bool = None, generation_strategy: str = "auto") -> Dict[str, Any]:
        response = super().process_query(query, context, session_id, use_generation, generation_strategy)
        
        # Add domain analysis
        detected = self.detect_query_domains(query)
        response['domain_analysis'] = {
            'detected_domains': detected,
            'primary_domain': detected[0] if detected else 'general',
            'cross_domain_synthesis': self.generate_cross_domain_synthesis(detected, query)
        }
        return response

    def detect_query_domains(self, query: str) -> List[str]:
        q = query.lower()
        domain_keywords = {
            'technical': ['code', 'programming', 'system', 'software', 'database'],
            'mathematical': ['calculate', 'equation', 'formula', 'solve', 'math'],
            'business': ['business', 'strategy', 'market', 'profit', 'revenue'],
            'creative': ['write', 'story', 'creative', 'narrative', 'character'],
            'analytical': ['analyze', 'compare', 'evaluate', 'assess']
        }
        domains = [d for d, kws in domain_keywords.items() if any(k in q for k in kws)]
        return domains if domains else ['general']

    def generate_cross_domain_synthesis(self, domains: List[str], query: str) -> str:
        if len(domains) <= 1:
            return f"Analysis focused on {domains[0] if domains else 'general'} domain."
        
        synthesis = f"Multi-domain analysis: {', '.join(domains)}. "
        if 'technical' in domains and 'business' in domains:
            synthesis += "Consider technical feasibility for business requirements. "
        if 'creative' in domains and 'analytical' in domains:
            synthesis += "Balance creative expression with analytical structure. "
        return synthesis.strip()

# ==================== VNI REGISTRY ====================
class VNIRegistry:
    """Global registry of all active VNIs"""
    _instance = None
    _vnis: Dict[str, EnhancedBaseVNI] = {}
    _capability_index: Dict[str, List[str]] = {}  # capability -> [vni_ids]
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = VNIRegistry()
        return cls._instance
    
    def register_vni(self, vni: EnhancedBaseVNI):
        self._vnis[vni.instance_id] = vni
        
        # Index by capabilities
        for capability in vni.available_capabilities.specializations:
            if capability not in self._capability_index:
                self._capability_index[capability] = []
            self._capability_index[capability].append(vni.instance_id)
    
    def find_vnis_by_capability(self, capability: str) -> List[str]:
        return self._capability_index.get(capability, [])
    
    def get_vni(self, vni_id: str) -> Optional[EnhancedBaseVNI]:
        return self._vnis.get(vni_id)

# ==================== VNI MANAGEMENT SYSTEM ====================
class VNIManager:
    """Manager for coordinating multiple VNI instances with generation"""
    
    def __init__(self, enable_generation: bool = True):
        self.vni_instances: Dict[str, EnhancedBaseVNI] = {}
        self.neural_pathways: Dict[str, NeuralPathway] = {}
        self.session_manager = SessionManager()
        self.attention_scores: Dict[str, float] = {}
        self.enable_generation = enable_generation
        
    def create_vni(self, vni_type: str, instance_id: str) -> EnhancedBaseVNI:
        vni_classes = {
            'medical': EnhancedMedicalVNI,
            'legal': EnhancedLegalVNI,
            'general': EnhancedGeneralVNI
        }
        if vni_type not in vni_classes:
            raise ValueError(f"Unknown VNI type: {vni_type}")
        
        vni = vni_classes[vni_type](instance_id)
        self.vni_instances[instance_id] = vni
        
        # Create neural pathways to other VNIs
        for existing_id in self.vni_instances:
            if existing_id != instance_id:
                pathway_id = f"{instance_id}->{existing_id}"
                self.neural_pathways[pathway_id] = NeuralPathway(instance_id, existing_id)
                
                reverse_id = f"{existing_id}->{instance_id}"
                self.neural_pathways[reverse_id] = NeuralPathway(existing_id, instance_id)
        
        return vni

    def route_query(self, query: str, context: Dict = None, session_id: str = "default",
                   use_generation: bool = None) -> Dict[str, Any]:
        """Route query with attention mechanism and generation"""
        if use_generation is None:
            use_generation = self.enable_generation
            
        self.session_manager.get_session(session_id)
        
        # Calculate attention scores
        self.attention_scores = self._calculate_attention_scores(query)
        
        # Activate VNIs based on attention
        activated = []
        for vni_id, vni in self.vni_instances.items():
            score = self.attention_scores.get(vni_id, 0.5)
            if score > 0.3:
                activated.append(vni_id)
        
        # Process with activated VNIs
        responses = []
        for vni_id in activated:
            vni = self.vni_instances[vni_id]
            if vni.attention_enabled:
                response = vni.process_query_with_attention(query, context or {}, self.attention_scores)
            else:
                response = vni.process_query(query, context, session_id, use_generation=use_generation)
            responses.append(response)
            
            # Update neural pathways
            self._update_pathways(vni_id, response.get('confidence', 0.5) > 0.6)
        
        return self._combine_responses(responses, query)

    def test_generation(self, query: str = "How can AI help healthcare?") -> Dict[str, Any]:
        """Test generation capability of all VNIs"""
        results = {}
        
        for vni_id, vni in self.vni_instances.items():
            if vni.generation_enabled:
                try:
                    generated = vni.generate_with_transformer(query)
                    results[vni_id] = {
                        'response': generated,
                        'success': True,
                        'model': 'DialoGPT-medium',
                        'vni_type': vni.vni_type
                    }
                except Exception as e:
                    results[vni_id] = {
                        'response': str(e),
                        'success': False,
                        'model': 'none',
                        'error': str(e)
                    }
            else:
                results[vni_id] = {
                    'response': "Generation not enabled",
                    'success': False,
                    'model': 'none'
                }
        
        return results

    def _calculate_attention_scores(self, query: str) -> Dict[str, float]:
        """Calculate attention scores for each VNI"""
        scores = {}
        
        for vni_id, vni in self.vni_instances.items():
            concepts, patterns = vni.extract_concepts_and_patterns(query)
            
            # Base score from concept/pattern matching
            base_score = min(1.0, (len(concepts) * 0.2 + len(patterns) * 0.3))
            
            # Boost from neural pathway strengths
            pathway_boost = 0.0
            for pid, pathway in self.neural_pathways.items():
                if pathway.target == vni_id:
                    pathway_boost += pathway.strength * 0.1
            
            scores[vni_id] = min(1.0, base_score + pathway_boost + 0.3)
        
        return scores

    def _update_pathways(self, vni_id: str, success: bool):
        """Update neural pathways based on success"""
        for pid, pathway in self.neural_pathways.items():
            if pathway.source == vni_id:
                pathway.activate(success)
            pathway.decay()

    def _combine_responses(self, responses: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        if not responses:
            return {
                'response': "I'm not sure how to help. Could you provide more context?",
                'confidence': 0.1,
                'sources': [],
                'combined': True
            }
        
        # Weight by attention and confidence
        for r in responses:
            r['weighted_score'] = r.get('confidence', 0.5) * r.get('attention_score', 1.0)
        
        best = max(responses, key=lambda r: r.get('weighted_score', 0))
        
        return {
            'response': best['response'],
            'confidence': best['confidence'],
            'attention_score': best.get('attention_score', 1.0),
            'generation_used': best.get('generation_used', False),
            'sources': [r['vni_instance'] for r in responses],
            'combined': len(responses) > 1,
            'all_responses': responses if len(responses) > 1 else None
        }

    def get_system_status(self) -> Dict[str, Any]:
        return {
            'vni_count': len(self.vni_instances),
            'pathway_count': len(self.neural_pathways),
            'active_sessions': len(self.session_manager.sessions),
            'generation_enabled': self.enable_generation,
            'generation_available': GENERATION_AVAILABLE,
            'vni_capabilities': {vid: vni.get_capabilities() for vid, vni in self.vni_instances.items()}
        }


class SessionManager:
    """Manage user sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = timedelta(hours=2)
        
    def get_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'created': datetime.now(),
                'last_activity': datetime.now(),
                'interaction_count': 0,
                'preferences': {}
            }
        else:
            self.sessions[session_id]['last_activity'] = datetime.now()
            self.sessions[session_id]['interaction_count'] += 1
        return self.sessions[session_id]
    
    def cleanup_expired_sessions(self) -> int:
        now = datetime.now()
        expired = [sid for sid, s in self.sessions.items() if now - s['last_activity'] > self.session_timeout]
        for sid in expired:
            del self.sessions[sid]
        return len(expired)


# ==================== MAIN ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("🧠 Enhanced VNI System with Text Generation")
    print("="*60)
    
    # Create manager and VNIs
    manager = VNIManager(enable_generation=True)
    medical = manager.create_vni('medical', 'med_001')
    legal = manager.create_vni('legal', 'legal_001')
    general = manager.create_vni('general', 'gen_001')
    
    print("\n=== VNI System Status ===")
    print(json.dumps(manager.get_system_status(), indent=2, default=str))
    
    # Test generation capability
    if GENERATION_AVAILABLE:
        print("\n=== Testing Generation Capability ===")
        test_results = manager.test_generation("What are the benefits of AI in healthcare?")
        for vni_id, result in test_results.items():
            print(f"\n{vni_id}:")
            print(f"  Success: {result['success']}")
            if result['success']:
                print(f"  Response: {result['response'][:100]}...")
    
    # Test queries
    queries = [
        "I have a headache and fever, what should I do?",
        "I need help reviewing a contract agreement",
        "How can I analyze my business data effectively?",
        "What are the latest developments in AI?",
        "Can you help me debug this code error?",
        "What are my legal rights in this situation?"
    ]
    
    print("\n=== Processing Test Queries ===\n")
    
    for i, query in enumerate(queries):
        print(f"Query {i+1}: {query}")
        print("-" * 50)
        
        response = manager.route_query(query, session_id="test_session")
        
        print(f"Response: {response['response'][:200]}...")
        print(f"Confidence: {response['confidence']:.2f}")
        print(f"Generation Used: {response.get('generation_used', False)}")
        print(f"Attention Score: {response.get('attention_score', 'N/A')}")
        print(f"Sources: {response['sources']}")
        print("\n")
    
    # Save all knowledge bases
    for vni in manager.vni_instances.values():
        vni.save_knowledge_base()
    
    print("\n=== System Statistics ===")
    for vni_id, vni in manager.vni_instances.items():
        stats = vni.get_knowledge_stats()
        print(f"\n{vni_id}:")
        print(f"  Concepts: {stats['concepts_count']}")
        print(f"  Learned Responses: {stats['learned_responses_count']}")
        print(f"  Generation Enabled: {stats['generation_enabled']}")
    
    print("\n✅ All knowledge bases saved")
    print("🚀 System ready for production use with text generation!")
