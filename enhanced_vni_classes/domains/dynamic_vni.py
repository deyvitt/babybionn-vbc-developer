# enhanced_vni_classes/domains/dynamic_vni.py
"""DynamicVNI: Full-featured version with knowledge systems & async support - Standardized"""
import json
import torch
import asyncio
import logging
import hashlib
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Set, Tuple, Callable

# Keep ALL original imports
from neuron.vni_memory import VNIMemory
from ..modules.knowledge_base import KnowledgeBase
from ..core.pipeline_steps import PipelineStep
from .base_knowledge_loader import BaseKnowledgeLoader
from neuron.demoHybridAttention import DemoHybridAttention
from neuron.smart_activation_router import SmartActivationRouter
from ..core.base_vni import EnhancedBaseVNI
from ..core.capabilities import VNICapabilities, VNIType
from ..modules.classifier import DomainClassifier
# +from llm_Gateway import LLMGateway  # ADDED: Import LLM Gateway

logger = logging.getLogger(__name__)

# ========== KEEP ALL ORIGINAL ENUMS ==========
class BiologicalMode(Enum):
    FULL = "full"
    HYBRID = "hybrid"
    MINIMAL = "minimal"
    OFF = "off"

class GenerationStyle(Enum):
    CONCISE = "concise"
    DETAILED = "detailed"
    CREATIVE = "creative"
    FORMAL = "formal"
    TECHNICAL = "technical"
    EMERGENCY = "emergency"
    INTUITIVE = "intuitive"
    ASSOCIATIVE = "associative"
    REFLECTIVE = "reflective"

# ========== KEEP ALL ORIGINAL DATACLASSES ==========
@dataclass
class BiologicalDomainConfig:
    """Full original configuration"""
    attention_weights: Dict[str, float] = field(default_factory=dict)
    memory_config: Dict[str, Any] = field(default_factory=dict)
    activation_patterns: List[str] = field(default_factory=list)
    synaptic_strength: float = 0.7
    neural_pathways: List[str] = field(default_factory=list)
    plasticity_enabled: bool = True
    cross_domain_resonance: bool = False
    hybrid_attention_enabled: bool = True
    attention_embed_dim: int = 512
    attention_num_heads: int = 8
    attention_dropout: float = 0.1
    memory_dim: int = 64
    smart_routing_enabled: bool = True
    routing_experts: int = 4
    routing_expert_dim: int = 256
    enable_llm_generation: bool = True  # ADDED: LLM generation toggle

@dataclass
class DomainEvolution:
    """Full evolution tracking with 1000+ history capacity"""
    learned_keywords: Set[str] = field(default_factory=set)
    query_patterns: Dict[str, int] = field(default_factory=dict)
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    confidence_trend: List[float] = field(default_factory=list)
    biological_metrics: Dict[str, Any] = field(default_factory=dict)
    synaptic_strength_history: List[float] = field(default_factory=list)
    attention_patterns: List[Dict[str, Any]] = field(default_factory=list)
    routing_decisions: List[Dict[str, Any]] = field(default_factory=list)
    attention_weights_history: List[Dict[str, float]] = field(default_factory=list)
    llm_generation_history: List[Dict[str, Any]] = field(default_factory=list)  # ADDED: Track LLM usage
    
    def add_interaction(self, query: str, confidence: float, response_success: bool, 
                       attention_metrics: Optional[Dict[str, Any]] = None,
                       routing_metrics: Optional[Dict[str, Any]] = None,
                       biological_metrics: Optional[Dict[str, Any]] = None,
                       llm_generation: Optional[Dict[str, Any]] = None):  # ADDED: LLM metrics
        """Full interaction tracking with 1000+ capacity"""
        self.confidence_trend.append(confidence)
        
        # Extract keywords for learning
        words = {w.lower() for w in query.split() if len(w) > 3}
        self.learned_keywords.update(words)
        
        # Update query patterns
        pattern_key = f"len_{len(query.split())}_confidence_{confidence:.2f}"
        self.query_patterns[pattern_key] = self.query_patterns.get(pattern_key, 0) + 1
        
        # Store biological metrics
        if biological_metrics:
            self.biological_metrics.update(biological_metrics)
            if 'synaptic_strength' in biological_metrics:
                self.synaptic_strength_history.append(biological_metrics['synaptic_strength'])
        
        # Store attention metrics
        if attention_metrics:
            self.attention_patterns.append({
                'timestamp': datetime.now().isoformat(),
                'metrics': attention_metrics
            })
            if 'attention_weights' in attention_metrics:
                self.attention_weights_history.append(attention_metrics['attention_weights'])
        
        # Store routing metrics
        if routing_metrics:
            self.routing_decisions.append({
                'timestamp': datetime.now().isoformat(),
                'metrics': routing_metrics
            })
        
        # Store LLM generation metrics
        if llm_generation:
            self.llm_generation_history.append({
                'timestamp': datetime.now().isoformat(),
                'metrics': llm_generation
            })
        
        # Store full adaptation history
        self.adaptation_history.append({
            'timestamp': datetime.now().isoformat(),
            'query_preview': query[:100],
            'confidence': confidence,
            'success': response_success,
            'biological_metrics': biological_metrics,
            'attention_metrics': attention_metrics,
            'routing_metrics': routing_metrics,
            'llm_generation': llm_generation,  # ADDED
            'word_count': len(query.split())
        })
        
        # Prune history to maintain 1000+ capacity
        if len(self.adaptation_history) > 1500:
            self.adaptation_history = self.adaptation_history[-1500:]
        if len(self.confidence_trend) > 1000:
            self.confidence_trend = self.confidence_trend[-1000:]
        if len(self.attention_patterns) > 500:
            self.attention_patterns = self.attention_patterns[-500:]
        if len(self.routing_decisions) > 500:
            self.routing_decisions = self.routing_decisions[-500:]
        if len(self.llm_generation_history) > 500:  # ADDED
            self.llm_generation_history = self.llm_generation_history[-500:]

@dataclass
class DomainConfig:
    """Full domain configuration with knowledge system support"""
    name: str
    description: str
    keywords: List[str]
    priority_keywords: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.3
    generation_temperature: float = 0.7
    response_templates: Dict[str, List[str]] = field(default_factory=dict)
    default_concepts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    default_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    biological_config: BiologicalDomainConfig = field(default_factory=BiologicalDomainConfig)
    max_keywords: int = 100
    learning_rate: float = 0.1
    biological_mode: BiologicalMode = BiologicalMode.HYBRID
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "2.0"
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['biological_mode'] = self.biological_mode.value
        data['biological_config'] = asdict(self.biological_config)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DomainConfig':
        if 'biological_mode' in data and isinstance(data['biological_mode'], str):
            data['biological_mode'] = BiologicalMode(data['biological_mode'])
        
        if 'biological_config' in data and isinstance(data['biological_config'], dict):
            data['biological_config'] = BiologicalDomainConfig(**data['biological_config'])
        
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

# ========== KEEP KNOWLEDGE MANAGEMENT SYSTEM ==========
class DynamicKnowledgeManager(BaseKnowledgeLoader):
    """Full knowledge management system with concepts and patterns"""
    
    def __init__(self, domain_config: DomainConfig):
        super().__init__()
        self.domain_config = domain_config
        self.knowledge_base = KnowledgeBase()
        self.concepts = self._load_domain_concepts()
        self.patterns = self._load_domain_patterns()
        self.response_templates = domain_config.response_templates
    
    def _load_domain_concepts(self) -> Dict[str, Any]:
        """Load domain-specific concepts"""
        concepts = self.domain_config.default_concepts.copy()
        
        # Add medical concepts if applicable
        if self.domain_config.name == 'medical':
            concepts.update({
                'symptoms': ['fever', 'headache', 'pain', 'nausea', 'fatigue'],
                'conditions': ['flu', 'cold', 'migraine', 'infection', 'injury'],
                'treatments': ['rest', 'hydration', 'medication', 'consultation']
            })
        elif self.domain_config.name == 'legal':
            concepts.update({
                'documents': ['contract', 'agreement', 'will', 'deed', 'license'],
                'processes': ['filing', 'hearing', 'settlement', 'appeal'],
                'roles': ['lawyer', 'judge', 'plaintiff', 'defendant']
            })
        elif self.domain_config.name == 'creative':
            concepts.update({
                'forms': ['story', 'poem', 'painting', 'music', 'design'],
                'elements': ['character', 'plot', 'color', 'composition', 'theme'],
                'techniques': ['brainstorming', 'sketching', 'drafting', 'editing']
            })
        
        return concepts
    
    def _load_domain_patterns(self) -> Dict[str, Any]:
        """Load domain-specific patterns"""
        patterns = self.domain_config.default_patterns.copy()
        
        # Add common patterns
        patterns.update({
            'problem_solving': ['identify', 'analyze', 'solve', 'verify'],
            'explanation': ['define', 'describe', 'explain', 'example'],
            'instruction': ['step1', 'step2', 'step3', 'verify']
        })
        
        return patterns
    
    async def load_domain_knowledge_async(self, domain_name: str):
        """Async knowledge loading"""
        try:
            knowledge = await self.load_knowledge_from_files_async(domain_name)
            if knowledge:
                self.knowledge_base.merge(knowledge)
                logger.info(f"Loaded knowledge for {domain_name}")
        except Exception as e:
            logger.error(f"Failed to load knowledge for {domain_name}: {e}")
    
    def get_concepts_for_query(self, query: str) -> List[str]:
        """Extract relevant concepts from query"""
        query_lower = query.lower()
        relevant_concepts = []
        
        for category, concepts in self.concepts.items():
            for concept in concepts:
                if isinstance(concept, str) and concept.lower() in query_lower:
                    relevant_concepts.append(f"{category}:{concept}")
        
        return relevant_concepts
    
    def get_matching_patterns(self, query: str) -> List[str]:
        """Find matching patterns for query"""
        matching_patterns = []
        
        for pattern_name, pattern_steps in self.patterns.items():
            # Check if any step keywords are in query
            for step in pattern_steps:
                if isinstance(step, str) and step.lower() in query.lower():
                    matching_patterns.append(pattern_name)
                    break
        
        return matching_patterns

# ========== KEEP ATTENTION PROCESSOR WITH FULL FEATURES ==========
class BiologicalAttentionProcessor:
    """Full-featured attention processor"""
    
    def __init__(self, domain_config: DomainConfig):
        self.domain_config = domain_config
        bio_config = domain_config.biological_config
        
        # Initialize Hybrid Attention
        self.hybrid_attention = None
        if bio_config.hybrid_attention_enabled:
            try:
                self.hybrid_attention = DemoHybridAttention(
                    dim=bio_config.attention_embed_dim,
                    num_heads=bio_config.attention_num_heads,
                    window_size=256,
                    use_sliding=True,
                    use_global=True,
                    use_hierarchical=True,
                    memory_tokens=bio_config.memory_dim
                )
                logger.info(f"✅ Initialized DemoHybridAttention for {domain_config.name}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize DemoHybridAttention: {e}")
        
        # Initialize Smart Activation Router
        self.smart_router = None
        if bio_config.smart_routing_enabled:
            try:
                self.smart_router = SmartActivationRouter(
                    input_dim=bio_config.attention_embed_dim,
                    num_experts=bio_config.routing_experts,
                    expert_dim=bio_config.routing_expert_dim
                )
                logger.info(f"✅ Initialized SmartActivationRouter for {domain_config.name}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize SmartActivationRouter: {e}")
        
        # Full attention statistics
        self.attention_stats = {
            'total_queries': 0,
            'hybrid_attention_used': 0,
            'smart_routing_used': 0,
            'avg_attention_weights': {},
            'expert_distribution': {},
            'error_count': 0
        }

    def _encode_query_to_features(self, query: str, context: Dict = None) -> torch.Tensor:
        """Encode query to feature tensor for attention processing."""
        # Placeholder: return a random tensor of the expected embedding dimension
        embed_dim = self.domain_config.biological_config.attention_embed_dim
        return torch.randn(1, embed_dim)

    def process_with_attention(self, query: str, context: Dict[str, Any] = None, 
                             memory_context: torch.Tensor = None) -> Dict[str, Any]:
        """Full attention processing"""
        self.attention_stats['total_queries'] += 1
        result = {
            'attention_processing': False,
            'attention_weights': {},
            'routing_decision': None,
            'activated_experts': [],
            'attention_confidence': 0.0
        }
        
        try:
            # Encode query to features
            query_features = self._encode_query_to_features(query, context)
            if query_features is None:
                return result
            
            # Apply Hybrid Attention
            if self.hybrid_attention:
                try:
                    attention_output, attention_weights = self.hybrid_attention(
                        query=query_features,
                        key=query_features,
                        value=query_features,
                    )
                    result['attention_processing'] = True
                    result['attention_weights'] = self._extract_attention_summary(attention_weights)
                    result['attention_confidence'] = self._calculate_attention_confidence(attention_weights)
                    result['attention_result'] = attention_output
                    self.attention_stats['hybrid_attention_used'] += 1
                except Exception as e:
                    logger.error(f"❌ Hybrid attention failed: {e}")
                    self.attention_stats['error_count'] += 1
            
            # Apply Smart Routing
            if self.smart_router and result.get('attention_result') is not None:
                try:
                    routed_output, routing_weights = self.smart_router(
                        result['attention_result'],
                        task=self.domain_config.name
                    )
                    activated_experts = self._get_activated_experts(routing_weights)
                    result['activated_experts'] = activated_experts
                    result['routing_decision'] = {
                        'experts': activated_experts,
                        'confidence': float(routing_weights.max().item()) if routing_weights is not None else 0.0
                    }
                    self.attention_stats['smart_routing_used'] += 1
                    
                    # Update expert distribution
                    for expert_idx in activated_experts:
                        exp_key = f'expert_{expert_idx}'
                        self.attention_stats['expert_distribution'][exp_key] = \
                            self.attention_stats['expert_distribution'].get(exp_key, 0) + 1
                    
                except Exception as e:
                    logger.error(f"❌ Smart routing failed: {e}")
                    self.attention_stats['error_count'] += 1
            
            # Update attention weight statistics
            if result.get('attention_weights'):
                self._update_attention_statistics(result['attention_weights'])
                
        except Exception as e:
            logger.error(f"❌ Attention processing failed: {e}")
            self.attention_stats['error_count'] += 1
        
        return result
    
    # ... (keep all the original helper methods: _encode_query_to_features, 
    # _extract_attention_summary, _calculate_attention_confidence, 
    # _get_activated_experts, _update_attention_statistics, get_attention_stats)

# ========== KEEP BIOLOGICAL PROCESSOR ==========
class BiologicalProcessor:
    """Full biological processor"""
    def __init__(self, domain_config: DomainConfig):
        self.domain_config = domain_config
        self.attention_processor = BiologicalAttentionProcessor(domain_config)

    def _calculate_activation(self, query: str, attention_result: Dict) -> float:
        """Calculate activation level based on query and attention."""
        # Base activation from attention confidence (if available)
        base = attention_result.get('attention_confidence', 0.5)
        
        # Boost for urgency keywords
        urgency_keywords = ['urgent', 'emergency', 'immediate', 'critical', 'asap']
        query_lower = query.lower()
        urgency_boost = 0.0
        for kw in urgency_keywords:
            if kw in query_lower:
                urgency_boost = 0.3
                break
        
        # Boost for question complexity (e.g., longer queries)
        word_count = len(query.split())
        if word_count > 10:
            complexity_boost = 0.1
        elif word_count > 5:
            complexity_boost = 0.05
        else:
            complexity_boost = 0.0
        
        # Combine and cap at 1.0
        activation = base + urgency_boost + complexity_boost
        return min(activation, 1.0)
    
    def _identify_pathways(self, query: str, attention_result: Dict) -> List[str]:
        """Identify neural pathways based on the query and attention result.
        Returns a list of pathway names."""
        pathways = []
        base_pathway = f"{self.domain_config.name}_pathway"
        pathways.append(base_pathway)
    
        # Add pathways based on activated experts from attention
        if attention_result.get('activated_experts'):
            for expert_idx in attention_result['activated_experts']:
                pathways.append(f"expert_{expert_idx}_pathway")
    
        # Add pathways based on keywords in query
        query_lower = query.lower()
        keyword_pathways = {
            'urgent': 'emergency_pathway',
            'emergency': 'emergency_pathway',
            'technical': 'technical_pathway',
            'medical': 'medical_pathway',
            'legal': 'legal_pathway',
        }
        for keyword, pathway in keyword_pathways.items():
            if keyword in query_lower and pathway not in pathways:
                pathways.append(pathway)
    
        # Remove duplicates and return
        return list(set(pathways))
    
    def _calculate_resonance(self, query: str, context: Dict, attention_result: Dict) -> float:
        """Calculate synaptic resonance based on query, context, and attention.
        Returns a float between 0 and 1."""
        base_resonance = attention_result.get('attention_confidence', 0.5)
    
        # Boost if multiple experts are activated
        expert_count = len(attention_result.get('activated_experts', []))
        if expert_count > 0:
            base_resonance += 0.1 * min(expert_count, 3)  # cap at +0.3
    
        # Boost if there are memory hits in context (if context contains memory info)
        memory_hits = context.get('memory_retrieval', {}).get('memory_hits', 0)
        if memory_hits > 0:
            base_resonance += 0.05 * min(memory_hits, 2)  # +0.1 max
    
        # Urgency boost
        urgency_keywords = ['urgent', 'emergency', 'immediate', 'critical', 'asap']
        query_lower = query.lower()
        if any(kw in query_lower for kw in urgency_keywords):
            base_resonance += 0.2
    
        # Cap at 1.0
        return min(base_resonance, 1.0)
    
    async def process_with_biological_systems_async(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Async biological processing"""
        return await asyncio.to_thread(
            self.process_with_biological_systems,
            query, context
        )
    
    def process_with_biological_systems(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Full biological processing"""
        # Get memory context if available
        memory_context = None
        if context and 'memory_context' in context:
            memory_context = context['memory_context']
        
        # Process with attention
        attention_result = self.attention_processor.process_with_attention(
            query, context, memory_context
        )
        
        # Calculate biological metrics
        activation = self._calculate_activation(query, attention_result)
        pathways = self._identify_pathways(query, attention_result)
        resonance = self._calculate_resonance(query, context, attention_result)
        
        return {
            'biological_processing': True,
            'activation_level': activation,
            'attention_weights': attention_result.get('attention_weights', {}),
            'neural_pathways': pathways,
            'synaptic_resonance': resonance,
            'attention_processing': attention_result['attention_processing'],
            'attention_results': attention_result if attention_result['attention_processing'] else None,
            'activated_experts': attention_result.get('activated_experts', [])
        }
    # ... (keep all original helper methods)

# ========== STANDARDIZED MAIN VNI CLASS WITH ALL FEATURES ==========
class DynamicVNI(EnhancedBaseVNI, BaseKnowledgeLoader):
    """Full-featured Dynamic VNI with knowledge systems & async support - Standardized"""
    def __init__(self, vni_id: str = None, name: str = None, 
                 capabilities: VNICapabilities = None,
                 memory_toolkit: VNIMemory = None,
                 domain_config: DomainConfig = None,
                 enable_learning: bool = True,
                 biological_mode: BiologicalMode = None):
                 #llm_gateway: LLMGateway = None):  # ADDED: LLM Gateway parameter
        
        """Standardized initialization with all features"""
        # Handle domain config
        if domain_config is None:
            domain_config = DomainConfig(
                name="dynamic",
                description="Dynamic VNI with full biological systems",
                keywords=["help", "information", "assistance"],
                biological_mode=BiologicalMode.HYBRID
            )
        
        # Override biological mode if specified
        if biological_mode:
            domain_config.biological_mode = biological_mode
        
        self.domain_config = domain_config
        
        # Generate VNI ID
        if vni_id is None:
            timestamp = int(datetime.now().timestamp() * 1000) % 10000
            domain_hash = hashlib.md5(domain_config.name.encode()).hexdigest()[:8]
            vni_id = f"dynamic_{domain_config.name}_{domain_hash}_{timestamp}"
        
        print(f"🔍 DEBUG: vni_id = {vni_id}, type = {type(vni_id)}")

        # Initialize base classes
        BaseKnowledgeLoader.__init__(self)
        
        # Set up capabilities following technical.py pattern
        capabilities = VNICapabilities(
            domains=[domain_config.name],  # ← Use domains parameter
            can_search=True,
            can_learn=True,
            can_collaborate=True,
            max_context_length=8192,
            special_abilities=[
                'dynamic_domain',
                'biological_systems_integration'
            ] + (['has_hybrid_attention'] if domain_config.biological_config.hybrid_attention_enabled else [])
              + (['has_smart_routing'] if domain_config.biological_config.smart_routing_enabled else []),
            vni_type='dynamic'
        )

        # 1. Initialize memory toolkit (you have this)
        self.memory_toolkit = memory_toolkit or VNIMemory(
            vni_id=vni_id,
            domain=domain_config.name,
            memory_type="episodic",
            embedding_dim=256,
            auto_save=True,
            max_memories=1000        
        )
        
        # 2. Initialize LLM Gateway (ADDED)
        # self.llm_gateway = llm_gateway or LLMGateway()
        
        # 3. Initialize activation_router LIKE technical.py
        if domain_config.biological_config.smart_routing_enabled:
            try:
                self.activation_router = SmartActivationRouter(
                    vni_id=vni_id,  # REQUIRED parameter like technical.py
                    domain=domain_config.name,  # REQUIRED parameter like technical.py
                    input_dim=domain_config.biological_config.attention_embed_dim,
                    num_experts=domain_config.biological_config.routing_experts,
                    expert_dim=domain_config.biological_config.routing_expert_dim
                )
                logger.info(f"✅ Initialized Activation Router for {domain_config.name}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Activation Router: {e}")
                self.activation_router = None
        
        # 4. Initialize knowledge manager (you have this)
        self.knowledge_manager = DynamicKnowledgeManager(domain_config)
        
        # 5. Initialize biological processor (you have this)
        self.biological_processor = BiologicalProcessor(domain_config)
        
        # 6. Initialize evolution tracking (you have this)
        self.evolution = DomainEvolution()
        self.interaction_count = 0
        self.enable_learning = enable_learning
        
        # 7. REGISTER FUNCTIONS with activation_router LIKE technical.py
        self._register_dynamic_functions()
        
        # 8. Performance metrics (following technical.py pattern)
        self.performance_metrics = {
            'total_queries': 0,
            'successful_responses': 0,
            'avg_confidence': 0.0,
            'memory_hit_rate': 0,
            'attention_usage': 0,
            'routing_usage': 0,
            'knowledge_base_usage': 0,
            'pipeline_executions': 0,
            'llm_generation_count': 0,  # ADDED
            'fallback_generation_count': 0  # ADDED
        }
        self.domain = "dynamic"  # or self.domain_config.name
        self.vni_type = "dynamic"
        self.name = "name"
        self.description = "description"  

        # Initialize EnhancedBaseVNI
        super().__init__(
            instance_id=vni_id,
            name=name or f"Dynamic VNI - {domain_config.name}",
            capabilities=capabilities
        )
        print(f"🔍 DEBUG AFTER SUPER: self.instance_id = {self.instance_id}, type = {type(self.instance_id)}")
        
        # ← THEN set it again if needed
        if not isinstance(self.instance_id, str):
            self.instance_id = vni_id
            print(f"🔍 DEBUG FORCED: self.instance_id = {self.instance_id}")

        # Initialize knowledge management system
        self.knowledge_manager = DynamicKnowledgeManager(domain_config)
        
        # Initialize memory system
        #self.memory_toolkit = memory_toolkit or VNIMemory(
        #    vni_id=vni_id,
        #    domain=domain_config.name,
        #    memory_type="episodic",
        #    embedding_dim=256,
        #    auto_save=True,
        #    max_memories=1000,
        #)
        
        # Initialize biological processor
        self.biological_processor = BiologicalProcessor(domain_config)
        
        # Initialize evolution tracking
        self.evolution = DomainEvolution()
        self.interaction_count = 0
        self.enable_learning = enable_learning
        
        # Initialize classifier
        self.classifier = DomainClassifier(enable_context=True, max_context=15)
        
        # Initialize pipeline
        self.pipeline_steps = self._initialize_pipeline_steps()
        
        # Performance metrics (technical.py pattern)
        self.performance_metrics = {
            'total_queries': 0,
            'successful_responses': 0,
            'avg_confidence': 0.0,
            'memory_hit_rate': 0,
            'attention_usage': 0,
            'routing_usage': 0,
            'knowledge_base_usage': 0,
            'pipeline_executions': 0,
            'llm_generation_count': 0,  # ADDED
            'fallback_generation_count': 0  # ADDED
        }
        
        # Load domain knowledge
        self._load_domain_knowledge()
        
        logger.info(f"🚀 Full-featured DynamicVNI '{vni_id}' initialized")
        logger.info(f"   Domain: {domain_config.name}")
        logger.info(f"   Biological Mode: {domain_config.biological_mode.value}")
        logger.info(f"   Knowledge System: Enabled")
        logger.info(f"   Async Support: Enabled")
        logger.info(f"   Learning: {enable_learning}")
        #logger.info(f"   LLM Gateway: {'Enabled' if self.llm_gateway else 'Disabled'}")  # ADDED
        
    def _register_dynamic_functions(self):
        """Register functions with activation router - LIKE technical.py"""
        if not hasattr(self, 'activation_router') or self.activation_router is None:
            return
        
        # Register main processing function
        self.activation_router.register_function(
            function_name="process_dynamic_query",
            function=self.process,
            domain=self.domain_config.name,
            priority=1
        )
        
        # Register async processing function
        self.activation_router.register_function(
            function_name="process_dynamic_query_async",
            function=self.process_async,
            domain=self.domain_config.name,
            priority=1
        )
        
        # Register knowledge lookup function
        self.activation_router.register_function(
            function_name="knowledge_lookup",
            function=self._knowledge_lookup,
            domain=self.domain_config.name,
            priority=2
        )
        
        # Register LLM generation function (ADDED)
        #self.activation_router.register_function(
        #    function_name="generate_with_llm",
        #    function=self._generate_response_with_llm,
        #    domain=self.domain_config.name,
        #    priority=2
        #)
        
        logger.info(f"Registered {len(self.activation_router.get_registered_functions())} dynamic functions")

    def _initialize_pipeline_steps(self) -> Dict[str, PipelineStep]:
        """Initialize pipeline steps with proper concrete classes"""
        steps = {}
        
        # Define concrete step classes inside the method
        class DynamicClassifyStep(PipelineStep):
            def __init__(self, vni):
                self.vni = vni
                
            def execute(self, query: str, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
                # Call the VNI's classification method
                result = self.vni._classify_query(query, context or {})
                data['classification'] = result
                return data

            async def execute_async(self, query: str, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
                """Async wrapper for execute"""
                return await asyncio.to_thread(self.execute, query, data, context)

        class DynamicKnowledgeLookupStep(PipelineStep):
            def __init__(self, vni):
                self.vni = vni
                
            def execute(self, query: str, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
                result = self.vni._knowledge_lookup(query, context or {})
                data['knowledge'] = result
                return data

            async def execute_async(self, query: str, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
                """Async wrapper for execute"""
                return await asyncio.to_thread(self.execute, query, data, context)
        
        class DynamicBiologicalAttentionStep(PipelineStep):
            def __init__(self, vni):
                self.vni = vni
                
            def execute(self, query: str, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
                result = self.vni._apply_biological_attention(query, context or {})
                data['biological'] = result
                return data

            async def execute_async(self, query: str, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
                """Async wrapper for execute"""
                return await asyncio.to_thread(self.execute, query, data, context)
        
        class DynamicMemoryRetrievalStep(PipelineStep):
            def __init__(self, vni):
                self.vni = vni
                
            def execute(self, query: str, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
                result = self.vni._memory_retrieval(query, context or {})
                data['memory'] = result
                return data
    
            async def execute_async(self, query: str, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
                """Async wrapper for execute"""
                return await asyncio.to_thread(self.execute, query, data, context)

        class DynamicGenerateResponseStep(PipelineStep):
            def __init__(self, vni):
                self.vni = vni
                
            def execute(self, query: str, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
                result = self.vni._generate_response(query, context or {})
                data['response_data'] = result
                return data

            async def execute_async(self, query: str, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
                """Async wrapper for execute"""
                return await asyncio.to_thread(self.execute, query, data, context)
        
        # Instantiate the steps with a reference to 'self' (the VNI)
        steps['classify'] = DynamicClassifyStep(self)
        steps['knowledge_lookup'] = DynamicKnowledgeLookupStep(self)
        steps['biological_attention'] = DynamicBiologicalAttentionStep(self)
        steps['memory_retrieval'] = DynamicMemoryRetrievalStep(self)
        steps['generate_response'] = DynamicGenerateResponseStep(self)
        
        return steps
    
    def _load_domain_knowledge(self):
        """Load domain knowledge"""
        try:
            self.load_domain_knowledge(self.domain_config.name)
            logger.info(f"Loaded knowledge for {self.domain_config.name}")
        except Exception as e:
            logger.warning(f"Could not load knowledge: {e}")
    
    # ========== PIPELINE STEP METHODS ==========
    def _classify_query(self, query: str, context: Dict) -> Dict:
        """Classify query domain"""
        classification = self.classifier.classify(query)
        return {
            'classification': classification,
            'confidence': classification.confidence if hasattr(classification, 'confidence') else 0.5
        }
    
    def _knowledge_lookup(self, query: str, context: Dict) -> Dict:
        """Look up knowledge"""
        concepts = self.knowledge_manager.get_concepts_for_query(query)
        patterns = self.knowledge_manager.get_matching_patterns(query)
        
        self.performance_metrics['knowledge_base_usage'] += 1
        
        return {
            'concepts': concepts,
            'patterns': patterns,
            'concept_count': len(concepts),
            'pattern_count': len(patterns)
        }
    
    def _apply_biological_attention(self, query: str, context: Dict) -> Dict:
        """Apply biological attention"""
        if self.domain_config.biological_mode == BiologicalMode.OFF:
            return {'skipped': True, 'reason': 'biological_mode_off'}
        
        biological_result = self.biological_processor.process_with_biological_systems(
            query, context
        )
        
        if biological_result.get('attention_processing'):
            self.performance_metrics['attention_usage'] += 1
        if biological_result.get('activated_experts'):
            self.performance_metrics['routing_usage'] += 1
        
        return biological_result
    
    def _memory_retrieval(self, query: str, context: Dict) -> Dict:
        """Retrieve from memory"""
        if not self.memory_toolkit:
            return {'memory_hits': 0}
        
        try:
            similar = self.memory_toolkit.retrieve_relevant_memory(
                query=query,
                limit=3
                #category='interactions',
                #max_results=3
            )
            hits = len(similar) if similar else 0
            if hits > 0:
                self.performance_metrics['memory_hit_rate'] += 1
            
            return {
                'memory_hits': hits,
                'similar_queries': [s.get('content', '')[:50] for s in similar[:2]] if similar else []
            }
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            return {'memory_hits': 0, 'error': str(e)}
    
    def _generate_response(self, query: str, context: Dict) -> Dict:
        """Generate response data for aggregator - NO LLM calls here"""
        classification = context.get('classify', {}).get('classification', {})
        domain = classification.get('primary_domain', 'general')
        
        # Extract concepts and patterns for context
        concepts = context.get('knowledge_lookup', {}).get('concepts', [])
        patterns = context.get('knowledge_lookup', {}).get('patterns', [])
        
        # Determine generation style (but don't generate)
        generation_style = self._determine_generation_style(query, context)
        
        # Prepare data for aggregator
        response_data = {
            'query': query,
            'domain': domain,
            'confidence': context.get('classify', {}).get('confidence', 0.5),
            'generation_data': {
                'needs_generation': True,
                'generation_style': generation_style,
                'domain': domain,
                'temperature': self.domain_config.generation_temperature,
                'context': {
                    'classification': classification,
                    'concepts': concepts[:10],  # Limit to top 10 concepts
                    'patterns': patterns,
                    'biological_context': context.get('biological_attention', {}),
                    'memory_hits': context.get('memory_retrieval', {}).get('memory_hits', 0),
                    'synaptic_strength': self.domain_config.biological_config.synaptic_strength
                },
                'priority_keywords': self.domain_config.priority_keywords,
                'max_tokens': 500,
                'requires_safety_check': domain in ['medical', 'legal'],
                'requires_disclaimer': domain in ['medical', 'legal']
            },
            'analysis_metrics': {
                'biological_activation': context.get('biological_attention', {}).get('activation_level', 0.5),
                'concept_coverage': len(concepts),
                'pattern_match': len(patterns),
                'confidence_score': context.get('classify', {}).get('confidence', 0.5)
            }
        }
        # Track generation metrics
        if self.domain_config.biological_config.enable_llm_generation:
            self.performance_metrics['llm_generation_count'] += 1
            generation_type = 'llm_gateway'
        else:
            self.performance_metrics['fallback_generation_count'] += 1
            generation_type = 'fallback'
        
        return {
            'response_data': response_data,
            'generated_by': generation_type,
            'generation_style': generation_style,
            'confidence': context.get('classify', {}).get('confidence', 0.5)
        }
    
    def _generate_fallback_response(self, query: str, context: Dict) -> Dict:
        """Generate fallback response data for aggregator"""
        classification = context.get('classify', {}).get('classification', {})
        domain = classification.get('primary_domain', 'general')
        
        # Prepare minimal data for aggregator
        response_data = {
            'query': query,
            'domain': domain,
            'confidence': context.get('classify', {}).get('confidence', 0.5) * 0.8,
            'generation_data': {
                'needs_generation': True,
                'generation_style': 'concise',
                'domain': domain,
                'temperature': 0.7,
                'context': {
                    'classification': classification,
                    'fallback_mode': True,
                    'note': 'Using fallback generation mode'
                },
                'priority_keywords': [],
                'max_tokens': 300
            }
        }
        
        return {
            'response_data': response_data,
            'generated_by': 'fallback',
            'confidence': context.get('classify', {}).get('confidence', 0.5) * 0.8
        }
    
    def _determine_generation_style(self, query: str, context: Dict) -> str:
        """Determine the appropriate generation style"""
        classification = context.get('classify', {}).get('classification', {})
        domain = classification.get('primary_domain', 'general')
        
        # Map domains to generation styles
        style_mapping = {
            'medical': GenerationStyle.FORMAL.value,
            'legal': GenerationStyle.FORMAL.value,
            'technical': GenerationStyle.TECHNICAL.value,
            'creative': GenerationStyle.CREATIVE.value,
            'emergency': GenerationStyle.EMERGENCY.value,
            'general': GenerationStyle.CONCISE.value
        }
        
        # Check for urgency signals
        urgency_keywords = ['urgent', 'emergency', 'immediately', 'critical', 'now']
        if any(keyword in query.lower() for keyword in urgency_keywords):
            return GenerationStyle.EMERGENCY.value
        
        return style_mapping.get(domain, GenerationStyle.CONCISE.value)
    
    # ========== MAIN PROCESSING METHODS ==========
    def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Synchronous processing - follows technical.py pattern"""
        return asyncio.run(self.process_async(input_text, context))
    
    async def process_async(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Async processing with full pipeline"""
        self.performance_metrics['total_queries'] += 1
        self.performance_metrics['pipeline_executions'] += 1
        self.interaction_count += 1
        
        start_time = datetime.now()
        context = context or {}
        context['query'] = input_text
        context['vni_id'] = self.vni_id
        
        # Define pipeline based on biological mode
        pipeline = self.get_biological_pipeline()
        
        # Execute pipeline asynchronously
        results = {}
        for step_name in pipeline:
            if step_name in self.pipeline_steps:
                try:
                    step_result = await self.pipeline_steps[step_name].execute_async(
                        input_text, context
                    )
                    results[step_name] = step_result
                    context.update({step_name: step_result})
                except Exception as e:
                    logger.error(f"Pipeline step {step_name} failed: {e}")
                    results[step_name] = {'error': str(e)}
        
        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(results)
        
        # Get response from results
        generate_response_result = results.get('generate_response', {})
        response_data = generate_response_result.get('response_data', {})
        generated_by = generate_response_result.get('generated_by', 'unknown')
        
        # Update evolution with LLM metrics
        llm_generation_metrics = None
        if 'generate_response' in results:
            llm_generation_metrics = {
                'generated_by': results['generate_response'].get('generated_by'),
                'generation_style': results['generate_response'].get('generation_style'),
                'response_data_present': bool(response_data)
            }
        
        self._update_evolution(input_text, confidence, results, llm_generation_metrics)
        
        # Store in memory - use text from response_data if available
        response_text = response_data.get('query', input_text)[:200]  # Fallback to query
        if confidence > self.domain_config.confidence_threshold:
            await self._store_in_memory_async(input_text, confidence, results, response_text)
        
        # Adaptive learning
        if self.enable_learning:
            self._adaptive_learning(confidence, results)
        
        processing_time = (datetime.now() - start_time).total_seconds()

        if response_data:
            domain_str = response_data.get('domain', self.domain_config.name)
            style = response_data.get('generation_data', {}).get('generation_style', 'general')
            opinion_text = f"Dynamic {domain_str} query. Style: {style}. Confidence: {confidence:.0%}."
        else:
            opinion_text = f"Dynamic {self.domain_config.name} query processed. Confidence: {confidence:.0%}."

        # Return standardized result (technical.py pattern)
        return {
            'vni_id': self.vni_id,
            'query': input_text,
            'domain': self.domain_config.name,
            'response_data': response_data,  # CHANGED: Now returns structured data
            'generated_by': generated_by,  # CHANGED: Renamed field
            'confidence': confidence,
            'processing_steps': pipeline,
            'step_results': {k: v for k, v in results.items() if not isinstance(v, dict) or 'error' not in v},
            'processing_time': processing_time,
            'biological_mode': self.domain_config.biological_mode.value,
            'uses_hybrid_attention': self.domain_config.biological_config.hybrid_attention_enabled,
            'uses_smart_routing': self.domain_config.biological_config.smart_routing_enabled,
            'provides_generation_data': True,  # CHANGED: Always provides data for aggregator
            'performance_metrics': self._get_current_metrics(),
            'pipeline_executed': True,
            'vni_metadata': {
                'vni_id': self.vni_id,
                'success': True,  # ← CRITICAL!
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }      
        }
    
    def get_biological_pipeline(self) -> List[str]:
        """Get pipeline based on biological mode"""
        base_pipeline = ["classify", "knowledge_lookup"]
        
        if self.domain_config.biological_mode != BiologicalMode.OFF:
            base_pipeline.insert(1, "biological_attention")
        
        base_pipeline.extend(["memory_retrieval", "generate_response"])
        
        # Add domain-specific steps
        if self.domain_config.name == "medical":
            base_pipeline.insert(2, "medical_safety_check")
        elif self.domain_config.name == "legal":
            base_pipeline.insert(2, "legal_disclaimer")
        
        return base_pipeline
    
    def _calculate_overall_confidence(self, results: Dict) -> float:
        """Calculate overall confidence from pipeline results"""
        confidence = 0.5
        
        # Boost from classification
        if 'classify' in results:
            classify_conf = results['classify'].get('confidence', 0.5)
            confidence = confidence * 0.3 + classify_conf * 0.7
        
        # Boost from knowledge
        if 'knowledge_lookup' in results:
            concept_count = results['knowledge_lookup'].get('concept_count', 0)
            if concept_count > 0:
                confidence = min(1.0, confidence + 0.1)
        
        # Boost from biological activation
        if 'biological_attention' in results:
            bio_result = results['biological_attention']
            if not bio_result.get('skipped'):
                activation = bio_result.get('activation_level', 0.5)
                confidence = confidence * 0.6 + activation * 0.4
        
        # Boost from memory
        if 'memory_retrieval' in results:
            memory_hits = results['memory_retrieval'].get('memory_hits', 0)
            if memory_hits > 0:
                confidence = min(1.0, confidence + 0.05 * memory_hits)
        
        # Adjust based on generation method
        if 'generate_response' in results:
            gen_result = results['generate_response']
            if gen_result.get('generated_by') == 'llm_gateway':
                confidence = min(1.0, confidence * 1.1)  # Boost for LLM generation
            elif gen_result.get('generated_by') == 'fallback':
                confidence = confidence * 0.9  # Reduce for fallback
        
        return min(confidence, 1.0)
    
    def _update_evolution(self, query: str, confidence: float, results: Dict, 
                         llm_generation_metrics: Dict = None):
        """Update evolution tracking"""
        attention_metrics = None
        routing_metrics = None
        biological_metrics = None
        
        if 'biological_attention' in results:
            bio_result = results['biological_attention']
            if not bio_result.get('skipped'):
                attention_metrics = bio_result.get('attention_results')
                routing_metrics = {
                    'activated_experts': bio_result.get('activated_experts', []),
                    'attention_processing': bio_result.get('attention_processing', False)
                }
                biological_metrics = {
                    'activation_level': bio_result.get('activation_level', 0.5),
                    'synaptic_strength': self.domain_config.biological_config.synaptic_strength,
                    'neural_pathways': bio_result.get('neural_pathways', [])
                }
        
        self.evolution.add_interaction(
            query, confidence, confidence > self.domain_config.confidence_threshold,
            attention_metrics=attention_metrics,
            routing_metrics=routing_metrics,
            biological_metrics=biological_metrics,
            llm_generation=llm_generation_metrics
        )
    
    async def _store_in_memory_async(self, query: str, confidence: float, results: Dict, response: str):
        """Async memory storage"""
        if not self.memory_toolkit:
            return
        
        try:
            memory_data = {
                'query': query[:200],
                'response': response[:200],
                'confidence': confidence,
                'domain': self.domain_config.name,
                'biological_mode': self.domain_config.biological_mode.value,
                'pipeline_steps': list(results.keys()),
                'generated_by': results.get('generate_response', {}).get('generated_by', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
            
            await self.memory_toolkit.store_interaction_async(
                query=query,
                response=response,
                context=memory_data,
                metadata={
                    'vni_id': self.vni_id,
                    'interaction_id': self.interaction_count,
                    'synaptic_strength': self.domain_config.biological_config.synaptic_strength,
                    'llm_used': results.get('generate_response', {}).get('generated_by') == 'llm_gateway'
                }
            )
        except Exception as e:
            logger.warning(f"Async memory storage failed: {e}")
    
    def _adaptive_learning(self, confidence: float, results: Dict):
        """Adaptive learning based on performance"""
        current_strength = self.domain_config.biological_config.synaptic_strength
        
        if confidence > 0.7:
            # Strengthen for good performance
            new_strength = min(1.0, current_strength * 1.05)
            self.domain_config.biological_config.synaptic_strength = new_strength
            
            # Learn from knowledge patterns
            if 'knowledge_lookup' in results:
                concepts = results['knowledge_lookup'].get('concepts', [])
                for concept in concepts[:3]:  # Learn top 3 concepts
                    if ':' in concept:
                        category, value = concept.split(':', 1)
                        self.evolution.learned_keywords.add(value.lower())
        
        elif confidence < 0.3:
            # Weaken for poor performance
            new_strength = max(0.1, current_strength * 0.95)
            self.domain_config.biological_config.synaptic_strength = new_strength
            
            # Adjust LLM usage based on performance
            if results.get('generate_response', {}).get('generated_by') == 'llm_gateway':
                # If LLM performed poorly, consider reducing usage
                self.domain_config.biological_config.enable_llm_generation = False
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        total = max(self.performance_metrics['total_queries'], 1)
        
        return {
            'total_queries': self.performance_metrics['total_queries'],
            'success_rate': self.performance_metrics['successful_responses'] / total,
            'avg_confidence': self.performance_metrics['avg_confidence'],
            'memory_hit_rate': self.performance_metrics['memory_hit_rate'] / total,
            'attention_usage': self.performance_metrics['attention_usage'] / total,
            'routing_usage': self.performance_metrics['routing_usage'] / total,
            'knowledge_base_usage': self.performance_metrics['knowledge_base_usage'] / total,
            'llm_generation_rate': self.performance_metrics['llm_generation_count'] / total,
            'fallback_generation_rate': self.performance_metrics['fallback_generation_count'] / total,
            'pipeline_executions': self.performance_metrics['pipeline_executions']
        }
    
    def get_insights(self) -> Dict[str, Any]:
        """Get insights following technical.py pattern"""
        return {
            'vni_id': self.vni_id,
            'domain': self.domain_config.name,
            'configuration': {
                'biological_mode': self.domain_config.biological_mode.value,
                'confidence_threshold': self.domain_config.confidence_threshold,
                'hybrid_attention': self.domain_config.biological_config.hybrid_attention_enabled,
                'smart_routing': self.domain_config.biological_config.smart_routing_enabled,
                'llm_generation': self.domain_config.biological_config.enable_llm_generation,
                'synaptic_strength': self.domain_config.biological_config.synaptic_strength,
                'neural_pathways': self.domain_config.biological_config.neural_pathways
            },
            'performance': self._get_current_metrics(),
            'knowledge_system': {
                'concepts_loaded': len(self.knowledge_manager.concepts) if hasattr(self, 'knowledge_manager') else 0,
                'patterns_loaded': len(self.knowledge_manager.patterns) if hasattr(self, 'knowledge_manager') else 0
            },
            'evolution': {
                'interaction_count': self.interaction_count,
                'learned_keywords_count': len(self.evolution.learned_keywords),
                'adaptation_history_count': len(self.evolution.adaptation_history),
                'attention_patterns_count': len(self.evolution.attention_patterns),
                'llm_generation_count': len(self.evolution.llm_generation_history)
            },
            'memory_stats': self.memory_toolkit.get_stats() if self.memory_toolkit else {'available': False},
            'llm_gateway_available': hasattr(self, 'llm_gateway') and self.llm_gateway is not None
        }
    
    # ========== COMPATIBILITY METHODS ==========
    def get_superior_status(self) -> Dict[str, Any]:
        """Original method for compatibility"""
        insights = self.get_insights()
        return {
            'superior_dynamic': True,
            **insights,
            'biological_metrics': self.evolution.biological_metrics,
            'synaptic_strength': self.domain_config.biological_config.synaptic_strength,
            'plasticity_enabled': self.domain_config.biological_config.plasticity_enabled,
            'cross_domain_resonance': self.domain_config.biological_config.cross_domain_resonance,
            'attention_patterns_count': len(self.evolution.attention_patterns),
            'routing_decisions_count': len(self.evolution.routing_decisions),
            'llm_generation_count': len(self.evolution.llm_generation_history)
        }
    
    def export_biological_config(self, include_memory: bool = True) -> Dict[str, Any]:
        """Original method for compatibility"""
        config_dict = self.domain_config.to_dict()
        config_dict['_evolution_data'] = {
            'interaction_count': self.interaction_count,
            'learned_keywords': list(self.evolution.learned_keywords),
            'synaptic_strength_history': self.evolution.synaptic_strength_history,
            'confidence_trend': self.evolution.confidence_trend[-20:],
            'llm_generation_history': self.evolution.llm_generation_history[-10:],
            'exported_at': datetime.now().isoformat()
        }
        if include_memory and self.memory_toolkit:
            try:
                memory_stats = self.memory_toolkit.get_stats()
                config_dict['_memory_summary'] = memory_stats
            except Exception as e:
                config_dict['_memory_summary'] = {'error': str(e)}
        return config_dict

# ========== KEEP ALL ORIGINAL FACTORY FUNCTIONS ==========
class EnhancedDomainFactory:
    """Keep original factory with all methods"""
    @staticmethod
    def create_medical_vni(instance_id: str = None, 
                          biological_mode: BiologicalMode = BiologicalMode.FULL) -> DynamicVNI: #,
                          #llm_gateway: LLMGateway = None) -> DynamicVNI:  
        config = DomainConfig(
            name='medical',
            description='Medical domain with full biological integration',
            keywords=['medical', 'health', 'doctor', 'hospital', 'pain', 'fever', 'symptom', 'treatment', 'emergency'],
            priority_keywords=['emergency', 'urgent', 'pain', 'fever', 'symptom'],
            biological_config=BiologicalDomainConfig(
                neural_pathways=['medical_pathway', 'emergency_pathway', 'safety_pathway'],
                synaptic_strength=0.8,
                cross_domain_resonance=True,
                enable_llm_generation=True  # Enable LLM for medical domain
            ),
            biological_mode=biological_mode,
            confidence_threshold=0.4,
            generation_temperature=0.6
        )
        return DynamicVNI(
            vni_id=instance_id,
            name="Medical Dynamic VNI",
            domain_config=config,
            biological_mode=biological_mode#,
            #llm_gateway=llm_gateway  # ADDED: Pass LLM Gateway
        )
    
    @staticmethod
    def create_legal_vni(instance_id: str = None,
                        biological_mode: BiologicalMode = BiologicalMode.FULL) -> DynamicVNI:
        config = DomainConfig(
            name='legal',
            description='Legal domain with full biological integration',
            keywords=['legal', 'law', 'contract', 'court', 'agreement', 'rights', 'dispute', 'liability'],
            priority_keywords=['urgent', 'contract', 'agreement', 'liability'],
            biological_config=BiologicalDomainConfig(
                neural_pathways=['legal_pathway', 'document_pathway', 'compliance_pathway'],
                synaptic_strength=0.7,
                cross_domain_resonance=True,
                enable_llm_generation=True
            ),
            biological_mode=biological_mode,
            confidence_threshold=0.5,
            generation_temperature=0.5
        )
        return DynamicVNI(
            vni_id=instance_id,
            name="Legal Dynamic VNI",
            domain_config=config,
            biological_mode=biological_mode
        )
    
    @staticmethod
    def create_creative_vni(instance_id: str = None,
                           biological_mode: BiologicalMode = BiologicalMode.FULL) -> DynamicVNI:
        config = DomainConfig(
            name='creative',
            description='Creative domain with full biological integration',
            keywords=['creative', 'art', 'design', 'story', 'poem', 'music', 'painting', 'writing', 'inspiration'],
            priority_keywords=['creative', 'design', 'story', 'art'],
            biological_config=BiologicalDomainConfig(
                neural_pathways=['creative_pathway', 'associative_pathway', 'inspiration_pathway'],
                synaptic_strength=0.9,
                cross_domain_resonance=True,
                enable_llm_generation=True
            ),
            biological_mode=biological_mode,
            confidence_threshold=0.3,
            generation_temperature=0.8
        )
        return DynamicVNI(
            vni_id=instance_id,
            name="Creative Dynamic VNI",
            domain_config=config,
            biological_mode=biological_mode
        )
    
    @staticmethod
    def create_custom_vni(name: str,
                         keywords: List[str],
                         biological_mode: BiologicalMode = BiologicalMode.HYBRID,
                         instance_id: str = None) -> DynamicVNI:
        config = DomainConfig(
            name=name,
            description=f'Custom {name} domain with biological integration',
            keywords=keywords,
            priority_keywords=keywords[:3] if len(keywords) > 3 else keywords,
            biological_config=BiologicalDomainConfig(
                neural_pathways=[f'{name}_pathway', 'general_pathway'],
                synaptic_strength=0.6,
                cross_domain_resonance=False,
                enable_llm_generation=True
            ),
            biological_mode=biological_mode,
            confidence_threshold=0.3,
            generation_temperature=0.7
        )
        return DynamicVNI(
            vni_id=instance_id,
            name=f"Custom {name} VNI",
            domain_config=config,
            biological_mode=biological_mode
        )    
def create_superior_vni(domain_name: str, 
                       biological_mode: str = "hybrid",
                       enable_learning: bool = True) -> DynamicVNI:#,
                       #llm_gateway: LLMGateway = None) -> DynamicVNI:
    """Original quick access function"""
    mode_map = {
        'full': BiologicalMode.FULL,
        'hybrid': BiologicalMode.HYBRID,
        'minimal': BiologicalMode.MINIMAL,
        'off': BiologicalMode.OFF
    }
    biological_mode_enum = mode_map.get(biological_mode.lower(), BiologicalMode.HYBRID)
    
    if domain_name == 'medical':
        return EnhancedDomainFactory.create_medical_vni(
            biological_mode=biological_mode_enum#,
            #llm_gateway=llm_gateway
        )
    elif domain_name == 'legal':
        return EnhancedDomainFactory.create_legal_vni(
            biological_mode=biological_mode_enum#,
            #llm_gateway=llm_gateway
        )
    elif domain_name == 'creative':
        return EnhancedDomainFactory.create_creative_vni(
            biological_mode=biological_mode_enum#,
            #llm_gateway=llm_gateway
        )
    else:
        return EnhancedDomainFactory.create_custom_vni(
            name=domain_name,
            keywords=[domain_name, 'information', 'help', 'assistance'],
            biological_mode=biological_mode_enum#,
            #llm_gateway=llm_gateway
        )

# ========== TEST ==========
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("🧬 Testing Full-Featured DynamicVNI (with Knowledge & Async)")
    print("=" * 50)
    
    # Test creation with LLM Gateway
    medical_vni = create_superior_vni("medical", biological_mode="full")#, llm_gateway=LLMGateway())
    
    # Async test
    async def test_async():
        query = "I have severe headache and high fever, what should I do?"
        print(f"\n📝 Query: {query}")
        
        result = await medical_vni.process_async(query)
        print(f"✅ Confidence: {result.get('confidence', 0):.2f}")
        print(f"🧠 Biological Mode: {result.get('biological_mode')}")
        print(f"🤖 Generated by: {result.get('generated_by', 'unknown')}")  # FIXED: Use 'generated_by'
        print(f"🔧 Processing Steps: {result.get('processing_steps', [])}")
        
        # Check response data
        response_data = result.get('response_data', {})
        print(f"📊 Response Data Keys: {list(response_data.keys())}")
        
        if response_data:
            generation_data = response_data.get('generation_data', {})
            print(f"📝 Generation Style: {generation_data.get('generation_style')}")
            print(f"🌡️ Temperature: {generation_data.get('temperature')}")
            print(f"🏷️ Domain: {response_data.get('domain')}")
            print(f"⚠️ Safety Check: {generation_data.get('requires_safety_check', False)}")
        
        # Get insights
        insights = medical_vni.get_insights()
        print(f"\n📊 Knowledge System:")
        print(f"   - Concepts: {insights.get('knowledge_system', {}).get('concepts_loaded', 0)}")
        print(f"   - Patterns: {insights.get('knowledge_system', {}).get('patterns_loaded', 0)}")
        print(f"📈 Performance:")
        print(f"   - Total Queries: {insights.get('performance', {}).get('total_queries', 0)}")
        print(f"   - Memory Hit Rate: {insights.get('performance', {}).get('memory_hit_rate', 0):.2f}")
        print(f"   - LLM Generation Rate: {insights.get('performance', {}).get('llm_generation_rate', 0):.2f}")
        
    asyncio.run(test_async())
    print(f"\n✅ Full-featured DynamicVNI test complete!")
