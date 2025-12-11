"""neural_mesh.py - Complete Neural Mesh Implementation for BabyBIONN
Dynamically connects VNIs into emergent, cross-domain reasoning networks"""
import asyncio
import logging
import hashlib
import json
import uuid
import random
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import numpy as np

# Import existing BabyBIONN components
from enhanced_vni_classes import EnhancedBaseVNI, VNIManager, NeuralPathway, VNIRegistry
from specialized_vni_base import SpecializedBaseVNI
from vni_orchestrator import VNIOrchestrator
from neuron.vni_storage import VNIStorage
from neuron.vni_messenger import VNIMessenger
from neuron.smart_activation_router import SmartActivationRouter
from neuron.demoHybridAttention import HybridAttentionMechanism

logger = logging.getLogger("neural_mesh")

# ==================== CORE NEURAL MESH TYPES ====================
class MeshNodeState(Enum):
    """States of a neural mesh node"""
    IDLE = "idle"
    ACTIVATING = "activating"
    ACTIVE = "active"
    REFRACTORY = "refractory"
    INTEGRATING = "integrating"

class SynapseType(Enum):
    """Types of synaptic connections"""
    EXCITATORY = "excitatory"  # Strengthens activation
    INHIBITORY = "inhibitory"  # Weakens activation
    MODULATORY = "modulatory"  # Modifies behavior
    GAP_JUNCTION = "gap_junction"  # Direct coupling
    ATTENTIONAL = "attentional"  # Attention-based

@dataclass
class ActivationPulse:
    """Propagation of activation through the mesh"""
    source_node: str
    target_node: str
    strength: float
    activation_type: str  # semantic, structural, attentional
    propagation_path: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def clone(self) -> 'ActivationPulse':
        return ActivationPulse(
            source_node=self.source_node,
            target_node=self.target_node,
            strength=self.strength,
            activation_type=self.activation_type,
            propagation_path=self.propagation_path.copy(),
            timestamp=self.timestamp
        )

@dataclass
class MeshNode:
    """A VNI wrapped as a neural mesh node"""
    vni_id: str
    vni_instance: EnhancedBaseVNI
    node_type: str  # sensor, processor, integrator, output
    activation_threshold: float = 0.3
    current_activation: float = 0.0
    resting_potential: float = -0.07  # Membrane potential analogy
    membrane_time_constant: float = 20.0  # ms
    refractory_period: float = 2.0  # ms
    last_fired: Optional[datetime] = None
    
    # Neural properties
    dendrites: List[str] = field(default_factory=list)  # Input connections
    axons: List[str] = field(default_factory=list)      # Output connections
    neurotransmitter_levels: Dict[str, float] = field(default_factory=dict)
    
    # Dynamic state
    state: MeshNodeState = MeshNodeState.IDLE
    activation_history: deque = field(default_factory=lambda: deque(maxlen=100))
    integration_buffer: List[ActivationPulse] = field(default_factory=list)
    
    def activate(self, pulse: ActivationPulse) -> float:
        """Process activation pulse and update node state"""
        
        # Check refractory period
        if self.state == MeshNodeState.REFRACTORY:
            if self.last_fired:
                elapsed = (datetime.now() - self.last_fired).total_seconds() * 1000
                if elapsed < self.refractory_period:
                    return 0.0
        
        # Calculate membrane potential change
        delta_v = pulse.strength * self._get_receptor_sensitivity(pulse.activation_type)
        
        # Integrate with current activation (leaky integrate-and-fire model)
        decay = np.exp(-self.membrane_time_constant / 1000.0)  # Simplified
        self.current_activation = self.current_activation * decay + delta_v
        
        # Store in buffer
        self.integration_buffer.append(pulse)
        
        # Check for firing threshold
        if self.current_activation >= self.activation_threshold:
            return self._fire()
        
        return 0.0
    
    def _fire(self) -> float:
        """Node fires, sending activation to connected nodes"""
        self.state = MeshNodeState.ACTIVE
        self.last_fired = datetime.now()
        
        # Record activation
        self.activation_history.append({
            'timestamp': datetime.now().isoformat(),
            'activation': self.current_activation,
            'state': self.state.value
        })
        
        # Calculate output strength (action potential)
        output_strength = min(1.0, self.current_activation * 1.2)
        
        # Reset after firing
        self.current_activation = self.resting_potential
        self.state = MeshNodeState.REFRACTORY
        self.integration_buffer.clear()
        
        return output_strength
    
    def _get_receptor_sensitivity(self, activation_type: str) -> float:
        """Get sensitivity to different activation types"""
        sensitivities = {
            'semantic': 1.0,
            'structural': 0.8,
            'attentional': 1.2,
            'emotional': 0.9,
            'procedural': 0.7
        }
        return sensitivities.get(activation_type, 1.0)
    
    def get_readiness(self) -> float:
        """Get node readiness for activation"""
        if self.state == MeshNodeState.REFRACTORY:
            return 0.0
        return 1.0 - (self.current_activation / self.activation_threshold)

@dataclass
class MeshSynapse:
    """Dynamic connection between mesh nodes"""
    id: str
    source_node: str
    target_node: str
    base_strength: float = 0.5
    current_strength: float = 0.5
    synapse_type: SynapseType = SynapseType.EXCITATORY
    plasticity_rate: float = 0.05  # Learning rate
    last_activated: Optional[datetime] = None
    activation_count: int = 0
    success_count: int = 0
    
    # Short-term plasticity
    facilitation: float = 0.0  # Short-term potentiation
    depression: float = 0.0    # Short-term depression
    stp_time_constant: float = 1000.0  # ms
    
    # Long-term plasticity
    ltp_threshold: float = 0.7  # Long-term potentiation threshold
    ltd_threshold: float = 0.3  # Long-term depression threshold
    
    def transmit(self, input_strength: float) -> float:
        """Transmit signal through synapse"""
        self.last_activated = datetime.now()
        self.activation_count += 1
        
        # Calculate short-term plasticity effects
        stp_factor = self._calculate_stp_factor()
        
        # Calculate transmission
        if self.synapse_type == SynapseType.EXCITATORY:
            output = input_strength * self.current_strength * stp_factor
        elif self.synapse_type == SynapseType.INHIBITORY:
            output = -input_strength * self.current_strength * stp_factor
        elif self.synapse_type == SynapseType.MODULATORY:
            output = input_strength * (0.5 + self.current_strength * 0.5) * stp_factor
        else:
            output = input_strength * self.current_strength
        
        return max(-1.0, min(1.0, output))  # Clamp to [-1, 1]
    
    def _calculate_stp_factor(self) -> float:
        """Calculate short-term plasticity factor"""
        # Simplified model: facilitation increases, depression decreases
        stp_factor = 1.0 + self.facilitation - self.depression
        
        # Decay over time
        if self.last_activated:
            elapsed = (datetime.now() - self.last_activated).total_seconds() * 1000
            decay = np.exp(-elapsed / self.stp_time_constant)
            self.facilitation *= decay
            self.depression *= decay
        
        return max(0.1, min(2.0, stp_factor))
    
    def update_plasticity(self, success: bool):
        """Update synaptic plasticity based on success"""
        if success:
            self.success_count += 1
            
            # Hebbian learning: neurons that fire together, wire together
            if self.activation_count > 10 and self.success_count / self.activation_count > self.ltp_threshold:
                self.current_strength = min(1.0, self.current_strength + self.plasticity_rate)
                self.facilitation += 0.1  # Short-term facilitation
        else:
            # Anti-Hebbian learning
            if self.activation_count > 5 and self.success_count / self.activation_count < self.ltd_threshold:
                self.current_strength = max(0.1, self.current_strength - self.plasticity_rate * 0.5)
                self.depression += 0.1  # Short-term depression
    
    def get_efficiency(self) -> float:
        """Get synaptic transmission efficiency"""
        recency_bonus = 0.0
        if self.last_activated:
            elapsed = (datetime.now() - self.last_activated).total_seconds()
            recency_bonus = max(0.0, 1.0 - elapsed / 3600.0)  # 1-hour half-life
        
        return self.current_strength * (0.8 + 0.2 * recency_bonus)

@dataclass
class SynapticPattern:
    """Emergent pattern of synaptic activation"""
    pattern_id: str
    participating_nodes: Set[str]
    connection_matrix: Dict[Tuple[str, str], float]  # (node1, node2) -> connection_strength
    activation_sequence: List[str]  # Sequence of node activations
    pattern_strength: float = 0.5
    pattern_frequency: int = 1
    last_activated: Optional[datetime] = None
    
    # Pattern properties
    coherence_score: float = 0.0  # How coherent the pattern is
    stability_score: float = 0.0  # How stable over time
    utility_score: float = 0.0    # How useful the pattern is
    
    def activate(self, activation_vector: Dict[str, float]) -> float:
        """Activate pattern and return activation level"""
        activation_sum = 0.0
        activated_nodes = 0
        
        for node in self.participating_nodes:
            node_activation = activation_vector.get(node, 0.0)
            activation_sum += node_activation
            if node_activation > 0.3:
                activated_nodes += 1
        
        # Calculate pattern activation
        if activated_nodes > 0:
            pattern_activation = activation_sum / len(self.participating_nodes)
            self.pattern_strength = min(1.0, self.pattern_strength + 0.05)
            self.last_activated = datetime.now()
            return pattern_activation
        
        return 0.0
    
    def evolve(self, success_metric: float):
        """Evolve pattern based on success"""
        if success_metric > 0.7:
            # Strengthen successful patterns
            for connection in self.connection_matrix:
                self.connection_matrix[connection] = min(
                    1.0, self.connection_matrix[connection] + 0.1
                )
            self.coherence_score = min(1.0, self.coherence_score + 0.05)
        else:
            # Weaken unsuccessful patterns
            for connection in self.connection_matrix:
                self.connection_matrix[connection] = max(
                    0.1, self.connection_matrix[connection] - 0.05
                )
        
        self.pattern_frequency += 1

# ==================== NEURAL MESH CORE ====================
class NeuralMeshCore:
    """Core neural mesh coordinating VNI interactions"""
    
    def __init__(self, vni_manager: VNIManager):
        self.vni_manager = vni_manager
        self.mesh_nodes: Dict[str, MeshNode] = {}
        self.mesh_synapses: Dict[str, MeshSynapse] = {}  # key: "source->target"
        self.synaptic_patterns: Dict[str, SynapticPattern] = {}
        
        # Integration with existing modules
        self.router = SmartActivationRouter()
        self.orchestrator = VNIOrchestrator()
        self.storage = VNIStorage()
        self.messenger = VNIMessenger()
        self.attention = HybridAttentionMechanism()
        
        # Mesh properties
        self.global_inhibition: float = 0.0
        self.global_excitation: float = 1.0
        self.mesh_learning_rate: float = 0.1
        self.activation_wave_speed: float = 100.0  # ms per hop
        
        # State tracking
        self.activation_waves: List[Dict] = []
        self.causal_chains: List[Dict] = []
        self.emergent_patterns: List[Dict] = []
        
        # Initialize mesh
        self._initialize_mesh_from_existing()
        
        logger.info(f"🧠 Neural Mesh Core initialized with {len(self.mesh_nodes)} nodes, "
                   f"{len(self.mesh_synapses)} synapses")
    
    def _initialize_mesh_from_existing(self):
        """Initialize mesh from existing VNIs and pathways"""
        
        # Create mesh nodes from existing VNIs
        for vni_id, vni in self.vni_manager.vni_instances.items():
            # Determine node type based on VNI type
            node_type = self._determine_node_type(vni)
            
            node = MeshNode(
                vni_id=vni_id,
                vni_instance=vni,
                node_type=node_type,
                activation_threshold=self._calculate_activation_threshold(vni),
                resting_potential=-0.07 if node_type == "processor" else -0.05
            )
            self.mesh_nodes[vni_id] = node
        
        # Create synapses from existing neural pathways
        for pathway_id, pathway in self.vni_manager.neural_pathways.items():
            synapse_key = f"{pathway.source}->{pathway.target}"
            
            # Determine synapse type based on VNI types
            source_vni = self.vni_manager.vni_instances.get(pathway.source)
            target_vni = self.vni_manager.vni_instances.get(pathway.target)
            synapse_type = self._determine_synapse_type(source_vni, target_vni)
            
            synapse = MeshSynapse(
                id=synapse_key,
                source_node=pathway.source,
                target_node=pathway.target,
                current_strength=pathway.strength,
                synapse_type=synapse_type,
                last_activated=pathway.last_activated
            )
            self.mesh_synapses[synapse_key] = synapse
            
            # Update node connections
            if pathway.source in self.mesh_nodes:
                self.mesh_nodes[pathway.source].axons.append(pathway.target)
            if pathway.target in self.mesh_nodes:
                self.mesh_nodes[pathway.target].dendrites.append(pathway.source)
        
        # Initialize with random connections for emergent patterns
        self._create_initial_random_connections()
        
        logger.info(f"Mesh initialized: {len(self.mesh_nodes)} nodes, "
                   f"{len(self.mesh_synapses)} synapses")
    
    def _determine_node_type(self, vni: EnhancedBaseVNI) -> str:
        """Determine mesh node type from VNI type"""
        vni_type = vni.vni_type.lower()
        
        if vni_type in ['medical', 'legal', 'technical']:
            return "processor"
        elif vni_type in ['sensor', 'input', 'perception']:
            return "sensor"
        elif vni_type in ['integrator', 'analytical', 'synthetic']:
            return "integrator"
        elif vni_type in ['output', 'actuator', 'executive']:
            return "output"
        else:
            return "processor"
    
    def _calculate_activation_threshold(self, vni: EnhancedBaseVNI) -> float:
        """Calculate activation threshold based on VNI properties"""
        # More specialized VNIs have higher thresholds (more selective)
        base_threshold = 0.3
        
        # Adjust based on VNI type
        if vni.vni_type == 'medical':
            return 0.4  # Medical needs higher confidence
        elif vni.vni_type == 'legal':
            return 0.35  # Legal needs precision
        elif vni.vni_type == 'general':
            return 0.25  # General is more flexible
        else:
            return base_threshold
    
    def _determine_synapse_type(self, source_vni: Optional[EnhancedBaseVNI], 
                               target_vni: Optional[EnhancedBaseVNI]) -> SynapseType:
        """Determine synapse type based on VNI relationships"""
        if not source_vni or not target_vni:
            return SynapseType.EXCITATORY
        
        # Complementary domains have excitatory connections
        complementary_pairs = [
            ('medical', 'legal'), ('legal', 'technical'), 
            ('technical', 'analytical'), ('analytical', 'creative')
        ]
        
        source_type = source_vni.vni_type
        target_type = target_vni.vni_type
        
        if (source_type, target_type) in complementary_pairs:
            return SynapseType.EXCITATORY
        elif source_type == target_type:
            return SynapseType.MODULATORY  # Same type modulates
        else:
            return SynapseType.EXCITATORY  # Default
    
    def _create_initial_random_connections(self):
        """Create initial random connections for emergent pattern formation"""
        node_ids = list(self.mesh_nodes.keys())
        
        # Create random connections (small-world network)
        for i in range(min(20, len(node_ids) * 3)):  # Limited random connections
            source = random.choice(node_ids)
            target = random.choice(node_ids)
            
            if source != target:
                synapse_key = f"{source}->{target}"
                if synapse_key not in self.mesh_synapses:
                    synapse = MeshSynapse(
                        id=synapse_key,
                        source_node=source,
                        target_node=target,
                        current_strength=random.uniform(0.1, 0.4),  # Weak initial connections
                        synapse_type=random.choice(list(SynapseType))
                    )
                    self.mesh_synapses[synapse_key] = synapse
                    
                    # Update node connections
                    self.mesh_nodes[source].axons.append(target)
                    self.mesh_nodes[target].dendrites.append(source)
    
    async def process_with_mesh(self, query: str, context: Dict = None, 
                               session_id: str = "default") -> Dict[str, Any]:
        """Process query using neural mesh for emergent intelligence 
        Args:
            query: User query
            context: Additional context
            session_id: Session identifier  
        Returns:
            Enhanced response with mesh intelligence"""
        # Phase 1: Query analysis and mesh preparation
        logger.info(f"🔍 Phase 1: Analyzing query for mesh activation")
        query_analysis = self._analyze_query_for_mesh(query, context)
        
        # Phase 2: Neural mesh activation
        logger.info(f"⚡ Phase 2: Activating neural mesh")
        mesh_activation = await self._activate_neural_mesh(query, query_analysis)
        
        # Phase 3: Parallel VNI processing
        logger.info(f"🌀 Phase 3: Parallel VNI processing")
        vni_responses = await self._process_parallel_vnis(
            mesh_activation, query, context, session_id
        )
        
        # Phase 4: Causal reasoning and synthesis
        logger.info(f"🔗 Phase 4: Causal reasoning and synthesis")
        final_response = await self._synthesize_with_causal_reasoning(
            vni_responses, query, query_analysis, mesh_activation
        )
        
        # Phase 5: Mesh learning and adaptation
        logger.info(f"📚 Phase 5: Mesh learning and adaptation")
        await self._learn_from_interaction(
            mesh_activation, vni_responses, final_response
        )
        
        # Add mesh context to response
        final_response['neural_mesh_context'] = {
            'activated_nodes': list(mesh_activation['activated_nodes']),
            'activation_pattern': mesh_activation['activation_pattern'],
            'causal_depth': len(final_response.get('causal_chain', [])),
            'mesh_synthesis': True,
            'timestamp': datetime.now().isoformat()
        }
        
        return final_response
    
    def _analyze_query_for_mesh(self, query: str, context: Dict) -> Dict[str, Any]:
        """Analyze query to determine mesh activation strategy"""
        
        # Use existing router for initial analysis
        router_analysis = self.router.analyze_query(query, context)
        
        # Calculate domain relevance for each VNI
        domain_relevance = {}
        for vni_id, vni in self.vni_manager.vni_instances.items():
            relevance = self._calculate_vni_relevance(vni, query, context)
            domain_relevance[vni_id] = {
                'relevance': relevance,
                'vni_type': vni.vni_type,
                'confidence': vni.calculate_confidence([], [])
            }
        
        # Detect query complexity
        complexity = self._assess_query_complexity(query)
        
        # Detect cross-domain requirements
        cross_domain_needs = self._detect_cross_domain_needs(query, domain_relevance)
        
        return {
            'query_text': query,
            'query_hash': hashlib.md5(query.encode()).hexdigest()[:8],
            'router_analysis': router_analysis,
            'domain_relevance': domain_relevance,
            'complexity_score': complexity,
            'cross_domain_needs': cross_domain_needs,
            'requires_mesh': complexity > 0.4 or len(cross_domain_needs) > 1,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_vni_relevance(self, vni: EnhancedBaseVNI, query: str, 
                                context: Dict) -> float:
        """Calculate relevance of VNI to query"""
        
        # Check if VNI has should_handle method
        if hasattr(vni, 'should_handle'):
            try:
                return 1.0 if vni.should_handle(query) else 0.0
            except:
                pass
        
        # Fallback: keyword matching
        concepts, patterns = vni.extract_concepts_and_patterns(query)
        confidence = vni.calculate_confidence(concepts, patterns)
        
        # Boost relevance if VNI type matches query context
        if context and 'preferred_domains' in context:
            if vni.vni_type in context['preferred_domains']:
                confidence = min(1.0, confidence * 1.5)
        
        return confidence
    
    def _assess_query_complexity(self, query: str) -> float:
        """Assess query complexity"""
        complexity_factors = []
        
        # Length factor
        word_count = len(query.split())
        complexity_factors.append(min(1.0, word_count / 50.0))
        
        # Domain diversity (count unique domain keywords)
        domain_keywords = {
            'medical': ['health', 'doctor', 'patient', 'treatment', 'symptom'],
            'legal': ['law', 'contract', 'legal', 'rights', 'compliance'],
            'technical': ['code', 'system', 'technical', 'software', 'algorithm'],
            'analytical': ['analyze', 'compare', 'evaluate', 'statistics'],
            'creative': ['create', 'design', 'innovate', 'imagine']
        }
        
        domain_counts = defaultdict(int)
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in query.lower():
                    domain_counts[domain] += 1
        
        # Diversity factor
        diversity = len([d for d in domain_counts if domain_counts[d] > 0])
        complexity_factors.append(min(1.0, diversity / 5.0))
        
        # Syntactic complexity (rough measure)
        syntactic_complex = len([c for c in query if c in [',', ';', ':']])
        complexity_factors.append(min(1.0, syntactic_complex / 5.0))
        
        # Average complexity factors
        return sum(complexity_factors) / len(complexity_factors) if complexity_factors else 0.5
    
    def _detect_cross_domain_needs(self, query: str, 
                                  domain_relevance: Dict) -> List[str]:
        """Detect which domains are needed for the query"""
        
        # Find domains with relevance above threshold
        relevant_domains = []
        threshold = 0.3
        
        for vni_id, relevance_info in domain_relevance.items():
            if relevance_info['relevance'] > threshold:
                domain = relevance_info['vni_type']
                if domain not in relevant_domains:
                    relevant_domains.append(domain)
        
        # Check for explicit cross-domain indicators
        cross_domain_indicators = [
            'both', 'and', 'also', 'combined', 'together', 
            'multiple', 'various', 'different'
        ]
        
        has_cross_domain_indicators = any(
            indicator in query.lower() for indicator in cross_domain_indicators
        )
        
        if has_cross_domain_indicators and len(relevant_domains) == 1:
            # Query suggests multiple domains but only one detected
            # Add complementary domains
            complementary = {
                'medical': ['legal', 'technical'],
                'legal': ['medical', 'technical'],
                'technical': ['analytical', 'creative'],
                'analytical': ['technical', 'creative'],
                'creative': ['analytical', 'technical']
            }
            
            primary = relevant_domains[0] if relevant_domains else 'general'
            if primary in complementary:
                relevant_domains.extend(complementary[primary][:2])
        
        return relevant_domains
    
    async def _activate_neural_mesh(self, query: str, 
                                   query_analysis: Dict) -> Dict[str, Any]:
        """Activate neural mesh based on query analysis"""
        
        # Initial activation based on domain relevance
        initial_activation = {}
        for vni_id, relevance_info in query_analysis['domain_relevance'].items():
            initial_activation[vni_id] = relevance_info['relevance']
        
        # Create activation wave
        activation_wave = {
            'query': query,
            'initial_activation': initial_activation,
            'activated_nodes': set(),
            'activation_pattern': {},
            'propagation_paths': [],
            'start_time': datetime.now(),
            'wave_id': str(uuid.uuid4())[:8]
        }
        
        # Propagate activation through mesh
        await self._propagate_activation_wave(activation_wave)
        
        # Extract activation pattern
        activation_pattern = self._extract_activation_pattern(activation_wave)
        
        return {
            'wave_id': activation_wave['wave_id'],
            'activated_nodes': activation_wave['activated_nodes'],
            'activation_pattern': activation_pattern,
            'propagation_paths': activation_wave['propagation_paths'],
            'activation_duration': (datetime.now() - activation_wave['start_time']).total_seconds(),
            'query_analysis': query_analysis
        }
    
    async def _propagate_activation_wave(self, activation_wave: Dict):
        """Propagate activation wave through neural mesh"""
        
        # Initial activation queue
        activation_queue = []
        for vni_id, activation in activation_wave['initial_activation'].items():
            if activation > 0.1:  # Threshold for initial activation
                activation_queue.append((vni_id, activation, []))
        
        # Breadth-first propagation
        visited = set()
        max_iterations = 100  # Safety limit
        iteration = 0
        
        while activation_queue and iteration < max_iterations:
            iteration += 1
            
            # Process current activation front
            next_queue = []
            
            for vni_id, activation_strength, path in activation_queue:
                if vni_id in visited:
                    continue
                
                visited.add(vni_id)
                
                # Activate node
                node_activation = self._activate_mesh_node(
                    vni_id, activation_strength, path
                )
                
                if node_activation > 0.2:  # Node activated significantly
                    activation_wave['activated_nodes'].add(vni_id)
                    activation_wave['activation_pattern'][vni_id] = node_activation
                    
                    # Record propagation path
                    full_path = path + [vni_id]
                    if len(full_path) > 1:
                        activation_wave['propagation_paths'].append({
                            'path': full_path,
                            'strength': node_activation,
                            'length': len(full_path)
                        })
                    
                    # Propagate to connected nodes
                    if vni_id in self.mesh_nodes:
                        node = self.mesh_nodes[vni_id]
                        
                        for target_id in node.axons:
                            if target_id not in visited:
                                synapse_key = f"{vni_id}->{target_id}"
                                if synapse_key in self.mesh_synapses:
                                    synapse = self.mesh_synapses[synapse_key]
                                    transmitted = synapse.transmit(node_activation)
                                    
                                    if transmitted > 0.1:  # Significant transmission
                                        next_queue.append((
                                            target_id, 
                                            transmitted, 
                                            full_path.copy()
                                        ))
            
            # Update queue for next iteration
            activation_queue = next_queue
            
            # Small delay for simulation
            await asyncio.sleep(0.001)
        
        logger.info(f"Activation wave completed: {len(activation_wave['activated_nodes'])} "
                   f"nodes activated in {iteration} iterations")
    
    def _activate_mesh_node(self, vni_id: str, activation_strength: float, 
                           path: List[str]) -> float:
        """Activate a specific mesh node"""
        
        if vni_id not in self.mesh_nodes:
            return 0.0
        
        node = self.mesh_nodes[vni_id]
        
        # Create activation pulse
        pulse = ActivationPulse(
            source_node=path[-1] if path else "query",
            target_node=vni_id,
            strength=activation_strength,
            activation_type="semantic",
            propagation_path=path.copy()
        )
        
        # Activate node
        node_activation = node.activate(pulse)
        
        return node_activation
    
    def _extract_activation_pattern(self, activation_wave: Dict) -> Dict:
        """Extract activation pattern from wave"""
        
        pattern = {
            'node_activations': activation_wave['activation_pattern'],
            'pathways': activation_wave['propagation_paths'],
            'node_count': len(activation_wave['activated_nodes']),
            'activation_summary': {
                'max_activation': max(activation_wave['activation_pattern'].values()) 
                    if activation_wave['activation_pattern'] else 0.0,
                'avg_activation': np.mean(list(activation_wave['activation_pattern'].values())) 
                    if activation_wave['activation_pattern'] else 0.0,
                'activation_variance': np.var(list(activation_wave['activation_pattern'].values())) 
                    if activation_wave['activation_pattern'] else 0.0
            }
        }
        
        # Check for emergent patterns
        emergent = self._detect_emergent_patterns(activation_wave)
        if emergent:
            pattern['emergent_patterns'] = emergent
        
        return pattern
    
    def _detect_emergent_patterns(self, activation_wave: Dict) -> List[Dict]:
        """Detect emergent patterns in activation"""
        
        emergent_patterns = []
        activated_nodes = list(activation_wave['activated_nodes'])
        
        if len(activated_nodes) >= 3:
            # Check for known patterns
            for pattern_id, pattern in self.synaptic_patterns.items():
                overlap = len(set(activated_nodes) & pattern.participating_nodes)
                if overlap >= 3:  # Significant overlap
                    pattern_activation = pattern.activate(
                        activation_wave['activation_pattern']
                    )
                    
                    if pattern_activation > 0.4:
                        emergent_patterns.append({
                            'pattern_id': pattern_id,
                            'activation': pattern_activation,
                            'overlap': overlap,
                            'coherence': pattern.coherence_score
                        })
        
        return emergent_patterns
    
    async def _process_parallel_vnis(self, mesh_activation: Dict, query: str,
                                    context: Dict, session_id: str) -> Dict[str, Any]:
        """Process query in parallel across activated VNIs"""
        
        vni_responses = {}
        activated_nodes = mesh_activation['activated_nodes']
        
        # Prepare async tasks for each activated VNI
        async_tasks = []
        
        for vni_id in activated_nodes:
            if vni_id in self.vni_manager.vni_instances:
                vni = self.vni_manager.vni_instances[vni_id]
                
                task = asyncio.create_task(
                    self._process_single_vni(vni, query, context, session_id, mesh_activation)
                )
                async_tasks.append((vni_id, task))
        
        # Wait for all tasks to complete
        for vni_id, task in async_tasks:
            try:
                response = await task
                vni_responses[vni_id] = response
            except Exception as e:
                logger.error(f"Error processing VNI {vni_id}: {e}")
                vni_responses[vni_id] = {
                    'error': str(e),
                    'response': f"VNI {vni_id} processing failed",
                    'confidence': 0.0
                }
        
        return vni_responses
    
    async def _process_single_vni(self, vni: EnhancedBaseVNI, query: str,
                                 context: Dict, session_id: str, 
                                 mesh_activation: Dict) -> Dict[str, Any]:
        """Process query through single VNI with mesh context"""
        
        # Enhance context with mesh information
        enhanced_context = context.copy() if context else {}
        enhanced_context['neural_mesh'] = {
            'activation_level': mesh_activation['activation_pattern'].get(vni.instance_id, 0.0),
            'mesh_context': mesh_activation,
            'collaborating_vnis': list(mesh_activation['activated_nodes'])
        }
        
        # Process query
        response = vni.process_query(
            query=query,
            context=enhanced_context,
            session_id=session_id,
            use_generation=True,  # Use generation for mesh-enhanced responses
            generation_strategy="bridge"  # Use bridge for better integration
        )
        
        # Add VNI metadata
        response['vni_metadata'] = {
            'instance_id': vni.instance_id,
            'vni_type': vni.vni_type,
            'confidence': response.get('confidence', 0.5),
            'generation_used': response.get('generation_used', False),
            'activation_level': mesh_activation['activation_pattern'].get(vni.instance_id, 0.0),
            'processing_time': datetime.now().isoformat()
        }
        return response
    
    async def _synthesize_with_causal_reasoning(self, vni_responses: Dict, 
                                                query: str, 
                                                query_analysis: Dict,
                                                mesh_activation: Dict) -> Dict[str, Any]:
        """Synthesize responses with causal reasoning"""
        
        if not vni_responses:
            return {
                'response': "No VNIs could process this query effectively.",
                'confidence': 0.1,
                'sources': [],
                'causal_chain': []
            }
        
        # Build causal chain
        causal_chain = self._build_causal_chain(vni_responses, query, query_analysis)
        
        # Weight responses by activation and confidence
        weighted_responses = []
        for vni_id, response in vni_responses.items():
            if 'error' in response:
                continue
            
            activation = mesh_activation['activation_pattern'].get(vni_id, 0.3)
            confidence = response.get('confidence', 0.5)
            
            # Calculate weight
            weight = (activation * 0.4) + (confidence * 0.6)
            
            weighted_responses.append({
                'vni_id': vni_id,
                'response': response['response'],
                'weight': weight,
                'confidence': confidence,
                'activation': activation,
                'vni_type': response.get('vni_metadata', {}).get('vni_type', 'unknown'),
                'full_response': response
            })
        
        # Sort by weight
        weighted_responses.sort(key=lambda x: x['weight'], reverse=True)
        
        # Synthesis strategy
        if len(weighted_responses) == 1:
            synthesized = self._synthesize_single_response(weighted_responses[0], causal_chain)
        elif len(weighted_responses) >= 3:
            synthesized = self._synthesize_multiple_responses(weighted_responses, causal_chain, query)
        else:
            synthesized = self._synthesize_pair_response(weighted_responses, causal_chain)
        
        # Build final response
        final_response = {
            'response': synthesized['response'],
            'confidence': synthesized['confidence'],
            'sources': [r['vni_id'] for r in weighted_responses],
            'causal_chain': causal_chain,
            'synthesis_method': synthesized['method'],
            'contributing_vnis': [
                {
                    'id': r['vni_id'],
                    'type': r['vni_type'],
                    'weight': r['weight'],
                    'activation': r['activation']
                }
                for r in weighted_responses[:5]  # Top 5 contributors
            ],
            'timestamp': datetime.now().isoformat()
        }
        return final_response
    
    def _build_causal_chain(self, vni_responses: Dict, query: str, 
                           query_analysis: Dict) -> List[Dict]:
        """Build causal chain across VNI responses"""
        
        causal_chain = []
        
        # Extract key concepts from query
        query_concepts = self._extract_key_concepts(query)
        
        # For each VNI response, extract causal statements
        for vni_id, response in vni_responses.items():
            if 'error' in response or 'response' not in response:
                continue
            
            response_text = response['response']
            vni_type = response.get('vni_metadata', {}).get('vni_type', 'unknown')
            
            # Extract causal patterns from response
            causal_statements = self._extract_causal_statements(response_text, vni_type)
            
            for statement in causal_statements:
                causal_chain.append({
                    'vni_id': vni_id,
                    'vni_type': vni_type,
                    'cause': statement['cause'],
                    'effect': statement['effect'],
                    'confidence': statement.get('confidence', 0.7),
                    'domain': vni_type
                })
        
        # Connect causal statements across domains
        connected_chain = self._connect_causal_statements(causal_chain, query_concepts)
        return connected_chain
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple implementation - can be enhanced with NLP
        words = text.lower().split()
        
        # Filter out stop words and get meaningful concepts
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        concepts = [word for word in words if word not in stop_words and len(word) > 3]
        
        return list(set(concepts))[:10]  # Return top 10 unique concepts
    
    def _extract_causal_statements(self, text: str, vni_type: str) -> List[Dict]:
        """Extract causal statements from text"""
        
        causal_statements = []
        
        # Look for causal patterns
        causal_patterns = [
            (r'(\w+) causes? (\w+)', 'direct'),
            (r'(\w+) leads? to (\w+)', 'direct'),
            (r'(\w+) results? in (\w+)', 'direct'),
            (r'(\w+) affects? (\w+)', 'influence'),
            (r'(\w+) influences? (\w+)', 'influence'),
            (r'if (\w+) then (\w+)', 'conditional'),
            (r'(\w+) because (\w+)', 'reason'),
        ]
        
        text_lower = text.lower()
        
        for pattern, relation_type in causal_patterns:
            import re
            matches = re.findall(pattern, text_lower)
            
            for match in matches:
                if len(match) == 2:
                    cause, effect = match
                    
                    # Clean up the cause and effect
                    cause = cause.strip()
                    effect = effect.strip()
                    
                    if len(cause) > 2 and len(effect) > 2:  # Meaningful words
                        causal_statements.append({
                            'cause': cause,
                            'effect': effect,
                            'relation': relation_type,
                            'confidence': 0.7 if relation_type == 'direct' else 0.5,
                            'source': 'extracted',
                            'domain': vni_type
                        })
        return causal_statements
    
    def _connect_causal_statements(self, statements: List[Dict], 
                                  query_concepts: List[str]) -> List[Dict]:
        """Connect causal statements into a chain"""
        
        if not statements:
            return []
        
        # Group by domain
        domain_statements = defaultdict(list)
        for statement in statements:
            domain_statements[statement['domain']].append(statement)
        
        # Try to build chains
        chains = []
        
        for start_concept in query_concepts[:3]:  # Try first 3 concepts as starting points
            chain = []
            current_concept = start_concept
            
            # Try to build chain of length up to 5
            for _ in range(5):
                # Find statements where current concept is cause
                possible_next = []
                
                for domain, domain_stmts in domain_statements.items():
                    for stmt in domain_stmts:
                        if stmt['cause'] in current_concept.lower() or current_concept.lower() in stmt['cause']:
                            possible_next.append((stmt, domain))
                
                if possible_next:
                    # Choose highest confidence statement
                    possible_next.sort(key=lambda x: x[0]['confidence'], reverse=True)
                    next_stmt, next_domain = possible_next[0]
                    
                    chain.append({
                        'from': current_concept,
                        'to': next_stmt['effect'],
                        'domain': next_domain,
                        'confidence': next_stmt['confidence'],
                        'relation': next_stmt['relation']
                    })
                    
                    current_concept = next_stmt['effect']
                else:
                    break
            
            if chain:
                chains.append(chain)
        
        # Return the longest chain
        if chains:
            chains.sort(key=len, reverse=True)
            return chains[0]
        return []
    
    def _synthesize_single_response(self, top_response: Dict, 
                                    causal_chain: List) -> Dict[str, Any]:
        """Synthesize when only one VNI is strongly activated"""
        
        response = top_response['response']
        vni_type = top_response['vni_type']
        
        # Enhance with causal chain if available
        if causal_chain:
            causal_summary = self._summarize_causal_chain(causal_chain, vni_type)
            response = f"{response}\n\n**Causal Analysis:** {causal_summary}"
        
        return {
            'response': response,
            'confidence': top_response['confidence'],
            'method': 'single_vni_enhanced'
        }
    
    def _synthesize_pair_response(self, responses: List[Dict], 
                                  causal_chain: List) -> Dict[str, Any]:
        """Synthesize when two VNIs are activated"""
        
        if len(responses) < 2:
            return self._synthesize_single_response(responses[0], causal_chain)
        
        resp1, resp2 = responses[0], responses[1]
        
        # Check if responses are complementary
        if self._are_responses_complementary(resp1, resp2):
            synthesized = self._combine_complementary_responses(resp1, resp2)
        else:
            # Use the stronger response
            if resp1['weight'] > resp2['weight']:
                synthesized = resp1['response']
            else:
                synthesized = resp2['response']
        
        # Add causal context
        if causal_chain:
            causal_context = self._add_causal_context(causal_chain, [resp1['vni_type'], resp2['vni_type']])
            synthesized = f"{synthesized}\n\n**Cross-Domain Analysis:** {causal_context}"
        
        # Calculate combined confidence
        combined_confidence = (resp1['confidence'] + resp2['confidence']) / 2
        return {
            'response': synthesized,
            'confidence': combined_confidence,
            'method': 'pair_synthesis'
        }
    
    def _synthesize_multiple_responses(self, responses: List[Dict], 
                                       causal_chain: List, 
                                       query: str) -> Dict[str, Any]:
        """Synthesize when multiple VNIs are activated"""
        
        # Group by VNI type
        type_groups = defaultdict(list)
        for resp in responses:
            type_groups[resp['vni_type']].append(resp)
        
        # Get best response from each type
        best_by_type = []
        for vni_type, type_responses in type_groups.items():
            type_responses.sort(key=lambda x: x['weight'], reverse=True)
            best_by_type.append(type_responses[0])
        
        # Build multi-perspective synthesis
        synthesis_parts = []
        
        for i, resp in enumerate(best_by_type):
            if i == 0:
                # Primary perspective
                synthesis_parts.append(f"**Primary Analysis ({resp['vni_type']}):** {resp['response']}")
            else:
                # Secondary perspectives
                synthesis_parts.append(f"**{resp['vni_type'].title()} Perspective:** {resp['response'][:150]}...")
        
        synthesized = "\n\n".join(synthesis_parts)
        
        # Add integrative conclusion
        if len(best_by_type) >= 2:
            conclusion = self._generate_integrative_conclusion(best_by_type, query)
            synthesized = f"{synthesized}\n\n**Integrative Conclusion:** {conclusion}"
        
        # Calculate average confidence
        avg_confidence = np.mean([r['confidence'] for r in best_by_type])
        return {
            'response': synthesized,
            'confidence': avg_confidence,
            'method': 'multi_perspective_synthesis'
        }
    
    def _are_responses_complementary(self, resp1: Dict, resp2: Dict) -> bool:
        """Check if two responses are complementary"""
        # Responses are complementary if:
        # 1. They're from different VNI types
        # 2. They don't contradict each other
        # 3. They address different aspects
        
        if resp1['vni_type'] == resp2['vni_type']:
            return False
        
        # Simple contradiction check
        contradictions = [
            ('yes', 'no'), ('should', 'should not'), ('recommend', 'avoid'),
            ('true', 'false'), ('correct', 'incorrect')
        ]
        
        resp1_lower = resp1['response'].lower()
        resp2_lower = resp2['response'].lower()
        
        for contra1, contra2 in contradictions:
            if contra1 in resp1_lower and contra2 in resp2_lower:
                return False
        return True
    
    def _combine_complementary_responses(self, resp1: Dict, resp2: Dict) -> str:
        """Combine two complementary responses"""
        # Determine order based on domain hierarchy
        domain_order = ['medical', 'legal', 'technical', 'analytical', 'creative', 'general']
        
        resp1_idx = domain_order.index(resp1['vni_type']) if resp1['vni_type'] in domain_order else 99
        resp2_idx = domain_order.index(resp2['vni_type']) if resp2['vni_type'] in domain_order else 99
        
        if resp1_idx < resp2_idx:
            first, second = resp1, resp2
        else:
            first, second = resp2, resp1
        return f"{first['response']}\n\nFrom a {second['vni_type']} perspective: {second['response']}"
    
    def _summarize_causal_chain(self, causal_chain: List, primary_domain: str) -> str:
        """Summarize causal chain for inclusion in response"""
        if not causal_chain:
            return "No causal relationships identified."
        
        # Filter for primary domain or strongest relationships
        relevant_links = []
        for link in causal_chain:
            if link['domain'] == primary_domain or link['confidence'] > 0.7:
                relevant_links.append(link)
        
        if not relevant_links:
            relevant_links = causal_chain[:2]  # Take first two
        
        # Build summary
        parts = []
        for link in relevant_links[:3]:  # Max 3 links
            parts.append(f"{link['from']} → {link['to']} ({link['domain']})")
        return "; ".join(parts)
    
    def _add_causal_context(self, causal_chain: List, domains: List[str]) -> str:
        """Add causal context for multiple domains"""
        if not causal_chain:
            return "Cross-domain relationships require further analysis."
        
        # Group by domain
        domain_links = defaultdict(list)
        for link in causal_chain:
            domain_links[link['domain']].append(link)
        
        # Build context
        context_parts = []
        for domain in domains:
            if domain in domain_links:
                links = domain_links[domain][:2]  # Max 2 links per domain
                for link in links:
                    context_parts.append(f"In {domain}: {link['from']} affects {link['to']}")
        return " | ".join(context_parts)
    
    def _generate_integrative_conclusion(self, responses: List[Dict], query: str) -> str:
        """Generate integrative conclusion from multiple perspectives"""
        domains = [r['vni_type'] for r in responses]
        
        if 'medical' in domains and 'legal' in domains:
            return "This requires balancing healthcare considerations with legal compliance."
        elif 'technical' in domains and 'analytical' in domains:
            return "Technical implementation should be guided by analytical rigor."
        elif 'creative' in domains and 'analytical' in domains:
            return "Creative solutions should be evaluated with analytical precision."
        elif len(domains) >= 3:
            return f"Multiple perspectives from {', '.join(domains)} suggest a comprehensive, multi-faceted approach."
        else:
            return "Integrating these perspectives provides a more complete understanding."
    
    async def _learn_from_interaction(self, mesh_activation: Dict,
                                     vni_responses: Dict,
                                     final_response: Dict):
        """Learn from the interaction and update mesh"""
        
        # Calculate success metric
        success_metric = self._calculate_success_metric(
            final_response, mesh_activation, vni_responses
        )
        
        # Update synapses based on success
        self._update_synapses(mesh_activation, success_metric)
        
        # Update or create synaptic patterns
        self._update_patterns(mesh_activation, success_metric)
        
        # Update node properties
        self._update_nodes(mesh_activation, vni_responses, success_metric)
        
        # Store interaction for learning
        self._store_interaction_for_learning(
            mesh_activation, vni_responses, final_response, success_metric
        )
    
    def _calculate_success_metric(self, final_response: Dict,
                                 mesh_activation: Dict,
                                 vni_responses: Dict) -> float:
        """Calculate overall success metric"""
        
        # Base success on response confidence
        base_success = final_response.get('confidence', 0.5)
        
        # Adjust based on number of contributing VNIs
        num_contributors = len(final_response.get('sources', []))
        if num_contributors > 1:
            # Multiple VNIs collaborating is good
            base_success = min(1.0, base_success * (1.0 + 0.1 * (num_contributors - 1)))
        
        # Adjust based on causal chain depth
        causal_depth = len(final_response.get('causal_chain', []))
        if causal_depth > 2:
            base_success = min(1.0, base_success * (1.0 + 0.05 * causal_depth))
        return base_success
    
    def _update_synapses(self, mesh_activation: Dict, success_metric: float):
        """Update synaptic strengths based on activation and success"""
        # Get activated pathways
        for pathway in mesh_activation.get('propagation_paths', []):
            path = pathway['path']
            
            # Update each synapse in the path
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                synapse_key = f"{source}->{target}"
                
                if synapse_key in self.mesh_synapses:
                    synapse = self.mesh_synapses[synapse_key]
                    
                    # Update based on success
                    synapse.update_plasticity(success_metric > 0.6)
                    
                    # Update strength based on activation
                    activation_strength = mesh_activation['activation_pattern'].get(target, 0.0)
                    if activation_strength > 0.3:
                        synapse.strengthen()
    
    def _update_patterns(self, mesh_activation: Dict, success_metric: float):
        """Update or create synaptic patterns"""
        
        activated_nodes = mesh_activation['activated_nodes']
        
        if len(activated_nodes) >= 3:
            # Check for existing pattern
            pattern_found = False
            
            for pattern_id, pattern in self.synaptic_patterns.items():
                overlap = len(activated_nodes & pattern.participating_nodes)
                if overlap >= 3:  # Significant overlap
                    pattern.evolve(success_metric)
                    pattern_found = True
                    break
            
            # Create new pattern if none found
            if not pattern_found and len(activated_nodes) >= 3:
                pattern_id = f"pattern_{len(self.synaptic_patterns) + 1}"
                
                # Create connection matrix
                connection_matrix = {}
                nodes_list = list(activated_nodes)
                
                # Create connections between activated nodes
                for i, node1 in enumerate(nodes_list):
                    for j, node2 in enumerate(nodes_list[i+1:], i+1):
                        synapse_key = f"{node1}->{node2}"
                        if synapse_key in self.mesh_synapses:
                            connection_matrix[(node1, node2)] = self.mesh_synapses[synapse_key].current_strength
                        else:
                            connection_matrix[(node1, node2)] = 0.3  # Default weak connection
                
                pattern = SynapticPattern(
                    pattern_id=pattern_id,
                    participating_nodes=activated_nodes,
                    connection_matrix=connection_matrix,
                    activation_sequence=list(activated_nodes),
                    pattern_strength=0.5,
                    pattern_frequency=1,
                    last_activated=datetime.now(),
                    coherence_score=0.6,
                    stability_score=0.5,
                    utility_score=success_metric
                )
                
                self.synaptic_patterns[pattern_id] = pattern
                logger.info(f"Created new synaptic pattern: {pattern_id} with {len(activated_nodes)} nodes")
    
    def _update_nodes(self, mesh_activation: Dict, vni_responses: Dict, 
                     success_metric: float):
        """Update node properties based on performance"""
        for vni_id, activation in mesh_activation['activation_pattern'].items():
            if vni_id in self.mesh_nodes:
                node = self.mesh_nodes[vni_id]
                
                # Adjust activation threshold based on performance
                if vni_id in vni_responses:
                    vni_response = vni_responses[vni_id]
                    confidence = vni_response.get('confidence', 0.5)
                    
                    if confidence > 0.7 and activation > 0.5:
                        # Good performance - slightly lower threshold
                        node.activation_threshold = max(0.2, node.activation_threshold - 0.02)
                    elif confidence < 0.3 and activation > 0.5:
                        # Poor performance - slightly raise threshold
                        node.activation_threshold = min(0.8, node.activation_threshold + 0.02)
    
    def _store_interaction_for_learning(self, mesh_activation: Dict,
                                       vni_responses: Dict,
                                       final_response: Dict,
                                       success_metric: float):
        """Store interaction for offline learning"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'query': mesh_activation.get('query_analysis', {}).get('query_text', ''),
            'mesh_activation': {
                'activated_nodes': list(mesh_activation['activated_nodes']),
                'activation_pattern': mesh_activation['activation_pattern'],
                'wave_id': mesh_activation.get('wave_id', '')
            },
            'vni_responses': {
                vni_id: {
                    'confidence': resp.get('confidence', 0.0),
                    'generation_used': resp.get('generation_used', False)
                }
                for vni_id, resp in vni_responses.items()
            },
            'final_response': {
                'confidence': final_response.get('confidence', 0.0),
                'causal_depth': len(final_response.get('causal_chain', [])),
                'sources': final_response.get('sources', [])
            },
            'success_metric': success_metric,
            'learning_timestamp': datetime.now().isoformat()
        }
        
        # Store in memory (could be persisted to storage)
        if hasattr(self, 'interaction_history'):
            self.interaction_history.append(interaction)
        
        logger.info(f"Stored interaction for learning: success={success_metric:.2f}, "
                   f"nodes={len(mesh_activation['activated_nodes'])}")
    
    def get_mesh_status(self) -> Dict[str, Any]:
        """Get current mesh status"""
        active_nodes = sum(1 for node in self.mesh_nodes.values() 
                          if node.state == MeshNodeState.ACTIVE)
        
        strong_synapses = sum(1 for synapse in self.mesh_synapses.values() 
                             if synapse.current_strength > 0.7)
        
        # Calculate mesh health metrics
        avg_node_activation = np.mean([node.current_activation 
                                      for node in self.mesh_nodes.values()])
        
        avg_synapse_strength = np.mean([synapse.current_strength 
                                       for synapse in self.mesh_synapses.values()])
        
        return {
            'total_nodes': len(self.mesh_nodes),
            'active_nodes': active_nodes,
            'total_synapses': len(self.mesh_synapses),
            'strong_synapses': strong_synapses,
            'synaptic_patterns': len(self.synaptic_patterns),
            'mesh_health': {
                'avg_node_activation': avg_node_activation,
                'avg_synapse_strength': avg_synapse_strength,
                'global_inhibition': self.global_inhibition,
                'global_excitation': self.global_excitation
            },
            'recent_activity': {
                'last_activation_wave': self.activation_waves[-1] if self.activation_waves else None,
                'pattern_activations': len([p for p in self.synaptic_patterns.values() 
                                          if p.last_activated and 
                                          (datetime.now() - p.last_activated).seconds < 3600])
            }
        }
    
    def visualize_mesh(self) -> Dict:
        """Generate visualization data for the mesh"""
        nodes = []
        for vni_id, node in self.mesh_nodes.items():
            nodes.append({
                'id': vni_id,
                'type': node.node_type,
                'activation': node.current_activation,
                'state': node.state.value,
                'vni_type': node.vni_instance.vni_type if hasattr(node.vni_instance, 'vni_type') else 'unknown'
            })
        
        edges = []
        for synapse_key, synapse in self.mesh_synapses.items():
            edges.append({
                'source': synapse.source_node,
                'target': synapse.target_node,
                'strength': synapse.current_strength,
                'type': synapse.synapse_type.value
            })
        
        patterns = []
        for pattern_id, pattern in self.synaptic_patterns.items():
            patterns.append({
                'id': pattern_id,
                'nodes': list(pattern.participating_nodes),
                'strength': pattern.pattern_strength,
                'coherence': pattern.coherence_score
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'patterns': patterns,
            'timestamp': datetime.now().isoformat()
        }

# ==================== INTEGRATION WITH EXISTING SYSTEM ====================
def integrate_neural_mesh_with_babybionn(existing_system: Any) -> NeuralMeshCore:
    """Integrate neural mesh with existing BabyBIONN system
    Args:
        existing_system: Your existing BabyBIONN system (must have vni_manager attribute)
    Returns:
        Integrated NeuralMeshCore instance"""
    # Extract VNI manager from existing system
    vni_manager = None
    
    if hasattr(existing_system, 'vni_manager'):
        vni_manager = existing_system.vni_manager
    elif hasattr(existing_system, 'manager') and isinstance(existing_system.manager, VNIManager):
        vni_manager = existing_system.manager
    elif isinstance(existing_system, VNIManager):
        vni_manager = existing_system
    else:
        # Try to find VNI manager in attributes
        for attr_name in dir(existing_system):
            attr = getattr(existing_system, attr_name)
            if isinstance(attr, VNIManager):
                vni_manager = attr
                break
    
    if vni_manager is None:
        raise ValueError("Could not find VNIManager in existing system")
    
    # Create neural mesh core
    neural_mesh = NeuralMeshCore(vni_manager)
    
    # Patch existing system with mesh-enhanced methods
    def create_mesh_enhanced_process(original_process):
        """Create mesh-enhanced version of process method"""
        
        async def mesh_enhanced_process(query: str, context: Dict = None, 
                                       session_id: str = "default",
                                       use_mesh: bool = True) -> Dict[str, Any]:
            """Enhanced query processing with neural mesh
            Args:
                query: User query
                context: Additional context
                session_id: Session identifier
                use_mesh: Whether to use neural mesh enhancement
            Returns:
                Response with optional mesh enhancement"""
            if not use_mesh or len(vni_manager.vni_instances) <= 1:
                # Use original processing for simple cases
                return original_process(query, context, session_id)
            
            # Use neural mesh for enhanced processing
            return await neural_mesh.process_with_mesh(query, context, session_id)
        return mesh_enhanced_process
    
    # Enhance existing system if it has a process method
    if hasattr(existing_system, 'process_query'):
        existing_system.process_query_mesh = create_mesh_enhanced_process(
            existing_system.process_query
        )
    
    # Add mesh methods to existing system
    existing_system.neural_mesh = neural_mesh
    existing_system.get_mesh_status = neural_mesh.get_mesh_status
    existing_system.visualize_mesh = neural_mesh.visualize_mesh
    
    logger.info(f"✅ Neural Mesh integrated with existing system")
    return neural_mesh

# ==================== MAIN INTEGRATION EXAMPLE ====================
def main_integration_example():
    """Example of neural mesh integration"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 BabyBIONN Neural Mesh Integration")
    print("=" * 60)
    
    # Create existing BabyBIONN system
    from enhanced_vni_classes import VNIManager
    
    print("1. Initializing existing VNI system...")
    manager = VNIManager(enable_generation=True)
    
    # Create VNIs
    medical = manager.create_vni('medical', 'med_001')
    legal = manager.create_vni('legal', 'legal_001')
    technical = manager.create_vni('general', 'tech_001')
    analytical = manager.create_vni('general', 'analyst_001')
    
    print(f"   Created {len(manager.vni_instances)} VNIs")
    
    # Integrate neural mesh
    print("2. Integrating Neural Mesh...")
    neural_mesh = integrate_neural_mesh_with_babybionn(manager)
    
    # Test queries
    test_queries = [
        "How should we design a healthcare app that complies with GDPR and uses AI for diagnostics?",
        "What are the implications of a data breach involving patient medical records?",
        "How can AI help with legal document analysis while ensuring data privacy?",
        "What's the best approach to analyze clinical trial data with machine learning?"
    ]
    
    print("\n3. Testing Neural Mesh with complex queries...")
    print("=" * 60)
    
    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: {query}")
        print("-" * 40)
        
        # Process with neural mesh
        response = asyncio.run(
            neural_mesh.process_with_mesh(query)
        )
        
        print(f"Response preview: {response['response'][:150]}...")
        print(f"Confidence: {response['confidence']:.2f}")
        print(f"Sources: {response['sources']}")
        print(f"Causal Depth: {len(response.get('causal_chain', []))}")
        
        if i == len(test_queries) - 1:
            print("\n📊 Final Mesh Status:")
            mesh_status = neural_mesh.get_mesh_status()
            print(f"   Active Nodes: {mesh_status['active_nodes']}/{mesh_status['total_nodes']}")
            print(f"   Synaptic Patterns: {mesh_status['synaptic_patterns']}")
            print(f"   Mesh Health: {mesh_status['mesh_health']['avg_node_activation']:.2f}")
    
    print("\n✅ Neural Mesh Integration Complete!")
    return neural_mesh

if __name__ == "__main__":
    # Run integration example
    neural_mesh = main_integration_example() 
