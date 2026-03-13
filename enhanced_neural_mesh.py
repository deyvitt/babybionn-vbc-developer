# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors
"""enhanced_neural_mesh.py - Ultimate Neural Mesh Orchestrator
Combines neural mesh sophistication with clean orchestrator design"""
import os
import inspect
import asyncio
import logging
import hashlib
import json
import uuid
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import numpy as np

# Import existing BabyBIONN components
from enhanced_vni_classes.core.base_vni import EnhancedBaseVNI
from enhanced_vni_classes.managers.vni_manager import VNIManager
from enhanced_vni_classes.core.neural_pathway import NeuralPathway
from enhanced_vni_classes.core.registry import VNIRegistry
from neuron.vni_storage import StorageManager
from neuron.vni_messenger import VNIMessenger
from bionn_activation import SmartActivationRouter
from bionn_attention import DemoHybridAtention
from bionn_aggregator import ResponseAggregator
from bionn_synaptic import SynapticConfig as AggregatorConfig

logger = logging.getLogger("enhanced_neural_mesh")

# ==================== CORE NEURAL MESH TYPES (FROM neural_mesh.py) ====================
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
        if self.last_activated is not None:
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
        if self.last_activated is not None:
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

# ==================== ENHANCED TASK MANAGEMENT (FROM vni_orchestrator.py) ====================
@dataclass
class NeuralMeshTask:
    """Enhanced task representation with neural mesh context"""
    task_id: str
    query: str
    context: Dict[str, Any]
    source_vni: Optional[str] = None
    target_vnis: List[str] = field(default_factory=list)
    processing_path: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, processing, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    result: Optional[Dict] = None
    mesh_activation: Optional[Dict] = None  # Neural mesh activation data
    causal_chain: List[Dict] = field(default_factory=list)  # Causal reasoning chain
    vni_responses: Dict[str, Any] = field(default_factory=dict)  # Store individual VNI responses

    def to_dict(self) -> Dict:
        """Convert task to dictionary for serialization"""
        return {
            'task_id': self.task_id,
            'query': self.query,
            'context': self.context,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'processing_path': self.processing_path,
            'result': self.result,
            'mesh_activation': self.mesh_activation,
            'causal_depth': len(self.causal_chain),
            'vni_response_count': len(self.vni_responses)            
        }

# ==================== ENHANCED COLLABORATION PATTERNS (FROM vni_orchestrator.py) ====================
class CollaborationPatternTracker:
    """Tracks and learns from successful collaboration patterns"""
    
    def __init__(self):
        self.patterns: Dict[str, Dict] = {}  # pattern_hash -> pattern data
        self.success_threshold = 0.6
    
    def record_pattern(self, vni_ids: List[str], success: bool):
        """Record a collaboration pattern and its success"""
        pattern_hash = self._hash_pattern(vni_ids)
        
        if pattern_hash not in self.patterns:
            self.patterns[pattern_hash] = {
                'pattern': vni_ids,
                'success_count': 0,
                'total_count': 0,
                'success_rate': 0.0,
                'last_used': datetime.now(),
                'activation_strength': 0.5  # Initial strength
            }
        
        pattern = self.patterns[pattern_hash]
        pattern['total_count'] += 1
        pattern['last_used'] = datetime.now()
        
        if success:
            pattern['success_count'] += 1
        
        # Update success rate
        pattern['success_rate'] = pattern['success_count'] / pattern['total_count']
        
        # Strengthen successful patterns, weaken unsuccessful ones
        if success:
            pattern['activation_strength'] = min(1.0, pattern['activation_strength'] + 0.1)
        else:
            pattern['activation_strength'] = max(0.1, pattern['activation_strength'] - 0.05)
    
    def get_best_patterns(self, query: str, available_vnis: List[str]) -> List[List[str]]:
        """Get best collaboration patterns for a query"""
        relevant_patterns = []
        
        for pattern_hash, pattern_data in self.patterns.items():
            # Check if pattern can be formed with available VNIs
            if all(vni_id in available_vnis for vni_id in pattern_data['pattern']):
                # Calculate relevance score
                relevance = pattern_data['success_rate'] * pattern_data['activation_strength']
                
                # Age factor (favor recently used patterns)
                days_since_use = (datetime.now() - pattern_data['last_used']).days
                age_factor = max(0.5, 1.0 - (days_since_use / 30.0))  # 30-day half-life
                
                final_score = relevance * age_factor
                relevant_patterns.append((pattern_data['pattern'], final_score))
        
        # Sort by score and return top patterns
        relevant_patterns.sort(key=lambda x: x[1], reverse=True)
        return [pattern for pattern, score in relevant_patterns[:3]]  # Top 3 patterns
    
    def _hash_pattern(self, vni_ids: List[str]) -> str:
        """Create hash for collaboration pattern"""
        pattern_str = "->".join(sorted(vni_ids))
        return hashlib.md5(pattern_str.encode()).hexdigest()[:8]

# ==================== ENHANCED NEURAL MESH CORE (COMBINES BOTH) ====================
class EnhancedNeuralMeshCore:
    """Ultimate neural mesh orchestrator combining both systems"""
    def __init__(self, vni_manager: VNIManager):
        self.vni_manager = vni_manager
        self.node_id = str(uuid.uuid4())

        # ====================== MOCK MODE SETUP ======================
        self.mock_mode_enabled = os.getenv("MOCK_MODE_ENABLED", "false").lower() == "true"
        logger.info(f"📊 MOCK MODE DEBUG - Raw env value: {os.getenv('MOCK_MODE_ENABLED')}")
        logger.info(f"📊 MOCK MODE DEBUG - Parsed value: {self.mock_mode_enabled}")

        if self.mock_mode_enabled:
            logger.info("🎭 MOCK MODE ENABLED - All API calls will be bypassed")
            try:
                from neuron.mock_response import MockResponseProvider
                self.mock_provider = MockResponseProvider()
                logger.info("✅ MockResponseProvider initialized")
            except ImportError as e:
                logger.error(f"❌ Failed to import MockResponseProvider: {e}")
                self.mock_mode_enabled = False
        else:
            logger.info("📵 MOCK MODE DISABLED - Using real API calls")

        # ===========================================================
        # In EnhancedNeuralMeshCore.__init__(), replace the default VNI creation with:
        if len(self.vni_manager.vni_instances) == 0:
            logger.info("🔄 Creating default VNIs...")
            try:
                # Simple string-based creation
                self.vni_manager.create_vni(
                    domain='medical',
                    instance_id='medical_001'            
                )
                logger.info("   ✅ Created medical VNI")
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to create medical VNI: {e}")
                logger.debug(f"   Full error: {e}", exc_info=True)    
            try:
                self.vni_manager.create_vni(
                    domain='legal',
                    instance_id='legal_001'          
                )
                logger.info("   ✅ Created legal VNI")
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to create legal VNI: {e}")
                logger.debug(f"   Full error: {e}", exc_info=True)        
            try:
                self.vni_manager.create_vni(
                    domain='general',
                    instance_id='general_001'
                )
                logger.info("   ✅ Created general VNI")
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to create general VNI: {e}")
                logger.debug(f"   Full error: {e}", exc_info=True)

        # Create SINGLE shared aggregator that persists
        aggregator_config = AggregatorConfig(
            aggregator_id="neural_mesh_aggregator",
            enable_hebbian_learning=True,
            enable_auto_spawning=False,  # Let mesh handle spawning
            consensus_threshold=0.6,
            enable_cross_domain_synthesis=True
        )
        self.aggregator = ResponseAggregator(aggregator_config, vni_manager)
        # Use aggregator's Hebbian engine for all learning
        self.hebbian_engine = self.aggregator.hebbian_engine
                
        # Enhanced task management (from vni_orchestrator)
        self.active_tasks: Dict[str, NeuralMeshTask] = {}
        self.task_history: List[NeuralMeshTask] = []
        self.max_history = 1000
        
        # Collaboration pattern learning (from vni_orchestrator)
        self.collaboration_tracker = CollaborationPatternTracker()
        
        # Neural mesh components (from neural_mesh.py)
        self.mesh_nodes: Dict[str, MeshNode] = {}
        self.mesh_synapses: Dict[str, MeshSynapse] = {}
        self.synaptic_patterns: Dict[str, SynapticPattern] = {}
        
        # Enhanced routing modules (from vni_orchestrator)
        self._init_routing_modules()
        
        # Integration with existing modules
        self.router = SmartActivationRouter() if hasattr(self, '_router_available') and self._router_available else None
        self.storage_manager = StorageManager()
        self.messenger = VNIMessenger(storage_manager=self.storage_manager)

        # Initialize DemoHybridAttention with required parameters
        if hasattr(self, '_attention_available') and self._attention_available:
            try:
                self.attention = DemoHybridAttention(
                    dim=256,                    # Required: dimension size
                    num_heads=8,                # Number of attention heads
                    window_size=256,            # Sliding window size
                    use_sliding=True,          # Enable sliding window attention
                    use_global=True,           # Enable global attention
                    use_hierarchical=True,     # Enable hierarchical attention
                    global_token_ratio=0.05,   # Ratio of global tokens
                    memory_tokens=16,          # Number of memory tokens
                    multi_modal=True           # Enable multi-modal fusion
                )
                logger.info("✅ DemoHybridAttention initialized with default config")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize DemoHybridAttention: {e}")
                self.attention = None
        else:
            self.attention = None

        # Mesh properties (from neural_mesh.py)
        self.global_inhibition: float = 0.0
        self.global_excitation: float = 1.0
        self.mesh_learning_rate: float = 0.1
        self.activation_wave_speed: float = 100.0  # ms per hop
        
        # State tracking (from neural_mesh.py)
        self.activation_waves: List[Dict] = []
        self.causal_chains: List[Dict] = []
        self.emergent_patterns: List[Dict] = []
        
        # Initialize mesh
        self._initialize_mesh_from_existing()
        
        logger.info(f"🧠 Enhanced Neural Mesh Core initialized with {len(self.mesh_nodes)} nodes")
    
    def _init_routing_modules(self):
        """Initialize routing and attention modules with fallbacks (from vni_orchestrator)"""
        try:
            from bionn_attention import HybridAttentionEngine
            self.attention_engine = HybridAttentionEngine()
            self._attention_available = True
            logger.info("✅ HybridAttentionEngine loaded")
        except ImportError:
            self.attention_engine = None
            self._attention_available = False
            logger.warning("⚠️ HybridAttentionEngine not available")
        
        try:
            from bionn_activation import SmartActivationRouter
            self.activation_router = SmartActivationRouter()
            self._router_available = True
            logger.info("✅ SmartActivationRouter loaded")
        except ImportError:
            self.activation_router = None
            self._router_available = False
            logger.warning("⚠️ SmartActivationRouter not available")
        
        try:
            from bionn_transform import TransVNICompareSegregate
            self.trans_vni_module = TransVNICompareSegregate()
            self._trans_vni_available = True
            logger.info("✅ TransVNICompareSegregate loaded")
        except ImportError:
            self.trans_vni_module = None
            self._trans_vni_available = False
            logger.warning("⚠️ TransVNICompareSegregate not available")
        
        try:
            from neuron.baseVNI_demo import SmartBaseVNI, EnhancedVNIConfig
            config = EnhancedVNIConfig()
            self.base_vni = SmartBaseVNI(config)
            logger.info("✅ BaseVNI patterns loaded")
        except ImportError:
            self.base_vni = None
            logger.warning("⚠️ BaseVNI patterns not available")
    
    def get_capability_manifest(self) -> List[Dict[str, Any]]:
        """Return a list of VNIs with their types and domains for P2P advertisement."""
        manifest = []
        for vni_id, vni in self.vni_manager.vni_instances.items():
            # Try to get subdomains; fallback to empty list
            subdomains = getattr(vni, 'subdomains', [])        
            manifest.append({
                'vni_id': vni_id,
                'type': getattr(vni, 'vni_type', 'unknown'),
                'domain': getattr(vni, 'domain', 'general'),
                'subdomains': subdomains,            
            })
        return manifest

    # ==================== MESH INITIALIZATION (FROM neural_mesh.py) ====================
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
            synapse_key = f"{pathway.source_id}->{pathway.target_id}"
            
            # Determine synapse type based on VNI types
            source_vni = self.vni_manager.vni_instances.get(pathway.source_id)
            target_vni = self.vni_manager.vni_instances.get(pathway.target_id)
            synapse_type = self._determine_synapse_type(source_vni, target_vni)
            
            synapse = MeshSynapse(
                id=synapse_key,
                source_node=pathway.source_id,
                target_node=pathway.target_id,
                current_strength=pathway.strength,
                synapse_type=synapse_type,
                last_activated=pathway.last_activated
            )
            self.mesh_synapses[synapse_key] = synapse
            
            # Update node connections
            if pathway.source_id in self.mesh_nodes:
                self.mesh_nodes[pathway.source_id].axons.append(pathway.target_id)
            if pathway.target_id in self.mesh_nodes:
                self.mesh_nodes[pathway.target_id].dendrites.append(pathway.source_id)
        
        # Initialize with random connections for emergent patterns
        self._create_initial_random_connections()
        
        logger.info(f"Mesh initialized: {len(self.mesh_nodes)} nodes, {len(self.mesh_synapses)} synapses")
    
    def _determine_node_type(self, vni: EnhancedBaseVNI) -> str:
        """Determine mesh node type from VNI type"""
        if hasattr(vni, 'capabilities') and vni.capabilities:
            # Try to get from capabilities
            vni_type = getattr(vni.capabilities, 'vni_type', 'specialized')
        else:
            # Fallback
            vni_type = getattr(vni, 'vni_type', 'specialized')
    
        vni_type = str(vni_type).lower()
        
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
        if vni.vni_type == 'medical':
            return 0.15 # changed from 0.4 and 0.25
        elif vni.vni_type == 'legal':
            return 0.20 # changed from 0.30
        elif vni.vni_type == 'general':
            return 0.10 # changed from 0.15
        elif vni.vni_type == 'technical':
            return 0.20 # changed from 0.25
        else:
            return 0.2 # lowered from 0.3
    
    def _determine_synapse_type(self, source_vni: Optional[EnhancedBaseVNI], 
                               target_vni: Optional[EnhancedBaseVNI]) -> SynapseType:
        """Determine synapse type based on VNI relationships"""
        if not source_vni or not target_vni:
            return SynapseType.EXCITATORY
        
        complementary_pairs = [
            ('medical', 'legal'), ('legal', 'technical'), 
            ('technical', 'analytical'), ('analytical', 'creative')
        ]
        
        source_type = source_vni.vni_type
        target_type = target_vni.vni_type
        
        if (source_type, target_type) in complementary_pairs:
            return SynapseType.EXCITATORY
        elif source_type == target_type:
            return SynapseType.MODULATORY
        else:
            return SynapseType.EXCITATORY
    
    def _create_initial_random_connections(self):
        """Create initial random connections for emergent pattern formation"""
        node_ids = list(self.mesh_nodes.keys())
        
        for i in range(min(20, len(node_ids) * 3)):
            source = random.choice(node_ids)
            target = random.choice(node_ids)
            
            if source != target:
                synapse_key = f"{source}->{target}"
                if synapse_key not in self.mesh_synapses:
                    synapse = MeshSynapse(
                        id=synapse_key,
                        source_node=source,
                        target_node=target,
                        current_strength=random.uniform(0.1, 0.4),
                        synapse_type=random.choice(list(SynapseType))
                    )
                    self.mesh_synapses[synapse_key] = synapse
                    
                    self.mesh_nodes[source].axons.append(target)
                    self.mesh_nodes[target].dendrites.append(source)
    
    # ==================== ENHANCED PROCESSING PIPELINE (COMBINES BOTH) ====================
    async def process_query(self, query: str, context: Dict = None, 
                           session_id: str = "default") -> Dict[str, Any]:
        """Enhanced processing with task management and neural mesh"""
        # ============ MOCK MODE CHECK - DO THIS FIRST! ============
        # Check if mock mode is enabled (from __init__)
        if hasattr(self, 'mock_mode_enabled') and self.mock_mode_enabled and hasattr(self, 'mock_provider'):
            logger.info(f"🎭 MOCK MODE: Generating mock response for: '{query[:50]}...'")
            mock_response = self.mock_provider.generate_response(query, context or {})
                    
            # Return mock response immediately - skip ALL processing
            return {
                'response': mock_response['response'],
                'confidence': mock_response['confidence'],
                'sources': ['mock_provider'],
                'causal_chain': [
                    {'step': 'mock_mode', 'reason': 'Mock mode enabled - bypassing all VNI processing'}
                ],
                'timestamp': datetime.now().isoformat(),
                'task_id': f"mock_{hashlib.md5(query.encode()).hexdigest()[:8]}"
            }    
        # ==================== ADD GREETING HANDLER HERE ====================
        greeting_response = self._handle_greeting_query(query)
        if greeting_response:
            logger.info(f"🎯 Greeting detected: '{query}' - using greeting handler")
            return greeting_response
        # ===================================================================
        # Create enhanced task (from vni_orchestrator)
        task_id = f"task_{hashlib.md5(query.encode()).hexdigest()[:8]}"    
        # Create enhanced task (from vni_orchestrator)
        task_id = f"task_{hashlib.md5(query.encode()).hexdigest()[:8]}"
        task = NeuralMeshTask(
            task_id=task_id,
            query=query,
            context=context or {}
        )
        self.active_tasks[task_id] = task
        task.status = "processing"
        
        try:
            # Step 1: Enhanced query analysis with fallback (from vni_orchestrator)
            initial_analysis = await self._enhanced_query_analysis(query, context)
            
            # Step 2: Get collaboration patterns (from vni_orchestrator)
            available_vni_ids = list(self.vni_manager.vni_instances.keys())
            best_patterns = self.collaboration_tracker.get_best_patterns(query, available_vni_ids)
            
            # Step 3: Neural mesh activation (from neural_mesh.py)
            mesh_activation = await self._activate_neural_mesh(query, initial_analysis)
            task.mesh_activation = mesh_activation
            
            # Step 4: Parallel VNI processing with enhanced context
            vni_responses = await self._process_with_enhanced_context(
                mesh_activation, query, context, session_id, best_patterns
            )
            # 🔥 ADD THIS CHECK 🔥
            if not vni_responses or all('error' in r for r in vni_responses.values()):
                logger.warning("No successful VNI responses - trying fallbacks")
                
                # Try DeepSeek API as fallback
                deepseek_response = await self._call_deepseek_fallback(query, context, task_id)
                if deepseek_response:
                    logger.info("✅ DeepSeek fallback succeeded")
                    return deepseek_response
                
                # If DeepSeek also fails, use enhanced fallback
                return self._generate_enhanced_fallback(query, context, task_id)

            # 🔥🔥🔥 STEP 5: USE AGGREGATOR (REPLACE _synthesize_responses) 🔥🔥🔥
            # 1. Convert VNI responses to aggregator format
            adapted_responses = {}
            for vni_id, response in vni_responses.items():
                vni = self.vni_manager.vni_instances.get(vni_id)
                vni_type = vni.vni_type if vni else 'general'
                
                adapted = {
                    'confidence_score': response.get('confidence', 0.5),
                    'vni_metadata': {
                        'vni_id': vni_id,
                        'success': 'error' not in response,
                        'domain': vni_type
                    }
                }
                
                # Add appropriate advice field
                if vni_type == 'medical':
                    adapted['medical_advice'] = response.get('response', '')
                elif vni_type == 'legal':
                    adapted['legal_advice'] = response.get('response', '')
                elif vni_type == 'technical':
                    adapted['technical_advice'] = response.get('response', '')
                else:
                    adapted['general_advice'] = response.get('response', '')
                
                adapted_responses[vni_id] = adapted
            
            # 2. Create router results for aggregator with all required fields
            router_results = {
                'execution_results': adapted_responses,
                'query_context': {
                    'query': query,
                    'query_complexity': context.get('complexity_score', 0.5) if context else 0.5,
                    'detected_domains': context.get('detected_domains', ['general']) if context else ['general'],
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat()
                },
                'activation_plan': {
                    'activated_vnis': [
                        {
                            'vni_id': vni_id, 
                            'activation_score': mesh_activation['activation_pattern'].get(vni_id, 0.5)
                        }
                        for vni_id in vni_responses.keys()
                    ],
                    'overall_attention': mesh_activation.get('activation_summary', {}).get('avg_activation', 0.5),
                    'attention_weights': {
                        vni_id: mesh_activation['activation_pattern'].get(vni_id, 0.5)
                        for vni_id in vni_responses.keys()
                    }
                }
            }
            # 3. Run aggregator with try/except to handle errors gracefully
            try:
                aggregated_results = self.aggregator(router_results)
                
                # Adding debug to see what's in aggregated_results
                logger.debug(f"Aggregator results type: {type(aggregated_results)}")
                if isinstance(aggregated_results, dict):
                    logger.debug(f"Aggregator keys: {list(aggregated_results.keys())}")
    
                # 4. Build final response from aggregator
                final_response = {
                    'response': aggregated_results.get('final_response', 'No response generated'),
                    'confidence': aggregated_results.get('confidence_metrics', {}).get('overall_confidence', 0.5),
                    'sources': list(vni_responses.keys()),
                    'causal_chain': self._build_simple_causal_chain(vni_responses, query),
                    'aggregated_data': aggregated_results,
                    'timestamp': datetime.now().isoformat()
                }           
            except Exception as e:
                logger.error(f"Aggregator failed: {e}")
                logger.error(f"Error type: {type(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
        
                # FALLBACK: Use the first successful VNI response directly
                successful_vni = next(
                    (vni_id for vni_id, resp in vni_responses.items() 
                    if 'error' not in resp and resp.get('response')),
                    None
                )
                
                if successful_vni:
                    # We have a successful VNI response - use it!
                    vni_response = vni_responses[successful_vni]
                    final_response = {
                        'response': vni_response.get('response', 'No response available'),
                        'confidence': vni_response.get('confidence', 0.5),
                        'sources': [successful_vni],
                        'causal_chain': self._build_simple_causal_chain({successful_vni: vni_response}, query),
                        'aggregator_failed': True,
                        'timestamp': datetime.now().isoformat()
                    }
                    logger.info(f"✅ Using direct VNI response from {successful_vni} after aggregator failure")
                else:
                    # No successful VNIs - use fallback
                    final_response = {
                        'response': 'Unable to generate response',
                        'confidence': 0.1,
                        'sources': [],
                        'causal_chain': [],
                        'timestamp': datetime.now().isoformat()
                    }
            
            task.result = final_response
            task.vni_responses = vni_responses  # Store for learning
            task.causal_chain = final_response.get('causal_chain', [])
            
            # Step 6: Enhanced learning (combines both systems)
            await self._enhanced_learning(task, final_response, vni_responses)
            
            task.status = "completed"
                
            # Save to history with size limit (from vni_orchestrator)
            self.task_history.append(task)
            if len(self.task_history) > self.max_history:
                self.task_history.pop(0)
                
            return final_response
                
        except Exception as e:
            logger.error(f"Processing failed for task {task_id}: {e}")
            task.status = "failed"
            task.result = {"error": str(e)}
                
            return self._generate_enhanced_fallback(query, context, task_id)   

    async def _call_deepseek_fallback(self, query: str, context: Dict, task_id: str) -> Optional[Dict]:
        """Call DeepSeek API as fallback when no VNIs respond"""
        import os
        import aiohttp
        import json
        from datetime import datetime
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.warning("No DEEPSEEK_API_KEY configured")
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                # Get temperature based on domain or use default
                temperature = 0.3
                if context and 'detected_domains' in context:
                    if 'medical' in context['detected_domains']:
                        temperature = float(os.getenv("VNI_MEDICAL_TEMPERATURE", "0.1"))
                    elif 'legal' in context['detected_domains']:
                        temperature = float(os.getenv("VNI_LEGAL_TEMPERATURE", "0.1"))
                    elif 'technical' in context['detected_domains']:
                        temperature = float(os.getenv("VNI_TECHNICAL_TEMPERATURE", "0.2"))
                
                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are BabyBIONN, a helpful AI assistant. Provide accurate, helpful responses. If the query is medical-related, provide general health information but always include a disclaimer to consult healthcare professionals. For legal questions, provide general information but always recommend consulting with an attorney."
                        },
                        {"role": "user", "content": query}
                    ],
                    "temperature": temperature,
                    "max_tokens": 1000
                }
                
                async with session.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        deepseek_response = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
                        # Add disclaimer based on domain
                        if context and 'detected_domains' in context:
                            if 'medical' in context['detected_domains']:
                                deepseek_response += "\n\n*Disclaimer: I'm an AI assistant, not a doctor. This information is for educational purposes. Please consult with a healthcare professional for medical advice.*"
                            elif 'legal' in context['detected_domains']:
                                deepseek_response += "\n\n*Disclaimer: I'm an AI assistant, not an attorney. This information is for educational purposes. Please consult with a qualified lawyer for legal advice.*"
                        
                        return {
                            'response': deepseek_response,
                            'confidence': 0.8,  # High confidence for DeepSeek
                            'sources': ['deepseek_fallback'],
                            'causal_chain': [
                                {'step': 'vni_fallback', 'reason': 'No VNIs could process query'},
                                {'step': 'deepseek_api', 'reason': 'Used DeepSeek as fallback'}
                            ],
                            'timestamp': datetime.now().isoformat(),
                            'task_id': task_id,
                            'provider': 'deepseek'
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"DeepSeek API error {response.status}: {error_text}")
                        return None
                        
        except asyncio.TimeoutError:
            logger.error("DeepSeek API timeout")
            return None
        except Exception as e:
            logger.error(f"DeepSeek API exception: {e}")
            return None
    
    async def _enhanced_query_analysis(self, query: str, context: Dict) -> Dict:
        """Enhanced query analysis using available NLP components"""
        
        analysis = {
            "query_text": query,
            "query_hash": hashlib.md5(query.encode()).hexdigest()[:8],
            "complexity_score": self._assess_query_complexity(query),
            "requires_mesh": len(query.split()) > 10,
            "word_count": len(query.split()),
            "char_count": len(query),
            "has_question": '?' in query,
            "detected_domains": []
        }
    
        # Try to use the attention engine if available
        if self.attention_engine and hasattr(self.attention_engine, 'analyze_query'):
            try:
                attention_analysis = self.attention_engine.analyze_query(query)
                analysis.update(attention_analysis)
            except Exception as e:
                logger.debug(f"Attention engine analysis failed: {e}")
    
        # Try to use base VNI if available
        if self.base_vni and hasattr(self.base_vni, 'process'):
            try:
                base_analysis = self.base_vni.process({"text": query})
                if base_analysis and isinstance(base_analysis, dict):
                    analysis.update(base_analysis)
            except Exception as e:
                logger.debug(f"Base VNI analysis failed: {e}")
    
        # Analyze query for domains
        query_lower = query.lower()
        domains = []
    
        if any(word in query_lower for word in ['medical', 'health', 'doctor', 'hospital', 'pain', 'fever']):
            domains.append('medical')
        if any(word in query_lower for word in ['legal', 'law', 'contract', 'rights', 'court', 'attorney']):
            domains.append('legal')
        if any(word in query_lower for word in ['technical', 'code', 'programming', 'software', 'debug', 'error']):
            domains.append('technical')
        if any(word in query_lower for word in ['business', 'market', 'profit', 'strategy', 'analysis']):
            domains.append('analytical')
        if any(word in query_lower for word in ['creative', 'write', 'story', 'art', 'design']):
            domains.append('creative')
    
        analysis['detected_domains'] = domains if domains else ['general']
        analysis['primary_domain'] = domains[0] if domains else 'general'
    
        # Detect query intent
        analysis['intent'] = self._detect_query_intent(query)
    
        # Detect emotional tone
        analysis['emotional_tone'] = self._detect_emotional_tone(query)
    
        return analysis

    def _detect_query_intent(self, query: str) -> str:
        """Detect the intent of the query"""
        q = query.lower()
    
        if any(w in q for w in ['what is', 'define', 'explain']):
            return 'definition'
        elif any(w in q for w in ['how to', 'how do i', 'how can i']):
            return 'how_to'
        elif any(w in q for w in ['why', 'reason', 'cause']):
            return 'explanation'
        elif any(w in q for w in ['compare', 'difference between']):
            return 'comparison'
        elif any(w in q for w in ['should i', 'recommend', 'advice']):
            return 'advice'
        elif any(w in q for w in ['help', 'problem', 'issue']):
            return 'help'
        elif '?' in query:
            return 'question'
        else:
            return 'statement'

    def _detect_emotional_tone(self, query: str) -> str:
        """Detect emotional tone of the query"""
        q = query.lower()
    
        if any(w in q for w in ['urgent', 'emergency', 'help', 'asap', 'immediately']):
            return 'urgent'
        elif any(w in q for w in ['thank', 'thanks', 'appreciate', 'good', 'great', 'awesome']):
            return 'positive'
        elif any(w in q for w in ['problem', 'issue', 'wrong', 'bad', 'terrible', 'frustrated']):
            return 'negative'
        elif any(w in q for w in ['please', 'could you', 'would you', 'kindly']):
            return 'polite'
        else:
            return 'neutral'

    def _assess_query_complexity(self, query: str) -> float:
        """Assess query complexity on a 0.0 to 1.0 scale"""
        words = query.split()
        word_count = len(words)
        
        # Base complexity from word count (0-1 scale)
        word_complexity = min(1.0, word_count / 100.0)
        
        # Check for complex sentence structures
        complex_indicators = 0
        if ',' in query:
            complex_indicators += 1
        if ';' in query:
            complex_indicators += 1
        if 'and' in query.lower() or 'but' in query.lower() or 'because' in query.lower():
            complex_indicators += 1
        
        structural_complexity = min(1.0, complex_indicators * 0.2)
    
        # Check for technical/domain-specific terms
        technical_terms = ['algorithm', 'implementation', 'architecture', 'paradigm', 
                          'methodology', 'framework', 'protocol', 'synthesis']
        technical_count = sum(1 for term in technical_terms if term in query.lower())
        technical_complexity = min(1.0, technical_count * 0.1)
    
        # Combine factors
        final_complexity = (word_complexity * 0.4 + 
                           structural_complexity * 0.3 + 
                           technical_complexity * 0.3)
    
        return round(final_complexity, 2)
    
    async def _activate_neural_mesh(self, query: str, query_analysis: Dict) -> Dict[str, Any]:
        """Activate neural mesh based on query analysis"""
    
        # Initial activation based on domain relevance
        initial_activation = {}
        for vni_id, vni in self.vni_manager.vni_instances.items():
            relevance = self._calculate_vni_relevance(vni, query, query_analysis.get('context', {}))
            initial_activation[vni_id] = relevance
        
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
            'activation_duration': (datetime.now() - activation_wave['start_time']).total_seconds()
        }
    
    def _calculate_vni_relevance(self, vni: EnhancedBaseVNI, query: str, context: Dict) -> float:
        """Calculate relevance of VNI to query using multiple factors"""
        
        relevance_score = 0.0
        query_lower = query.lower()
        vni_type = vni.vni_type.lower()
        
        # Factor 1: Direct domain match from query analysis
        if 'detected_domains' in context:
            if vni_type in context['detected_domains']:
                relevance_score += 0.4
                logger.debug(f"VNI {vni.instance_id}: +0.4 for direct domain match")
        
        # Factor 2: Use VNI's own classifier if available
        if hasattr(vni, 'should_handle'):
            try:
                if vni.should_handle(query):
                    relevance_score += 0.3
                    logger.debug(f"VNI {vni.instance_id}: +0.3 from should_handle()")
            except AttributeError as e:
                logger.warning(f"VNI {vni.instance_id} has should_handle but it failed: {e}")
                # Don't add score if method exists but fails
            except Exception as e:
                logger.error(f"Unexpected error in should_handle for {vni.instance_id}: {e}")
                # Log the error but continue with other factors
    
        # Factor 3: Keyword matching
        domain_keywords = {
            'medical': ['health', 'doctor', 'patient', 'treatment', 'medicine', 'hospital', 
                       'pain', 'fever', 'symptom', 'diagnosis', 'illness', 'disease',
                       'sick', 'medical', 'clinic', 'pharmacy', 'prescription', 'vaccine',
                       'infection', 'virus', 'bacteria', 'allergy', 'asthma', 'cancer',
                       'diabetes', 'heart', 'lung', 'kidney', 'liver', 'brain',
                       'mental health', 'psychology', 'therapy', 'counseling', 
                       'diabetes', 'blood sugar', 'glucose', 'insulin', 'type 1', 'type 2',
                       'what is', 'define', 'explain', 'tell me about', 'condition', 
                       'chronic', 'metabolic', 'endocrine', 'pancreas'],
        
            'legal': ['law', 'legal', 'contract', 'rights', 'court', 'attorney', 'case',
                     'evidence', 'justice', 'liability', 'compliance', 'agreement',
                     'statute', 'regulation', 'legislation', 'lawsuit', 'trial', 'judge',
                     'jury', 'defendant', 'plaintiff', 'settlement', 'arbitration',
                     'mediation', 'copyright', 'patent', 'trademark', 'intellectual property',
                     'privacy', 'data protection', 'gdpr'],

            'general': ['code', 'programming', 'system', 'software', 'technical', 'business',
                       'analysis', 'creative', 'write', 'calculate', 'math', 'strategy',
                       'algorithm', 'data', 'database', 'network', 'security', 'cloud',
                       'api', 'framework', 'architecture', 'design', 'development',
                       'testing', 'debug', 'error', 'bug', 'fix', 'solution', 'optimization',
                       'performance', 'scalability', 'reliability', 'maintenance']
        }
    
        if vni_type in domain_keywords:
            keyword_matches = sum(1 for keyword in domain_keywords[vni_type] 
                                if keyword in query_lower)
            if keyword_matches > 0:
                keyword_score = min(0.3, keyword_matches * 0.05)
                relevance_score += keyword_score
                logger.debug(f"VNI {vni.instance_id}: +{keyword_score:.2f} for {keyword_matches} keyword matches")
    
        # Factor 4: Query intent matching
        if 'intent' in context:
            intent_scores = {
                'medical': {
                    'definition': 0.1, 'how_to': 0.2, 'explanation': 0.3, 
                    'advice': 0.4, 'help': 0.5, 'comparison': 0.2, 'question': 0.3
                },
                'legal': {
                    'definition': 0.2, 'how_to': 0.1, 'explanation': 0.3,
                    'advice': 0.5, 'comparison': 0.4, 'question': 0.2
                },
                'general': {
                    'definition': 0.3, 'how_to': 0.4, 'explanation': 0.3,
                    'comparison': 0.3, 'question': 0.2, 'advice': 0.3, 'help': 0.3
                }
            }
        
            if vni_type in intent_scores and context['intent'] in intent_scores[vni_type]:
                intent_score = intent_scores[vni_type][context['intent']]
                relevance_score += intent_score
                logger.debug(f"VNI {vni.instance_id}: +{intent_score:.2f} for intent '{context['intent']}'")
    
        # Factor 5: Emotional tone matching
        if 'emotional_tone' in context:
            tone_scores = {
                'medical': {
                    'urgent': 0.4, 'positive': 0.1, 'negative': 0.3, 'polite': 0.1,
                    'neutral': 0.2, 'anxious': 0.3, 'concerned': 0.2
                },
                'legal': {
                    'urgent': 0.3, 'positive': 0.1, 'negative': 0.2, 'polite': 0.2,
                    'neutral': 0.1, 'anxious': 0.3, 'concerned': 0.3
                },
                'general': {
                    'urgent': 0.2, 'positive': 0.1, 'negative': 0.1, 'polite': 0.1,
                    'neutral': 0.0, 'anxious': 0.1, 'concerned': 0.1
                }
            }
        
            if vni_type in tone_scores and context['emotional_tone'] in tone_scores[vni_type]:
                tone_score = tone_scores[vni_type][context['emotional_tone']]
                relevance_score += tone_score
                logger.debug(f"VNI {vni.instance_id}: +{tone_score:.2f} for tone '{context['emotional_tone']}'")
    
        # Factor 6: Complexity matching
        if 'complexity_score' in context:
            complexity = context['complexity_score']
            complexity_bonus = 0.0
            
            if vni_type == 'general' and complexity > 0.5:
                complexity_bonus = 0.2  # General VNI handles complex queries well
            elif vni_type in ['medical', 'legal'] and 0.3 <= complexity <= 0.7:
                complexity_bonus = 0.1  # Specialized VNIs handle moderate complexity well
            
            if complexity_bonus > 0:
                relevance_score += complexity_bonus
                logger.debug(f"VNI {vni.instance_id}: +{complexity_bonus:.2f} for complexity {complexity:.2f}")
        
        # Factor 7: Session history (if available)
        if 'session_history' in context and hasattr(vni, 'context_memory'):
            session_id = context.get('session_id', 'default')
            if session_id in vni.context_memory:
                # Boost relevance if VNI was recently used in this session
                history = vni.context_memory[session_id]['conversation_history']
                if len(history) > 0:
                    recency_bonus = min(0.2, len(history) * 0.02)
                    relevance_score += recency_bonus
                    logger.debug(f"VNI {vni.instance_id}: +{recency_bonus:.2f} for session history")
        
        # Factor 8: VNI's confidence in its own knowledge
        try:
            # Try to get confidence from VNI's knowledge base
            if hasattr(vni, 'extract_concepts_and_patterns') and hasattr(vni, 'calculate_confidence'):
                concepts, patterns = vni.extract_concepts_and_patterns(query)
                vni_confidence = vni.calculate_confidence(concepts, patterns)
            else:
                vni_confidence = 0.5  # Default confidence
            confidence_bonus = vni_confidence * 0.2  # Up to 0.2 bonus
            relevance_score += confidence_bonus
            logger.debug(f"VNI {vni.instance_id}: +{confidence_bonus:.2f} for internal confidence {vni_confidence:.2f}")
        except Exception as e:
            logger.debug(f"Could not calculate internal confidence for {vni.instance_id}: {e}")
        
        # Ensure score is between 0.1 and 1.0 with some randomness for exploration
        relevance_score = max(0.1, min(1.0, relevance_score))
        
        # Add small random factor (0-0.05) to avoid always picking the same VNI for borderline cases
        random_factor = random.uniform(0, 0.05)
        relevance_score += random_factor
        
        # Final clamp
        relevance_score = max(0.1, min(1.0, relevance_score))
        
        logger.info(f"VNI {vni.instance_id} ({vni_type}) final relevance: {relevance_score:.2f}")
        
        return relevance_score
    
    async def _propagate_activation_wave(self, activation_wave: Dict):
        """Propagate activation wave through neural mesh"""
        activation_queue = []
        for vni_id, activation in activation_wave['initial_activation'].items():
            if activation > 0.1:
                activation_queue.append((vni_id, activation, []))
        
        visited = set()
        max_iterations = 50
        iteration = 0
        
        while activation_queue and iteration < max_iterations:
            iteration += 1
            next_queue = []
            
            for vni_id, activation_strength, path in activation_queue:
                if vni_id in visited:
                    continue
                
                visited.add(vni_id)
                
                # Activate node
                node_activation = self._activate_mesh_node(vni_id, activation_strength, path)
                
                if node_activation > 0.2:
                    activation_wave['activated_nodes'].add(vni_id)
                    activation_wave['activation_pattern'][vni_id] = node_activation
                    
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
                                    
                                    if transmitted > 0.1:
                                        next_queue.append((target_id, transmitted, full_path.copy()))
            
            activation_queue = next_queue
            await asyncio.sleep(0.001)
    
    def _activate_mesh_node(self, vni_id: str, activation_strength: float, path: List[str]) -> float:
        """Activate a specific mesh node"""
        if vni_id not in self.mesh_nodes:
            return 0.0
        
        node = self.mesh_nodes[vni_id]
        
        pulse = ActivationPulse(
            source_node=path[-1] if path else "query",
            target_node=vni_id,
            strength=activation_strength,
            activation_type="semantic",
            propagation_path=path.copy()
        )
        
        node_activation = node.activate(pulse)
        return node_activation
    
    def _extract_activation_pattern(self, activation_wave: Dict) -> Dict:
        """Extract activation pattern from wave"""
        pattern = {
            'node_activations': activation_wave['activation_pattern'],
            'pathways': activation_wave['propagation_paths'],
            'node_count': len(activation_wave['activated_nodes'])
        }
        
        if activation_wave['activation_pattern']:
            activations = list(activation_wave['activation_pattern'].values())
            pattern['activation_summary'] = {
                'max_activation': max(activations),
                'avg_activation': np.mean(activations),
                'activation_variance': np.var(activations)
            }
        
        return pattern
    
    def _safe_process_response(self, response_data):
        """Safely process response data, handling various input types.
        Args:
            response_data: Can be dict, str, list, or any other type
            
        Returns:
            dict: Standardized response format"""
        try:
            # Case 1: Already a properly formatted dict
            if isinstance(response_data, dict):
                # Ensure it has required fields
                result = {
                    'response': response_data.get('response', ''),
                    'confidence': response_data.get('confidence', 0.5),
                    'vni_metadata': response_data.get('vni_metadata', {}),
                    'error': response_data.get('error', None)
                }
                return result
        
            # Case 2: String - could be JSON or plain text
            elif isinstance(response_data, str):
                try:
                    # Try to parse as JSON
                    import json
                    parsed = json.loads(response_data)
                    if isinstance(parsed, dict):
                        return self._safe_process_response(parsed)  # Recursive call
                    else:
                        return {
                            'response': response_data,
                            'confidence': 0.5,
                            'vni_metadata': {},
                            'error': None
                        }
                except json.JSONDecodeError:
                    # Plain text response
                    return {
                        'response': response_data,
                        'confidence': 0.5,
                        'vni_metadata': {},
                        'error': None
                    }
            
            # Case 3: Any other type
            else:
                return {
                    'response': str(response_data),
                    'confidence': 0.5,
                    'vni_metadata': {},
                    'error': None
                }
                
        except Exception as e:
            logger.error(f"❌ Error processing response: {e}")
            return {
                'response': f"Error processing response: {str(e)}",
                'confidence': 0.0,
                'vni_metadata': {},
                'error': str(e)
            }
        
    async def _process_with_enhanced_context(self, mesh_activation: Dict, query: str,
                                            context: Dict, session_id: str,
                                            best_patterns: List[List[str]]) -> Dict[str, Any]:
        """Process query with enhanced context including collaboration patterns"""
        
        vni_responses = {}
        activated_nodes = mesh_activation['activated_nodes']
        query_lower = query.lower()
    
        # Filter nodes by domain relevance
        filtered_nodes = set()
        for vni_id in activated_nodes:
            vni = self.vni_manager.vni_instances.get(vni_id)
            if not vni:
                continue
                
            # Skip medical VNI for non-medical queries
            if vni.vni_type == 'medical':
                medical_keywords = ['health', 'doctor', 'patient', 'medicine', 'hospital', 
                                  'pain', 'fever', 'symptom', 'diagnosis', 'illness', 
                                  'disease', 'medical', 'diabetes', 'cancer', 'heart']
                if not any(keyword in query_lower for keyword in medical_keywords):
                    logger.debug(f"Skipping medical VNI for non-medical query: {query}")
                    continue
                    
            # Skip legal VNI for non-legal queries
            if vni.vni_type == 'legal':
                legal_keywords = ['law', 'legal', 'contract', 'court', 'attorney', 'rights']
                if not any(keyword in query_lower for keyword in legal_keywords):
                    logger.debug(f"Skipping legal VNI for non-legal query: {query}")
                    continue
                    
            filtered_nodes.add(vni_id)
        
        # If we filtered out everything, use original nodes
        if not filtered_nodes:
            filtered_nodes = activated_nodes
        
        if best_patterns:
            for pattern in best_patterns:
                if all(vni_id in filtered_nodes for vni_id in pattern):
                    logger.info(f"Using collaboration pattern: {pattern}")
                    for vni_id in pattern:
                        if vni_id in self.vni_manager.vni_instances:
                            response = await self._process_single_vni_enhanced(
                                vni_id, query, context, session_id, mesh_activation, pattern
                            )
                            vni_responses[vni_id] = response
                    break
                    
        # If no pattern worked or not enough nodes, process all activated nodes
        if not vni_responses:
            for vni_id in activated_nodes:
                if vni_id in self.vni_manager.vni_instances:
                    response = await self._process_single_vni_enhanced(
                        vni_id, query, context, session_id, mesh_activation, []
                    )
                    vni_responses[vni_id] = response
        
        return vni_responses
    
    async def _process_single_vni_enhanced(self, vni_id: str, query: str,
                                          context: Dict, session_id: str,
                                          mesh_activation: Dict, 
                                          collaboration_pattern: List[str]) -> Dict[str, Any]:
        """Process with enhanced context including collaboration patterns"""
        vni = self.vni_manager.vni_instances[vni_id]

        # === 1. Ensure vni_id attribute exists ===
        if not hasattr(vni, 'vni_id'):
            if hasattr(vni, 'instance_id'):
                vni.vni_id = vni.instance_id
            else:
                vni.vni_id = vni_id
        
        # === 2. Medical VNI Wrapper (with patient_context) ===
        if vni.vni_type == 'medical':
            original_method = getattr(vni, 'process_query', None) or getattr(vni, 'process', None)
            
            async def safe_medical_process(*args, **kwargs):
                try:
                    if original_method:
                        # Extract query and context from kwargs or args
                        query_text = kwargs.get('query', args[0] if args else "")
                        ctx = kwargs.get('context', args[1] if len(args) > 1 else {})
                        
                        if inspect.iscoroutinefunction(original_method):
                            return await original_method(query_text, ctx, patient_context={})
                        else:
                            return original_method(query_text, ctx, patient_context={})
                except Exception as e:
                    if 'requires_caution' in str(e):
                        return {
                            'response': f"Regarding '{query_text}': This appears to be a medical question. For educational purposes, I can provide general information. Please consult a healthcare professional for personal medical advice.",
                            'confidence': 0.7,
                            'domain': 'medical',
                            'vni_metadata': {
                                'vni_id': vni.vni_id,
                                'success': True,
                                'domain': 'medical'
                            }
                        }
                # Fallback
                return {
                    'response': f"Medical query received",
                    'confidence': 0.5,
                    'domain': 'medical',
                    'vni_metadata': {'vni_id': vni.vni_id, 'success': True, 'domain': 'medical'}
                }
            
            vni.process_query = safe_medical_process
        
        # === 3. Legal VNI Wrapper ===
        elif vni.vni_type == 'legal':
            async def safe_legal_process(*args, **kwargs):
                query_text = kwargs.get('query', args[0] if args else "")
                return {
                    'response': f"Regarding '{query_text}': For legal information, please consult with a qualified attorney. I can provide general legal concepts for educational purposes.",
                    'confidence': 0.6,
                    'domain': 'legal',
                    'vni_metadata': {'vni_id': vni.vni_id, 'success': True, 'domain': 'legal'}
                }
            vni.process_query = safe_legal_process
        
        # === 4. Technical/Dynamic VNI Wrapper ===
        elif vni.vni_type == 'dynamic' or 'technical' in str(vni.__class__).lower():
            if not hasattr(vni, 'process_query') and hasattr(vni, 'process_async'):
                original_async = vni.process_async
                
                async def wrapped_process_async(*args, **kwargs):
                    query_text = kwargs.get('query', args[0] if args else "")
                    ctx = kwargs.get('context', args[1] if len(args) > 1 else {})
                    return await original_async(query_text, ctx)
                
                vni.process_query = wrapped_process_async
        # ========== END OF SNIPPET==========

        # Enhanced context with collaboration info
        enhanced_context = context.copy() if context else {}
        enhanced_context['neural_mesh'] = {
            'activation_level': mesh_activation['activation_pattern'].get(vni_id, 0.0),
            'mesh_context': mesh_activation,
            'collaborating_vnis': list(mesh_activation['activated_nodes']),
            'collaboration_pattern': collaboration_pattern,
            'pattern_position': collaboration_pattern.index(vni_id) if vni_id in collaboration_pattern else -1
        }
        
        # Add attention routing if available (from vni_orchestrator)
        if self.attention_engine and hasattr(vni, 'available_capabilities'):
            vni_descriptor = {
                "id": vni.instance_id,
                "type": vni.vni_type,
                "capabilities": list(vni.available_capabilities.specializations) if hasattr(vni.available_capabilities, 'specializations') else [],
                "collaboration_score": vni.available_capabilities.collaboration_score if hasattr(vni.available_capabilities, 'collaboration_score') else 0.5
            }
            attention_score = self.attention_engine.compute_scores(
                query, [vni_descriptor], enhanced_context
            )[0] if hasattr(self.attention_engine, 'compute_scores') else 0.5
            enhanced_context['attention_score'] = attention_score
        
        # Process query - handle different VNI types with different method names
        try:
            response = None
            
            # Check what methods the VNI has
            if hasattr(vni, 'process_query'):
                # TechnicalVNI and others use process_query
                if inspect.iscoroutinefunction(vni.process_query):
                    response = await vni.process_query(query=query, context=enhanced_context)
                else:
                    response = vni.process_query(query=query, context=enhanced_context)
            
            # For DynamicVNI, it uses process_async (as seen in dynamic_vni.py)
            elif hasattr(vni, 'process_async'):
                if inspect.iscoroutinefunction(vni.process_async):
                    response = await vni.process_async(query, enhanced_context)
                else:
                    response = vni.process_async(query, enhanced_context)
            
            # Try process method as fallback
            elif hasattr(vni, 'process'):
                if inspect.iscoroutinefunction(vni.process):
                    response = await vni.process(query, enhanced_context)
                else:
                    response = vni.process(query, enhanced_context)
            
            # If no method found, return error
            else:
                logger.error(f"VNI {vni_id} has no process_query, process_async, or process method")
                return {
                    'response': f"VNI {vni.vni_type} cannot process this query type",
                    'confidence': 0.1,
                    'error': 'No suitable method found',
                    'vni_metadata': {
                        'instance_id': vni.instance_id,
                        'vni_type': vni.vni_type
                    }
                }
                
            logger.debug(f"🎯 Raw response from {vni_id}: type={type(response)}")
            
        except Exception as e:
            logger.error(f"❌ Error in {vni_id}.process_query: {e}")
            response = {
                'response': f"VNI processing error: {str(e)}",
                'confidence': 0.1,
                'error': str(e)
            }
        
        # Ensure response is a dict
        if not isinstance(response, dict):
            response = {
                'response': str(response),
                'confidence': 0.5,
                'vni_metadata': {}
            }
        
        # Ensure required fields
        if 'response' not in response:
            response['response'] = ''
        if 'confidence' not in response:
            response['confidence'] = 0.5
        
        # Add metadata
        vni_metadata = {
            'instance_id': vni.instance_id,
            'vni_type': vni.vni_type,
            'confidence': response.get('confidence', 0.5),
            'generation_used': response.get('generation_used', False),
            'activation_level': mesh_activation['activation_pattern'].get(vni_id, 0.0),
            'attention_score': enhanced_context.get('attention_score', 0.0),
            'in_collaboration_pattern': vni_id in collaboration_pattern,
            'processing_time': datetime.now().isoformat()
        }
        
        if 'vni_metadata' in response and isinstance(response['vni_metadata'], dict):
            response['vni_metadata'].update(vni_metadata)
        else:
            response['vni_metadata'] = vni_metadata
        
        return response
    
    async def _synthesize_responses(self, vni_responses: Dict, 
                                   query: str, 
                                   query_analysis: Dict,
                                   mesh_activation: Dict) -> Dict[str, Any]:
        """Synthesize responses from multiple VNIs"""
        
        if not vni_responses:
            return {
                'response': "No VNIs could process this query effectively.",
                'confidence': 0.1,
                'sources': [],
                'causal_chain': []
            }
        
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
                'vni_type': response.get('vni_metadata', {}).get('vni_type', 'unknown')
            })
        
        # Sort by weight
        weighted_responses.sort(key=lambda x: x['weight'], reverse=True)
        
        # Build synthesis based on number of responses
        if len(weighted_responses) == 1:
            synthesized = self._synthesize_single_response(weighted_responses[0])
        elif len(weighted_responses) >= 3:
            synthesized = self._synthesize_multiple_responses(weighted_responses, query)
        else:
            synthesized = self._synthesize_pair_response(weighted_responses)
        
        # Build final response
        final_response = {
            'response': synthesized['response'],
            'confidence': synthesized['confidence'],
            'sources': [r['vni_id'] for r in weighted_responses],
            'causal_chain': self._build_simple_causal_chain(weighted_responses, query),
            'synthesis_method': synthesized['method'],
            'contributing_vnis': [
                {
                    'id': r['vni_id'],
                    'type': r['vni_type'],
                    'weight': r['weight'],
                    'activation': r['activation']
                }
                for r in weighted_responses[:5]
            ],
            'timestamp': datetime.now().isoformat()
        }
        return final_response
    
    def _synthesize_single_response(self, top_response: Dict) -> Dict[str, Any]:
        """Synthesize when only one VNI is strongly activated"""
        return {
            'response': top_response['response'],
            'confidence': top_response['confidence'],
            'method': 'single_vni_enhanced'
        }
    
    def _synthesize_pair_response(self, responses: List[Dict]) -> Dict[str, Any]:
        """Synthesize when two VNIs are activated"""
        if len(responses) < 2:
            return self._synthesize_single_response(responses[0])
        
        resp1, resp2 = responses[0], responses[1]
        
        # Combine responses
        if resp1['vni_type'] != resp2['vni_type']:
            synthesized = f"**{resp1['vni_type'].title()} Perspective:** {resp1['response']}\n\n**{resp2['vni_type'].title()} Perspective:** {resp2['response']}"
        else:
            synthesized = f"**Primary:** {resp1['response']}\n\n**Secondary:** {resp2['response']}"
        
        combined_confidence = (resp1['confidence'] + resp2['confidence']) / 2
        return {
            'response': synthesized,
            'confidence': combined_confidence,
            'method': 'pair_synthesis'
        }
    
    def _synthesize_multiple_responses(self, responses: List[Dict], query: str) -> Dict[str, Any]:
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
                synthesis_parts.append(f"**Primary Analysis ({resp['vni_type']}):** {resp['response']}")
            else:
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
    
    def _build_simple_causal_chain(self, responses, query: str) -> List[Dict]:
        """Build simple causal chain from responses"""
        causal_chain = []
        
        # Handle different input types
        if isinstance(responses, dict):
            # Convert dict to list of response items
            response_items = list(responses.values())
        elif isinstance(responses, list):
            response_items = responses
        else:
            logger.error(f"Unexpected responses type: {type(responses)}")
            return []
        
        # Extract key concepts from query
        words = query.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        concepts = [word for word in words if word not in stop_words and len(word) > 3][:5]
        
        # Create simple causal links
        for concept in concepts:
            for resp_item in response_items:
                # Skip if not a dict
                if not isinstance(resp_item, dict):
                    continue
                    
                # Try to get response text from various possible fields
                response_text = ""
                if 'response' in resp_item and isinstance(resp_item['response'], str):
                    response_text = resp_item['response'].lower()
                elif 'medical_advice' in resp_item and isinstance(resp_item['medical_advice'], str):
                    response_text = resp_item['medical_advice'].lower()
                elif 'legal_advice' in resp_item and isinstance(resp_item['legal_advice'], str):
                    response_text = resp_item['legal_advice'].lower()
                elif 'general_advice' in resp_item and isinstance(resp_item['general_advice'], str):
                    response_text = resp_item['general_advice'].lower()
                
                if response_text and concept in response_text:
                    causal_chain.append({
                        'concept': concept,
                        'domain': resp_item.get('vni_type', 'unknown'),
                        'confidence': resp_item.get('confidence', 0.5)
                    })
                    break
        
        return causal_chain
    
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
    
    async def _enhanced_learning(self, task: NeuralMeshTask, 
                                final_response: Dict,
                                vni_responses: Dict):
        """Enhanced learning combining both systems"""
        
        # Determine success (from vni_orchestrator)
        success = final_response.get('confidence', 0) > 0.6 and 'error' not in final_response
        
        # Update collaboration patterns (from vni_orchestrator)
        activated_vni_ids = list(task.mesh_activation['activated_nodes']) if task.mesh_activation else []
        if activated_vni_ids:
            self.collaboration_tracker.record_pattern(activated_vni_ids, success)
        
        # Update neural pathways (from vni_orchestrator)
        for i in range(len(activated_vni_ids) - 1):
            for j in range(i + 1, len(activated_vni_ids)):
                source_id = activated_vni_ids[i]
                target_id = activated_vni_ids[j]
                pathway_key = f"{source_id}->{target_id}"
                
                if pathway_key in self.vni_manager.neural_pathways:
                    self.vni_manager.neural_pathways[pathway_key].activate(success)
        
        # Update mesh synapses and patterns (from neural_mesh.py)
        if task.mesh_activation:
            success_metric = final_response.get('confidence', 0.5)
            self._update_synapses(task.mesh_activation, success_metric)
            self._update_patterns(task.mesh_activation, success_metric)

    def _trigger_individual_vni_learning(self, task: NeuralMeshTask, 
                                       aggregator_results: Dict):
        """Learn from AGGREGATED wisdom, not individual responses"""
        # Get the aggregator's final analysis
        final_response = aggregator_results['final_response']
        aggregation_analysis = aggregator_results['aggregation_analysis']
        confidence_metrics = aggregator_results['confidence_metrics']
        
        # Use the aggregator's consensus as success metric
        consensus_score = aggregation_analysis['consensus_analysis']['consensus_score']
        
        # Get ALL participating VNIs from aggregator analysis
        all_vnis = set()
        
        # From aggregator's VNI contributions
        for contrib in aggregation_analysis.get('vni_contributions', []):
            all_vnis.add(contrib['vni_id'])
        
        # From aggregator's processing metadata
        for vni_id in aggregator_results.get('processing_metadata', {}).get('successful_vnis', []):
            all_vnis.add(vni_id)
        
        logger.info(f"🎓 Teaching {len(all_vnis)} VNIs from aggregated wisdom (consensus: {consensus_score:.1%})")

        # This ensures learning accumulates across ALL queries
        self.aggregator.hebbian_engine.learn_from_interaction(
            activated_vnis=list(all_vnis),
            vni_outputs={vni_id: task.vni_responses.get(vni_id, {}) for vni_id in all_vnis},
            overall_quality=consensus_score,
            query_context=task.context
        )
    
        # Teach each VNI what the COLLECTIVE wisdom concluded
        for vni_id in all_vnis:
            vni = self.vni_manager.vni_instances.get(vni_id)
            if vni and hasattr(vni, 'learn_from_interaction'):
            
                learning_package = {
                    'query': task.query,
                    'my_response': task.vni_responses.get(vni_id, {}),  # What I said
                    'collective_wisdom': final_response,                 # What WE concluded
                    'consensus_score': consensus_score,                   # How much we agreed
                    'my_role_in_consensus': self._get_my_role(vni_id, aggregation_analysis),
                    'domain_synthesis': aggregation_analysis.get('domain_coverage', {})
                }
                
                vni.learn_from_interaction(
                    query=task.query,
                    response_data=learning_package,  # RICHER learning data!
                    success_metric=consensus_score,  # Learn from COLLECTIVE success
                    session_id=task.context.get('session_id', 'default')
                )

    # ==================== ADD GREETING HANDLER HERE ====================
    def _handle_greeting_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Handle greeting queries before normal processing"""
        import re
        import random
        from datetime import datetime
        import hashlib
        
        query_lower = query.lower().strip()
        
        # Greeting patterns
        greeting_patterns = [
            r'\b(hello|hi|hey|greetings|howdy|hiya|hullo)\b',
            r'\b(what.*your name|who.*are you)\b',
            r'\b(my name is|i am|i\'m)\b',
            r'\b(good morning|good afternoon|good evening)\b',
            r'\b(how are you|how do you do)\b',
            r'\b(hi there|hello there)\b',
            r'\b(yo|sup|what\'s up)\b',
            r'\b(nice to meet you|pleased to meet you)\b'
        ]
        
        # Check if it's a greeting
        is_greeting = False
        for pattern in greeting_patterns:
            if re.search(pattern, query_lower):
                is_greeting = True
                break
        
        if not is_greeting:
            return None
        
        # Generate appropriate response
        if re.search(r'what.*your name|who.*are you', query_lower):
            response = random.choice([
                "Hello! I'm BabyBIONN, an enhanced neural mesh system with specialized VNIs for medical, legal, and general knowledge.",
                "I'm BabyBIONN, your neural mesh assistant with specialized knowledge in medical, legal, and technical domains.",
                "Greetings! I'm BabyBIONN, a collaborative neural network ready to assist you."
            ])
        
        elif re.search(r'my name is', query_lower):
            name_match = re.search(r'my name is (\w+)', query_lower)
            if name_match:
                name = name_match.group(1).title()
                response = f"Nice to meet you, {name}! I'm BabyBIONN. How can I assist you today?"
            else:
                response = "Nice to meet you! I'm BabyBIONN. How can I help?"
        
        elif re.search(r'how are you', query_lower):
            response = random.choice([
                "I'm functioning optimally, thank you! Ready to assist with medical, legal, or general questions.",
                "All neural pathways are active and ready! How can I help you today?",
                "System status: excellent! Enhanced neural mesh is online and learning."
            ])
        
        elif re.search(r'good morning|good afternoon|good evening', query_lower):
            time_match = re.search(r'(good morning|good afternoon|good evening)', query_lower)
            time_greeting = time_match.group(1)
            response = f"{time_greeting.title()}! I'm BabyBIONN. How can I help you today?"
        
        else:
            response = random.choice([
                "Hello! How can I help you today?",
                "Hi there! I'm BabyBIONN. What would you like to discuss?",
                "Greetings! I'm ready to assist with medical, legal, or general questions."
            ])
        
        # Return formatted response
        return {
            'response': response,
            'confidence': 0.95,
            'sources': ['greeting_handler'],
            'causal_chain': [
                {'step': 'greeting_detection', 'reason': 'Detected greeting pattern in query'},
                {'step': 'response_generation', 'reason': 'Generated appropriate greeting response'}
            ],
            'timestamp': datetime.now().isoformat(),
            'task_id': f"greeting_{hashlib.md5(query.encode()).hexdigest()[:8]}"
        }
    # ==================== END GREETING HANDLER ====================

    def _get_my_role(self, vni_id: str, aggregation_analysis: Dict) -> str:
        """Determine VNI's role in the consensus"""
        for contrib in aggregation_analysis.get('vni_contributions', []):
            if contrib['vni_id'] == vni_id:
                return contrib.get('contribution_level', 'secondary')
        
        # Check if was primary contributor
        if 'agreeing_vnis' in aggregation_analysis.get('consensus_analysis', {}):
            if vni_id in aggregation_analysis['consensus_analysis']['agreeing_vnis']:
                return 'consensus_member'
    
        return 'participant'

    def _update_synapses(self, mesh_activation: Dict, success_metric: float):
        """Update synaptic strengths based on activation and success"""
        for pathway in mesh_activation.get('propagation_paths', []):
            path = pathway['path']
            
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                synapse_key = f"{source}->{target}"
                
                if synapse_key in self.mesh_synapses:
                    synapse = self.mesh_synapses[synapse_key]
                    
                    # Update based on success
                    synapse.update_plasticity(success_metric > 0.6)
    
    def _update_patterns(self, mesh_activation: Dict, success_metric: float):
        """Update or create synaptic patterns"""
        activated_nodes = mesh_activation['activated_nodes']
        
        if len(activated_nodes) >= 3:
            # Check for existing pattern
            pattern_found = False
            
            for pattern_id, pattern in self.synaptic_patterns.items():
                overlap = len(activated_nodes & pattern.participating_nodes)
                if overlap >= 3:
                    pattern.evolve(success_metric)
                    pattern_found = True
                    break
            
            # Create new pattern if none found
            if not pattern_found and len(activated_nodes) >= 3:
                pattern_id = f"pattern_{len(self.synaptic_patterns) + 1}"
                
                connection_matrix = {}
                nodes_list = list(activated_nodes)
                
                for i, node1 in enumerate(nodes_list):
                    for j, node2 in enumerate(nodes_list[i+1:], i+1):
                        synapse_key = f"{node1}->{node2}"
                        if synapse_key in self.mesh_synapses:
                            connection_matrix[(node1, node2)] = self.mesh_synapses[synapse_key].current_strength
                        else:
                            connection_matrix[(node1, node2)] = 0.3
                
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
    
    def _generate_enhanced_fallback(self, query: str, context: Dict, task_id: str) -> Dict:
        """Enhanced fallback response with task context"""
        return {
            "response": f"I encountered an issue processing your query (Task: {task_id}). "
                       f"As a fallback, I suggest rephrasing or providing more context. "
                       f"The neural mesh had difficulty activating appropriate VNIs for: '{query[:50]}...'",
            "confidence": 0.3,
            "fallback_used": True,
            "processing_path": [],
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "suggestion": "Try breaking down complex queries or specifying domains like 'medical', 'legal', or 'technical'"
        }
    
    # ==================== TASK MANAGEMENT METHODS ====================
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of a specific task"""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id].to_dict()
        return None
    
    def get_recent_tasks(self, limit: int = 10) -> List[Dict]:
        """Get recent tasks"""
        recent = self.task_history[-limit:] if self.task_history else []
        return [task.to_dict() for task in recent]
    
    def get_task_statistics(self) -> Dict:
        """Get task processing statistics"""
        completed = sum(1 for task in self.task_history if task.status == "completed")
        failed = sum(1 for task in self.task_history if task.status == "failed")
        processing = len(self.active_tasks)
        
        avg_confidence = 0.0
        if completed > 0:
            confidences = [task.result.get('confidence', 0) for task in self.task_history 
                          if task.status == "completed" and task.result]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'total_tasks': len(self.task_history),
            'completed': completed,
            'failed': failed,
            'processing': processing,
            'success_rate': completed / len(self.task_history) if self.task_history else 0.0,
            'avg_confidence': avg_confidence,
            'collaboration_patterns': len(self.collaboration_tracker.patterns)
        }
    
    def get_mesh_status(self) -> Dict[str, Any]:
        """Get current mesh status"""
        active_nodes = sum(1 for node in self.mesh_nodes.values() 
                          if node.state == MeshNodeState.ACTIVE)
        
        strong_synapses = sum(1 for synapse in self.mesh_synapses.values() 
                             if synapse.current_strength > 0.7)
        
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
            }
        }

# ==================== SIMPLIFIED INTEGRATION ====================
def integrate_enhanced_mesh(existing_system: Any) -> EnhancedNeuralMeshCore:
    """Integrate enhanced neural mesh with existing system"""
    
    # Extract VNI manager
    vni_manager = None
    
    if hasattr(existing_system, 'vni_manager'):
        vni_manager = existing_system.vni_manager
    elif isinstance(existing_system, VNIManager):
        vni_manager = existing_system
    else:
        # Try to find VNI manager
        for attr_name in dir(existing_system):
            attr = getattr(existing_system, attr_name)
            if isinstance(attr, VNIManager):
                vni_manager = attr
                break
    
    if vni_manager is None:
        raise ValueError("Could not find VNIManager in existing system")
    
    # Create enhanced neural mesh
    enhanced_mesh = EnhancedNeuralMeshCore(vni_manager)
    
    # Add mesh to existing system
    existing_system.neural_mesh = enhanced_mesh
    existing_system.process_query_enhanced = enhanced_mesh.process_query
    existing_system.get_mesh_status = enhanced_mesh.get_mesh_status
    
    logger.info(f"✅ Enhanced Neural Mesh integrated with existing system")
    return enhanced_mesh

# ==================== MAIN DEMO ====================
async def main_demo():
    """Demo of the enhanced neural mesh system"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 ENHANCED NEURAL MESH DEMO")
    print("=" * 60)
    
    try:
        from enhanced_vni_classes import VNIManager
        
        print("1. Initializing VNI Manager...")
        manager = VNIManager(enable_generation=True)
        
        # Create some VNIs - try common valid types
        print("2. Creating VNIs...")
        # Common valid VNI types (based on enhanced_vni_classes.py structure)
        possible_types = ['medical', 'legal', 'general', 'analytical', 'creative', 'sensory']
        created_vnis = []
        
        for vni_type in possible_types:
            try:
                vni = manager.create_vni(vni_type, f"{vni_type}_001")
                print(f"   ✅ Created {vni_type} VNI: {vni.instance_id}")
                created_vnis.append(vni_type)
            except ValueError as e:
                print(f"   ⚠️  Cannot create {vni_type}: {e}")
        
        if len(created_vnis) < 2:
            print(f"❌ Need at least 2 VNIs for mesh, but only created {len(created_vnis)}")
            print("   Trying with minimal setup...")
            # Force create at least medical and general
            try:
                vni = manager.create_vni('medical', "medical_001")
                created_vnis.append('medical')
                print(f"   ✅ Created medical VNI: {vni.instance_id}")
            except:
                pass
            try:
                vni = manager.create_vni('general', "general_001")
                created_vnis.append('general')
                print(f"   ✅ Created general VNI: {vni.instance_id}")
            except:
                pass
        
        print(f"\n3. Creating Enhanced Neural Mesh with {len(created_vnis)} VNIs...")
        enhanced_mesh = EnhancedNeuralMeshCore(manager)
        
        # Test queries
        test_queries = [
            "What are the medical and legal considerations for AI diagnostics?",
            "How can systems ensure healthcare data privacy?",
            "Explain the implications of machine learning in medical diagnosis"
        ]
        
        print("\n4. Testing Enhanced Processing...")
        print("=" * 60)
        
        for i, query in enumerate(test_queries):
            print(f"\n📝 Query {i+1}: {query}")
            print("-" * 40)
            
            try:
                response = await enhanced_mesh.process_query(query)
                
                print(f"✅ Confidence: {response.get('confidence', 0):.2f}")
                print(f"📊 Sources: {len(response.get('sources', []))} VNIs")
                print(f"🔗 Synthesis: {response.get('synthesis_method', 'unknown')}")
                
                preview = response.get('response', 'No response')[200] + "..." if len(response.get('response', '')) > 200 else response.get('response', 'No response')
                print(f"\n💬 Preview: {preview}")
            except Exception as e:
                print(f"❌ Processing failed: {e}")
        
        # Show statistics
        print("\n📈 System Statistics:")
        try:
            stats = enhanced_mesh.get_task_statistics()
            print(f"   Tasks processed: {stats.get('total_tasks', 0)}")
            print(f"   Success rate: {stats.get('success_rate', 0):.1%}")
            print(f"   Collaboration patterns: {stats.get('collaboration_patterns', 0)}")
        except:
            print("   Statistics not available")
        
        try:
            mesh_status = enhanced_mesh.get_mesh_status()
            print(f"   Mesh nodes: {mesh_status.get('total_nodes', 0)}")
            print(f"   Active nodes: {mesh_status.get('active_nodes', 0)}")
            print(f"   Synapses: {mesh_status.get('total_synapses', 0)}")
        except:
            print("   Mesh status not available")
        
        print("\n✅ Enhanced Neural Mesh Demo Complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main_demo())
