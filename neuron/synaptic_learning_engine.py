# neuron/shared/synaptic_learning_engine.py (REFACTORED - NO DUPLICATION)
"""
BabyBIONN-specific extensions to the HebbianLearningEngine from aggregator.py
Adds visualization, autonomous routing, and BabyBIONN-specific features.
"""

import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from synaptic_visualization import SynapticVisualizer
from neuron.aggregator import HebbianLearningEngine
from neuron.shared.types import ConnectionType, SynapticConnection, PatternRecord
from neuron.shared.synaptic_config import SynapticConfig as AggregatorConfig
from neuron.shared.constants import (
    DOMAIN_KEYWORDS, STRONG_CONNECTION_THRESHOLD, 
    WEAK_CONNECTION_THRESHOLD, STDP_TAU
)
logger = logging.getLogger("synaptic_learning")

# ========== BABYBIONN VISUALIZATION EXTENSION ==========

class VisualizedHebbianEngine(HebbianLearningEngine):
    """Extends HebbianLearningEngine with visualization capabilities"""
    
    def __init__(self, config: AggregatorConfig = None, enable_visualization: bool = True):
        if config is None:
            config = AggregatorConfig(
                enable_hebbian_learning=True,
                learning_rate=0.1,
                decay_rate=0.01,
                strengthening_threshold=0.7,
                weakening_threshold=0.4
            )
        
        super().__init__(config)
        
        # Add visualization capability
        self.visualizer = SynapticVisualizer() if enable_visualization else None
        self.enable_visualization = enable_visualization
        
        logger.info("VisualizedHebbianEngine initialized with BabyBIONN visualization")
    
    def hebbian_update_with_visualization(self, **kwargs):
        """Hebbian update with automatic visualization"""
        result = super().hebbian_update(**kwargs)
        
        # Update visualization after learning
        if self.visualizer:
            self.visualizer.update_connections(self._convert_to_visualization_format())
        
        return result
    
    def create_visualization(self, filename: str = "synaptic_network.png"):
        """Create visualization of current network state"""
        if self.visualizer:
            self.visualizer.create_static_visualization(filename)
            logger.info(f"Visualization saved to {filename}")
        else:
            logger.info("Visualization disabled")
    
    def _convert_to_visualization_format(self) -> Dict[str, Any]:
        """Convert Hebbian connections to visualization format"""
        pathways = {}
        
        for key, conn in self.connections.items():
            conn_id = f"{conn.source_vni}->{conn.target_vni}"
            pathways[conn_id] = {
                'source': conn.source_vni,
                'target': conn.target_vni,
                'strength': conn.strength,
                'activation_count': conn.activation_count,
                'last_activated': conn.last_activated
            }
        
        return pathways

# ========== BABYBIONN-SPECIFIC FEATURES ==========

class BabyBIONNLearningExtensions:
    """Adds BabyBIONN-specific learning features to Hebbian engine"""
    
    def __init__(self, hebbian_engine: HebbianLearningEngine):
        self.hebbian_engine = hebbian_engine
        self.spontaneous_threshold = 0.7
    
    def detect_spontaneous_activation(self) -> List[Tuple[str, str, float]]:
        """Detect connections that spontaneously strengthened without direct input"""
        spontaneous = []
        
        for conn in self.hebbian_engine.connections.values():
            # BabyBIONN-specific spontaneous detection logic:
            # Spontaneous if: high strength but low activation count AND recent increase
            if (conn.strength > self.spontaneous_threshold and 
                conn.activation_count < 3 and
                conn.last_activated and
                (datetime.now() - conn.last_activated).total_seconds() < 300):  # Last 5 minutes
                
                if conn.connection_type != ConnectionType.COMPETITIVE:
                    spontaneous.append((conn.source_vni, conn.target_vni, conn.strength))
        
        return spontaneous
    
    def predict_next_strong_connection(self, current_context: Dict) -> Optional[Tuple[str, str]]:
        """Predict which connection will strengthen next based on context"""
        if not self.hebbian_engine.connections:
            return None
        
        context_hash = self._hash_context(current_context)
        
        # 1. Look for patterns where this context was successful
        if context_hash in self.hebbian_engine.context_patterns:
            patterns = self.hebbian_engine.context_patterns[context_hash]
            patterns.sort(key=lambda x: x[1], reverse=True)
            
            if patterns and patterns[0][0]:
                vni_list = patterns[0][0]
                if len(vni_list) >= 2:
                    return (vni_list[0], vni_list[1])
        
        # 2. Look for connections that are strengthening recently
        strengthening_connections = []
        for conn in self.hebbian_engine.connections.values():
            if len(conn.performance_history) >= 3:
                recent = conn.performance_history[-3:]
                if recent[-1] > recent[-2] > recent[-3]:  # Consistently strengthening
                    score = conn.strength * conn.success_rate()
                    strengthening_connections.append(((conn.source_vni, conn.target_vni), score))
        
        if strengthening_connections:
            strengthening_connections.sort(key=lambda x: x[1], reverse=True)
            return strengthening_connections[0][0]
        
        # 3. Fallback: strongest overall connection
        strongest = None
        max_strength = 0
        for conn in self.hebbian_engine.connections.values():
            if conn.strength > max_strength:
                max_strength = conn.strength
                strongest = (conn.source_vni, conn.target_vni)
        
        return strongest
    
    def suggest_reroute(self, current_vni: str, target_vni: str, max_depth: int = 3) -> List[str]:
        """Suggest optimal routing path between VNIs using Hebbian connections"""
        # Use Dijkstra-like algorithm with connection strengths as weights
        # This is BabyBIONN-specific routing logic
        if current_vni == target_vni:
            return [current_vni]
        
        # Implementation uses self.hebbian_engine.connections
        # (Same as before but using the single source of truth)
        
        distances = {current_vni: 0}
        previous = {}
        unvisited = set()
        
        # Initialize with all VNIs in connections
        for conn in self.hebbian_engine.connections.values():
            unvisited.add(conn.source_vni)
            unvisited.add(conn.target_vni)
        
        if target_vni not in unvisited:
            return []
        
        while unvisited:
            current = min(unvisited, key=lambda x: distances.get(x, float('inf')))
            
            if current == target_vni:
                break
            
            unvisited.remove(current)
            current_distance = distances[current]
            
            # Check all connections from current node
            for conn in self.hebbian_engine.connections.values():
                next_vni = None
                if conn.source_vni == current:
                    next_vni = conn.target_vni
                elif conn.target_vni == current:
                    next_vni = conn.source_vni
                
                if next_vni and next_vni in unvisited:
                    cost = 1.0 / max(0.1, conn.strength)
                    new_distance = current_distance + cost
                    
                    if new_distance < distances.get(next_vni, float('inf')):
                        distances[next_vni] = new_distance
                        previous[next_vni] = current
        
        if target_vni not in previous:
            return []
        
        path = []
        current = target_vni
        while current is not None:
            path.append(current)
            current = previous.get(current)
        
        path.reverse()
        
        return path if len(path) <= max_depth + 1 else []
    
    def get_babybionn_insights(self) -> Dict[str, Any]:
        """Get BabyBIONN-specific insights about connections"""
        stats = self.hebbian_engine.get_learning_statistics()
        
        spontaneous_count = 0
        strengthening_count = 0
        
        for conn in self.hebbian_engine.connections.values():
            if conn.strength > 0.7 and conn.activation_count < 3:
                spontaneous_count += 1
            
            if len(conn.performance_history) >= 3:
                recent = conn.performance_history[-3:]
                if recent[-1] > recent[-2] > recent[-3]:
                    strengthening_count += 1
        
        return {
            **stats,
            "spontaneous_connections": spontaneous_count,
            "strengthening_connections": strengthening_count,
            "total_context_patterns": sum(len(v) for v in self.hebbian_engine.context_patterns.values()),
            "babybionn_specific": True
        }
    
    def _hash_context(self, context: Dict) -> str:
        """Create hash of context for comparison"""
        return hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()[:16]

# ========== AUTONOMOUS LEARNING ORCHESTRATOR (SIMPLIFIED) ==========

class AutonomousLearningOrchestrator:
    """Orchestrator that adds autonomous learning to BabyBIONN
    USING ONLY THE EXISTING HEBBIAN ENGINE
    """
    
    def __init__(self, base_orchestrator, hebbian_engine: HebbianLearningEngine = None):
        self.base_orchestrator = base_orchestrator
        
        # Use existing Hebbian engine or create one
        if hebbian_engine is None:
            config = AggregatorConfig(
                enable_hebbian_learning=True,
                learning_rate=0.1,
                decay_rate=0.01
            )
            self.hebbian_engine = VisualizedHebbianEngine(config, enable_visualization=True)
        else:
            self.hebbian_engine = hebbian_engine
        
        # Add BabyBIONN extensions
        self.babybionn_extensions = BabyBIONNLearningExtensions(self.hebbian_engine)
        
        logger.info("AutonomousLearningOrchestrator using existing HebbianLearningEngine")
    
    def process_cycle(self, input_data: Dict) -> Dict:
        """Process a cycle with autonomous learning capabilities"""
        base_result = self.base_orchestrator.process(input_data)
        
        # 1. Learn from this interaction
        self._learn_from_interaction(input_data, base_result)
        
        # 2. Add BabyBIONN insights
        base_result["babybionn_insights"] = self.babybionn_extensions.get_babybionn_insights()
        
        # 3. Detect spontaneous learning
        spontaneous = self.babybionn_extensions.detect_spontaneous_activation()
        if spontaneous:
            base_result["spontaneous_learning"] = [
                {"connection": f"{s[0]}→{s[1]}", "strength": s[2]} 
                for s in spontaneous
            ]
        
        # 4. Save periodically
        if hasattr(self.hebbian_engine, 'save_network') and "cycle_count" in base_result:
            if base_result["cycle_count"] % 10 == 0:
                self.hebbian_engine.save_network("babybionn_synaptic_memory.json")
        
        return base_result
    
    def _learn_from_interaction(self, input_data: Dict, result: Dict):
        """Learn from the current interaction using Hebbian engine"""
        activated_vnis = result.get("activated_vnis", [])
        if len(activated_vnis) < 2:
            return
        
        # Calculate outcome quality
        outcome_quality = self._calculate_outcome_quality(result)
        
        # Update all pairs of activated VNIs
        for i in range(len(activated_vnis)):
            for j in range(i+1, len(activated_vnis)):
                vni1 = activated_vnis[i]
                vni2 = activated_vnis[j]
                
                # Create context for this interaction
                context = {
                    **input_data,
                    "vni_pair": [vni1, vni2],
                    "outcome_quality": outcome_quality,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Use the existing Hebbian learning method
                self.hebbian_engine.hebbian_update(
                    vni1=vni1,
                    vni2=vni2,
                    pre_activation=0.5,  # Default
                    post_activation=0.5,  # Default
                    outcome_quality=outcome_quality,
                    context_hash=self._hash_context(context)
                )
    
    def _calculate_outcome_quality(self, result: Dict) -> float:
        """Calculate quality score for learning"""
        quality = 0.5  # Default
        
        if "confidence" in result:
            quality = result["confidence"]
        elif "success" in result:
            quality = 0.8 if result["success"] else 0.3
        
        return min(1.0, max(0.0, quality))
    
    def _hash_context(self, context: Dict) -> str:
        return hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()[:16]
    
    def create_visualization(self, filename: str = "synaptic_network.png"):
        """Create visualization of the network"""
        if hasattr(self.hebbian_engine, 'create_visualization'):
            self.hebbian_engine.create_visualization(filename)
        elif hasattr(self.hebbian_engine, 'visualizer') and self.hebbian_engine.visualizer:
            self.hebbian_engine.visualizer.create_static_visualization(filename)

# ========== MAIN INTEGRATION ==========

def integrate_with_babybionn(babybionn_orchestrator, existing_hebbian_engine=None):
    """Main integration function - adds BabyBIONN features using existing engine"""
    logger.info("=" * 60)
    logger.info("INTEGRATING BABYBIONN FEATURES WITH EXISTING HEBBIAN ENGINE")
    logger.info("=" * 60)
    
    # Create autonomous orchestrator using existing engine
    autonomous_orchestrator = AutonomousLearningOrchestrator(
        babybionn_orchestrator,
        hebbian_engine=existing_hebbian_engine
    )
    
    logger.info("✅ BabyBIONN features integrated!")
    logger.info("")
    logger.info("🔗 INTEGRATION DETAILS:")
    logger.info("   • Using existing HebbianLearningEngine from aggregator.py")
    logger.info("   • NO duplication of learning logic")
    logger.info("   • Adding visualization capabilities")
    logger.info("   • Adding BabyBIONN-specific routing and insights")
    logger.info("")
    logger.info("🎯 NEW BABYBIONN CAPABILITIES:")
    logger.info("   • Network visualization")
    logger.info("   • Spontaneous learning detection")
    logger.info("   • Intelligent rerouting suggestions")
    logger.info("   • BabyBIONN-specific insights")
    logger.info("=" * 60)
    
    return autonomous_orchestrator

# ========== DEMONSTRATION ==========

def demonstrate_integration():
    """Demonstrate the integrated system"""
    
    # Create or get existing Hebbian engine
    from neuron.aggregator import HebbianLearningEngine, AggregatorConfig
    
    config = AggregatorConfig(
        enable_hebbian_learning=True,
        learning_rate=0.1,
        decay_rate=0.01
    )
    
    # Use the extended version with visualization
    engine = VisualizedHebbianEngine(config, enable_visualization=True)
    
    # Simulate some learning
    engine.hebbian_update(
        vni1="VNI_medical_diabetes",
        vni2="VNI_legal_employment",
        pre_activation=0.7,
        post_activation=0.8,
        outcome_quality=0.9,
        context_hash="medical_legal"
    )
    
    # Add BabyBIONN extensions
    extensions = BabyBIONNLearningExtensions(engine)
    
    # Get insights
    insights = extensions.get_babybionn_insights()
    print("\nBabyBIONN Insights:")
    for key, value in insights.items():
        print(f"  {key}: {value}")
    
    # Create visualization
    engine.create_visualization("babybionn_network.png")
    
    # Detect spontaneous connections
    spontaneous = extensions.detect_spontaneous_activation()
    if spontaneous:
        print("\nSpontaneous connections detected:")
        for src, tgt, strength in spontaneous:
            print(f"  {src} → {tgt} (strength: {strength:.2f})")
    
    return engine

if __name__ == "__main__":
    demonstrate_integration() 
