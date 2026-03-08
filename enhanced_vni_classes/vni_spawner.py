# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

"""
BIONN-AUTO-SPAWN-001: VNI Spawner System
Coordinates between VNI creation and mesh integration.
"""

import time
from typing import List, Dict, Any, Optional, Callable
import random

class VNISpawner:
    """
    Intelligent VNI spawning system that:
    1. Receives new VNIs from enhanced_vni_classes.py
    2. Analyzes when to spawn new VNIs
    3. Integrates VNIs into the neural mesh
    4. Coordinates with enhanced_neural_mesh.py
    """
    
    def __init__(self, 
                 mesh_coordinator,  # Reference to enhanced_neural_mesh instance
                 vni_factory=None,   # Optional: reference to VNI factory
                 config: Optional[Dict] = None):
        """
        Initialize the spawner with references to other components.
        
        Args:
            mesh_coordinator: Instance of EnhancedNeuralMesh (coordinator)
            vni_factory: Callable or instance that creates VNIs
            config: Configuration dictionary
        """
        self.mesh = mesh_coordinator
        self.vni_factory = vni_factory
        self.config = config or {}
        
        # Spawner state
        self.spawn_log = []
        self.analytics = {
            'total_spawned': 0,
            'last_spawn_time': 0,
            'average_integration_time': 0
        }
        
        # Thresholds and parameters
        self.thresholds = {
            'min_neurons': 5,
            'ideal_density': 0.6,
            'spawn_cooldown': 2.0,  # seconds
            'max_connections_per_vni': 7
        }
        
        # Pattern preferences (can be learned)
        self.pattern_weights = {
            'oscillator': 0.25,
            'resonator': 0.25,
            'integrator': 0.25,
            'differentiator': 0.25
        }
        
        print("[VNI-SPAWNER] Initialized with mesh coordinator")
    
    # ========== PUBLIC API ==========
    
    def spawn_new_vni(self, pattern: Optional[str] = None, **kwargs) -> Any:
        """
        Main entry point: Spawn a new VNI and integrate it.
        
        Args:
            pattern: Optional specific pattern to create
            **kwargs: Additional parameters for VNI creation
            
        Returns:
            The created VNI instance
        """
        # Check cooldown
        if time.time() - self.analytics['last_spawn_time'] < self.thresholds['spawn_cooldown']:
            print("[VNI-SPAWNER] In cooldown, skipping spawn")
            return None
        
        # 1. Create VNI (either using factory or default)
        new_vni = self._create_vni(pattern, **kwargs)
        if not new_vni:
            print("[VNI-SPAWNER] Failed to create VNI")
            return None
        
        # 2. Integrate into mesh
        start_time = time.time()
        connections = self._integrate_into_mesh(new_vni)
        integration_time = time.time() - start_time
        
        # 3. Update analytics
        self.analytics['total_spawned'] += 1
        self.analytics['last_spawn_time'] = time.time()
        self.analytics['average_integration_time'] = (
            self.analytics['average_integration_time'] * (self.analytics['total_spawned'] - 1) + 
            integration_time
        ) / self.analytics['total_spawned']
        
        # 4. Log the spawn
        self.spawn_log.append({
            'timestamp': time.time(),
            'vni_id': getattr(new_vni, 'id', 'unknown'),
            'pattern': pattern,
            'connections_made': len(connections),
            'integration_time': integration_time
        })
        
        print(f"[VNI-SPAWNER] Spawned VNI-{new_vni.id} with {len(connections)} connections")
        return new_vni
    
    def analyze_and_spawn_if_needed(self) -> List[Any]:
        """
        Intelligent analysis: Check mesh state and spawn VNIs if needed.
        
        Returns:
            List of spawned VNIs (empty if none spawned)
        """
        spawned_vnis = []
        
        # Get analysis from mesh coordinator
        mesh_analysis = self._get_mesh_analysis()
        
        # Determine if spawning is needed
        spawn_decisions = self._make_spawn_decisions(mesh_analysis)
        
        # Execute spawn decisions
        for decision in spawn_decisions:
            if decision['should_spawn']:
                vni = self.spawn_new_vni(
                    pattern=decision.get('recommended_pattern'),
                    priority=decision.get('priority', 'normal')
                )
                if vni:
                    spawned_vnis.append(vni)
        
        return spawned_vnis
    
    def integrate_external_vni(self, vni) -> List[Dict]:
        """
        Integrate a VNI created externally (e.g., from enhanced_vni_classes.py).
        
        Args:
            vni: VNI instance to integrate
            
        Returns:
            List of connections created
        """
        print(f"[VNI-SPAWNER] Integrating external VNI-{vni.id}")
        return self._integrate_into_mesh(vni)
    
    def get_spawn_recommendation(self) -> Dict[str, Any]:
        """
        Get recommendation on whether/what to spawn.
        
        Returns:
            Dictionary with spawn recommendations
        """
        mesh_analysis = self._get_mesh_analysis()
        return self._make_spawn_decisions(mesh_analysis)
    
    # ========== INTEGRATION WITH ENHANCED_VNI_CLASSES ==========
    
    def set_vni_factory(self, factory_func: Callable):
        """
        Set the function/object that creates VNIs.
        
        Args:
            factory_func: Callable that returns VNI instances
        """
        self.vni_factory = factory_func
        print("[VNI-SPAWNER] VNI factory set")
    
    def request_vni_creation(self, specifications: Dict) -> Any:
        """
        Request creation of a VNI with specific properties.
        This interfaces with enhanced_vni_classes.py.
        
        Args:
            specifications: Dict with VNI properties
            
        Returns:
            Created VNI instance or None
        """
        if not self.vni_factory:
            print("[VNI-SPAWNER] No VNI factory available")
            return self._create_default_vni(specifications)
        
        try:
            # Try to use the factory
            if callable(self.vni_factory):
                vni = self.vni_factory(**specifications)
            elif hasattr(self.vni_factory, 'create_vni'):
                vni = self.vni_factory.create_vni(**specifications)
            else:
                vni = self.vni_factory(**specifications)
            
            print(f"[VNI-SPAWNER] Factory created VNI with specs: {specifications}")
            return vni
        except Exception as e:
            print(f"[VNI-SPAWNER] Factory failed: {e}")
            return self._create_default_vni(specifications)
    
    # ========== CORE SPAWNING LOGIC ==========
    
    def _create_vni(self, pattern: Optional[str] = None, **kwargs) -> Any:
        """
        Create a new VNI instance.
        
        Args:
            pattern: Pattern type for the VNI
            **kwargs: Additional VNI properties
            
        Returns:
            VNI instance
        """
        # Determine pattern if not specified
        if not pattern:
            pattern = self._select_optimal_pattern()
        
        # Prepare specifications
        specs = {
            'pattern': pattern,
            **kwargs
        }
        
        # Add frequency based on pattern
        if 'frequency' not in specs:
            specs['frequency'] = self._get_pattern_frequency(pattern)
        
        # Create the VNI
        return self.request_vni_creation(specs)
    
    def _integrate_into_mesh(self, vni) -> List[Dict]:
        """
        Integrate a VNI into the neural mesh.
        Interfaces with enhanced_neural_mesh.py.
        
        Args:
            vni: VNI instance to integrate
            
        Returns:
            List of connections created
        """
        # Check if mesh has integration method
        if hasattr(self.mesh, 'add_virtual_neuron'):
            # Use mesh's own method if available
            self.mesh.add_virtual_neuron(vni)
            connections = []
        elif hasattr(self.mesh, 'integrate_vni'):
            # Use custom integration method
            connections = self.mesh.integrate_vni(vni)
        else:
            # Default integration
            connections = self._default_mesh_integration(vni)
        
        # Create automatic connections
        auto_connections = self._create_automatic_connections(vni)
        connections.extend(auto_connections)
        
        return connections
    
    def _create_automatic_connections(self, new_vni) -> List[Dict]:
        """
        Create automatic connections for a new VNI.
        
        Args:
            new_vni: New VNI instance
            
        Returns:
            List of connection dictionaries
        """
        connections = []
        
        # Get existing VNIs from mesh
        existing_vnis = self._get_existing_vnis()
        if len(existing_vnis) <= 1:
            return connections
        
        # Calculate compatibility scores
        compat_scores = []
        for existing in existing_vnis:
            if getattr(existing, 'id', None) == getattr(new_vni, 'id', None):
                continue
            
            score = self._calculate_compatibility(new_vni, existing)
            compat_scores.append((existing, score))
        
        # Sort by compatibility
        compat_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Create connections
        max_conn = min(self.thresholds['max_connections_per_vni'], len(compat_scores))
        for target_vni, score in compat_scores[:max_conn]:
            if score > 0.2:  # Minimum threshold
                connection = self._create_connection_record(new_vni, target_vni, score)
                connections.append(connection)
                
                # Also add to mesh if it tracks connections
                if hasattr(self.mesh, 'connections'):
                    self.mesh.connections.append(connection)
        
        return connections
    
    def _get_mesh_analysis(self) -> Dict[str, Any]:
        """
        Get analysis from the mesh coordinator.
        
        Returns:
            Dictionary with mesh analysis
        """
        analysis = {
            'neuron_count': 0,
            'connection_count': 0,
            'connection_density': 0,
            'pattern_distribution': {},
            'activity_level': 0.5,
            'requires_attention': False
        }
        
        # Try to get analysis from mesh
        if hasattr(self.mesh, 'analyze_mesh_state'):
            try:
                mesh_analysis = self.mesh.analyze_mesh_state()
                analysis.update(mesh_analysis)
            except:
                pass
        
        # Fallback: direct inspection
        if hasattr(self.mesh, 'virtual_neurons'):
            neurons = self.mesh.virtual_neurons
            analysis['neuron_count'] = len(neurons)
            
            # Count patterns
            for vni in neurons:
                if hasattr(vni, 'pattern'):
                    pattern = vni.pattern
                    analysis['pattern_distribution'][pattern] = \
                        analysis['pattern_distribution'].get(pattern, 0) + 1
        
        if hasattr(self.mesh, 'connections'):
            analysis['connection_count'] = len(self.mesh.connections)
        
        # Calculate density
        n = analysis['neuron_count']
        if n > 1:
            max_possible = n * (n - 1) / 2
            analysis['connection_density'] = analysis['connection_count'] / max_possible
        
        return analysis
    
    def _make_spawn_decisions(self, mesh_analysis: Dict) -> List[Dict]:
        """
        Make decisions about spawning based on mesh analysis.
        
        Args:
            mesh_analysis: Analysis from mesh coordinator
            
        Returns:
            List of spawn decisions
        """
        decisions = []
        
        # Decision 1: Minimum neuron count
        if mesh_analysis['neuron_count'] < self.thresholds['min_neurons']:
            needed = self.thresholds['min_neurons'] - mesh_analysis['neuron_count']
            for _ in range(min(needed, 3)):  # Max 3 at once
                decisions.append({
                    'should_spawn': True,
                    'priority': 'high',
                    'reason': f'Below minimum threshold ({mesh_analysis["neuron_count"]} < {self.thresholds["min_neurons"]})',
                    'recommended_pattern': self._select_optimal_pattern()
                })
        
        # Decision 2: Connection density
        elif mesh_analysis['connection_density'] < self.thresholds['ideal_density'] * 0.7:
            decisions.append({
                'should_spawn': True,
                'priority': 'medium',
                'reason': f'Low connection density ({mesh_analysis["connection_density"]:.2f})',
                'recommended_pattern': 'resonator'  # Resonators create many connections
            })
        
        # Decision 3: Pattern balance
        pattern_balance = self._analyze_pattern_balance(mesh_analysis['pattern_distribution'])
        if pattern_balance.get('needs_balance', False):
            for pattern in pattern_balance.get('underrepresented', []):
                decisions.append({
                    'should_spawn': True,
                    'priority': 'low',
                    'reason': f'Underrepresented pattern: {pattern}',
                    'recommended_pattern': pattern
                })
        
        # If no decisions, maybe don't spawn
        if not decisions:
            decisions.append({
                'should_spawn': False,
                'reason': 'Mesh is in good state',
                'priority': 'none'
            })
        
        return decisions
    
    # ========== HELPER METHODS ==========
    
    def _select_optimal_pattern(self) -> str:
        """Select optimal pattern based on current mesh state."""
        analysis = self._get_mesh_analysis()
        distribution = analysis['pattern_distribution']
        
        if not distribution:
            return random.choice(list(self.pattern_weights.keys()))
        
        # Find least represented pattern
        total = sum(distribution.values())
        least_ratio = float('inf')
        least_pattern = None
        
        for pattern, weight in self.pattern_weights.items():
            current = distribution.get(pattern, 0)
            ratio = current / max(1, total) / weight
            
            if ratio < least_ratio:
                least_ratio = ratio
                least_pattern = pattern
        
        return least_pattern or random.choice(list(self.pattern_weights.keys()))
    
    def _analyze_pattern_balance(self, distribution: Dict) -> Dict:
        """Analyze if patterns are balanced."""
        if not distribution:
            return {'needs_balance': False}
        
        total = sum(distribution.values())
        underrepresented = []
        
        for pattern, weight in self.pattern_weights.items():
            expected = total * weight
            actual = distribution.get(pattern, 0)
            
            if actual < expected * 0.5:  # Less than half of expected
                underrepresented.append(pattern)
        
        return {
            'needs_balance': len(underrepresented) > 0,
            'underrepresented': underrepresented[:2]  # Max 2 at once
        }
    
    def _calculate_compatibility(self, vni_a, vni_b) -> float:
        """Calculate compatibility between two VNIs."""
        # Simplified compatibility
        score = 0.5
        
        # Pattern compatibility
        if hasattr(vni_a, 'pattern') and hasattr(vni_b, 'pattern'):
            if vni_a.pattern == vni_b.pattern:
                score += 0.2
            elif self._are_patterns_compatible(vni_a.pattern, vni_b.pattern):
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _are_patterns_compatible(self, pattern_a: str, pattern_b: str) -> bool:
        """Check if two patterns are compatible."""
        compatible = {
            'oscillator': ['resonator'],
            'resonator': ['oscillator', 'integrator'],
            'integrator': ['resonator', 'differentiator'],
            'differentiator': ['integrator']
        }
        return pattern_b in compatible.get(pattern_a, [])
    
    def _get_pattern_frequency(self, pattern: str) -> float:
        """Get typical frequency for a pattern."""
        base_frequencies = {
            'oscillator': 1.0,
            'resonator': 0.5,
            'integrator': 0.1,
            'differentiator': 2.0
        }
        return base_frequencies.get(pattern, 1.0) * random.uniform(0.8, 1.2)
    
    def _get_existing_vnis(self) -> List:
        """Get existing VNIs from mesh."""
        if hasattr(self.mesh, 'virtual_neurons'):
            return self.mesh.virtual_neurons
        elif hasattr(self.mesh, 'get_virtual_neurons'):
            return self.mesh.get_virtual_neurons()
        else:
            return []
    
    def _default_mesh_integration(self, vni) -> List[Dict]:
        """Default integration if mesh doesn't have specific method."""
        if not hasattr(self.mesh, 'virtual_neurons'):
            self.mesh.virtual_neurons = []
        self.mesh.virtual_neurons.append(vni)
        return []
    
    def _create_default_vni(self, specifications: Dict) -> Any:
        """Create a default VNI if factory is unavailable."""
        # This is a minimal VNI class for fallback
        class DefaultVNI:
            def __init__(self, **kwargs):
                self.id = f"vni_{int(time.time()*1000)}_{random.randint(1000,9999)}"
                self.pattern = kwargs.get('pattern', 'oscillator')
                self.frequency = kwargs.get('frequency', 1.0)
                self.created = time.time()
        
        return DefaultVNI(**specifications)
    
    def _create_connection_record(self, source, target, strength: float) -> Dict:
        """Create a connection record."""
        return {
            'id': f"conn_{getattr(source, 'id', 'src')}_{getattr(target, 'id', 'tgt')}_{int(time.time()*1000)}",
            'source': getattr(source, 'id', 'unknown'),
            'target': getattr(target, 'id', 'unknown'),
            'strength': strength,
            'created': time.time(),
            'type': 'auto_spawn'
        }
    
    def get_status(self) -> Dict:
        """Get current status of the spawner."""
        return {
            **self.analytics,
            'total_log_entries': len(self.spawn_log),
            'thresholds': self.thresholds
        } 
