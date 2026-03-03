# enhanced_vni_classes/managers/vni_manager.py
import time
import logging
from datetime import datetime
from ..core.base_vni import EnhancedBaseVNI
from typing import Dict, Any, List, Optional
from .dynamic_factory import DynamicVNIFactory
from ..core.neural_pathway import NeuralPathway
from ..managers.session_manager import SessionManager
from ..core.capabilities import VNICapabilities, VNIType
from ..domains.dynamic_vni import DynamicVNI, EnhancedDomainFactory

logger = logging.getLogger(__name__)

class VNIManager:
    """Enhanced manager with dynamic VNI creation support"""
    
    def __init__(self, enable_generation: bool = True, 
                 enable_dynamic_vnis: bool = True):
        self.vni_instances: Dict[str, EnhancedBaseVNI] = {}
        self.neural_pathways: Dict[str, NeuralPathway] = {}
        self.session_manager = SessionManager()
        self.attention_scores: Dict[str, float] = {}
        self.enable_generation = enable_generation
        self.enable_dynamic_vnis = enable_dynamic_vnis
        self.dynamic_factory = DynamicVNIFactory()        
        # Dynamic VNI registry
        self.dynamic_vni_configs: Dict[str, Any] = {}
        self.spawner = None  # Will be set when mesh is connected
    
    def connect_spawner_to_mesh(self, mesh_coordinator, config=None):
        """
        Connect auto-spawner to mesh coordinator.
        
        Args:
            mesh_coordinator: EnhancedNeuralMeshCore instance
            config: Optional spawner configuration
        """
        from enhanced_vni_classes.vni_spawner import VNISpawner
        
        self.spawner = VNISpawner(
            vni_manager=self,
            mesh_coordinator=mesh_coordinator,
            config=config or {}
        )
        print(f"[VNI-SPAWNER] Connected to mesh coordinator")
        return self.spawner
    
    def auto_spawn_if_needed(self):
        """Auto-spawn VNIs if analysis says we need more."""
        if self.spawner:
            analysis = self.spawner.analyze_spawning_need()
            if analysis.get('should_spawn', False):
                return self.spawner.spawn_new_vni(
                    pattern=analysis.get('recommended_pattern')
                )
        return None
    
    # ========== CREATE VNI METHOD ==========
    def create_vni(self, 
                   domain: str, 
                   instance_id: str = None, 
                   enable_generation: bool = False,
                   capabilities: Optional[VNICapabilities] = None,
                   vni_config: Optional[Dict[str, Any]] = None,
                   **kwargs) -> EnhancedBaseVNI:
        """Create a VNI instance with optional capabilities"""
        
        if instance_id is None:
            instance_id = f"{domain}-vni-{len(self.vni_instances)}"

        # Prepare configuration
        config = vni_config or {}
        
        # If capabilities provided, update config
        if capabilities:
            config['capabilities'] = capabilities
            # Use domain from capabilities if provided
            if hasattr(capabilities, 'domain') and capabilities.domain:
                domain = capabilities.domain
            elif hasattr(capabilities, 'domains') and capabilities.domains:
                domain = capabilities.domains[0]
        
        # Merge any additional kwargs into config
        config.update(kwargs)
        # ====== END OF ADDED SECTION ======

        # Import here to avoid circular imports
        from ..domains.medical import MedicalVNI
        from ..domains.legal import LegalVNI
        from ..domains.general import GeneralVNI
    
        # Create appropriate VNI based on domain
        if domain == "medical":
            # FIXED: MedicalVNI expects (vni_id: str, name: str = "Medical Assistant")
            vni = MedicalVNI(vni_id=instance_id, name=f"Medical Specialist {instance_id}")
        elif domain == "legal":
            # FIXED: LegalVNI expects (instance_id: str, vni_config: Dict = None)
            vni = LegalVNI(
                instance_id=instance_id,
                vni_config={"domain": "legal", "enable_generation": enable_generation}
            )
        elif domain == "general":
            # FIXED: GeneralVNI expects (instance_id: str, vni_config: Dict = None)
            vni = GeneralVNI(
                instance_id=instance_id,
                vni_config={"domain": "general", "enable_generation": enable_generation}
            )
        else:
            # ===== USE DYNAMIC FACTORY FOR ANY OTHER DOMAIN =====
            logger.info(f"🔄 Using DynamicVNIFactory for custom domain: {domain}")
            
            # Create base configuration
            base_config = {
                "domain": domain,
                "enable_generation": enable_generation,
                "safety_level": "medium",
                "is_custom_domain": True
            }
            
            # Use the factory to create VNI for ANY topic
            vni = self.dynamic_factory.create_for_topic(
                topic=domain,
                instance_id=instance_id,
                # base_config=base_config
            )
        
        # Register the VNI
        self._register_vni(vni)
        logger.info(f"✅ Created {domain} VNI: {instance_id}")
        
        return vni
    
    # ========== ADD NEW STANDARDIZED METHOD ==========
    def create_vni_standardized(self, 
                               capabilities: VNICapabilities = None,
                               instance_id: str = None,
                               vni_config: Dict[str, Any] = None) -> EnhancedBaseVNI:
        """New standardized method using VNICapabilities.
        Args:
            capabilities: VNICapabilities object (must include domain)
            instance_id: Optional custom instance ID
            vni_config: Optional additional configuration"""
        if capabilities is None:
            capabilities = VNICapabilities(domain="general")
        
        # For backward compatibility, call the old method with extracted domain
        return self.create_vni(
            domain=capabilities.domain,
            instance_id=instance_id,
            enable_generation=capabilities.generation_enabled
        )
    
    # ========== END OF CREATE VNI METHOD ==========
        
    def create_dynamic_vni(self, domain_config: Dict[str, Any], 
                          instance_id: str = None) -> DynamicVNI:
        """Create a dynamic VNI from configuration dictionary"""
        try:
            from ..domains.dynamic_vni import DomainConfig
            vni_config = DomainConfig.from_dict(domain_config)
            vni = DynamicVNI(vni_config, instance_id)
            
            self._register_vni(vni)
            logger.info(f"✅ Created dynamic VNI: {vni.instance_id} "
                       f"for domain: {vni_config.name}")
            
            return vni
        except Exception as e:
            logger.error(f"❌ Failed to create dynamic VNI: {e}")
            raise
        # ========== UPDATE CREATE_VNI_FROM_DOMAIN_NAME METHOD ==========
    def create_vni_from_domain_name(self, domain_name: str, 
                                   instance_id: str = None) -> EnhancedBaseVNI:
        """Create VNI from domain name (supports both predefined and dynamic)"""
        # Try predefined domain first
        if domain_name in ['medical', 'legal', 'general']:
            return self.create_vni(domain_name, instance_id)
        
        # Try dynamic domain from predefined configurations
        try:
            vni = EnhancedDomainFactory.create_dynamic_vni(domain_name, instance_id)
            self._register_vni(vni)
            return vni
        except ValueError:
            # Domain not found in predefined - use our dynamic factory
            logger.info(f"🔄 Using DynamicVNIFactory for: {domain_name}")
            return self._create_auto_detected_vni(domain_name, instance_id)
    
    def _create_auto_detected_vni(self, domain_hint: str, 
                                 instance_id: str = None) -> EnhancedBaseVNI:
        """Automatically create a VNI based on domain hint - ENHANCED VERSION"""
        logger.info(f"🔍 Auto-detecting domain configuration for: {domain_hint}")
        
        # Use the factory instead of hardcoded expansions
        base_config = {
            'confidence_threshold': 0.4,
            'generation_temperature': 0.7,
            'is_auto_detected': True
        }
        
        vni = self.dynamic_factory.create_for_any_topic(
            topic=domain_hint,
            instance_id=instance_id,
            base_config=base_config
        )
        
        # Update description to indicate auto-detection
        if hasattr(vni, 'domain_config'):
            vni.domain_config.description = f"Auto-detected domain: {domain_hint}. {vni.domain_config.description}"
        
        return vni
    
    def _extract_keywords_from_domain(self, domain_hint: str) -> List[str]:
        """Extract relevant keywords from domain hint - ENHANCED VERSION"""
        # Delegate to the factory's keyword extraction
        return self.dynamic_factory._extract_keywords(domain_hint)
    
    # ========== ADD NEW METHOD FOR ADVANCED DYNAMIC CREATION ==========
    def create_dynamic_vni_for_topic(self, topic: str, instance_id: str = None,
                                    custom_config: Dict[str, Any] = None) -> DynamicVNI:
        """
        Advanced method for creating dynamic VNIs with custom configuration.
        
        Args:
            topic: Any topic name (e.g., "quantum physics", "blockchain")
            instance_id: Optional custom instance ID
            custom_config: Optional custom configuration overrides
        
        Returns:
            DynamicVNI instance
        """
        logger.info(f"🎯 Creating advanced dynamic VNI for topic: {topic}")
        
        # Merge custom config with defaults
        base_config = {
            "enable_generation": self.enable_generation,
            "created_by": "create_dynamic_vni_for_topic",
            "timestamp": datetime.now().isoformat()
        }
        
        if custom_config:
            base_config.update(custom_config)
        
        # Use factory
        vni = self.dynamic_factory.create_for_any_topic(
            topic=topic,
            instance_id=instance_id,
            base_config=base_config
        )
        
        # Register and store config
        self._register_vni(vni)
        self.dynamic_vni_configs[vni.instance_id] = base_config
        
        return vni
    
    def analyze_and_spawn_dynamic(self, query: str, 
                                 context: Dict[str, Any] = None) -> List[str]:
        """Enhanced spawning with dynamic domain detection"""
        # Detect relevant domains
        detected_domains = EnhancedDomainFactory.detect_domain_from_query(query)
        
        # Also detect custom domains from context
        if context and 'custom_domains' in context:
            for domain_name in context['custom_domains']:
                if domain_name not in detected_domains:
                    # Check if domain name appears in query
                    if domain_name.lower() in query.lower():
                        detected_domains.append(domain_name)
        
        if not detected_domains:
            detected_domains = ['general']
        
        # Spawn VNIs for missing domains
        spawned = []
        for domain in detected_domains:
            if not self._has_domain_coverage(domain):
                try:
                    vni = self.create_vni_from_domain_name(domain)
                    spawned.append(vni.instance_id)
                    logger.info(f"🔄 Spawned VNI for domain: {domain}")
                except Exception as e:
                    logger.error(f"❌ Failed to spawn VNI for {domain}: {e}")
        
        return spawned
    
    def _has_domain_coverage(self, domain: str) -> bool:
        """Check if we already have a VNI for this domain"""
        for vni in self.vni_instances.values():
            if vni.vni_type == domain:
                return True
            # Also check dynamic VNIs with similar domains
            if hasattr(vni, 'domain_config'):
                if vni.domain_config.name == domain:
                    return True
        return False
    
    def _register_vni(self, vni: EnhancedBaseVNI):

        """Register VNI and create neural pathways"""
        self.vni_instances[vni.instance_id] = vni
        
        # Create neural pathways to existing VNIs
        for existing_id in self.vni_instances:
            if existing_id != vni.instance_id:
                pathway_id = f"{vni.instance_id}->{existing_id}"
                self.neural_pathways[pathway_id] = NeuralPathway(
                    source_id=vni.instance_id, 
                    target_id=existing_id,
                    pathway_type="bidirectional",  # Add this
                    #initial_strength=0.1  # Might also need this
                )                
                reverse_id = f"{existing_id}->{vni.instance_id}"
                self.neural_pathways[reverse_id] = NeuralPathway(
                    source_id=existing_id,
                    target_id=vni.instance_id,
                    pathway_type="bidirectional"
                )


# ========== VNI SPAWNER INTEGRATION ==========
class VNISpawner:
    """Auto-spawning system integrated directly into VNI management."""
    
    def __init__(self, vni_manager, mesh_coordinator, config=None):
        """
        Initialize with references to VNI manager and mesh.
        
        Args:
            vni_manager: The VNIManager instance (this class)
            mesh_coordinator: EnhancedNeuralMeshCore instance
            config: Optional configuration
        """
        self.vni_manager = vni_manager
        self.mesh = mesh_coordinator
        self.config = config or {}
        self.spawn_log = []
        
        # Spawner thresholds
        self.thresholds = {
            'min_neurons': 10,
            'ideal_density': 0.6,
            'spawn_cooldown': 2.0
        }
        
        print(f"[VNI-SPAWNER] Integrated with VNIManager")
    
    def spawn_new_vni(self, pattern=None, **kwargs):
        """
        Spawn a new VNI through the VNI manager.
        
        Args:
            pattern: VNI pattern type
            **kwargs: Additional VNI parameters
        """
        # 1. Determine domain/pattern
        domain = kwargs.pop('domain', 'general')
        if pattern:
            domain = pattern  # Use pattern as domain
        
        # 2. Create unique ID
        vni_id = f"{domain}_auto_{len(self.vni_manager.vni_instances) + 1:04d}"
        
        # 3. Use existing VNIManager to create VNI
        self.vni_manager.create_vni(
            domain=domain,
            vni_id=vni_id,
            enable_generation=kwargs.pop('enable_generation', True)
        )
        
        # 4. Get the created VNI
        new_vni = self.vni_manager.vni_instances.get(vni_id)
        if not new_vni:
            print(f"[VNI-SPAWNER] Failed to create VNI {vni_id}")
            return None
        
        # 5. Set additional properties
        for key, value in kwargs.items():
            if hasattr(new_vni, key):
                setattr(new_vni, key, value)
        
        # 6. Integrate into mesh
        self._integrate_into_mesh(new_vni)
        
        # 7. Log
        self.spawn_log.append({
            'timestamp': time.time(),
            'vni_id': vni_id,
            'pattern': pattern,
            'domain': domain
        })
        
        print(f"[VNI-SPAWNER] Spawned VNI-{vni_id} ({domain})")
        return new_vni
    
    def _integrate_into_mesh(self, vni):
        """Integrate VNI into neural mesh."""
        # Check if mesh has integration method
        if hasattr(self.mesh, 'add_virtual_neuron'):
            self.mesh.add_virtual_neuron(vni)
        elif hasattr(self.mesh, 'virtual_neurons'):
            # Add to mesh's neuron list
            if not hasattr(self.mesh, 'virtual_neurons'):
                self.mesh.virtual_neurons = []
            self.mesh.virtual_neurons.append(vni)
        
        # Create automatic connections
        self._create_auto_connections(vni)
    
    def _create_auto_connections(self, new_vni):
        """Create automatic connections for new VNI."""
        # Get existing VNIs from manager
        existing_vnis = list(self.vni_manager.vni_instances.values())
        if len(existing_vnis) <= 1:
            return
        
        # Create connections to 3 most recent VNIs (simple heuristic)
        for existing in existing_vnis[-3:]:
            if existing.vni_id == new_vni.vni_id:
                continue
            
            # Create connection record
            connection = {
                'source': new_vni.vni_id,
                'target': existing.vni_id,
                'strength': 0.5,
                'type': 'auto_spawn',
                'timestamp': time.time()
            }
            
            # Add to mesh connections if available
            if hasattr(self.mesh, 'connections'):
                self.mesh.connections.append(connection)
    
    def analyze_spawning_need(self):
        """
        Analyze if we need to spawn more VNIs.
        
        Returns:
            Dict with spawn recommendation
        """
        total_vnis = len(self.vni_manager.vni_instances)
        
        # Simple analysis
        if total_vnis < self.thresholds['min_neurons']:
            return {
                'should_spawn': True,
                'priority': 'high',
                'recommended_count': self.thresholds['min_neurons'] - total_vnis,
                'reason': f'Below minimum ({total_vnis} < {self.thresholds["min_neurons"]})'
            }
        
        # Check domain distribution
        domain_counts = {}
        for vni in self.vni_manager.vni_instances.values():
            domain = getattr(vni, 'vni_type', 'unknown')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # If any domain has 0 VNIs, spawn one
        for domain in ['medical', 'technical', 'legal', 'general']:
            if domain_counts.get(domain, 0) == 0:
                return {
                    'should_spawn': True,
                    'priority': 'medium',
                    'recommended_pattern': domain,
                    'reason': f'Missing domain: {domain}'
                }
        
        return {
            'should_spawn': False,
            'reason': 'Adequate VNI distribution'
        }
