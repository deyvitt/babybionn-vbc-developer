# vni_orchestrator.py
import asyncio
import hashlib
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enhanced_vni_classes import VNIManager, EnhancedBaseVNI, VNIRegistry
# We'll import from other modules once they're updated

logger = logging.getLogger("vni_orchestrator")

@dataclass
class ProcessingTask:
    """Represents a task flowing through the VNI network"""
    task_id: str
    query: str
    context: Dict[str, Any]
    source_vni: Optional[str] = None
    target_vnis: List[str] = field(default_factory=list)
    processing_path: List[str] = field(default_factory=list)  # VNIs that processed this
    status: str = "pending"  # pending, processing, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    result: Optional[Dict] = None

class VNIOrchestrator:
    """Main orchestrator coordinating VNIs, attention, and routing"""
    
    def __init__(self, vni_manager: VNIManager):
        self.vni_manager = vni_manager
        self.vni_registry = VNIRegistry.get_instance()
        
        # Import and initialize routing modules
        self._init_routing_modules()
        
        # Task tracking
        self.active_tasks: Dict[str, ProcessingTask] = {}
        self.task_history: List[ProcessingTask] = []
        self.max_history = 1000
        
        # Collaboration patterns
        self.collaboration_patterns = {}  # Pattern hash -> success rate
        
    def _init_routing_modules(self):
        """Initialize routing and attention modules"""
        try:
            # Import from demoHybridAttention
            from neuron.demoHybridAttention import HybridAttentionEngine
            self.attention_engine = HybridAttentionEngine()
            logger.info("✅ HybridAttentionEngine loaded")
        except ImportError:
            self.attention_engine = None
            logger.warning("⚠️  HybridAttentionEngine not available")
            
        try:
            # Import from smart_activation_router
            from neuron.smart_activation_router import SmartActivationRouter
            self.activation_router = SmartActivationRouter()
            logger.info("✅ SmartActivationRouter loaded")
        except ImportError:
            self.activation_router = None
            logger.warning("⚠️  SmartActivationRouter not available")
            
        try:
            # Import from transVNI_compare_segregate
            from neuron.transVNI_compare_segregate import TransVNICompareSegregate
            self.trans_vni = TransVNICompareSegregate()
            logger.info("✅ TransVNICompareSegregate loaded")
        except ImportError:
            self.trans_vni = None
            logger.warning("⚠️  TransVNICompareSegregate not available")
            
        try:
            # Import from baseVNI_demo for abstraction patterns
            from neuron.baseVNI_demo import SmartBaseVNI, EnhancedVNIConfig
            config = EnhancedVNIConfig()
            self.base_vni = SmartBaseVNI(config)
            logger.info("✅ BaseVNI patterns loaded")
        except ImportError:
            self.base_vni = None
            logger.warning("⚠️  BaseVNI patterns not available")
    
    async def process_query(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """Main entry point for processing queries through VNI network"""
        
        # Create task
        task_id = f"task_{hashlib.md5(query.encode()).hexdigest()[:8]}"
        task = ProcessingTask(
            task_id=task_id,
            query=query,
            context=context or {}
        )
        
        self.active_tasks[task_id] = task
        task.status = "processing"
        
        try:
            # Step 1: Analyze query using baseVNI patterns if available
            initial_analysis = await self._analyze_query_with_basevni(query, context)
            
            # Step 2: Determine required capabilities
            required_capabilities = self._extract_required_capabilities(initial_analysis, query)
            
            # Step 3: Get or create VNIs with needed capabilities
            activated_vnis = await self._get_or_create_vnis(required_capabilities)
            
            # Step 4: Use attention/routing to organize VNIs
            if self.attention_engine and self.activation_router:
                organized_vnis = await self._organize_vnis_with_attention(
                    activated_vnis, query, context
                )
            else:
                organized_vnis = activated_vnis
            
            # Step 5: Process through VNI network
            result = await self._process_through_vni_network(
                task, organized_vnis, initial_analysis
            )
            
            # Step 6: Update learning
            await self._update_learning_from_task(task, result, organized_vnis)
            
            task.result = result
            task.status = "completed"
            
            # Save to history
            self.task_history.append(task)
            if len(self.task_history) > self.max_history:
                self.task_history.pop(0)
            
            return result
            
        except Exception as e:
            logger.error(f"Processing failed for task {task_id}: {e}")
            task.status = "failed"
            task.result = {"error": str(e)}
            
            # Fallback response
            return self._generate_fallback_response(query, context)
    
    async def _analyze_query_with_basevni(self, query: str, context: Dict) -> Dict:
        """Use baseVNI patterns for initial analysis"""
        if self.base_vni:
            try:
                # Assuming base_vni can process queries
                analysis = self.base_vni.process({"text": query})
                return analysis
            except:
                pass
        
        # Fallback analysis
        return {
            "modality": "text",
            "complexity": "medium",
            "estimated_topics": ["general"],
            "abstraction_hint": "semantic"
        }
    
    async def _get_or_create_vnis(self, capabilities: Dict) -> List[EnhancedBaseVNI]:
        """Get existing VNIs or create new ones with required capabilities"""
        available_vnis = []
        
        # Find existing VNIs with matching capabilities
        for vni_id, vni in self.vni_manager.vni_instances.items():
            if self._vni_has_capabilities(vni, capabilities):
                available_vnis.append(vni)
        
        # If not enough VNIs, create new ones
        if len(available_vnis) < 3:  # Minimum 3 VNIs for collaboration
            requirements = {
                "capabilities": capabilities,
                "priority": "medium"
            }
            new_vni = self.vni_manager.create_dynamic_vni(requirements)
            available_vnis.append(new_vni)
        
        return available_vnis
    
    async def _organize_vnis_with_attention(self, vnis: List[EnhancedBaseVNI], 
                                           query: str, context: Dict) -> List[EnhancedBaseVNI]:
        """Use attention and routing to organize VNIs"""
        
        # Convert VNIs to attention input format
        vni_descriptors = []
        for vni in vnis:
            descriptor = {
                "id": vni.instance_id,
                "type": vni.vni_type,
                "capabilities": list(vni.available_capabilities.specializations),
                "collaboration_score": vni.available_capabilities.collaboration_score
            }
            vni_descriptors.append(descriptor)
        
        # Get attention scores
        if self.attention_engine:
            attention_scores = self.attention_engine.compute_scores(query, vni_descriptors, context)
            
            # Sort VNIs by attention score
            sorted_pairs = sorted(
                zip(vnis, attention_scores),
                key=lambda x: x[1],
                reverse=True
            )
            vnis = [vni for vni, _ in sorted_pairs]
        
        # Use activation router to create processing pathway
        if self.activation_router and len(vnis) > 1:
            pathway = self.activation_router.create_pathway(vnis, query, context)
            if pathway:
                # Reorder VNIs according to pathway
                ordered_ids = [step["vni_id"] for step in pathway]
                vni_dict = {vni.instance_id: vni for vni in vnis}
                vnis = [vni_dict[vni_id] for vni_id in ordered_ids if vni_id in vni_dict]
        
        return vnis
    
    async def _process_through_vni_network(self, task: ProcessingTask, 
                                          vnis: List[EnhancedBaseVNI], 
                                          initial_analysis: Dict) -> Dict:
        """Process query through the VNI network with collaboration"""
        
        if not vnis:
            return {"error": "No VNIs available"}
        
        # For single VNI, just process directly
        if len(vnis) == 1:
            vni = vnis[0]
            task.processing_path.append(vni.instance_id)
            result = await self._process_with_vni(vni, task.query, task.context)
            return result
        
        # For multiple VNIs, use collaborative processing
        # Start with first VNI
        current_vni = vnis[0]
        task.processing_path.append(current_vni.instance_id)
        
        # Process with collaboration context
        collaboration_context = {
            "initial_analysis": initial_analysis,
            "collaborating_vnis": [v.instance_id for v in vnis[1:]],
            "task_id": task.task_id,
            "processing_mode": "cascade"  # or "parallel", "sequential"
        }
        
        current_context = {**(task.context or {}), **collaboration_context}
        
        # Main processing loop
        intermediate_result = None
        for i, next_vni in enumerate(vnis[1:], 1):
            try:
                # Current VNI processes with context
                result = await self._process_with_vni(
                    current_vni, 
                    task.query, 
                    current_context
                )
                
                # Add to processing path
                task.processing_path.append(next_vni.instance_id)
                
                # Prepare context for next VNI
                current_context = {
                    "previous_result": result,
                    "previous_vni": current_vni.instance_id,
                    "stage": i,
                    "total_stages": len(vnis)
                }
                
                intermediate_result = result
                current_vni = next_vni
                
            except Exception as e:
                logger.error(f"VNI {current_vni.instance_id} failed: {e}")
                break
        
        return intermediate_result or {"error": "Processing failed"}
    
    async def _process_with_vni(self, vni: EnhancedBaseVNI, query: str, context: Dict) -> Dict:
        """Process query with a single VNI"""
        try:
            # Update usage stats
            if vni.instance_id in self.vni_manager.usage_stats:
                stats = self.vni_manager.usage_stats[vni.instance_id]
                stats['last_used'] = datetime.now()
                stats['usage_count'] += 1
            
            # Process query
            result = vni.process_query(query, context)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing with VNI {vni.instance_id}: {e}")
            return {"error": str(e), "vni_id": vni.instance_id}
    
    async def _update_learning_from_task(self, task: ProcessingTask, 
                                        result: Dict, vnis: List[EnhancedBaseVNI]):
        """Update learning based on task outcome"""
        
        # Determine success
        success = result.get('confidence', 0) > 0.6 and 'error' not in result
        
        # Update neural pathways in VNI Manager
        for i in range(len(vnis) - 1):
            for j in range(i + 1, len(vnis)):
                source_id = vnis[i].instance_id
                target_id = vnis[j].instance_id
                
                pathway_key = f"{source_id}->{target_id}"
                if pathway_key in self.vni_manager.neural_pathways:
                    self.vni_manager.neural_pathways[pathway_key].activate(success)
        
        # Update collaboration patterns
        pattern_hash = self._hash_collaboration_pattern([v.instance_id for v in vnis])
        if pattern_hash not in self.collaboration_patterns:
            self.collaboration_patterns[pattern_hash] = {
                "pattern": [v.instance_id for v in vnis],
                "success_count": 0,
                "total_count": 0
            }
        
        stats = self.collaboration_patterns[pattern_hash]
        stats["total_count"] += 1
        if success:
            stats["success_count"] += 1
    
    def _hash_collaboration_pattern(self, vni_ids: List[str]) -> str:
        """Create hash for collaboration pattern"""
        pattern_str = "->".join(sorted(vni_ids))
        return hashlib.md5(pattern_str.encode()).hexdigest()[:8]
    
    def _generate_fallback_response(self, query: str, context: Dict) -> Dict:
        """Generate fallback response when processing fails"""
        return {
            "response": f"I encountered an issue processing your query. As a fallback, I suggest: Please try rephrasing or providing more context.",
            "confidence": 0.3,
            "fallback_used": True,
            "processing_path": [],
            "timestamp": datetime.now().isoformat()
        } 
