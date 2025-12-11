# neuron/smart_activation_router.py
import time
import torch
import asyncio
import logging
import torch.nn as nn
import concurrent.futures
import torch.nn.functional as F
from dataclasses import dataclass
from collections import defaultdict
from enhanced_vni_classes import EnhancedBaseVNI
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger("smart_activation_router")

@dataclass
class RouterConfig:
    """Configuration for smart activation router"""
    router_id: str = "smart_activation_router"
    activation_threshold: float = 0.3
    max_parallel_vnis: int = 5
    timeout_seconds: int = 30
    fallback_strategy: str = "sequential"
    
    # VNI performance weights - UPDATED for general VNI
    vni_performance_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.vni_performance_weights is None:
            self.vni_performance_weights = {
                'operAction_medical': 0.85,
                'operAction_legal': 0.80,
                'operAction_general': 0.82,  # CHANGED from technical to general
                'transVNI_compare_segregate': 0.90,
                'transVNI_merge_integrate': 0.88
            }

# UPDATE the specialization matching to include new domains
def calculate_specialization_match(self, baseVNI_output: Dict[str, Any], 
                                 specializations: List[str]) -> float:
    """Calculate how well VNI specializations match the input"""
    topic_scores = baseVNI_output.get('topic_classification', {})
    
    match_score = 0.0
    for specialization in specializations:
        # Map VNI specializations to topics - EXPANDED for general VNI
        specialization_to_topic = {
            'medical': 'medical',
            'legal': 'legal', 
            'technical': 'technical',
            'mathematical': 'technical',  # Map to technical for backward compatibility
            'business': 'technical',
            'creative': 'technical', 
            'analytical': 'technical',
            'scientific': 'technical',
            'educational': 'technical',
            'general': 'technical',
            'comparison': 'general',
            'merging': 'general'
        }
        topic = specialization_to_topic.get(specialization)
        if topic in topic_scores:
            match_score += topic_scores[topic]
    
    return min(match_score, 1.0)

# UPDATE cross-domain activation detection
def is_cross_domain_activation(self, activated_vnis: List[Dict[str, Any]]) -> bool:
    """Check if this is a cross-domain activation - EXPANDED domains"""
    domains = set()
    for vni in activated_vnis:
        for specialization in vni['specializations']:
            # Include all general VNI domains in cross-domain detection
            if specialization in ['medical', 'legal', 'technical', 
                                'mathematical', 'business', 'creative', 
                                'analytical', 'scientific', 'educational']:
                domains.add(specialization)
    
    return len(domains) >= 2

# UPDATE the test function to use general VNI
def test_smart_activation_router():
    """Test the smart activation router with sample VNIs - UPDATED for general"""
    
    # Initialize router
    config = RouterConfig(router_id="smart_router_test_001")
    router = SmartActivationRouter(config)
    
    # Create mock VNIs for testing - UPDATED for general VNI
    class MockVNI(nn.Module):
        def __init__(self, vni_id, domain):
            super().__init__()
            self.vni_id = vni_id
            self.domain = domain
        
        def forward(self, inputs):
            return {
                f"{self.domain}_analysis": {"result": f"Mock analysis from {self.vni_id}"},
                'confidence_score': 0.8,
                'vni_metadata': {
                    'vni_id': self.vni_id,
                    'success': True,
                    'domain': self.domain
                }
            }
        
        def get_capabilities(self):
            return {
                'input_types': [f'{self.domain}_data'],
                'output_types': [f'{self.domain}_analysis']
            }
    
    # Register mock VNIs - UPDATED for general VNI
    mock_vnis = [
        ('operAction_medical', ['medical'], MockVNI('operAction_medical', 'medical')),
        ('operAction_legal', ['legal'], MockVNI('operAction_legal', 'legal')),
        ('operAction_general', ['mathematical', 'business', 'creative', 'technical'],  # EXPANDED specializations
         MockVNI('operAction_general', 'general')),
        ('transVNI_compare_segregate', ['comparison'], MockVNI('transVNI_compare_segregate', 'comparison'))
    ]
    
    for vni_id, specializations, vni_instance in mock_vnis:
        router.register_vni(
            vni_id, 
            vni_instance, 
            specializations, 
            vni_instance.get_capabilities()
        )
    
    # Create test baseVNI output - UPDATED for multi-domain
    test_baseVNI_output = {
        'abstraction_levels': {
            'cognitive': {
                'tensor': torch.randn(256),
                'concepts': ['calculate', 'business', 'strategy', 'growth', 'profit'],
                'intent': 'problem_solving'
            },
            'structural': {
                'tensor': torch.randn(256)
            }
        },
        'topic_classification': {
            'medical': 0.2,
            'legal': 0.1,
            'technical': 0.7  # General VNI will handle this
        },
        'primary_topic': 'technical'
    }
    
    print("=== Smart Activation Router Demo Test (Updated for General VNI) ===\n")
    
    with torch.no_grad():
        results = router(test_baseVNI_output)
    
    # Display results
    activation_plan = results['activation_plan']
    print(f"Execution Strategy: {activation_plan['execution_strategy']}")
    print(f"VNIs Activated: {len(activation_plan['activated_vnis'])}")
    
    print("\nActivation Scores:")
    for vni_id, score in activation_plan['activation_scores'].items():
        print(f"  {vni_id}: {score:.3f}")
    
    print("\nExecution Results:")
    for vni_id, result in results['execution_results'].items():
        success = result.get('vni_metadata', {}).get('success', False)
        print(f"  {vni_id}: {'SUCCESS' if success else 'FAILED'}")
    
    return router

class VNIRegistry:
    """Registry for managing available VNIs"""
    
    def __init__(self):
        self.available_vnis = {}
        self.vni_performance = defaultdict(list)
        self.vni_specializations = {}
        
    def register_vni(self, vni_id: str, vni_instance: nn.Module, 
                    specializations: List[str], capabilities: Dict[str, Any]):
        """Register a VNI with the router"""
        self.available_vnis[vni_id] = {
            'instance': vni_instance,
            'specializations': specializations,
            'capabilities': capabilities,
            'status': 'available',
            'last_used': time.time(),
            'performance_history': []
        }
        self.vni_specializations[vni_id] = specializations
        
        logger.info(f"VNI registered: {vni_id} with specializations: {specializations}")
    
    def get_vni(self, vni_id: str) -> Optional[Dict[str, Any]]:
        """Get VNI instance by ID"""
        return self.available_vnis.get(vni_id)
    
    def get_vnis_by_specialization(self, specialization: str) -> List[str]:
        """Get VNIs that have a specific specialization"""
        matching_vnis = []
        for vni_id, specializations in self.vni_specializations.items():
            if specialization in specializations:
                matching_vnis.append(vni_id)
        return matching_vnis
    
    def update_performance(self, vni_id: str, performance_score: float):
        """Update VNI performance metrics"""
        if vni_id in self.available_vnis:
            self.available_vnis[vni_id]['performance_history'].append(performance_score)
            # Keep only last 100 performance records
            if len(self.available_vnis[vni_id]['performance_history']) > 100:
                self.available_vnis[vni_id]['performance_history'].pop(0)
    
    def get_average_performance(self, vni_id: str) -> float:
        """Get average performance score for VNI"""
        if vni_id in self.available_vnis:
            history = self.available_vnis[vni_id]['performance_history']
            if history:
                return sum(history) / len(history)
        return 0.5  # Default performance
    
    def get_all_vnis(self) -> List[str]:
        """Get all registered VNI IDs"""
        return list(self.available_vnis.keys())

class ActivationScorer(nn.Module):
    """Neural network for scoring VNI activations"""
    
    def __init__(self, config: RouterConfig):
        super().__init__()
        self.config = config
        
        # Activation scoring network
        self.activation_scorer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Tanh()
        )
        
        # Context integration
        self.context_integrator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Sigmoid()
        )
        
        # Cross-VNI relationship modeling
        self.relationship_model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Softmax(dim=-1)
        )
    
    def compute_activation_score(self, baseVNI_output: Dict[str, Any], 
                               vni_id: str, vni_specializations: List[str]) -> float:
        """Compute activation score for a specific VNI"""
        
        # Extract features from baseVNI output
        features = self.extract_activation_features(baseVNI_output, vni_specializations)
        
        # Get VNI performance weight
        performance_weight = self.config.vni_performance_weights.get(vni_id, 0.5)
        
        # Neural network scoring
        with torch.no_grad():
            feature_tensor = torch.tensor(features['numerical']).unsqueeze(0)
            activation_score = self.activation_scorer(feature_tensor)
            context_score = self.context_integrator(activation_score)
            final_score = context_score.mean().item()
        
        # Apply performance weighting
        weighted_score = final_score * performance_weight
        
        # Apply specialization matching boost
        specialization_match = self.calculate_specialization_match(
            baseVNI_output, vni_specializations
        )
        weighted_score *= (1.0 + specialization_match * 0.3)  # Up to 30% boost
        
        return min(weighted_score, 1.0)  # Cap at 1.0
    
    def extract_activation_features(self, baseVNI_output: Dict[str, Any], 
                                  specializations: List[str]) -> Dict[str, Any]:
        """Extract features for activation scoring"""
        
        features = {
            'numerical': [],
            'categorical': [],
            'contextual': []
        }
        
        # Topic classification scores
        topic_scores = baseVNI_output.get('topic_classification', {})
        for topic in ['medical', 'legal', 'technical']:
            features['numerical'].append(topic_scores.get(topic, 0.0))
        
        # Abstraction level features
        abstraction_data = baseVNI_output.get('abstraction_levels', {})
        if 'cognitive' in abstraction_data:
            cognitive_tensor = abstraction_data['cognitive'].get('tensor', torch.zeros(256))
            features['numerical'].extend(cognitive_tensor[:8].tolist())  # First 8 values
        
        # Intent features
        cognitive_data = abstraction_data.get('cognitive', {})
        intent = cognitive_data.get('intent', 'information')
        intent_mapping = {'question': 0.8, 'problem_solving': 0.9, 'explanation': 0.7, 'information': 0.5}
        features['numerical'].append(intent_mapping.get(intent, 0.5))
        
        # Specialization matching
        specialization_match = 0.0
        for specialization in specializations:
            if specialization in topic_scores:
                specialization_match += topic_scores[specialization]
        features['numerical'].append(specialization_match)
        
        # Pad or truncate to fixed size
        target_size = 64
        if len(features['numerical']) < target_size:
            features['numerical'].extend([0.0] * (target_size - len(features['numerical'])))
        else:
            features['numerical'] = features['numerical'][:target_size]
        
        return features
    
    def calculate_specialization_match(self, baseVNI_output: Dict[str, Any], 
                                     specializations: List[str]) -> float:
        """Calculate how well VNI specializations match the input"""
        topic_scores = baseVNI_output.get('topic_classification', {})
        
        match_score = 0.0
        for specialization in specializations:
            # Map VNI specializations to topics
            specialization_to_topic = {
                'medical': 'medical',
                'legal': 'legal', 
                'technical': 'technical',
                'comparison': 'general',
                'merging': 'general'
            }
            topic = specialization_to_topic.get(specialization)
            if topic in topic_scores:
                match_score += topic_scores[topic]
        
        return min(match_score, 1.0)
    
    def analyze_cross_vni_synergy(self, activation_scores: Dict[str, float]) -> List[Tuple[str, str, float]]:
        """Analyze potential synergies between VNI pairs"""
        synergies = []
        vni_pairs = []
        
        vni_ids = list(activation_scores.keys())
        for i in range(len(vni_ids)):
            for j in range(i + 1, len(vni_ids)):
                vni_pairs.append((vni_ids[i], vni_ids[j]))
        
        for vni1, vni2 in vni_pairs:
            score1 = activation_scores.get(vni1, 0.0)
            score2 = activation_scores.get(vni2, 0.0)
            
            # Simple synergy calculation (can be enhanced)
            synergy_score = (score1 + score2) * 0.6  # Synergy multiplier
            
            # Domain-specific synergy boosts
            if ('medical' in vni1 and 'legal' in vni2) or ('legal' in vni1 and 'medical' in vni2):
                synergy_score *= 1.3  # Medical-legal synergy
            elif ('technical' in vni1 and 'medical' in vni2) or ('medical' in vni1 and 'technical' in vni2):
                synergy_score *= 1.2  # Technical-medical synergy
            
            synergies.append((vni1, vni2, synergy_score))
        
        # Sort by synergy score
        synergies.sort(key=lambda x: x[2], reverse=True)
        return synergies

class SmartActivationRouter(nn.Module):
    """Main smart activation router class"""
    
    def __init__(self, config: RouterConfig = None):
        super().__init__()
        self.config = config or RouterConfig()
        self.vni_registry = VNIRegistry()
        self.activation_scorer = ActivationScorer(self.config)
        
        # Execution pool for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_parallel_vnis
        )
        
        logger.info(f"Smart Activation Router initialized with ID: {self.config.router_id}")
    
    def register_vni(self, vni_id: str, vni_instance: nn.Module, 
                    specializations: List[str], capabilities: Dict[str, Any]):
        """Register a VNI with the router"""
        self.vni_registry.register_vni(vni_id, vni_instance, specializations, capabilities)
    
    def forward(self, baseVNI_output: Dict[str, Any]) -> Dict[str, Any]:
        """Main routing function - determine which VNIs to activate"""
        
        try:
            # Step 1: Calculate activation scores for all VNIs
            activation_scores = self.calculate_activation_scores(baseVNI_output)
            
            # Step 2: Select VNIs to activate based on threshold
            activated_vnis = self.select_vnis_to_activate(activation_scores)
            
            # Step 3: Analyze synergies between selected VNIs
            synergies = self.activation_scorer.analyze_cross_vni_synergy(activation_scores)
            
            # Step 4: Generate execution plan
            execution_plan = self.generate_execution_plan(activated_vnis, synergies, baseVNI_output)
            
            # Step 5: Execute VNIs according to plan
            execution_results = self.execute_vnis(execution_plan)
            
            # Compile results
            results = {
                'activation_plan': {
                    'activated_vnis': activated_vnis,
                    'activation_scores': activation_scores,
                    'synergies_identified': synergies[:5],  # Top 5 synergies
                    'execution_strategy': execution_plan['strategy']
                },
                'execution_results': execution_results,
                'routing_metadata': {
                    'total_vnis_considered': len(activation_scores),
                    'vnis_activated': len(activated_vnis),
                    'average_activation_score': sum(activation_scores.values()) / len(activation_scores) if activation_scores else 0,
                    'cross_domain_activation': self.is_cross_domain_activation(activated_vnis)
                }
            }
            
            # Add router metadata
            results['router_metadata'] = {
                'router_id': self.config.router_id,
                'processing_stages': ['score_calculation', 'vni_selection', 
                                    'synergy_analysis', 'execution_planning', 'parallel_execution'],
                'success': True,
                'timestamp': time.time()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Smart activation routing failed: {str(e)}")
            return self._generate_error_output(str(e))
    
    def create_vni_pathway(self, vnis: List[EnhancedBaseVNI], query: str, 
                          context: Dict) -> List[Dict]:
        """Create optimal processing pathway for VNIs"""
        
        # Analyze query to determine processing flow
        analysis = self._analyze_processing_needs(query, context)
        
        # Determine optimal sequence
        if analysis.get("processing_mode") == "cascade":
            pathway = self._create_cascade_pathway(vnis, analysis)
        elif analysis.get("processing_mode") == "parallel":
            pathway = self._create_parallel_pathway(vnis, analysis)
        else:
            pathway = self._create_hybrid_pathway(vnis, analysis)
        
        return pathway
    
    def _create_cascade_pathway(self, vnis: List[EnhancedBaseVNI], analysis: Dict) -> List[Dict]:
        """Create sequential processing pathway"""
        pathway = []
        
        # Sort VNIs by specialization match
        sorted_vnis = sorted(vnis, 
            key=lambda v: self._compute_specialization_match(v, analysis),
            reverse=True
        )
        
        for i, vni in enumerate(sorted_vnis):
            pathway.append({
                "vni_id": vni.instance_id,
                "role": self._determine_vni_role(vni, i, analysis),
                "expected_output": analysis.get(f"stage_{i}_output", "abstraction"),
                "timeout": analysis.get("stage_timeout", 30)
            })
        
        return pathway
    
    def route_between_vnis(self, source_vni: EnhancedBaseVNI, 
                          target_vnis: List[EnhancedBaseVNI],
                          data: Dict) -> Dict:
        """Route data between VNIs intelligently"""
        
        routing_decisions = {}
        
        for target_vni in target_vnis:
            # Check if routing makes sense
            if self._should_route(source_vni, target_vni, data):
                routing_decisions[target_vni.instance_id] = {
                    "route": True,
                    "priority": self._compute_routing_priority(source_vni, target_vni, data),
                    "data_format": self._determine_data_format(source_vni, target_vni),
                    "expected_processing_time": self._estimate_processing_time(target_vni, data)
                }
            else:
                routing_decisions[target_vni.instance_id] = {
                    "route": False,
                    "reason": "Not compatible for this data type"
                }
        
        return routing_decisions

    def calculate_activation_scores(self, baseVNI_output: Dict[str, Any]) -> Dict[str, float]:
        """Calculate activation scores for all registered VNIs"""
        activation_scores = {}
        
        for vni_id in self.vni_registry.get_all_vnis():
            vni_info = self.vni_registry.get_vni(vni_id)
            if vni_info:
                specializations = vni_info['specializations']
                score = self.activation_scorer.compute_activation_score(
                    baseVNI_output, vni_id, specializations
                )
                activation_scores[vni_id] = score
        
        return activation_scores

    def select_vnis(self, attention_scores: Dict[str, float]) -> List[str]:
        """
        Select VNIs to activate based on attention scores
        Simple implementation for the orchestrator
        """
        if not attention_scores:
            return []
    
        # Apply threshold filtering
        activation_threshold = getattr(self, 'activation_threshold', 0.3)
        above_threshold = {
            vni_id: score for vni_id, score in attention_scores.items() 
            if score >= activation_threshold
        }
    
        # If nothing above threshold, take top 2
        if not above_threshold:
            sorted_scores = sorted(attention_scores.items(), key=lambda x: x[1], reverse=True)
            above_threshold = dict(sorted_scores[:2])
    
        # Ensure domain diversity
        selected_vnis = self._ensure_domain_diversity(above_threshold)
    
        return selected_vnis

    def _ensure_domain_diversity(self, candidate_vnis: Dict[str, float]) -> List[str]:
        """Ensure we get a diverse set of VNIs across different domains"""
        if len(candidate_vnis) <= 2:
            return list(candidate_vnis.keys())
    
        # Group by domain
        domain_groups = {}
        for vni_id in candidate_vnis:
            domain = vni_id.split('_')[0]
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(vni_id)
    
        # Select top from each domain
        selected = []
        for domain, vnis in domain_groups.items():
            # Sort by score and take top 1 per domain
            sorted_vnis = sorted(
                [(vni, candidate_vnis[vni]) for vni in vnis],
                key=lambda x: x[1],
                reverse=True
            )
            selected.extend([vni for vni, score in sorted_vnis[:1]])
    
        # If we have too many, take top overall
        if len(selected) > 3:
            selected_with_scores = [(vni, candidate_vnis[vni]) for vni in selected]
            selected_with_scores.sort(key=lambda x: x[1], reverse=True)
            selected = [vni for vni, score in selected_with_scores[:3]]
    
        return selected

    def select_vnis_to_activate(self, activation_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Select which VNIs to activate based on scores and threshold"""
        activated_vnis = []
        
        for vni_id, score in activation_scores.items():
            if score >= self.config.activation_threshold:
                vni_info = self.vni_registry.get_vni(vni_id)
                if vni_info:
                    activated_vnis.append({
                        'vni_id': vni_id,
                        'activation_score': score,
                        'specializations': vni_info['specializations'],
                        'capabilities': vni_info['capabilities'],
                        'performance_history': self.vni_registry.get_average_performance(vni_id)
                    })
        
        # Sort by activation score (highest first)
        activated_vnis.sort(key=lambda x: x['activation_score'], reverse=True)
        
        # Limit to max parallel VNIs
        if len(activated_vnis) > self.config.max_parallel_vnis:
            activated_vnis = activated_vnis[:self.config.max_parallel_vnis]
        
        return activated_vnis
    
    def generate_execution_plan(self, activated_vnis: List[Dict[str, Any]], 
                              synergies: List[Tuple[str, str, float]],
                              baseVNI_output: Dict[str, Any]) -> Dict[str, Any]:
        """Generate execution plan for activated VNIs"""
        
        execution_plan = {
            'vnis_to_execute': [],
            'execution_order': [],
            'data_flow': {},
            'strategy': self.determine_execution_strategy(activated_vnis, synergies),
            'timeout': self.config.timeout_seconds
        }
        
        # Prepare execution details for each VNI
        for vni_info in activated_vnis:
            vni_id = vni_info['vni_id']
            
            # Determine input data for this VNI
            input_data = self.prepare_vni_input(vni_id, baseVNI_output, activated_vnis)
            
            execution_plan['vnis_to_execute'].append({
                'vni_id': vni_id,
                'input_data': input_data,
                'expected_output_type': vni_info['capabilities'].get('output_types', ['general'])[0],
                'priority': vni_info['activation_score'],
                'dependencies': self.identify_dependencies(vni_id, activated_vnis, synergies)
            })
        
        # Determine execution order based on dependencies and priorities
        execution_plan['execution_order'] = self.determine_execution_order(
            execution_plan['vnis_to_execute']
        )
        
        return execution_plan
    
    def determine_execution_strategy(self, activated_vnis: List[Dict[str, Any]], 
                                   synergies: List[Tuple[str, str, float]]) -> str:
        """Determine the best execution strategy"""
        
        if len(activated_vnis) == 1:
            return 'direct'
        
        # Check for strong synergies
        strong_synergies = [s for s in synergies if s[2] > 0.7]
        if strong_synergies:
            return 'collaborative_parallel'
        
        # Check for cross-domain activation
        if self.is_cross_domain_activation(activated_vnis):
            return 'hybrid_parallel'
        
        return 'priority_parallel'
    
    def is_cross_domain_activation(self, activated_vnis: List[Dict[str, Any]]) -> bool:
        """Check if this is a cross-domain activation"""
        domains = set()
        for vni in activated_vnis:
            for specialization in vni['specializations']:
                if specialization in ['medical', 'legal', 'technical']:
                    domains.add(specialization)
        
        return len(domains) >= 2
    
    def prepare_vni_input(self, vni_id: str, baseVNI_output: Dict[str, Any],
                         activated_vnis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare input data for a specific VNI"""
        
        input_data = {
            'abstraction_data': baseVNI_output.get('abstraction_levels', {}),
            'topic_data': baseVNI_output.get('topic_classification', {}),
            'metadata': {
                'source_topics': list(baseVNI_output.get('topic_classification', {}).keys()),
                'activated_vnis': [vni['vni_id'] for vni in activated_vnis],
                'timestamp': time.time()
            }
        }
        
        # Add VNI-specific context
        if 'transVNI' in vni_id:
            input_data['processing_type'] = 'comparison_segregation'
        elif 'operAction' in vni_id:
            input_data['processing_type'] = 'reasoning_operation'
        
        return input_data
    
    def identify_dependencies(self, vni_id: str, activated_vnis: List[Dict[str, Any]],
                            synergies: List[Tuple[str, str, float]]) -> List[str]:
        """Identify VNI dependencies based on synergies"""
        dependencies = []
        
        for vni1, vni2, score in synergies:
            if vni1 == vni_id and score > 0.6:
                dependencies.append(vni2)
            elif vni2 == vni_id and score > 0.6:
                dependencies.append(vni1)
        
        return dependencies
    
    def determine_execution_order(self, vnis_to_execute: List[Dict[str, Any]]) -> List[str]:
        """Determine execution order considering dependencies"""
        # Simple priority-based ordering for demo
        # In production, this would use topological sorting for dependencies
        ordered_vnis = sorted(vnis_to_execute, key=lambda x: x['priority'], reverse=True)
        return [vni['vni_id'] for vni in ordered_vnis]
    
    def execute_vnis(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute VNIs according to the execution plan"""
        
        execution_results = {}
        vnis_to_execute = execution_plan['vnis_to_execute']
        
        # Execute VNIs based on strategy
        if execution_plan['strategy'] in ['direct', 'priority_parallel']:
            # Parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(vnis_to_execute)) as executor:
                future_to_vni = {
                    executor.submit(self.execute_single_vni, vni_info): vni_info['vni_id']
                    for vni_info in vnis_to_execute
                }
                
                for future in concurrent.futures.as_completed(future_to_vni, timeout=self.config.timeout_seconds):
                    vni_id = future_to_vni[future]
                    try:
                        result = future.result()
                        execution_results[vni_id] = result
                        # Update performance metrics
                        if result.get('vni_metadata', {}).get('success', False):
                            confidence = result.get('confidence_score', 0.5)
                            self.vni_registry.update_performance(vni_id, confidence)
                    except Exception as e:
                        logger.error(f"VNI {vni_id} execution failed: {str(e)}")
                        execution_results[vni_id] = self._generate_vni_error_result(vni_id, str(e))
        
        elif execution_plan['strategy'] == 'collaborative_parallel':
            # Enhanced parallel execution with collaboration awareness
            execution_results = self.execute_collaborative_parallel(vnis_to_execute)
        
        else:
            # Fallback sequential execution
            for vni_info in vnis_to_execute:
                vni_id = vni_info['vni_id']
                try:
                    result = self.execute_single_vni(vni_info)
                    execution_results[vni_id] = result
                    if result.get('vni_metadata', {}).get('success', False):
                        confidence = result.get('confidence_score', 0.5)
                        self.vni_registry.update_performance(vni_id, confidence)
                except Exception as e:
                    logger.error(f"VNI {vni_id} execution failed: {str(e)}")
                    execution_results[vni_id] = self._generate_vni_error_result(vni_id, str(e))
        
        return execution_results
    
    def execute_single_vni(self, vni_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single VNI"""
        vni_id = vni_info['vni_id']
        input_data = vni_info['input_data']
        
        vni_data = self.vni_registry.get_vni(vni_id)
        if not vni_data:
            raise ValueError(f"VNI {vni_id} not found in registry")
        
        vni_instance = vni_data['instance']
        
        # Execute VNI
        with torch.no_grad():
            result = vni_instance(input_data)
        
        return result
    
    def execute_collaborative_parallel(self, vnis_to_execute: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute VNIs in parallel with collaboration support"""
        # Simplified collaborative execution for demo
        # In production, this would include real-time data sharing between VNIs
        return self.execute_vnis({
            'vnis_to_execute': vnis_to_execute,
            'strategy': 'priority_parallel',
            'timeout': self.config.timeout_seconds
        })
    
    def _generate_error_output(self, error_msg: str) -> Dict[str, Any]:
        """Generate error output"""
        return {
            'activation_plan': {'activated_vnis': []},
            'execution_results': {},
            'router_metadata': {
                'router_id': self.config.router_id,
                'success': False,
                'error': error_msg,
                'timestamp': time.time()
            }
        }
    
    def _generate_vni_error_result(self, vni_id: str, error_msg: str) -> Dict[str, Any]:
        """Generate error result for VNI execution"""
        return {
            'vni_metadata': {
                'vni_id': vni_id,
                'success': False,
                'error': error_msg
            }
        }
    
    def get_router_status(self) -> Dict[str, Any]:
        """Get current router status and statistics"""
        total_vnis = len(self.vni_registry.get_all_vnis())
        performance_stats = {}
        
        for vni_id in self.vni_registry.get_all_vnis():
            performance_stats[vni_id] = self.vni_registry.get_average_performance(vni_id)
        
        return {
            'router_id': self.config.router_id,
            'total_registered_vnis': total_vnis,
            'vni_performance_stats': performance_stats,
            'activation_threshold': self.config.activation_threshold,
            'max_parallel_vnis': self.config.max_parallel_vnis
        }

# Demonstration and testing
def test_smart_activation_router():
    """Test the smart activation router with sample VNIs"""
    
    # Initialize router
    config = RouterConfig(router_id="smart_router_test_001")
    router = SmartActivationRouter(config)
    
    # Create mock VNIs for testing
    class MockVNI(nn.Module):
        def __init__(self, vni_id, domain):
            super().__init__()
            self.vni_id = vni_id
            self.domain = domain
        
        def forward(self, inputs):
            return {
                f"{self.domain}_analysis": {"result": f"Mock analysis from {self.vni_id}"},
                'confidence_score': 0.8,
                'vni_metadata': {
                    'vni_id': self.vni_id,
                    'success': True,
                    'domain': self.domain
                }
            }
        
        def get_capabilities(self):
            return {
                'input_types': [f'{self.domain}_data'],
                'output_types': [f'{self.domain}_analysis']
            }
    
    # Register mock VNIs
    mock_vnis = [
        ('operAction_medical', ['medical'], MockVNI('operAction_medical', 'medical')),
        ('operAction_legal', ['legal'], MockVNI('operAction_legal', 'legal')),
        ('operAction_technical', ['technical'], MockVNI('operAction_technical', 'technical')),
        ('transVNI_compare_segregate', ['comparison'], MockVNI('transVNI_compare_segregate', 'comparison'))
    ]
    
    for vni_id, specializations, vni_instance in mock_vnis:
        router.register_vni(
            vni_id, 
            vni_instance, 
            specializations, 
            vni_instance.get_capabilities()
        )
    
    # Create test baseVNI output
    test_baseVNI_output = {
        'abstraction_levels': {
            'cognitive': {
                'tensor': torch.randn(256),
                'concepts': ['patient', 'contract', 'software', 'compliance'],
                'intent': 'problem_solving'
            },
            'structural': {
                'tensor': torch.randn(256)
            }
        },
        'topic_classification': {
            'medical': 0.7,
            'legal': 0.6,
            'technical': 0.8
        },
        'primary_topic': 'technical'
    }
    
    print("=== Smart Activation Router Demo Test ===\n")
    
    with torch.no_grad():
        results = router(test_baseVNI_output)
    
    # Display results
    activation_plan = results['activation_plan']
    print(f"Execution Strategy: {activation_plan['execution_strategy']}")
    print(f"VNIs Activated: {len(activation_plan['activated_vnis'])}")
    
    print("\nActivation Scores:")
    for vni_id, score in activation_plan['activation_scores'].items():
        print(f"  {vni_id}: {score:.3f}")
    
    print("\nExecution Results:")
    for vni_id, result in results['execution_results'].items():
        success = result.get('vni_metadata', {}).get('success', False)
        print(f"  {vni_id}: {'SUCCESS' if success else 'FAILED'}")
    
    return router

if __name__ == "__main__":
    # Run demonstration
    test_smart_activation_router() 
