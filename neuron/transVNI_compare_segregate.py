# transVNI_compare_segregate.py
import time
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from enhanced_vni_classes.managers.vni_manager import VNIManager, EnhancedBaseVNI

logger = logging.getLogger("transVNI_compare_segregate")

import networkx as nx  # For potential future knowledge graphs

class RoutingMemory:
    def __init__(self):
        self.successful_routes = {}  # pattern -> successful VNI combinations
        self.performance_history = defaultdict(list)
        self.domain_patterns = self.initialize_domain_patterns()
    
    def initialize_domain_patterns(self):
        return {
            'medical_keywords': ['symptom', 'diagnosis', 'treatment', 'patient', 'disease', 'medical', 'health', 'hospital'],
            'legal_keywords': ['contract', 'liability', 'compliance', 'regulation', 'law', 'legal', 'court', 'agreement'],
            'technical_keywords': ['code', 'algorithm', 'system', 'implementation', 'debug', 'technical', 'software', 'programming']
        }
    
    def hash_pattern(self, input_data):
        """Create a simple hashable representation of input pattern"""
        if isinstance(input_data, dict):
            # Use abstraction level signatures
            signature_parts = []
            for level, data in input_data.items():
                if hasattr(data, 'shape'):
                    signature_parts.append(f"{level}_{data.shape}_{data.mean().item():.3f}")
                else:
                    signature_parts.append(f"{level}_{str(data)[:50]}")
            return tuple(signature_parts)
        else:
            return str(input_data)[:100]  # Truncate for hashability
    
    def extract_pattern(self, input_data):
        """Extract key features for pattern matching"""
        pattern = {}
        
        if isinstance(input_data, dict):
            for level, data in input_data.items():
                if hasattr(data, 'shape') and torch.is_tensor(data):
                    pattern[f'{level}_mean'] = data.mean().item()
                    pattern[f'{level}_std'] = data.std().item()
                    pattern[f'{level}_non_zero'] = (data != 0).float().mean().item()
        
        return pattern
    
    def pattern_similarity(self, pattern1, pattern2):
        """Calculate similarity between two patterns"""
        if not pattern1 or not pattern2:
            return 0.0
        
        common_keys = set(pattern1.keys()) & set(pattern2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = pattern1[key], pattern2[key]
            # Normalized difference for numerical values
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                diff = abs(val1 - val2)
                max_val = max(abs(val1), abs(val2), 1e-6)  # Avoid division by zero
                similarity = 1.0 - (diff / max_val)
                similarities.append(max(0.0, similarity))
        
        return np.mean(similarities) if similarities else 0.0
    
    def record_successful_route(self, input_pattern, activated_vnis, success_score):
        """Remember which VNI combinations worked for which inputs"""
        pattern_key = self.hash_pattern(input_pattern)
        
        if pattern_key in self.successful_routes:
            # Update existing entry
            existing = self.successful_routes[pattern_key]
            # Weighted average of scores
            total_count = existing['count'] + 1
            existing['score'] = (existing['score'] * existing['count'] + success_score) / total_count
            existing['count'] = total_count
            existing['timestamp'] = time.time()
        else:
            # New entry
            self.successful_routes[pattern_key] = {
                'vnis': activated_vnis,
                'score': success_score,
                'timestamp': time.time(),
                'count': 1
            }
    
    def find_similar_success(self, current_input, similarity_threshold=0.7):
        """Find historically successful routes for similar inputs"""
        current_pattern = self.extract_pattern(current_input)
        similarities = []
        
        for pattern_key, route_data in self.successful_routes.items():
            # Reconstruct the original pattern for comparison
            original_pattern = self.pattern_from_hash(pattern_key)
            similarity = self.pattern_similarity(current_pattern, original_pattern)
            
            if similarity > similarity_threshold:
                similarities.append({
                    'similarity': similarity,
                    'route_data': route_data,
                    'pattern_key': pattern_key
                })
        
        # Return most similar successful routes
        return sorted(similarities, key=lambda x: x['similarity'], reverse=True)[:5]
    
    def pattern_from_hash(self, pattern_key):
        """Reconstruct pattern from hash (simplified)"""
        # This is a simplified reconstruction - you might want to store the full pattern
        if isinstance(pattern_key, tuple):
            # Reconstruct from tuple signature
            pattern = {}
            for part in pattern_key:
                if '_mean' in part:
                    key = part.split('_mean')[0]
                    pattern[f'{key}_mean'] = float(part.split('_')[-1])
            return pattern
        else:
            return {'raw_pattern': pattern_key}
    
    def cleanup_old_entries(self, max_age_hours=24):
        """Remove old routing entries to prevent memory bloat"""
        current_time = time.time()
        old_keys = []
        
        for pattern_key, route_data in self.successful_routes.items():
            age_hours = (current_time - route_data['timestamp']) / 3600
            if age_hours > max_age_hours:
                old_keys.append(pattern_key)
        
        for key in old_keys:
            del self.successful_routes[key]
        
        return len(old_keys)

@dataclass
class TransVNIConfig:
    """Configuration for transVNI compare and segregate"""
    vni_id: str = "transVNI_compare_segregate"
    routing_threshold: float = 0.3
    max_specialists: int = 3
    cross_domain_threshold: float = 0.4
    similarity_threshold: float = 0.7
    
    # Specialist VNI mappings
    specialist_mappings: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.specialist_mappings is None:
            self.specialist_mappings = {
            'medical': ['med_001'],  # EnhancedMedicalVNI instance
            'legal': ['legal_001'],   # EnhancedLegalVNI instance  
            'technical': ['gen_001'], # EnhancedGeneralVNI instance
            'cross_domain': ['med_001', 'legal_001', 'gen_001']
            }

class ComparisonEngine(nn.Module):
    """Core comparison and relationship detection engine"""
    
    def __init__(self, config: TransVNIConfig):
        super().__init__()
        self.config = config
        
        # Similarity computation layers
        self.semantic_similarity = nn.CosineSimilarity(dim=1)
        
        # Relationship detection network
        self.relationship_detector = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5),  # 5 relationship types
            nn.Softmax(dim=-1)
        )
        
        # Cross-domain analysis
        self.domain_integrator = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.Tanh()
        )

        # ==================== NEW METHOD ====================
    def extract_topic_classification(self, baseVNI_output: Dict[str, Any]) -> Dict[str, float]:
        """Extract topic classification from enhanced VNI response"""
        # Look for domain hints in the response
        response_type = baseVNI_output.get('response_type', '')
        vni_instance = baseVNI_output.get('vni_instance', '')
        
        # Infer topics from VNI instance ID
        scores = {'medical': 0.0, 'legal': 0.0, 'technical': 0.0}
        
        if 'med' in vni_instance:
            scores['medical'] = baseVNI_output.get('confidence', 0.5)
        elif 'legal' in vni_instance:
            scores['legal'] = baseVNI_output.get('confidence', 0.5)
        elif 'gen' in vni_instance:
            scores['technical'] = baseVNI_output.get('confidence', 0.5)
        
        return scores
    # ====================================================

    def extract_comparison_features(self, baseVNI_output: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Extract features for comparison from baseVNI output"""
        features = {}
        
        # Extract from each abstraction level
        abstraction_data = baseVNI_output.get('abstraction_levels', {})
        
        # semantic level features (concepts, meaning)
        if 'semantic' in abstraction_data:
            semantic = abstraction_data['semantic']
            features['concepts'] = semantic.get('tensor', torch.zeros(256))
            features['intent_embedding'] = self.encode_intent(semantic.get('intent', 'information'))
        
        # Structural level features (relationships, patterns)
        if 'structural' in abstraction_data:
            structural = abstraction_data['structural']
            features['relationships'] = structural.get('tensor', torch.zeros(256))
            features['complexity'] = self.encode_complexity(structural.get('logical_flow', 'descriptive'))
        
        # Topic features
        topic_scores = baseVNI_output.get('topic_classification', {})
        features['topic_vector'] = self.encode_topics(topic_scores)
        
        return features
    
    def encode_intent(self, intent: str) -> torch.Tensor:
        """Encode intent into embedding"""
        intent_mapping = {
            'question': torch.tensor([1.0, 0.0, 0.0, 0.0]),
            'problem_solving': torch.tensor([0.0, 1.0, 0.0, 0.0]),
            'explanation': torch.tensor([0.0, 0.0, 1.0, 0.0]),
            'information': torch.tensor([0.0, 0.0, 0.0, 1.0])
        }
        return intent_mapping.get(intent, torch.tensor([0.25, 0.25, 0.25, 0.25]))
    
    def encode_complexity(self, complexity: str) -> torch.Tensor:
        """Encode complexity into embedding"""
        complexity_mapping = {
            'conditional': torch.tensor([1.0, 0.0, 0.0]),
            'conclusive': torch.tensor([0.0, 1.0, 0.0]),
            'contrastive': torch.tensor([0.0, 0.0, 1.0]),
            'descriptive': torch.tensor([0.5, 0.5, 0.0])
        }
        return complexity_mapping.get(complexity, torch.tensor([0.33, 0.33, 0.33]))
    
    def encode_topics(self, topic_scores: Dict[str, float]) -> torch.Tensor:
        """Encode topic scores into vector"""
        topics = ['medical', 'legal', 'technical']
        vector = torch.zeros(len(topics))
        
        for i, topic in enumerate(topics):
            vector[i] = topic_scores.get(topic, 0.0)
            
        return vector
    
    def compare_elements(self, features: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compare different elements to find patterns and relationships"""
        
        comparisons = {}
        
        # Compare topic strengths
        topic_vector = features.get('topic_vector', torch.zeros(3))
        comparisons['dominant_topic'] = self.identify_dominant_topic(topic_vector)
        comparisons['topic_balance'] = self.assess_topic_balance(topic_vector)
        
        # Analyze relationships between concepts and structure
        concepts = features.get('concepts', torch.zeros(256))
        relationships = features.get('relationships', torch.zeros(256))
        
        if concepts.numel() > 0 and relationships.numel() > 0:
            structural_similarity = F.cosine_similarity(concepts.unsqueeze(0), 
                                                       relationships.unsqueeze(0)).item()
            comparisons['concept_structure_alignment'] = structural_similarity
        
        # Detect cross-domain patterns
        comparisons['cross_domain_indicator'] = self.detect_cross_domain(topic_vector)
        
        return comparisons
    
    def identify_dominant_topic(self, topic_vector: torch.Tensor) -> str:
        """Identify the dominant topic"""
        topics = ['medical', 'legal', 'technical']
        max_idx = torch.argmax(topic_vector).item()
        return topics[max_idx] if max_idx < len(topics) else 'general'
    
    def assess_topic_balance(self, topic_vector: torch.Tensor) -> str:
        """Assess how balanced the topics are"""
        entropy = -torch.sum(topic_vector * torch.log(topic_vector + 1e-8)).item()
        
        if entropy > 1.0:
            return 'balanced'
        elif entropy > 0.5:
            return 'mixed'
        else:
            return 'focused'
    
    def detect_cross_domain(self, topic_vector: torch.Tensor) -> bool:
        """Detect if this is a cross-domain problem"""
        # Count topics above threshold
        active_topics = sum(score > self.config.cross_domain_threshold for score in topic_vector)
        return active_topics >= 2
    
    def segregate_data(self, baseVNI_output: Dict[str, Any], 
                      comparisons: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Segregate data into appropriate categories for routing"""
        
        segregated = defaultdict(dict)
        topic_scores = baseVNI_output.get('topic_classification', {})
        
        # Extract abstraction data
        abstraction_data = baseVNI_output.get('abstraction_levels', {})
        
        # Route based on topics and comparisons
        for topic, score in topic_scores.items():
            if score >= self.config.routing_threshold:
                # Prepare data for this topic specialist
                segregated[topic] = {
                    'abstraction_data': self.extract_topic_relevant_data(abstraction_data, topic),
                    'confidence': score,
                    'routing_priority': score,
                    'metadata': {
                        'source_topics': [topic],
                        'cross_domain': comparisons.get('cross_domain_indicator', False)
                    }
                }
        
        # Handle cross-domain cases
        if comparisons.get('cross_domain_indicator', False):
            cross_domain_data = self.prepare_cross_domain_data(abstraction_data, topic_scores)
            segregated['cross_domain'] = {
                'abstraction_data': cross_domain_data,
                'confidence': max(topic_scores.values()) if topic_scores else 0.0,
                'routing_priority': 1.0,  # High priority for complex cases
                'metadata': {
                    'source_topics': [t for t, s in topic_scores.items() if s > self.config.routing_threshold],
                    'cross_domain': True
                }
            }
        
        return dict(segregated)
    
    def extract_topic_relevant_data(self, abstraction_data: Dict[str, Any], topic: str) -> Dict[str, Any]:
        """Extract data relevant to specific topic"""
        relevant_data = {}
        
        # semantic level - focus on topic-relevant concepts
        if 'semantic' in abstraction_data:
            semantic = abstraction_data['semantic'].copy()
            # Enhance with topic context
            semantic['topic_context'] = topic
            relevant_data['semantic'] = semantic
        
        # Structural level - maintain relationships
        if 'structural' in abstraction_data:
            relevant_data['structural'] = abstraction_data['structural']
        
        # Signal level - keep embeddings and patterns
        if 'signal' in abstraction_data:
            relevant_data['signal'] = abstraction_data['signal']
        
        return relevant_data
    
    def prepare_cross_domain_data(self, abstraction_data: Dict[str, Any], 
                                 topic_scores: Dict[str, float]) -> Dict[str, Any]:
        """Prepare data for cross-domain processing"""
        cross_domain_data = {}
        
        # Integrate multiple topic perspectives
        active_topics = [t for t, s in topic_scores.items() if s > self.config.routing_threshold]
        
        # Enhanced semantic level with multi-topic context
        if 'semantic' in abstraction_data:
            semantic = abstraction_data['semantic'].copy()
            semantic['multi_topic_context'] = active_topics
            semantic['topic_interactions'] = self.analyze_topic_interactions(topic_scores)
            cross_domain_data['semantic'] = semantic
        
        # Enhanced structural level with cross-domain relationships
        if 'structural' in abstraction_data:
            structural = abstraction_data['structural'].copy()
            structural['cross_domain_relationships'] = self.identify_cross_domain_relationships(active_topics)
            cross_domain_data['structural'] = structural
        
        # Include all signal data
        if 'signal' in abstraction_data:
            cross_domain_data['signal'] = abstraction_data['signal']
        
        return cross_domain_data
    
    def analyze_topic_interactions(self, topic_scores: Dict[str, float]) -> List[str]:
        """Analyze how topics might interact"""
        interactions = []
        active_topics = [t for t, s in topic_scores.items() if s > self.config.routing_threshold]
        
        if 'medical' in active_topics and 'legal' in active_topics:
            interactions.append('medical_legal_interface')
        if 'medical' in active_topics and 'technical' in active_topics:
            interactions.append('medical_technical_integration')
        if 'legal' in active_topics and 'technical' in active_topics:
            interactions.append('legal_technical_compliance')
            
        return interactions
    
    def identify_cross_domain_relationships(self, active_topics: List[str]) -> List[str]:
        """Identify potential cross-domain relationships"""
        relationships = []
        
        if len(active_topics) >= 2:
            relationships.append('interdisciplinary')
            relationships.append('complex_integration')
            
        if len(active_topics) >= 3:
            relationships.append('multi_domain_synthesis')
            
        return relationships
    
    def generate_routing_plan(self, segregated_data: Dict[str, Dict[str, Any]], 
                            comparisons: Dict[str, Any]) -> Dict[str, Any]:
        """Generate routing plan to specialist VNIs"""
        
        routing_plan = {
            'activations': [],
            'data_distribution': {},
            'routing_strategy': 'parallel',  # or 'sequential', 'conditional'
            'priority_order': []
        }
        
        # Sort by priority
        sorted_categories = sorted(segregated_data.items(), 
                                 key=lambda x: x[1]['routing_priority'], 
                                 reverse=True)
        
        for category, data in sorted_categories:
            specialists = self.config.specialist_mappings.get(category, [])
            
            for specialist in specialists:
                activation = {
                    'vni_id': specialist,
                    'category': category,
                    'confidence': data['confidence'],
                    'input_data': data['abstraction_data'],
                    'metadata': data['metadata']
                }
                routing_plan['activations'].append(activation)
                
                # Track data distribution
                if specialist not in routing_plan['data_distribution']:
                    routing_plan['data_distribution'][specialist] = []
                routing_plan['data_distribution'][specialist].append(category)
        
        # Set priority order
        routing_plan['priority_order'] = [act['vni_id'] for act in routing_plan['activations']]
        
        # Determine routing strategy
        if comparisons.get('cross_domain_indicator', False):
            routing_plan['routing_strategy'] = 'collaborative_parallel'
        elif len(routing_plan['activations']) == 1:
            routing_plan['routing_strategy'] = 'direct'
        else:
            routing_plan['routing_strategy'] = 'priority_parallel'
        
        return routing_plan

class PerformanceTracker:
    def __init__(self):
        self.routing_decisions = []
        self.outcome_scores = []
        self.learning_rate = 0.1
        self.confidence_threshold = 0.6  # Starting threshold
    
    def record_decision(self, activation_plan, input_features, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
            
        self.routing_decisions.append({
            'plan': activation_plan,
            'features': input_features,
            'timestamp': timestamp
        })
        return len(self.routing_decisions) - 1  # Return index for future reference
    
    def update_with_outcome(self, decision_index, outcome_quality, human_feedback=None):
        if 0 <= decision_index < len(self.routing_decisions):
            self.outcome_scores.append((decision_index, outcome_quality))
            
            # Simple learning: adjust confidence threshold based on performance
            if outcome_quality < 0.4:
                # Too many poor outcomes - be more conservative
                self.adjust_confidence_threshold(0.02)  # Increase threshold
            elif outcome_quality > 0.8:
                # Good outcomes - can be more aggressive
                self.adjust_confidence_threshold(-0.01)  # Decrease threshold
    
    def adjust_confidence_threshold(self, adjustment):
        """Adjust the confidence threshold for routing decisions"""
        new_threshold = self.confidence_threshold + adjustment
        # Keep threshold in reasonable bounds
        self.confidence_threshold = max(0.3, min(0.9, new_threshold))
    
    def get_performance_stats(self):
        if not self.outcome_scores:
            return {'average_score': 0, 'total_cases': 0, 'confidence_threshold': self.confidence_threshold}
        
        scores = [score for _, score in self.outcome_scores]
        return {
            'average_score': np.mean(scores),
            'total_cases': len(scores),
            'recent_trend': np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores),
            'confidence_threshold': self.confidence_threshold
        }
    
class TransVNICompareSegregate(nn.Module):
    """Main transVNI compare and segregate class"""
    def __init__(self, config: TransVNIConfig = None, vni_manager: VNIManager = None):
        super().__init__()
        self.config = config or TransVNIConfig()
        self.comparison_engine = ComparisonEngine(self.config)
        self.vni_manager = vni_manager  # Add VNI manager reference

        # Initialize the new intelligent components
        self.routing_memory = RoutingMemory()
        self.performance_tracker = PerformanceTracker()
        
        # Add learnable domain preferences
        self.domain_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))  # medical, legal, technical
        
        print("Enhanced TransVNI with routing memory and learning capabilities")

        logger.info(f"transVNI Compare-Segregate initialized with ID: {self.config.vni_id}")
    
    def forward(self, baseVNI_output: Dict[str, Any]) -> Dict[str, Any]:
        """Process baseVNI output through comparison and segregation"""
        
        try:
            # Step 1: Extract features for comparison
            features = self.comparison_engine.extract_comparison_features(baseVNI_output)
            
            # Step 2: Compare elements and find relationships
            comparisons = self.comparison_engine.compare_elements(features)
            
            # Step 3: Segregate data into categories
            segregated_data = self.comparison_engine.segregate_data(baseVNI_output, comparisons)
            
            # Step 4: Generate routing plan
            routing_plan = self.comparison_engine.generate_routing_plan(segregated_data, comparisons)
            
            # Compile final output
            results = {
                'routing_plan': routing_plan,
                'segregated_data': segregated_data,
                'comparison_analysis': comparisons,
                'processing_metadata': {
                    'input_topics': baseVNI_output.get('topic_classification', {}),
                    'dominant_topic': comparisons.get('dominant_topic', 'unknown'),
                    'cross_domain_detected': comparisons.get('cross_domain_indicator', False)
                }
            }
            
            # Add VNI metadata
            results['vni_metadata'] = {
                'vni_id': self.config.vni_id,
                'vni_type': 'transVNI_compare_segregate',
                'processing_stages': ['feature_extraction', 'comparison_analysis', 'data_segregation', 'routing_generation'],
                'success': True,
                'activations_generated': len(routing_plan['activations'])
            }
            
            return results
            
        except Exception as e:
            logger.error(f"transVNI processing failed: {str(e)}")
            return self._generate_error_output(str(e))
    
    def _generate_error_output(self, error_msg: str) -> Dict[str, Any]:
        """Generate error output"""
        return {
            'routing_plan': {'activations': []},
            'segregated_data': {},
            'vni_metadata': {
                'vni_id': self.config.vni_id,
                'vni_type': 'transVNI_compare_segregate',
                'success': False,
                'error': error_msg
            }
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return VNI capabilities description"""
        return {
            'vni_type': 'transVNI_compare_segregate',
            'description': 'Data comparison, segregation, and routing VNI',
            'capabilities': [
                'Multi-level feature comparison',
                'Topic-based data segregation',
                'Cross-domain relationship detection',
                'Intelligent routing plan generation',
                'Specialist VNI activation'
            ],
            'input_types': ['baseVNI_output'],
            'output_types': ['routing_plan', 'segregated_data', 'comparison_analysis'],
            'specialist_mappings': self.config.specialist_mappings
        }

# Demonstration and testing
def test_transVNI_demo():
    """Test the transVNI with sample baseVNI outputs"""
    
    # Initialize transVNI
    config = TransVNIConfig(vni_id="transVNI_test_001")
    trans_vni = TransVNICompareSegregate(config)
    
    # Create mock baseVNI outputs for testing
    test_cases = [
        {
            'name': 'Medical Case',
            'baseVNI_output': {
                'abstraction_levels': {
                    'semantic': {
                        'tensor': torch.randn(256),
                        'intent': 'question',
                        'concepts': ['patient', 'diagnosis', 'treatment']
                    },
                    'structural': {
                        'tensor': torch.randn(256),
                        'logical_flow': 'conditional'
                    },
                    'signal': {
                        'tensor': torch.randn(256)
                    }
                },
                'topic_classification': {
                    'medical': 0.8,
                    'legal': 0.1,
                    'technical': 0.1
                },
                'primary_topic': 'medical'
            }
        },
        {
            'name': 'Cross-Domain Case',
            'baseVNI_output': {
                'abstraction_levels': {
                    'semantic': {
                        'tensor': torch.randn(256),
                        'intent': 'problem_solving',
                        'concepts': ['software', 'patient', 'compliance']
                    },
                    'structural': {
                        'tensor': torch.randn(256),
                        'logical_flow': 'complex'
                    },
                    'signal': {
                        'tensor': torch.randn(256)
                    }
                },
                'topic_classification': {
                    'medical': 0.6,
                    'legal': 0.5,
                    'technical': 0.7
                },
                'primary_topic': 'technical'
            }
        }
    ]
    
    print("=== transVNI Compare-Segregate Demo Test ===\n")
    
    for i, test_case in enumerate(test_cases):
        print(f"Test Case {i+1}: {test_case['name']}")
        
        with torch.no_grad():
            results = trans_vni(test_case['baseVNI_output'])
        
        # Display results
        routing_plan = results['routing_plan']
        print(f"Routing Strategy: {routing_plan['routing_strategy']}")
        print("Activations:")
        for activation in routing_plan['activations']:
            print(f"  - {activation['vni_id']} for {activation['category']} (conf: {activation['confidence']:.3f})")
        
        print(f"Cross-domain detected: {results['processing_metadata']['cross_domain_detected']}")
        print(f"Dominant topic: {results['processing_metadata']['dominant_topic']}")
        print("-" * 50)
    
    return trans_vni

if __name__ == "__main__":
    # Run demonstration
    test_transVNI_demo()
    
# Example Output
""" {
  'routing_plan': {
    'activations': [
      {
        'vni_id': 'operAction_medical',
        'category': 'medical', 
        'confidence': 0.8,
        'input_data': { ... medical-relevant abstractions ... }
      },
      {
        'vni_id': 'operAction_legal',
        'category': 'legal',
        'confidence': 0.6, 
        'input_data': { ... legal-relevant abstractions ... }
      }
    ],
    'routing_strategy': 'collaborative_parallel'
  }
}"""
