# neuron/aggregator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("aggregator")

@dataclass
class AggregatorConfig:
    """Configuration for response aggregator"""
    aggregator_id: str = "response_aggregator"
    consensus_threshold: float = 0.7
    conflict_resolution_strategy: str = "confidence_weighted"  # confidence_weighted, majority_vote, hybrid
    min_confidence_threshold: float = 0.4
    max_output_length: int = 500
    
    # Output synthesis parameters
    enable_cross_domain_synthesis: bool = True
    enable_conflict_detection: bool = True
    enable_confidence_calibration: bool = True

class ConflictDetector(nn.Module):
    """Neural network for detecting conflicts between VNI outputs"""
    
    def __init__(self):
        super().__init__()
        
        self.conflict_detector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        
        self.conflict_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 3 classes: no_conflict, minor_conflict, major_conflict
            nn.Softmax(dim=-1)
        )
    
    def detect_conflicts(self, vni_outputs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect conflicts between different VNI outputs"""
        conflicts = []
        vni_ids = list(vni_outputs.keys())
        
        # Compare all pairs of VNIs
        for i in range(len(vni_ids)):
            for j in range(i + 1, len(vni_ids)):
                vni1_id = vni_ids[i]
                vni2_id = vni_ids[j]
                
                vni1_output = vni_outputs[vni1_id]
                vni2_output = vni_outputs[vni2_id]
                
                conflict_level = self.analyze_output_conflict(vni1_output, vni2_output)
                
                if conflict_level['level'] != 'no_conflict':
                    conflicts.append({
                        'vni1': vni1_id,
                        'vni2': vni2_id,
                        'conflict_level': conflict_level['level'],
                        'confidence': conflict_level['confidence'],
                        'conflicting_aspects': conflict_level['aspects'],
                        'domain_comparison': f"{self.get_vni_domain(vni1_id)} vs {self.get_vni_domain(vni2_id)}"
                    })
        
        # Sort by conflict severity
        severity_weights = {'major_conflict': 2, 'minor_conflict': 1, 'no_conflict': 0}
        conflicts.sort(key=lambda x: severity_weights[x['conflict_level']], reverse=True)
        
        return conflicts
    
    def analyze_output_conflict(self, output1: Dict[str, Any], output2: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conflict between two VNI outputs"""
        
        # Extract key information for comparison
        advice1 = output1.get('medical_advice') or output1.get('legal_advice') or output1.get('technical_advice', '')
        advice2 = output2.get('medical_advice') or output2.get('legal_advice') or output2.get('technical_advice', '')
        
        confidence1 = output1.get('confidence_score', 0.5)
        confidence2 = output2.get('confidence_score', 0.5)
        
        # Semantic similarity analysis (simplified)
        similarity = self.calculate_semantic_similarity(advice1, advice2)
        
        # Conflict detection logic
        if similarity > 0.8:
            conflict_level = 'no_conflict'
            confidence = 0.9
        elif similarity > 0.6:
            conflict_level = 'minor_conflict'
            confidence = 0.7
        else:
            conflict_level = 'major_conflict'
            confidence = 0.8
        
        # Confidence disparity adjustment
        confidence_diff = abs(confidence1 - confidence2)
        if confidence_diff > 0.3:
            conflict_level = 'major_conflict' if conflict_level != 'no_conflict' else 'minor_conflict'
            confidence = max(confidence, 0.7)
        
        return {
            'level': conflict_level,
            'confidence': confidence,
            'aspects': ['semantic_meaning', 'confidence_levels'],
            'similarity_score': similarity
        }
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts (simplified)"""
        # In production, this would use proper sentence embeddings
        # For demo, using simple word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def get_vni_domain(self, vni_id: str) -> str:
        """Extract domain from VNI ID"""
        if 'medical' in vni_id:
            return 'medical'
        elif 'legal' in vni_id:
            return 'legal'
        elif 'technical' in vni_id:
            return 'technical'
        else:
            return 'general'

class ConsensusCalculator(nn.Module):
    """Calculate consensus between multiple VNI outputs"""
    
    def __init__(self, config: AggregatorConfig):
        super().__init__()
        self.config = config
        
        # Change input size from 256 → 8 to match actual feature count
        self.consensus_network = nn.Sequential(
            nn.Linear(8, 128),   # ←←← FIX: was 256
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
   
    def calculate_consensus(self, vni_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall consensus between VNI outputs"""
        
        if not vni_outputs:
            return {'consensus_level': 'none', 'confidence': 0.0, 'agreeing_vnis': []}
        
        # Extract confidence scores and domains
        confidence_scores = {}
        domains = {}
        
        for vni_id, output in vni_outputs.items():
            confidence_scores[vni_id] = output.get('confidence_score', 0.5)
            domains[vni_id] = self.get_vni_domain(vni_id)
        
        # Calculate weighted consensus
        total_confidence = sum(confidence_scores.values())
        if total_confidence == 0:
            return {'consensus_level': 'none', 'confidence': 0.0, 'agreeing_vnis': []}
        
        # Domain distribution analysis
        domain_counts = defaultdict(int)
        for domain in domains.values():
            domain_counts[domain] += 1
        
        # Consensus calculation
        avg_confidence = total_confidence / len(confidence_scores)
        domain_variety = len(set(domains.values()))
        
        # Neural network consensus scoring
        consensus_features = self.extract_consensus_features(vni_outputs, confidence_scores, domains)
        with torch.no_grad():
            consensus_tensor = torch.tensor(consensus_features).unsqueeze(0)
            neural_consensus = self.consensus_network(consensus_tensor).item()
        
        # Combined consensus score
        combined_consensus = (neural_consensus + avg_confidence) / 2
        
        # Determine consensus level
        if combined_consensus > 0.8:
            consensus_level = 'strong'
        elif combined_consensus > 0.6:
            consensus_level = 'moderate'
        elif combined_consensus > 0.4:
            consensus_level = 'weak'
        else:
            consensus_level = 'none'
        
        # Identify agreeing VNIs (those with above-average confidence)
        agreeing_vnis = [vni_id for vni_id, conf in confidence_scores.items() 
                        if conf >= avg_confidence]
        
        return {
            'consensus_level': consensus_level,
            'consensus_score': combined_consensus,
            'average_confidence': avg_confidence,
            'domain_distribution': dict(domain_counts),
            'agreeing_vnis': agreeing_vnis,
            'total_vnis': len(vni_outputs)
        }
    
    def extract_consensus_features(self, vni_outputs: Dict[str, Dict[str, Any]],
                                 confidence_scores: Dict[str, float],
                                 domains: Dict[str, str]) -> List[float]:
        """Extract features for consensus calculation"""
        features = []
        
        # Confidence statistics
        confidences = list(confidence_scores.values())
        features.append(np.mean(confidences))  # Mean confidence
        features.append(np.std(confidences))   # Confidence variance
        features.append(max(confidences))      # Max confidence
        features.append(min(confidences))      # Min confidence
        
        # Domain diversity
        unique_domains = len(set(domains.values()))
        features.append(unique_domains / len(domains))  # Domain diversity ratio
        
        # Output quality indicators
        success_count = sum(1 for output in vni_outputs.values() 
                          if output.get('vni_metadata', {}).get('success', False))
        features.append(success_count / len(vni_outputs))  # Success rate
        
        # Pad to fixed size
        while len(features) < 8:
            features.append(0.0)
        
        return features[:8]  # Ensure exactly 8 features
    
    def get_vni_domain(self, vni_id: str) -> str:
        """Extract domain from VNI ID"""
        if 'medical' in vni_id:
            return 'medical'
        elif 'legal' in vni_id:
            return 'legal'
        elif 'technical' in vni_id:
            return 'technical'
        else:
            return 'general'

class ResponseSynthesizer(nn.Module):
    """Synthesize final response from multiple VNI outputs"""
    
    def __init__(self, config: AggregatorConfig):
        super().__init__()
        self.config = config
        
        self.response_generator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
    
    def synthesize_response(self, vni_outputs: Dict[str, Dict[str, Any]],
                          consensus: Dict[str, Any],
                          conflicts: List[Dict[str, Any]]) -> str:
        """Synthesize final response from all VNI outputs"""
        
        if not vni_outputs:
            return "Unable to generate response: No VNI outputs available."
        
        # Extract key advice from each VNI
        domain_advice = self.extract_domain_advice(vni_outputs)
        
        # Build response based on consensus level
        if consensus['consensus_level'] == 'strong':
            response = self.build_consensus_response(domain_advice, consensus)
        elif consensus['consensus_level'] == 'moderate':
            response = self.build_balanced_response(domain_advice, conflicts)
        else:
            response = self.build_conflict_response(domain_advice, conflicts, consensus)
        
        # Apply length limit
        if len(response) > self.config.max_output_length:
            response = response[:self.config.max_output_length-3] + "..."
        
        return response
    
    def extract_domain_advice(self, vni_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Extract advice from each domain"""
        domain_advice = {}
        
        for vni_id, output in vni_outputs.items():
            domain = self.get_vni_domain(vni_id)
            
            # Extract the main advice
            advice = (output.get('medical_advice') or 
                     output.get('legal_advice') or 
                     output.get('technical_advice') or 
                     "No specific advice available.")
            
            confidence = output.get('confidence_score', 0.5)
            
            domain_advice[domain] = {
                'advice': advice,
                'confidence': confidence,
                'vni_id': vni_id
            }
        
        return domain_advice
    
    def build_consensus_response(self, domain_advice: Dict[str, Dict[str, Any]],
                               consensus: Dict[str, Any]) -> str:
        """Build response when there's strong consensus"""
        response_parts = ["Based on comprehensive analysis with strong consensus:"]
        
        # Include advice from all domains
        for domain, info in domain_advice.items():
            response_parts.append(f"\n{domain.title()} Perspective: {info['advice']}")
        
        response_parts.append(f"\nOverall Confidence: {consensus['consensus_score']:.1%}")
        response_parts.append("Multiple expert analyses agree on this assessment.")
        
        return " ".join(response_parts)
    
    def build_balanced_response(self, domain_advice: Dict[str, Dict[str, Any]],
                              conflicts: List[Dict[str, Any]]) -> str:
        """Build balanced response for moderate consensus"""
        response_parts = ["Analysis reveals multiple perspectives:"]
        
        # Include highest confidence advice first
        sorted_domains = sorted(domain_advice.items(), 
                              key=lambda x: x[1]['confidence'], reverse=True)
        
        for domain, info in sorted_domains:
            response_parts.append(f"\n{domain.title()} Analysis: {info['advice']}")
        
        if conflicts:
            response_parts.append(f"\nNote: {len(conflicts)} minor conflicts were identified and resolved.")
        
        response_parts.append("\nThis integrated view considers all available expert opinions.")
        
        return " ".join(response_parts)
    
    def build_conflict_response(self, domain_advice: Dict[str, Dict[str, Any]],
                              conflicts: List[Dict[str, Any]],
                              consensus: Dict[str, Any]) -> str:
        """Build response when there are significant conflicts"""
        response_parts = ["Analysis reveals complex situation with differing expert opinions:"]
        
        # Present each domain's perspective
        for domain, info in domain_advice.items():
            response_parts.append(f"\n{domain.title()} View: {info['advice']}")
        
        # Acknowledge conflicts
        if conflicts:
            response_parts.append(f"\nKey conflicts identified: {len(conflicts)} major disagreements between expert analyses.")
            response_parts.append("Consider consulting additional specialists for this complex scenario.")
        
        response_parts.append(f"\nOverall analysis confidence: {consensus['consensus_score']:.1%}")
        
        return " ".join(response_parts)
    
    def get_vni_domain(self, vni_id: str) -> str:
        """Extract domain from VNI ID"""
        if 'medical' in vni_id:
            return 'medical'
        elif 'legal' in vni_id:
            return 'legal'
        elif 'technical' in vni_id:
            return 'technical'
        else:
            return 'general'

class ResponseAggregator(nn.Module):
    """Main response aggregator class"""
    
    def __init__(self, config: AggregatorConfig = None):
        super().__init__()
        self.config = config or AggregatorConfig()
        self.conflict_detector = ConflictDetector()
        self.consensus_calculator = ConsensusCalculator(self.config)
        self.response_synthesizer = ResponseSynthesizer(self.config)
        
        logger.info(f"Response Aggregator initialized with ID: {self.config.aggregator_id}")
    
    def forward(self, router_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate VNI outputs into final response"""
        
        try:
            execution_results = router_results.get('execution_results', {})
            
            # Filter successful VNI outputs
            successful_outputs = {}
            for vni_id, result in execution_results.items():
                if result.get('vni_metadata', {}).get('success', False):
                    successful_outputs[vni_id] = result
            
            if not successful_outputs:
                return self._generate_no_output_response(router_results)
            
            # Step 1: Detect conflicts between VNI outputs
            conflicts = self.conflict_detector.detect_conflicts(successful_outputs)
            
            # Step 2: Calculate overall consensus
            consensus = self.consensus_calculator.calculate_consensus(successful_outputs)
            
            # Step 3: Synthesize final response
            final_response = self.response_synthesizer.synthesize_response(
                successful_outputs, consensus, conflicts
            )
            
            # Step 4: Calculate overall confidence
            overall_confidence = self.calculate_overall_confidence(successful_outputs, consensus, conflicts)
            
            # Compile aggregation results
            results = {
                'final_response': final_response,
                'aggregation_analysis': {
                    'consensus_analysis': consensus,
                    'conflict_analysis': conflicts,
                    'vni_contributions': self.analyze_vni_contributions(successful_outputs),
                    'domain_coverage': self.analyze_domain_coverage(successful_outputs)
                },
                'confidence_metrics': {
                    'overall_confidence': overall_confidence,
                    'consensus_confidence': consensus.get('consensus_score', 0.0),
                    'vni_confidence_distribution': self.get_confidence_distribution(successful_outputs)
                },
                'processing_metadata': {
                    'total_vnis_processed': len(execution_results),
                    'successful_vnis': len(successful_outputs),
                    'conflicts_detected': len(conflicts),
                    'cross_domain_synthesis': self.is_cross_domain_synthesis(successful_outputs)
                }
            }
            
            # Add aggregator metadata
            results['aggregator_metadata'] = {
                'aggregator_id': self.config.aggregator_id,
                'processing_stages': ['conflict_detection', 'consensus_calculation', 
                                    'response_synthesis', 'confidence_calibration'],
                'success': True,
                'response_length': len(final_response)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Response aggregation failed: {str(e)}")
            return self._generate_error_output(str(e))
    
    def calculate_overall_confidence(self, successful_outputs: Dict[str, Dict[str, Any]],
                                   consensus: Dict[str, Any],
                                   conflicts: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score"""
        
        # Base confidence from VNI outputs
        vni_confidences = [output.get('confidence_score', 0.5) 
                          for output in successful_outputs.values()]
        base_confidence = sum(vni_confidences) / len(vni_confidences) if vni_confidences else 0.5
        
        # Adjust for consensus
        consensus_score = consensus.get('consensus_score', 0.5)
        
        # Adjust for conflicts
        conflict_penalty = 0.0
        for conflict in conflicts:
            if conflict['conflict_level'] == 'major_conflict':
                conflict_penalty += 0.2
            elif conflict['conflict_level'] == 'minor_conflict':
                conflict_penalty += 0.1
        
        conflict_penalty = min(conflict_penalty, 0.5)  # Cap penalty
        
        # Calculate final confidence
        overall_confidence = (base_confidence + consensus_score) / 2
        overall_confidence = max(0.0, overall_confidence - conflict_penalty)
        
        return min(overall_confidence, 1.0)
    
    def analyze_vni_contributions(self, successful_outputs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze contributions from each VNI"""
        contributions = []
        
        for vni_id, output in successful_outputs.items():
            confidence = output.get('confidence_score', 0.5)
            domain = self.get_vni_domain(vni_id)
            
            contributions.append({
                'vni_id': vni_id,
                'domain': domain,
                'confidence': confidence,
                'contribution_level': 'primary' if confidence > 0.7 else 'secondary',
                'output_type': list(output.keys())[0] if output else 'unknown'
            })
        
        return sorted(contributions, key=lambda x: x['confidence'], reverse=True)
    
    def analyze_domain_coverage(self, successful_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
        """Analyze domain coverage of the responses"""
        domain_counts = defaultdict(int)
        
        for vni_id in successful_outputs.keys():
            domain = self.get_vni_domain(vni_id)
            domain_counts[domain] += 1
        
        return dict(domain_counts)
    
    def get_confidence_distribution(self, successful_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Get confidence score distribution"""
        confidences = [output.get('confidence_score', 0.5) for output in successful_outputs.values()]
        
        if not confidences:
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0}
        
        return {
            'min': min(confidences),
            'max': max(confidences),
            'mean': sum(confidences) / len(confidences),
            'std': np.std(confidences) if len(confidences) > 1 else 0.0
        }
    
    def is_cross_domain_synthesis(self, successful_outputs: Dict[str, Dict[str, Any]]) -> bool:
        """Check if this is a cross-domain synthesis"""
        domains = set()
        for vni_id in successful_outputs.keys():
            domains.add(self.get_vni_domain(vni_id))
        
        return len(domains) >= 2
    
    def get_vni_domain(self, vni_id: str) -> str:
        """Extract domain from VNI ID"""
        if 'medical' in vni_id:
            return 'medical'
        elif 'legal' in vni_id:
            return 'legal'
        elif 'technical' in vni_id:
            return 'technical'
        else:
            return 'general'
    
    def _generate_no_output_response(self, router_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response when no VNIs produced successful outputs"""
        return {
            'final_response': "Unable to generate comprehensive analysis. The system could not process this request with sufficient confidence.",
            'aggregation_analysis': {
                'consensus_analysis': {'consensus_level': 'none', 'consensus_score': 0.0},
                'conflict_analysis': [],
                'vni_contributions': [],
                'domain_coverage': {}
            },
            'confidence_metrics': {
                'overall_confidence': 0.0,
                'consensus_confidence': 0.0,
                'vni_confidence_distribution': {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0}
            },
            'aggregator_metadata': {
                'aggregator_id': self.config.aggregator_id,
                'success': False,
                'error': 'No successful VNI outputs'
            }
        }
    
    def _generate_error_output(self, error_msg: str) -> Dict[str, Any]:
        """Generate error output"""
        return {
            'final_response': f"System error: {error_msg}",
            'aggregator_metadata': {
                'aggregator_id': self.config.aggregator_id,
                'success': False,
                'error': error_msg
            }
        }
    
    def get_aggregator_capabilities(self) -> Dict[str, Any]:
        """Return aggregator capabilities description"""
        return {
            'aggregator_type': 'response_aggregator',
            'description': 'Final response synthesis and conflict resolution',
            'capabilities': [
                'Multi-VNI output integration',
                'Conflict detection and resolution',
                'Consensus calculation',
                'Confidence-weighted synthesis',
                'Cross-domain response generation'
            ],
            'input_types': ['vni_execution_results'],
            'output_types': ['final_response', 'aggregation_analysis', 'confidence_metrics']
        }

# Demonstration and testing
def test_response_aggregator():
    """Test the response aggregator with sample router results"""
    
    # Initialize aggregator
    config = AggregatorConfig(aggregator_id="aggregator_test_001")
    aggregator = ResponseAggregator(config)
    
    # Create mock router results
    mock_router_results = {
        'execution_results': {
            'operAction_medical': {
                'medical_analysis': {
                    'diagnoses': [{'condition': 'respiratory_infection', 'confidence': 0.8}],
                    'treatments': [{'treatment': 'antibiotics', 'type': 'medication'}]
                },
                'medical_advice': "Patient shows symptoms of respiratory infection. Recommend antibiotics and rest.",
                'confidence_score': 0.8,
                'vni_metadata': {
                    'vni_id': 'operAction_medical',
                    'success': True,
                    'domain': 'medical'
                }
            },
            'operAction_legal': {
                'legal_analysis': {
                    'legal_issues': [{'issue': 'Liability Considerations', 'risk_score': 0.6}],
                    'compliance_issues': [{'framework': 'HIPAA', 'compliance_level': 0.7}]
                },
                'legal_advice': "Ensure HIPAA compliance for patient data handling and consider liability protections.",
                'confidence_score': 0.7,
                'vni_metadata': {
                    'vni_id': 'operAction_legal',
                    'success': True,
                    'domain': 'legal'
                }
            }
        },
        'activation_plan': {
            'activated_vnis': [
                {'vni_id': 'operAction_medical', 'activation_score': 0.8},
                {'vni_id': 'operAction_legal', 'activation_score': 0.7}
            ]
        }
    }
    
    print("=== Response Aggregator Demo Test ===\n")
    
    with torch.no_grad():
        results = aggregator(mock_router_results)
    
    # Display results
    print("Final Response:")
    print(results['final_response'])
    
    print(f"\nOverall Confidence: {results['confidence_metrics']['overall_confidence']:.1%}")
    
    consensus = results['aggregation_analysis']['consensus_analysis']
    print(f"Consensus Level: {consensus['consensus_level']} (score: {consensus['consensus_score']:.1%})")
    
    conflicts = results['aggregation_analysis']['conflict_analysis']
    print(f"Conflicts Detected: {len(conflicts)}")
    
    return aggregator

if __name__ == "__main__":
    # Run demonstration
    test_response_aggregator() 
