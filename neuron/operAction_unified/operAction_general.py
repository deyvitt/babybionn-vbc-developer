# operAction_general.py - COMPLETELY REVISED
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import re

logger = logging.getLogger("operAction_general")

@dataclass
class GeneralOperActionConfig:
    """Configuration for general-purpose operAction VNI"""
    vni_id: str = "operAction_general"
    reasoning_depth: str = "comprehensive"
    knowledge_base: Dict[str, Any] = None
    confidence_threshold: float = 0.6
    
    def __post_init__(self):
        if self.knowledge_base is None:
            self.knowledge_base = {
                # Expanded to cover multiple domains
                'domain_patterns': {
                    'mathematical': {
                        'concepts': ['calculate', 'equation', 'formula', 'solve', 'compute', 'algebra', 'geometry'],
                        'problem_types': ['calculation', 'proof', 'optimization', 'analysis'],
                        'solution_approaches': ['algorithmic', 'analytical', 'numerical', 'geometric']
                    },
                    'business': {
                        'concepts': ['strategy', 'market', 'profit', 'revenue', 'growth', 'competition'],
                        'problem_types': ['planning', 'analysis', 'decision', 'optimization'],
                        'solution_approaches': ['strategic', 'financial', 'operational', 'market_analysis']
                    },
                    'creative': {
                        'concepts': ['write', 'story', 'creative', 'poem', 'narrative', 'character', 'plot'],
                        'problem_types': ['composition', 'generation', 'refinement', 'structuring'],
                        'solution_approaches': ['brainstorming', 'outlining', 'drafting', 'revising']
                    },
                    'analytical': {
                        'concepts': ['analyze', 'compare', 'evaluate', 'assess', 'interpret'],
                        'problem_types': ['comparison', 'evaluation', 'interpretation', 'synthesis'],
                        'solution_approaches': ['comparative_analysis', 'critical_thinking', 'synthesis']
                    }
                },
                'reasoning_frameworks': {
                    'problem_solving': ['define_problem', 'analyze_causes', 'generate_solutions', 'evaluate_options'],
                    'decision_making': ['identify_options', 'weigh_pros_cons', 'assess_risks', 'make_choice'],
                    'creative_thinking': ['divergent_thinking', 'pattern_breaking', 'analogical_reasoning'],
                    'critical_thinking': ['evidence_evaluation', 'logical_analysis', 'bias_detection']
                }
            }

class DomainDetector(nn.Module):
    """Detect which domains are relevant to the input"""
    
    def __init__(self):
        super().__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),  # 8 potential domains
            nn.Softmax(dim=-1)
        )
    
    def forward(self, cognitive_tensor: torch.Tensor, concepts: List[str]) -> Dict[str, float]:
        """Detect domain relevance scores"""
        # Neural classification
        neural_scores = self.domain_classifier(cognitive_tensor.unsqueeze(0))[0]
        
        # Keyword-based classification
        domain_keywords = {
            'mathematical': ['calculate', 'equation', 'formula', 'solve', 'math', 'algebra', 'geometry'],
            'business': ['business', 'strategy', 'market', 'profit', 'revenue', 'sales', 'growth'],
            'creative': ['write', 'story', 'creative', 'poem', 'narrative', 'character', 'plot'],
            'analytical': ['analyze', 'compare', 'evaluate', 'assess', 'interpret', 'synthesize'],
            'technical': ['system', 'code', 'software', 'database', 'api', 'server', 'network'],
            'scientific': ['experiment', 'hypothesis', 'data', 'research', 'study', 'evidence'],
            'educational': ['learn', 'teach', 'education', 'study', 'knowledge', 'understand'],
            'general': ['information', 'explanation', 'description', 'overview']
        }
        
        keyword_scores = torch.zeros(8)
        concepts_lower = [str(c).lower() for c in concepts]
        
        for i, (domain, keywords) in enumerate(domain_keywords.items()):
            matches = sum(1 for concept in concepts_lower 
                         if any(keyword in concept for keyword in keywords))
            keyword_scores[i] = min(matches / 3, 1.0)  # Normalize
        
        # Combine neural and keyword scores
        combined_scores = 0.6 * neural_scores + 0.4 * keyword_scores
        domain_names = list(domain_keywords.keys())
        
        return {domain: score.item() for domain, score in zip(domain_names, combined_scores)}

class GeneralReasoningEngine(nn.Module):
    """General-purpose reasoning engine for multiple domains"""
    
    def __init__(self, config: GeneralOperActionConfig):
        super().__init__()
        self.config = config
        self.domain_detector = DomainDetector()
        
        # Multi-domain reasoning networks
        self.concept_analyzer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        
        self.problem_analyzer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Softmax(dim=-1)
        )
        
        self.solution_generator = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.Sigmoid()
        )
    
    def detect_primary_domain(self, abstraction_data: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
        """Detect the primary domain and all domain scores"""
        cognitive_tensor = abstraction_data.get('cognitive', {}).get('tensor', torch.zeros(256))
        concepts = abstraction_data.get('cognitive', {}).get('concepts', [])
        
        domain_scores = self.domain_detector(cognitive_tensor, concepts)
        primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
        
        return primary_domain, domain_scores
    
    def extract_general_concepts(self, abstraction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract concepts across multiple domains"""
        general_concepts = {
            'primary_domain': '',
            'domain_scores': {},
            'key_concepts': [],
            'problem_indicators': [],
            'solution_approaches': [],
            'reasoning_patterns': []
        }
        
        if 'cognitive' in abstraction_data:
            cognitive = abstraction_data['cognitive']
            concepts = cognitive.get('concepts', [])
            
            # Detect domains
            primary_domain, domain_scores = self.detect_primary_domain(abstraction_data)
            general_concepts['primary_domain'] = primary_domain
            general_concepts['domain_scores'] = domain_scores
            
            # Extract domain-specific concepts
            general_concepts['key_concepts'] = concepts
            
            # Problem indicators
            problem_words = ['problem', 'issue', 'challenge', 'difficulty', 'trouble']
            general_concepts['problem_indicators'] = [
                c for c in concepts if any(word in c.lower() for word in problem_words)
            ]
            
            # Solution approaches based on domain
            if domain_scores.get('mathematical', 0) > 0.3:
                general_concepts['solution_approaches'].extend(['algorithmic', 'analytical'])
            if domain_scores.get('business', 0) > 0.3:
                general_concepts['solution_approaches'].extend(['strategic', 'financial'])
            if domain_scores.get('creative', 0) > 0.3:
                general_concepts['solution_approaches'].extend(['brainstorming', 'composition'])
        
        return general_concepts
    
    def analyze_general_problem(self, general_concepts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze problems across multiple domains"""
        problems = []
        primary_domain = general_concepts.get('primary_domain', 'general')
        
        # Domain-specific problem analysis
        if primary_domain == 'mathematical':
            problems.extend(self.analyze_mathematical_problems(general_concepts))
        elif primary_domain == 'business':
            problems.extend(self.analyze_business_problems(general_concepts))
        elif primary_domain == 'creative':
            problems.extend(self.analyze_creative_problems(general_concepts))
        elif primary_domain == 'analytical':
            problems.extend(self.analyze_analytical_problems(general_concepts))
        else:
            problems.extend(self.analyze_general_problems(general_concepts))
        
        return problems
    
    def analyze_mathematical_problems(self, concepts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze mathematical problems"""
        problems = []
        key_concepts = concepts.get('key_concepts', [])
        
        if any('calculate' in str(c).lower() for c in key_concepts):
            problems.append({
                'problem': 'Mathematical Calculation Required',
                'domain': 'mathematical',
                'complexity': 'medium',
                'description': 'Requires numerical computation or formula application',
                'approach': 'Apply appropriate mathematical operations and verify results'
            })
        
        if any('equation' in str(c).lower() for c in key_concepts):
            problems.append({
                'problem': 'Equation Solving Needed',
                'domain': 'mathematical', 
                'complexity': 'medium',
                'description': 'Involves solving equations or systems of equations',
                'approach': 'Use algebraic manipulation or numerical methods'
            })
        
        return problems
    
    def analyze_business_problems(self, concepts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze business problems"""
        problems = []
        key_concepts = concepts.get('key_concepts', [])
        
        if any('strategy' in str(c).lower() for c in key_concepts):
            problems.append({
                'problem': 'Strategic Planning Required',
                'domain': 'business',
                'complexity': 'high',
                'description': 'Involves developing business strategies or plans',
                'approach': 'Conduct market analysis and develop strategic options'
            })
        
        return problems
    
    def analyze_creative_problems(self, concepts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze creative writing/problems"""
        problems = []
        key_concepts = concepts.get('key_concepts', [])
        
        if any('write' in str(c).lower() for c in key_concepts):
            problems.append({
                'problem': 'Creative Composition Required',
                'domain': 'creative',
                'complexity': 'medium', 
                'description': 'Involves creative writing or content generation',
                'approach': 'Use brainstorming and structured composition techniques'
            })
        
        return problems
    
    def analyze_analytical_problems(self, concepts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze analytical problems"""
        problems = []
        key_concepts = concepts.get('key_concepts', [])
        
        if any('analyze' in str(c).lower() for c in key_concepts):
            problems.append({
                'problem': 'In-depth Analysis Required',
                'domain': 'analytical',
                'complexity': 'high',
                'description': 'Requires systematic analysis and evaluation',
                'approach': 'Apply critical thinking and evidence-based evaluation'
            })
        
        return problems
    
    def analyze_general_problems(self, concepts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze general problems"""
        problems = []
        
        if concepts.get('problem_indicators'):
            problems.append({
                'problem': 'General Problem Resolution',
                'domain': 'general',
                'complexity': 'medium',
                'description': 'General problem-solving approach required',
                'approach': 'Apply systematic problem-solving framework'
            })
        
        return problems

class OperActionGeneral(nn.Module):
    """General-purpose reasoning VNI for multiple domains"""
    
    def __init__(self, config: GeneralOperActionConfig = None):
        super().__init__()
        self.config = config or GeneralOperActionConfig()
        self.reasoning_engine = GeneralReasoningEngine(self.config)
        
        logger.info(f"General operAction VNI initialized with ID: {self.config.vni_id}")
    
    def forward(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process general reasoning request across multiple domains"""
        
        try:
            abstraction_data = input_data.get('abstraction_data', {})
            
            # Extract general concepts and detect domain
            general_concepts = self.reasoning_engine.extract_general_concepts(abstraction_data)
            
            # Analyze problems based on detected domain
            problems = self.reasoning_engine.analyze_general_problem(general_concepts)
            
            # Generate domain-appropriate advice
            general_advice = self.generate_general_advice(general_concepts, problems)
            
            # Compile results
            results = {
                'general_analysis': {
                    'primary_domain': general_concepts['primary_domain'],
                    'domain_scores': general_concepts['domain_scores'],
                    'problems_identified': problems,
                    'key_concepts': general_concepts['key_concepts'],
                    'recommended_approaches': general_concepts['solution_approaches']
                },
                'general_advice': general_advice,
                'confidence_score': general_concepts['domain_scores'].get(general_concepts['primary_domain'], 0.0),
                'processing_metadata': {
                    'domains_detected': len([d for d, s in general_concepts['domain_scores'].items() if s > 0.3]),
                    'problems_identified': len(problems),
                    'primary_domain': general_concepts['primary_domain']
                }
            }
            
            # Add VNI metadata
            results['vni_metadata'] = {
                'vni_id': self.config.vni_id,
                'vni_type': 'operAction_general',
                'processing_stages': ['domain_detection', 'concept_extraction', 'problem_analysis'],
                'success': True,
                'domain': 'multi_domain'
            }
            
            return results
            
        except Exception as e:
            logger.error(f"General operAction processing failed: {str(e)}")
            return self._generate_error_output(str(e))
    
    def generate_general_advice(self, concepts: Dict[str, Any], problems: List[Dict[str, Any]]) -> str:
        """Generate domain-appropriate general advice"""
        primary_domain = concepts['primary_domain']
        
        if not problems:
            return f"No specific problems identified in {primary_domain} domain. General analysis complete."
        
        primary_problem = problems[0]
        
        advice_parts = [
            f"Analysis focused on {primary_domain} domain.",
            f"Primary concern: {primary_problem['problem']}",
            f"Recommended approach: {primary_problem['approach']}",
            f"Complexity level: {primary_problem['complexity']}"
        ]
        
        if len(problems) > 1:
            advice_parts.append(f"Additional {len(problems)-1} considerations identified.")
        
        return " ".join(advice_parts)
    
    def _generate_error_output(self, error_msg: str) -> Dict[str, Any]:
        """Generate error output"""
        return {
            'general_analysis': {},
            'general_advice': f"General analysis unavailable: {error_msg}",
            'vni_metadata': {
                'vni_id': self.config.vni_id,
                'vni_type': 'operAction_general',
                'success': False,
                'error': error_msg,
                'domain': 'multi_domain'
            }
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return VNI capabilities description"""
        return {
            'vni_type': 'operAction_general',
            'description': 'General-purpose multi-domain reasoning VNI',
            'capabilities': [
                'Multi-domain detection and classification',
                'Domain-appropriate problem analysis',
                'General reasoning and solution generation',
                'Cross-domain concept integration'
            ],
            'supported_domains': ['mathematical', 'business', 'creative', 'analytical', 'technical', 'scientific', 'educational', 'general'],
            'input_types': ['multi_domain_abstraction_data'],
            'output_types': ['general_analysis', 'domain_assessment', 'problem_identification'],
            'domain': 'multi_domain'
        }

# Test with multiple domains
def test_general_operaction():
    """Test the general operAction with various domain inputs"""
    
    general_vni = OperActionGeneral()
    
    test_cases = [
        {
            'name': 'Mathematical Query',
            'concepts': ['calculate', 'area', 'circle', 'radius', 'formula'],
            'expected_domain': 'mathematical'
        },
        {
            'name': 'Business Query', 
            'concepts': ['business', 'strategy', 'market', 'growth', 'competition'],
            'expected_domain': 'business'
        },
        {
            'name': 'Creative Query',
            'concepts': ['write', 'story', 'character', 'plot', 'narrative'],
            'expected_domain': 'creative'
        }
    ]
    
    for test_case in test_cases:
        print(f"\n=== Testing: {test_case['name']} ===")
        
        test_input = {
            'abstraction_data': {
                'cognitive': {
                    'tensor': torch.randn(256),
                    'concepts': test_case['concepts'],
                    'intent': 'analysis'
                }
            }
        }
        
        with torch.no_grad():
            results = general_vni(test_input)
        
        primary_domain = results['general_analysis']['primary_domain']
        print(f"Detected Domain: {primary_domain}")
        print(f"Confidence: {results['confidence_score']:.1%}")
        print(f"Advice: {results['general_advice']}")

if __name__ == "__main__":
    test_general_operaction()
