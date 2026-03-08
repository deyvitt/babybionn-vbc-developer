# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

# _general.py - HYBRID VERSION with multi-domain support + dynamic adaptation
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import re
from specialized_vni_base import SpecializedVNIBase

logger = logging.getLogger("operAction_general_hybrid")

@dataclass
class GeneralOperActionConfig:
    """Configuration for general-purpose VNI"""
    vni_id: str = "operAction_general_hybrid"
    reasoning_depth: str = "adaptive"
    knowledge_base: Dict[str, Any] = None
    enable_dynamic_adaptation: bool = True
    max_domains_per_query: int = 3
    
    def __post_init__(self):
        if self.knowledge_base is None:
            self.knowledge_base = {
                'domain_definitions': {
                    'mathematical': {
                        'concepts': ['calculate', 'equation', 'formula', 'solve', 'compute', 'math'],
                        'problem_types': ['calculation', 'proof', 'optimization', 'analysis'],
                        'solution_strategies': ['algorithmic', 'analytical', 'numerical']
                    },
                    'scientific': {
                        'concepts': ['experiment', 'hypothesis', 'theory', 'data', 'research', 'study'],
                        'problem_types': ['experimental_design', 'data_analysis', 'theoretical_modeling'],
                        'solution_strategies': ['empirical', 'theoretical', 'computational']
                    },
                    'business': {
                        'concepts': ['strategy', 'market', 'profit', 'revenue', 'growth', 'competition'],
                        'problem_types': ['planning', 'analysis', 'decision_making', 'optimization'],
                        'solution_strategies': ['strategic', 'financial', 'operational']
                    },
                    'creative': {
                        'concepts': ['write', 'story', 'creative', 'poem', 'narrative', 'character'],
                        'problem_types': ['composition', 'generation', 'refinement', 'structuring'],
                        'solution_strategies': ['brainstorming', 'outlining', 'drafting']
                    },
                    'technical': {
                        'concepts': ['system', 'code', 'software', 'database', 'api', 'server'],
                        'problem_types': ['debugging', 'design', 'implementation', 'optimization'],
                        'solution_strategies': ['systematic', 'modular', 'iterative']
                    },
                    'educational': {
                        'concepts': ['learn', 'teach', 'education', 'study', 'knowledge', 'understanding'],
                        'problem_types': ['explanation', 'instruction', 'assessment', 'curriculum'],
                        'solution_strategies': ['pedagogical', 'scaffolded', 'interactive']
                    },
                    'analytical': {
                        'concepts': ['analyze', 'compare', 'evaluate', 'assess', 'interpret', 'synthesize'],
                        'problem_types': ['comparison', 'evaluation', 'interpretation', 'synthesis'],
                        'solution_strategies': ['critical_thinking', 'comparative_analysis', 'synthesis']
                    }
                },
                'reasoning_frameworks': {
                    'problem_solving': {
                        'steps': ['define_problem', 'analyze_causes', 'generate_solutions', 'evaluate_options'],
                        'applicability': ['mathematical', 'technical', 'business', 'analytical']
                    },
                    'decision_making': {
                        'steps': ['identify_options', 'weigh_pros_cons', 'assess_risks', 'make_choice'],
                        'applicability': ['business', 'technical', 'analytical']
                    },
                    'creative_thinking': {
                        'steps': ['divergent_thinking', 'pattern_breaking', 'analogical_reasoning'],
                        'applicability': ['creative', 'educational', 'scientific']
                    },
                    'critical_thinking': {
                        'steps': ['evidence_evaluation', 'logical_analysis', 'bias_detection'],
                        'applicability': ['analytical', 'scientific', 'educational']
                    }
                }
            }

class DomainDetectionEngine(nn.Module):
    """Enhanced domain detection with neural networks"""
    
    def __init__(self, config: GeneralOperActionConfig):
        super().__init__()
        self.config = config
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(config.knowledge_base['domain_definitions'])),
            nn.Softmax(dim=-1)
        )
        
        self.domain_importance = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
    
    def detect_domains(self, text: str, cognitive_tensor: torch.Tensor) -> Dict[str, float]:
        """Detect relevant domains with confidence scores"""
        domain_names = list(self.config.knowledge_base['domain_definitions'].keys())
        
        # Neural classification
        neural_scores = self.domain_classifier(cognitive_tensor.unsqueeze(0))[0]
        
        # Keyword-based detection
        keyword_scores = torch.zeros(len(domain_names))
        text_lower = text.lower()
        
        for i, (domain, info) in enumerate(self.config.knowledge_base['domain_definitions'].items()):
            keyword_matches = sum(1 for concept in info['concepts'] if concept in text_lower)
            keyword_scores[i] = min(keyword_matches / 3, 1.0)  # Normalize
        
        # Combine scores
        combined_scores = 0.6 * neural_scores + 0.4 * keyword_scores
        
        # Calculate domain importance
        if cognitive_tensor.numel() > 0:
            importance_scores = self.domain_importance(cognitive_tensor.unsqueeze(0))
            combined_scores = combined_scores * importance_scores[0]
        
        return {domain: score.item() for domain, score in zip(domain_names, combined_scores)}
    
    def select_primary_domains(self, domain_scores: Dict[str, float]) -> List[str]:
        """Select most relevant domains for analysis"""
        # Sort by score
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top N domains above threshold
        threshold = 0.3
        selected = [domain for domain, score in sorted_domains 
                   if score > threshold][:self.config.max_domains_per_query]
        
        return selected

class GeneralReasoningEngine(nn.Module):
    """General reasoning engine with dynamic adaptation"""
    
    def __init__(self, config: GeneralOperActionConfig):
        super().__init__()
        self.config = config
        self.domain_detector = DomainDetectionEngine(config)
        
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
        
        # Dynamic adapter for cross-domain learning
        self.dynamic_adapter = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Learned cross-domain patterns
        self.cross_domain_patterns = []
        self.successful_responses = []
        
        # Performance tracking
        self.performance_stats = {
            'total_queries': 0,
            'domain_distribution': {},
            'avg_domains_per_query': 0,
            'common_query_patterns': {}
        }
    
    def extract_general_concepts(self, text: str, abstraction_data: Dict = None) -> Dict[str, Any]:
        """Extract general concepts across domains"""
        concepts = {
            'detected_domains': [],
            'domain_scores': {},
            'key_concepts': [],
            'problem_indicators': [],
            'query_intent': 'unknown',
            'complexity_estimate': 'medium'
        }
        
        # Detect domains
        cognitive_tensor = torch.zeros(256)
        if abstraction_data and 'cognitive' in abstraction_data:
            cognitive_tensor = abstraction_data['cognitive'].get('tensor', torch.zeros(256))
        
        domain_scores = self.domain_detector.detect_domains(text, cognitive_tensor)
        concepts['domain_scores'] = domain_scores
        
        # Select primary domains
        primary_domains = self.domain_detector.select_primary_domains(domain_scores)
        concepts['detected_domains'] = primary_domains
        
        # Extract key concepts
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Collect concepts from all detected domains
        all_domain_concepts = []
        for domain in primary_domains:
            if domain in self.config.knowledge_base['domain_definitions']:
                domain_info = self.config.knowledge_base['domain_definitions'][domain]
                all_domain_concepts.extend(domain_info['concepts'])
        
        # Find matches in text
        for word in words:
            if len(word) > 3 and word in all_domain_concepts:
                if word not in concepts['key_concepts']:
                    concepts['key_concepts'].append(word)
        
        # Detect problem indicators
        problem_words = ['problem', 'issue', 'challenge', 'difficulty', 'trouble', 'help', 'how to']
        for word in problem_words:
            if word in text_lower:
                concepts['problem_indicators'].append(word)
                concepts['query_intent'] = 'problem_solving'
        
        # Detect question intent
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', '?']
        if any(text_lower.startswith(word) for word in question_words) or '?' in text:
            concepts['query_intent'] = 'information_seeking'
        
        # Estimate complexity
        word_count = len(words)
        if word_count > 50:
            concepts['complexity_estimate'] = 'high'
        elif word_count > 20:
            concepts['complexity_estimate'] = 'medium'
        else:
            concepts['complexity_estimate'] = 'low'
        
        # Update stats
        self.performance_stats['total_queries'] += 1
        for domain in primary_domains:
            if domain not in self.performance_stats['domain_distribution']:
                self.performance_stats['domain_distribution'][domain] = 0
            self.performance_stats['domain_distribution'][domain] += 1
        
        total_domains = sum(len(concepts['detected_domains']) for _ in range(self.performance_stats['total_queries']))
        self.performance_stats['avg_domains_per_query'] = total_domains / self.performance_stats['total_queries']
        
        return concepts
    
    def analyze_general_problem(self, concepts: Dict, features: torch.Tensor) -> List[Dict]:
        """Analyze problems across detected domains"""
        problems = []
        primary_domains = concepts.get('detected_domains', [])
        
        for domain in primary_domains:
            domain_problems = self._analyze_domain_problem(domain, concepts, features)
            problems.extend(domain_problems)
        
        # Apply dynamic adjustment
        if self.config.enable_dynamic_adaptation and features is not None and problems:
            dynamic_adjustment = self.dynamic_adapter(features)
            adjustment_factor = torch.sigmoid(dynamic_adjustment.mean()).item()
            
            for problem in problems:
                problem['confidence'] = min(problem.get('confidence', 0.5) * (1 + adjustment_factor * 0.2), 1.0)
                problem['evidence_sources'].append('dynamic_adapter')
        
        # Sort by confidence
        problems.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return problems[:5]  # Top 5 problems
    
    def _analyze_domain_problem(self, domain: str, concepts: Dict, features: torch.Tensor) -> List[Dict]:
        """Analyze problems specific to a domain"""
        problems = []
        
        if domain not in self.config.knowledge_base['domain_definitions']:
            return problems
        
        domain_info = self.config.knowledge_base['domain_definitions'][domain]
        
        # Check for problem types
        for problem_type in domain_info['problem_types']:
            # Simple matching for now
            if any(word in str(concepts).lower() for word in problem_type.split('_')):
                confidence = 0.5
                
                # Neural analysis
                if features is not None:
                    problem_features = self.problem_analyzer(features[:128])
                    confidence = problem_features.mean().item()
                
                problems.append({
                    'domain': domain,
                    'problem_type': problem_type.replace('_', ' ').title(),
                    'description': f'This appears to be a {problem_type.replace("_", " ")} problem in {domain}',
                    'confidence': confidence,
                    'recommended_approach': self._get_recommended_approach(domain, problem_type),
                    'evidence_sources': ['domain_patterns']
                })
        
        return problems
    
    def _get_recommended_approach(self, domain: str, problem_type: str) -> str:
        """Get recommended approach for a domain problem"""
        # Find appropriate reasoning framework
        for framework_name, framework_info in self.config.knowledge_base['reasoning_frameworks'].items():
            if domain in framework_info['applicability']:
                steps = framework_info['steps']
                return f"Apply {framework_name.replace('_', ' ')}: {', '.join(steps)}"
        
        # Fallback to domain-specific strategy
        if domain in self.config.knowledge_base['domain_definitions']:
            strategies = self.config.knowledge_base['domain_definitions'][domain]['solution_strategies']
            return f"Consider {strategies[0]} approach"
        
        return "Apply systematic problem-solving framework"
    
    def generate_solutions(self, problems: List[Dict], concepts: Dict, features: torch.Tensor) -> List[Dict]:
        """Generate solution suggestions"""
        solutions = []
        
        for problem in problems[:3]:  # Top 3 problems
            domain = problem['domain']
            
            # Get domain strategies
            domain_strategies = []
            if domain in self.config.knowledge_base['domain_definitions']:
                domain_strategies = self.config.knowledge_base['domain_definitions'][domain]['solution_strategies']
            
            # Neural solution generation
            solution_details = []
            if features is not None:
                solution_scores = self.solution_generator(features[:96])
                solution_scores = solution_scores[0].tolist()
                
                for i, strategy in enumerate(domain_strategies[:len(solution_scores)]):
                    confidence = solution_scores[i] if i < len(solution_scores) else 0.5
                    solution_details.append({
                        'strategy': strategy,
                        'description': f'Use {strategy} approach for {domain} problem',
                        'confidence': confidence,
                        'steps': self._generate_strategy_steps(strategy)
                    })
            else:
                # Fallback to static strategies
                for strategy in domain_strategies[:3]:
                    solution_details.append({
                        'strategy': strategy,
                        'description': f'Consider {strategy} approach',
                        'confidence': 0.5,
                        'steps': self._generate_strategy_steps(strategy)
                    })
            
            solutions.append({
                'problem': problem['problem_type'],
                'domain': domain,
                'confidence': problem['confidence'],
                'solution_strategies': solution_details,
                'cross_domain_considerations': self._get_cross_domain_considerations(domain, concepts['detected_domains'])
            })
        
        return solutions
    
    def _generate_strategy_steps(self, strategy: str) -> List[str]:
        """Generate steps for a strategy"""
        strategy_steps = {
            'algorithmic': ['Define algorithm', 'Implement solution', 'Test with cases'],
            'analytical': ['Analyze components', 'Identify patterns', 'Derive solution'],
            'strategic': ['Assess situation', 'Develop plan', 'Execute and monitor'],
            'brainstorming': ['Generate ideas', 'Evaluate options', 'Select best approach'],
            'systematic': ['Define problem', 'Break down components', 'Solve systematically']
        }
        
        return strategy_steps.get(strategy, ['Define approach', 'Execute', 'Evaluate results'])
    
    def _get_cross_domain_considerations(self, primary_domain: str, all_domains: List[str]) -> List[str]:
        """Get cross-domain considerations"""
        considerations = []
        
        for domain in all_domains:
            if domain != primary_domain:
                considerations.append(f"Consider {domain} perspectives")
        
        return considerations[:2]  # Top 2 considerations
    
    def generate_general_advice(self, concepts: Dict, problems: List[Dict], 
                              solutions: List[Dict]) -> str:
        """Generate general advice"""
        
        if not problems:
            return "Based on the query, no specific problems were identified. Consider rephrasing or providing more context for targeted assistance."
        
        primary_problem = problems[0]
        advice_parts = []
        
        # Summary
        advice_parts.append(f"**Analysis Summary:**")
        advice_parts.append(f"- Primary Domain: {primary_problem['domain'].title()}")
        advice_parts.append(f"- Problem Type: {primary_problem['problem_type']}")
        advice_parts.append(f"- Confidence: {primary_problem['confidence']:.1%}")
        
        # Detected domains
        if len(concepts['detected_domains']) > 1:
            advice_parts.append(f"\n**Cross-Domain Analysis:**")
            advice_parts.append(f"Multiple domains detected: {', '.join([d.title() for d in concepts['detected_domains']])}")
        
        # Recommended approach
        advice_parts.append(f"\n**Recommended Approach:**")
        advice_parts.append(primary_problem['recommended_approach'])
        
        # Solution strategies
        if solutions:
            primary_solution = solutions[0]
            advice_parts.append(f"\n**Solution Strategies:**")
            for strategy in primary_solution['solution_strategies'][:2]:
                advice_parts.append(f"- {strategy['strategy'].title()}: {strategy['description']}")
        
        # Additional considerations
        if 'complexity_estimate' in concepts and concepts['complexity_estimate'] == 'high':
            advice_parts.append(f"\n**Note:** This appears to be a complex query. Consider breaking it down into smaller parts.")
        
        # Next steps
        advice_parts.append(f"\n**Next Steps:**")
        advice_parts.append("1. Clarify specific requirements if needed")
        advice_parts.append("2. Apply recommended approach systematically")
        advice_parts.append("3. Iterate based on results")
        
        return "\n".join(advice_parts)

class GeneralActionVNI(SpecializedVNIBase):
    """Hybrid General VNI for multi-domain reasoning"""
    
    def __init__(self, config: GeneralOperActionConfig = None):
        config_obj = config or GeneralOperActionConfig()
        super().__init__(topic_name="general", config=config_obj.__dict__)
        
        self.config = config_obj
        self.reasoning_engine = GeneralReasoningEngine(self.config)
        
        # Multi-domain dynamic adapter
        self.multidomain_adapter = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Replace base adapter
        self.dynamic_adapter = self.multidomain_adapter
        
        logger.info(f"✅ Hybrid General VNI initialized: {self.config.vni_id}")
        logger.info(f"   Supports {len(self.config.knowledge_base['domain_definitions'])} domains")
    
    def forward(self, base_features: Dict, input_data: Any) -> Dict:
        """Process general input with multi-domain analysis"""
        
        # Extract text
        text = self._extract_text(input_data)
        
        # Extract concepts
        abstraction_data = base_features.get('abstraction_levels', {})
        concepts = self.reasoning_engine.extract_general_concepts(text, abstraction_data)
        
        # Get features
        features = self._extract_features(base_features)
        
        # Analyze problems
        problems = self.reasoning_engine.analyze_general_problem(concepts, features)
        
        # Generate solutions
        solutions = self.reasoning_engine.generate_solutions(problems, concepts, features)
        
        # Generate advice
        general_advice = self.reasoning_engine.generate_general_advice(concepts, problems, solutions)
        
        # Compile result
        base_result = {
            'general_analysis': {
                'detected_domains': concepts['detected_domains'],
                'domain_scores': concepts['domain_scores'],
                'problems_identified': problems,
                'solutions': solutions,
                'query_intent': concepts['query_intent'],
                'complexity': concepts['complexity_estimate']
            },
            'general_advice': general_advice,
            'confidence_score': problems[0]['confidence'] if problems else 0.0,
            'processing_metadata': {
                'domains_detected': len(concepts['detected_domains']),
                'problems_identified': len(problems),
                'primary_domain': concepts['detected_domains'][0] if concepts['detected_domains'] else 'none',
                'dynamic_adaptation_used': self.config.enable_dynamic_adaptation
            }
        }
        
        # Apply dynamic adaptation
        if self.config.enable_dynamic_adaptation and self.adaptation_strength > 0.1:
            adapted_result = self.apply_dynamic_adaptation(base_result, base_features)
            adapted_result['dynamic_adaptation_applied'] = True
            result = adapted_result
        else:
            result = base_result
        
        # Add metadata
        result['vni_metadata'] = {
            'vni_id': self.config.vni_id,
            'vni_type': 'operAction_general_hybrid',
            'processing_stages': ['domain_detection', 'concept_extraction', 
                                'problem_analysis', 'solution_generation'],
            'success': True,
            'domain': 'multi_domain',
            'hybrid_system': True,
            'dynamic_learning_enabled': self.config.enable_dynamic_adaptation,
            'supported_domains_count': len(self.config.knowledge_base['domain_definitions'])
        }
        
        return result
    
    def _extract_text(self, input_data: Any) -> str:
        """Extract text from input"""
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict):
            return input_data.get('text', str(input_data))
        return str(input_data)
    
    def _extract_features(self, base_features: Dict) -> Optional[torch.Tensor]:
        """Extract features for neural processing"""
        if 'semantic' in base_features and 'tensor' in base_features['semantic']:
            return base_features['semantic']['tensor']
        elif 'tensor' in base_features:
            return base_features['tensor']
        elif 'abstraction_levels' in base_features:
            abstraction = base_features['abstraction_levels']
            if 'semantic' in abstraction and 'tensor' in abstraction['semantic']:
                return abstraction['semantic']['tensor']
        return torch.zeros(1, 256) 
