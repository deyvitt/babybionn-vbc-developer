# _technical.py - HYBRID VERSION with static technical knowledge + dynamic adaptation
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import re
from specialized_vni_base import SpecializedVNIBase

logger = logging.getLogger("operAction_technical_hybrid")

@dataclass
class TechnicalOperActionConfig:
    """Configuration for technical operAction VNI"""
    vni_id: str = "operAction_technical_hybrid"
    reasoning_depth: str = "comprehensive"
    technical_knowledge_base: Dict[str, Any] = None
    confidence_threshold: float = 0.6
    enable_dynamic_adaptation: bool = True
    
    def __post_init__(self):
        if self.technical_knowledge_base is None:
            self.technical_knowledge_base = {
                'technical_domains': {
                    'software_development': {
                        'concepts': ['code', 'programming', 'software', 'algorithm', 'bug', 'debug'],
                        'common_issues': ['syntax_error', 'runtime_error', 'performance_issue', 'security_vulnerability'],
                        'solution_patterns': ['refactoring', 'debugging', 'optimization', 'testing']
                    },
                    'system_administration': {
                        'concepts': ['server', 'database', 'network', 'configuration', 'deployment', 'backup'],
                        'common_issues': ['system_crash', 'performance_bottleneck', 'connectivity_issue', 'security_breach'],
                        'solution_patterns': ['troubleshooting', 'scaling', 'monitoring', 'recovery']
                    },
                    'data_science': {
                        'concepts': ['data', 'analysis', 'model', 'machine_learning', 'statistics', 'visualization'],
                        'common_issues': ['data_quality', 'model_overfitting', 'computational_complexity', 'interpretation'],
                        'solution_patterns': ['data_cleaning', 'feature_engineering', 'model_selection', 'validation']
                    },
                    'cybersecurity': {
                        'concepts': ['security', 'encryption', 'authentication', 'vulnerability', 'firewall', 'malware'],
                        'common_issues': ['unauthorized_access', 'data_breach', 'dos_attack', 'phishing'],
                        'solution_patterns': ['risk_assessment', 'penetration_testing', 'incident_response', 'hardening']
                    }
                },
                'complexity_levels': {
                    'beginner': ['basic syntax', 'simple debugging', 'routine configuration'],
                    'intermediate': ['algorithm design', 'system optimization', 'data modeling'],
                    'advanced': ['architectural design', 'machine_learning_models', 'security_auditing']
                },
                'urgency_categories': {
                    'critical': ['system_down', 'security_breach', 'data_loss'],
                    'high': ['performance_issues', 'functionality_broken', 'deadline_approaching'],
                    'medium': ['optimization_needed', 'feature_request', 'code_review'],
                    'low': ['general_inquiry', 'best_practices', 'learning_question']
                }
            }

class TechnicalKnowledgeGraph:
    """Static technical knowledge graph"""
    def __init__(self):
        self.technical_ontology = self.build_technical_ontology()
        self.solution_patterns = self.load_solution_patterns()
    
    def build_technical_ontology(self):
        """Build technical concept relationships"""
        return {
            'software_development': {
                'concepts': ['variable', 'function', 'class', 'module', 'package', 'framework'],
                'relationships': {
                    'bug': ['causes', 'symptoms', 'solutions'],
                    'performance': ['metrics', 'bottlenecks', 'optimizations'],
                    'testing': ['unit_tests', 'integration_tests', 'test_coverage']
                },
                'best_practices': ['code_review', 'version_control', 'documentation', 'testing']
            },
            'system_administration': {
                'concepts': ['server', 'load_balancer', 'database', 'cache', 'monitoring'],
                'relationships': {
                    'scalability': ['horizontal', 'vertical', 'load_distribution'],
                    'reliability': ['redundancy', 'backups', 'failover'],
                    'security': ['authentication', 'authorization', 'encryption']
                },
                'best_practices': ['automation', 'monitoring', 'disaster_recovery', 'capacity_planning']
            },
            'data_science': {
                'concepts': ['dataset', 'feature', 'model', 'training', 'validation', 'prediction'],
                'relationships': {
                    'modeling': ['supervised', 'unsupervised', 'reinforcement'],
                    'evaluation': ['accuracy', 'precision', 'recall', 'f1_score'],
                    'optimization': ['hyperparameter_tuning', 'regularization', 'ensemble_methods']
                },
                'best_practices': ['data_exploration', 'feature_selection', 'cross_validation', 'model_interpretation']
            }
        }
    
    def load_solution_patterns(self):
        """Load common technical solution patterns"""
        return {
            'debugging_patterns': {
                'isolate_problem': ['reproduce_issue', 'identify_scope', 'check_logs'],
                'root_cause_analysis': ['hypothesis_generation', 'experimentation', 'validation'],
                'solution_implementation': ['fix_implementation', 'test_fix', 'deploy_solution']
            },
            'optimization_patterns': {
                'performance_analysis': ['profiling', 'bottleneck_identification', 'metric_establishment'],
                'optimization_strategies': ['algorithm_improvement', 'resource_optimization', 'caching'],
                'validation': ['performance_testing', 'comparison_metrics', 'stability_checking']
            },
            'architecture_patterns': {
                'design_principles': ['modularity', 'scalability', 'maintainability', 'security'],
                'pattern_selection': ['monolithic', 'microservices', 'event_driven', 'serverless'],
                'implementation_guidance': ['technology_stack', 'deployment_strategy', 'monitoring_setup']
            }
        }

class TechnicalReasoningEngine(nn.Module):
    """Technical reasoning engine with dynamic learning"""
    
    def __init__(self, config: TechnicalOperActionConfig):
        super().__init__()
        self.config = config
        self.knowledge_graph = TechnicalKnowledgeGraph()
        
        # Technical reasoning networks
        self.technical_concept_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        
        self.problem_classifier = nn.Sequential(
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
        
        self.complexity_assessor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # beginner, intermediate, advanced
            nn.Softmax(dim=-1)
        )
        
        # Dynamic adaptation
        self.dynamic_adapter = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Learned patterns
        self.learned_solutions = []
        self.successful_resolutions = []
        
        # Performance tracking
        self.performance_stats = {
            'total_issues': 0,
            'resolved_issues': 0,
            'avg_complexity': 'intermediate',
            'common_technical_patterns': {}
        }
    
    def extract_technical_concepts(self, text: str, abstraction_data: Dict = None) -> Dict[str, Any]:
        """Extract technical concepts from text"""
        technical_concepts = {
            'technical_domains': [],
            'problem_indicators': [],
            'technology_stack': [],
            'error_messages': [],
            'complexity_indicators': [],
            'urgency_level': 'medium'
        }
        
        text_lower = text.lower()
        
        # Detect technical domains
        for domain, info in self.config.technical_knowledge_base['technical_domains'].items():
            if any(concept in text_lower for concept in info['concepts']):
                domain_concepts = [c for c in info['concepts'] if c in text_lower]
                technical_concepts['technical_domains'].append({
                    'domain': domain,
                    'concepts_found': domain_concepts,
                    'potential_issues': info['common_issues']
                })
        
        # Extract technology stack
        tech_keywords = {
            'programming_languages': ['python', 'javascript', 'java', 'c++', 'c#', 'go', 'rust', 'ruby'],
            'frameworks': ['react', 'angular', 'django', 'flask', 'spring', 'tensorflow', 'pytorch'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch'],
            'tools': ['docker', 'kubernetes', 'aws', 'azure', 'git', 'jenkins']
        }
        
        for category, items in tech_keywords.items():
            found_items = [item for item in items if item in text_lower]
            if found_items:
                technical_concepts['technology_stack'].append({
                    'category': category,
                    'items': found_items
                })
        
        # Extract error messages
        error_patterns = [r'error: (.+)', r'exception: (.+)', r'failed to (.+)', r'cannot (.+)']
        for pattern in error_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            technical_concepts['error_messages'].extend(matches)
        
        # Determine urgency
        for level, indicators in self.config.technical_knowledge_base['urgency_categories'].items():
            if any(indicator in text_lower for indicator in indicators):
                technical_concepts['urgency_level'] = level
                break
        
        # Determine complexity
        word_count = len(re.findall(r'\b\w+\b', text_lower))
        technical_concepts['complexity_indicators'].append(f'word_count: {word_count}')
        
        return technical_concepts
    
    def analyze_technical_problem(self, concepts: Dict, features: torch.Tensor) -> List[Dict]:
        """Analyze technical problems"""
        problems = []
        
        for domain_info in concepts.get('technical_domains', []):
            domain = domain_info['domain']
            domain_data = self.config.technical_knowledge_base['technical_domains'].get(domain, {})
            
            # Check for common issues
            for potential_issue in domain_data.get('common_issues', []):
                if any(keyword in str(concepts).lower() for keyword in potential_issue.split('_')):
                    confidence = 0.5
                    
                    # Neural analysis
                    if features is not None:
                        problem_features = self.problem_classifier(features[:128])
                        confidence = problem_features.mean().item()
                    
                    # Apply dynamic adjustment
                    if self.config.enable_dynamic_adaptation and features is not None:
                        dynamic_adjustment = self.dynamic_adapter(features)
                        adjustment_factor = torch.sigmoid(dynamic_adjustment.mean()).item()
                        confidence = min(confidence * (1 + adjustment_factor * 0.3), 1.0)
                    
                    # Assess complexity
                    complexity = self._assess_complexity(potential_issue, concepts, features)
                    
                    problems.append({
                        'domain': domain,
                        'issue': potential_issue.replace('_', ' ').title(),
                        'confidence': confidence,
                        'complexity': complexity,
                        'urgency': concepts.get('urgency_level', 'medium'),
                        'description': f'Potential {potential_issue.replace("_", " ")} in {domain}',
                        'evidence_sources': ['knowledge_base']
                    })
        
        # Sort by confidence
        problems.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Update stats
        self.performance_stats['total_issues'] += 1
        if problems and problems[0]['confidence'] > self.config.confidence_threshold:
            self.performance_stats['resolved_issues'] += 1
        
        return problems[:5]  # Top 5 problems
    
    def _assess_complexity(self, issue: str, concepts: Dict, features: torch.Tensor) -> str:
        """Assess technical complexity"""
        # Check predefined complexity levels
        for level, indicators in self.config.technical_knowledge_base['complexity_levels'].items():
            if any(indicator in issue for indicator in indicators):
                return level
        
        # Neural complexity assessment
        if features is not None:
            complexity_scores = self.complexity_assessor(features)
            levels = ['beginner', 'intermediate', 'advanced']
            max_idx = torch.argmax(complexity_scores[0]).item()
            if max_idx < len(levels):
                return levels[max_idx]
        
        # Fallback based on urgency
        if concepts.get('urgency_level') == 'critical':
            return 'advanced'
        elif concepts.get('urgency_level') == 'high':
            return 'intermediate'
        else:
            return 'beginner'
    
    def generate_solutions(self, problems: List[Dict], concepts: Dict, features: torch.Tensor) -> List[Dict]:
        """Generate technical solutions"""
        solutions = []
        
        for problem in problems[:3]:  # Top 3 problems
            domain = problem['domain']
            issue = problem['issue']
            
            # Get solution patterns from knowledge base
            domain_data = self.config.technical_knowledge_base['technical_domains'].get(domain, {})
            solution_patterns = domain_data.get('solution_patterns', [])
            
            # Neural solution generation
            solution_details = []
            if features is not None:
                solution_scores = self.solution_generator(features[:96])
                solution_scores = solution_scores[0].tolist()
                
                for i, pattern in enumerate(solution_patterns[:len(solution_scores)]):
                    confidence = solution_scores[i] if i < len(solution_scores) else 0.5
                    
                    # Get specific steps for this pattern
                    steps = self._get_solution_steps(pattern, issue, domain)
                    
                    solution_details.append({
                        'pattern': pattern,
                        'confidence': confidence,
                        'steps': steps,
                        'estimated_time': self._estimate_time(pattern, problem['complexity'])
                    })
            else:
                # Fallback to static patterns
                for pattern in solution_patterns[:3]:
                    steps = self._get_solution_steps(pattern, issue, domain)
                    solution_details.append({
                        'pattern': pattern,
                        'confidence': 0.5,
                        'steps': steps,
                        'estimated_time': self._estimate_time(pattern, problem['complexity'])
                    })
            
            solutions.append({
                'problem': issue,
                'domain': domain,
                'complexity': problem['complexity'],
                'urgency': problem['urgency'],
                'solution_patterns': solution_details,
                'additional_considerations': self._get_additional_considerations(problem, concepts)
            })
        
        return solutions
    
    def _get_solution_steps(self, pattern: str, issue: str, domain: str) -> List[str]:
        """Get specific steps for a solution pattern"""
        # Check knowledge graph for pattern-specific steps
        if pattern in self.knowledge_graph.solution_patterns:
            return self.knowledge_graph.solution_patterns[pattern].get('steps', [])
        
        # Fallback steps
        pattern_steps = {
            'debugging': ['Reproduce the issue', 'Check logs and error messages', 'Isolate the problem', 'Implement fix', 'Test solution'],
            'optimization': ['Profile performance', 'Identify bottlenecks', 'Implement optimizations', 'Test performance gains'],
            'refactoring': ['Analyze current code', 'Design improved structure', 'Implement changes incrementally', 'Test thoroughly'],
            'troubleshooting': ['Identify symptoms', 'Check system status', 'Test components', 'Apply fixes', 'Verify resolution']
        }
        
        return pattern_steps.get(pattern, ['Analyze problem', 'Develop solution', 'Implement', 'Test', 'Document'])
    
    def _estimate_time(self, pattern: str, complexity: str) -> str:
        """Estimate time for solution"""
        time_estimates = {
            'debugging': {'beginner': '1-2 hours', 'intermediate': '2-4 hours', 'advanced': '4-8+ hours'},
            'optimization': {'beginner': '2-4 hours', 'intermediate': '4-8 hours', 'advanced': '1-3 days'},
            'refactoring': {'beginner': '4-8 hours', 'intermediate': '1-2 days', 'advanced': '3-5 days'},
            'troubleshooting': {'beginner': '1-3 hours', 'intermediate': '3-6 hours', 'advanced': '6-12 hours'}
        }
        
        return time_estimates.get(pattern, {}).get(complexity, 'variable')
    
    def _get_additional_considerations(self, problem: Dict, concepts: Dict) -> List[str]:
        """Get additional technical considerations"""
        considerations = []
        
        if problem['urgency'] == 'critical':
            considerations.append('Immediate action required')
            considerations.append('Consider rollback or hotfix')
        
        if problem['complexity'] == 'advanced':
            considerations.append('May require senior technical expertise')
            considerations.append('Consider phased implementation')
        
        if concepts.get('technology_stack'):
            considerations.append('Consider technology-specific best practices')
        
        return considerations
    
    def generate_technical_advice(self, problems: List[Dict], solutions: List[Dict], 
                                concepts: Dict) -> str:
        """Generate technical advice"""
        
        if not problems:
            return "No specific technical issues identified. Please provide more details about the technical problem you're facing."
        
        primary_problem = problems[0]
        advice_parts = []
        
        # Summary
        advice_parts.append(f"**Technical Analysis:**")
        advice_parts.append(f"- Issue: {primary_problem['issue']}")
        advice_parts.append(f"- Domain: {primary_problem['domain'].replace('_', ' ').title()}")
        advice_parts.append(f"- Complexity: {primary_problem['complexity'].title()}")
        advice_parts.append(f"- Urgency: {primary_problem['urgency'].title()}")
        
        # Recommended approach
        if solutions:
            primary_solution = solutions[0]
            advice_parts.append(f"\n**Recommended Approach:**")
            
            for pattern in primary_solution['solution_patterns'][:2]:
                advice_parts.append(f"- {pattern['pattern'].title()}: Confidence {pattern['confidence']:.1%}")
                advice_parts.append(f"  Estimated time: {pattern['estimated_time']}")
                advice_parts.append(f"  Key steps:")
                for i, step in enumerate(pattern['steps'][:3], 1):
                    advice_parts.append(f"    {i}. {step}")
        
        # Additional considerations
        if solutions and solutions[0].get('additional_considerations'):
            advice_parts.append(f"\n**Important Considerations:**")
            for consideration in solutions[0]['additional_considerations'][:3]:
                advice_parts.append(f"- {consideration}")
        
        # Next steps
        advice_parts.append(f"\n**Next Steps:**")
        advice_parts.append("1. Gather all relevant error messages and logs")
        advice_parts.append("2. Implement recommended approach systematically")
        advice_parts.append("3. Test thoroughly before deployment")
        advice_parts.append("4. Document changes and lessons learned")
        
        # Resources
        advice_parts.append(f"\n**Additional Resources:**")
        advice_parts.append("- Official documentation for relevant technologies")
        advice_parts.append("- Community forums and Stack Overflow")
        advice_parts.append("- Consider peer review for complex issues")
        
        return "\n".join(advice_parts)

class TechnicalActionVNI(SpecializedVNIBase):
    """Hybrid Technical VNI with static knowledge + dynamic adaptation"""
    
    def __init__(self, config: TechnicalOperActionConfig = None):
        config_obj = config or TechnicalOperActionConfig()
        super().__init__(topic_name="technical", config=config_obj.__dict__)
        
        self.config = config_obj
        self.reasoning_engine = TechnicalReasoningEngine(self.config)
        
        # Technical-specific dynamic adapter
        self.technical_adapter = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Replace base adapter
        self.dynamic_adapter = self.technical_adapter
        
        logger.info(f"✅ Hybrid Technical VNI initialized: {self.config.vni_id}")
        logger.info(f"   Dynamic adaptation: {self.config.enable_dynamic_adaptation}")
    
    def forward(self, base_features: Dict, input_data: Any) -> Dict:
        """Process technical input with hybrid analysis"""
        
        # Extract text
        text = self._extract_text(input_data)
        
        # Extract technical concepts
        abstraction_data = base_features.get('abstraction_levels', {})
        technical_concepts = self.reasoning_engine.extract_technical_concepts(text, abstraction_data)
        
        # Get features
        features = self._extract_features(base_features)
        
        # Analyze problems
        problems = self.reasoning_engine.analyze_technical_problem(technical_concepts, features)
        
        # Generate solutions
        solutions = self.reasoning_engine.generate_solutions(problems, technical_concepts, features)
        
        # Generate advice
        technical_advice = self.reasoning_engine.generate_technical_advice(problems, solutions, technical_concepts)
        
        # Compile result
        base_result = {
            'technical_analysis': {
                'problems_identified': problems,
                'solutions': solutions,
                'technical_concepts': technical_concepts,
                'technology_stack': technical_concepts.get('technology_stack', []),
                'error_messages': technical_concepts.get('error_messages', [])
            },
            'technical_advice': technical_advice,
            'confidence_score': problems[0]['confidence'] if problems else 0.0,
            'processing_metadata': {
                'technical_domains': len(technical_concepts.get('technical_domains', [])),
                'problems_found': len(problems),
                'urgency_level': technical_concepts.get('urgency_level', 'medium'),
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
            'vni_type': 'operAction_technical_hybrid',
            'processing_stages': ['concept_extraction', 'problem_analysis', 
                                'solution_generation', 'complexity_assessment'],
            'success': True,
            'domain': 'technical',
            'hybrid_system': True,
            'static_knowledge_used': True,
            'dynamic_learning_enabled': self.config.enable_dynamic_adaptation
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
