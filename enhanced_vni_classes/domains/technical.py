# enhanced_vni_classes/domains/technical.py - ENHANCED VERSION with Memory Integration
import re
import time
import json
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from specialized_vni_base import SpecializedVNIBase
from neuron.vni_memory import VNIMemory  # Memory toolkit
from neuron.demoHybridAttention import DemoHybridAttention
from neuron.smart_activation_router import SmartActivationRouter, FunctionRegistry

logger = logging.getLogger("operAction_technical_enhanced")

@dataclass
class TechnicalOperActionConfig:
    """Configuration for technical operAction VNI"""
    vni_id: str = "operAction_technical_enhanced"
    reasoning_depth: str = "comprehensive"
    technical_knowledge_base: Dict[str, Any] = None
    confidence_threshold: float = 0.6
    enable_dynamic_adaptation: bool = True
    enable_memory_integration: bool = True
    memory_retention_days: int = 30
    enable_hybrid_attention: bool = True  # NEW: Hybrid attention toggle
    enable_smart_routing: bool = True  # NEW: Smart routing toggle
    
    def __post_init__(self):
        if self.technical_knowledge_base is None:
            self.technical_knowledge_base = {
                'technical_domains': {
                    'software_development': {
                        'concepts': ['code', 'programming', 'software', 'algorithm', 'bug', 'debug', 'git', 'repository'],
                        'common_issues': ['syntax_error', 'runtime_error', 'performance_issue', 'security_vulnerability', 'dependency_conflict'],
                        'solution_patterns': ['refactoring', 'debugging', 'optimization', 'testing', 'migration'],
                        'complexity_factors': ['algorithm_complexity', 'system_architecture', 'integration_points']
                    },
                    'devops_infrastructure': {
                        'concepts': ['server', 'database', 'network', 'configuration', 'deployment', 'backup', 'kubernetes', 'docker'],
                        'common_issues': ['system_crash', 'performance_bottleneck', 'connectivity_issue', 'security_breach', 'scaling_issue'],
                        'solution_patterns': ['troubleshooting', 'scaling', 'monitoring', 'recovery', 'automation'],
                        'complexity_factors': ['system_scale', 'high_availability', 'disaster_recovery']
                    },
                    'data_science_ai': {
                        'concepts': ['data', 'analysis', 'model', 'machine_learning', 'statistics', 'visualization', 'neural_network'],
                        'common_issues': ['data_quality', 'model_overfitting', 'computational_complexity', 'interpretation', 'training_instability'],
                        'solution_patterns': ['data_cleaning', 'feature_engineering', 'model_selection', 'validation', 'hyperparameter_tuning'],
                        'complexity_factors': ['data_volume', 'model_complexity', 'computation_requirements']
                    },
                    'cybersecurity': {
                        'concepts': ['security', 'encryption', 'authentication', 'vulnerability', 'firewall', 'malware', 'pentest'],
                        'common_issues': ['unauthorized_access', 'data_breach', 'dos_attack', 'phishing', 'configuration_misstep'],
                        'solution_patterns': ['risk_assessment', 'penetration_testing', 'incident_response', 'hardening', 'monitoring'],
                        'complexity_factors': ['attack_surface', 'compliance_requirements', 'threat_sophistication']
                    },
                    'cloud_computing': {
                        'concepts': ['aws', 'azure', 'gcp', 'cloud', 'serverless', 'microservices', 'containers'],
                        'common_issues': ['cost_optimization', 'vendor_lockin', 'multi_cloud_complexity', 'latency_issues'],
                        'solution_patterns': ['architecture_review', 'cost_analysis', 'performance_optimization', 'migration_planning'],
                        'complexity_factors': ['multi_cloud', 'hybrid_environment', 'legacy_integration']
                    }
                },
                'complexity_assessment': {
                    'beginner': {
                        'indicators': ['single_file', 'basic_syntax', 'routine_configuration', 'documentation_issue'],
                        'estimated_time': 'minutes_to_hours'
                    },
                    'intermediate': {
                        'indicators': ['multiple_modules', 'algorithm_design', 'system_optimization', 'data_modeling'],
                        'estimated_time': 'hours_to_days'
                    },
                    'advanced': {
                        'indicators': ['distributed_system', 'architectural_design', 'ml_model_training', 'security_auditing'],
                        'estimated_time': 'days_to_weeks'
                    },
                    'expert': {
                        'indicators': ['enterprise_architecture', 'research_problem', 'novel_algorithm', 'crisis_management'],
                        'estimated_time': 'weeks_to_months'
                    }
                },
                'urgency_matrix': {
                    'critical': {
                        'indicators': ['system_down', 'security_breach', 'data_loss', 'production_outage'],
                        'response_time': 'immediate',
                        'priority': 1
                    },
                    'high': {
                        'indicators': ['performance_degradation', 'functionality_broken', 'deadline_imminent', 'customer_blocked'],
                        'response_time': '< 4 hours',
                        'priority': 2
                    },
                    'medium': {
                        'indicators': ['optimization_needed', 'feature_development', 'code_review', 'technical_debt'],
                        'response_time': '< 24 hours',
                        'priority': 3
                    },
                    'low': {
                        'indicators': ['general_inquiry', 'best_practices', 'learning_question', 'planning_discussion'],
                        'response_time': 'when_possible',
                        'priority': 4
                    }
                },
                'technology_patterns': {
                    'stack_combinations': {
                        'web_development': ['javascript', 'react', 'nodejs', 'mongodb'],
                        'data_science': ['python', 'pandas', 'tensorflow', 'jupyter'],
                        'enterprise': ['java', 'spring', 'oracle', 'kubernetes'],
                        'mobile': ['swift', 'kotlin', 'react_native', 'firebase']
                    },
                    'anti_patterns': {
                        'spaghetti_code': ['tight_coupling', 'global_state', 'duplicate_logic'],
                        'premature_optimization': ['micro_optimizations', 'complexity_without_need'],
                        'over_engineering': ['unnecessary_abstractions', 'future_proofing_excess']
                    }
                }
            }

class TechnicalKnowledgeGraph:
    """Enhanced technical knowledge graph with memory integration"""
    
    def __init__(self, config: TechnicalOperActionConfig = None):
        self.config = config or TechnicalOperActionConfig()
        self.technical_ontology = self.build_enhanced_ontology()
        self.solution_patterns = self.load_solution_patterns()
        self.best_practices = self.load_best_practices()
        self.technology_taxonomy = self.build_technology_taxonomy()
    
    def build_enhanced_ontology(self):
        """Build comprehensive technical concept relationships"""
        return {
            'software_development': {
                'concepts': ['variable', 'function', 'class', 'module', 'package', 'framework', 'library', 'sdk'],
                'relationships': {
                    'bug': ['causes', 'symptoms', 'solutions', 'prevention'],
                    'performance': ['metrics', 'bottlenecks', 'optimizations', 'monitoring'],
                    'testing': ['unit_tests', 'integration_tests', 'e2e_tests', 'test_coverage'],
                    'architecture': ['design_patterns', 'separation_of_concerns', 'scalability_patterns']
                },
                'quality_attributes': ['maintainability', 'testability', 'deployability', 'observability'],
                'maturity_levels': ['prototype', 'mvp', 'production_ready', 'enterprise_grade']
            },
            'devops_infrastructure': {
                'concepts': ['server', 'load_balancer', 'database', 'cache', 'monitoring', 'logging', 'alerting'],
                'relationships': {
                    'scalability': ['horizontal', 'vertical', 'load_distribution', 'auto_scaling'],
                    'reliability': ['redundancy', 'backups', 'failover', 'disaster_recovery'],
                    'security': ['authentication', 'authorization', 'encryption', 'auditing'],
                    'observability': ['metrics', 'logs', 'traces', 'dashboards']
                },
                'operational_excellence': ['automation', 'documentation', 'runbooks', 'post_mortems'],
                'infrastructure_as_code': ['terraform', 'cloudformation', 'ansible', 'chef']
            },
            'data_science_ai': {
                'concepts': ['dataset', 'feature', 'model', 'training', 'validation', 'prediction', 'inference'],
                'relationships': {
                    'modeling': ['supervised', 'unsupervised', 'reinforcement', 'semi_supervised'],
                    'evaluation': ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'],
                    'optimization': ['hyperparameter_tuning', 'regularization', 'ensemble_methods', 'transfer_learning'],
                    'mlops': ['model_serving', 'monitoring', 'retraining', 'versioning']
                },
                'data_quality': ['completeness', 'accuracy', 'consistency', 'timeliness'],
                'ethical_considerations': ['bias_detection', 'fairness', 'explainability', 'privacy']
            },
            'cross_domain_patterns': {
                'problem_solving': ['divide_and_conquer', 'pattern_recognition', 'abstraction', 'system_thinking'],
                'collaboration': ['code_review', 'pair_programming', 'knowledge_sharing', 'documentation'],
                'learning': ['deliberate_practice', 'feedback_loops', 'knowledge_gaps', 'skill_progression']
            }
        }
    
    def load_solution_patterns(self):
        """Load comprehensive technical solution patterns"""
        return {
            'debugging_patterns': {
                'systematic_debugging': ['reproduce_issue', 'isolate_scope', 'check_logs', 'formulate_hypothesis', 'test_hypothesis'],
                'root_cause_analysis': ['five_whys', 'fishbone_diagram', 'fault_tree_analysis'],
                'preventive_measures': ['logging', 'monitoring', 'alerting', 'unit_tests', 'code_reviews']
            },
            'optimization_patterns': {
                'performance_analysis': ['profiling', 'bottleneck_identification', 'metric_establishment', 'benchmarking'],
                'optimization_strategies': ['algorithm_improvement', 'data_structure_selection', 'caching', 'parallelization'],
                'validation': ['performance_testing', 'comparison_metrics', 'stability_checking', 'regression_testing']
            },
            'architecture_patterns': {
                'design_principles': ['single_responsibility', 'open_closed', 'dependency_inversion', 'interface_segregation'],
                'pattern_selection': ['monolithic', 'microservices', 'event_driven', 'serverless', 'hexagonal'],
                'implementation_guidance': ['technology_stack', 'deployment_strategy', 'scaling_approach', 'monitoring_setup']
            },
            'refactoring_patterns': {
                'code_smells': ['long_method', 'large_class', 'duplicate_code', 'feature_envy'],
                'refactoring_techniques': ['extract_method', 'move_method', 'rename_variable', 'introduce_parameter_object'],
                'safety_measures': ['test_coverage', 'small_steps', 'continuous_integration', 'backup_points']
            }
        }
    
    def load_best_practices(self):
        """Load technology-specific best practices"""
        return {
            'version_control': ['feature_branches', 'pull_requests', 'semantic_versioning', 'changelog_maintenance'],
            'testing': ['test_pyramid', 'test_driven_development', 'property_based_testing', 'mutation_testing'],
            'deployment': ['blue_green_deployment', 'canary_releases', 'feature_flags', 'dark_launching'],
            'documentation': ['readme_first', 'api_documentation', 'architecture_decision_records', 'runbooks'],
            'security': ['least_privilege', 'defense_in_depth', 'security_by_design', 'regular_audits']
        }
    
    def build_technology_taxonomy(self):
        """Build taxonomy of technologies and their relationships"""
        return {
            'programming_languages': {
                'python': ['data_science', 'web_development', 'automation'],
                'javascript': ['frontend', 'backend', 'mobile'],
                'java': ['enterprise', 'android', 'big_data'],
                'go': ['cloud_services', 'microservices', 'devops_tools'],
                'rust': ['systems_programming', 'webassembly', 'performance_critical']
            },
            'frameworks': {
                'frontend': ['react', 'angular', 'vue', 'svelte'],
                'backend': ['spring', 'django', 'express', 'flask'],
                'mobile': ['react_native', 'flutter', 'swiftui', 'android_jetpack'],
                'ml': ['tensorflow', 'pytorch', 'scikit_learn', 'huggingface']
            },
            'databases': {
                'relational': ['postgresql', 'mysql', 'oracle', 'sql_server'],
                'nosql': ['mongodb', 'cassandra', 'redis', 'elasticsearch'],
                'new_sql': ['cockroachdb', 'tidb', 'yugabyte'],
                'time_series': ['influxdb', 'timescaledb', 'prometheus']
            },
            'cloud_platforms': {
                'aws': ['ec2', 's3', 'lambda', 'rds'],
                'azure': ['vm', 'blob_storage', 'functions', 'cosmosdb'],
                'gcp': ['compute_engine', 'cloud_storage', 'cloud_functions', 'bigquery'],
                'multi_cloud': ['terraform', 'pulumi', 'crossplane', 'anthos']
            }
        }
    
    def find_similar_issues(self, current_issue: Dict, memory_toolkit) -> List[Dict]:
        """Find similar past issues using memory toolkit"""
        if not memory_toolkit:
            return []
        
        # Create issue signature
        issue_signature = {
            'domain': current_issue.get('domain', ''),
            'primary_symptoms': current_issue.get('symptoms', []),
            'technology_stack': current_issue.get('technologies', []),
            'complexity_level': current_issue.get('complexity', 'intermediate')
        }
        
        # Query memory for similar issues
        similar_issues = memory_toolkit.retrieve_similar(
            category='technical_issues',
            query=issue_signature,
            similarity_threshold=0.6
        )
        
        # Filter and format results
        formatted_results = []
        for issue in similar_issues[:3]:  # Top 3 similar issues
            formatted_results.append({
                'past_issue': issue.get('description', ''),
                'solution_applied': issue.get('solution', ''),
                'effectiveness': issue.get('effectiveness_score', 0.5),
                'relevance_score': issue.get('similarity_score', 0.5),
                'learned_pattern': issue.get('pattern_identified', '')
            })
        
        return formatted_results
    
    def generate_contextual_advice(self, problem: Dict, past_issues: List[Dict], 
                                 technology_context: Dict) -> List[str]:
        """Generate contextual advice based on past issues and technology context"""
        advice_items = []
        
        # Add advice from similar past issues
        for past_issue in past_issues:
            if past_issue['relevance_score'] > 0.7:
                advice_items.append(
                    f"Similar past issue resolved with: {past_issue['solution_applied']} "
                    f"(effectiveness: {past_issue['effectiveness']:.0%})"
                )
        
        # Add technology-specific advice
        tech_stack = technology_context.get('identified_technologies', [])
        for tech in tech_stack:
            if tech in self.technology_taxonomy.get('programming_languages', {}):
                advice_items.append(f"Consider {tech}-specific best practices and patterns")
        
        # Add complexity-based advice
        complexity = problem.get('complexity', 'intermediate')
        if complexity == 'advanced':
            advice_items.append("Consider breaking down into smaller, manageable components")
            advice_items.append("Implement comprehensive monitoring from the start")
        elif complexity == 'beginner':
            advice_items.append("Focus on understanding core concepts before optimization")
            advice_items.append("Leverage established libraries and frameworks")
        
        return list(set(advice_items))[:5]  # Unique, top 5 items

class TechnicalReasoningEngine(nn.Module):
    """Enhanced technical reasoning engine with memory integration"""
    
    def __init__(self, config: TechnicalOperActionConfig, memory_toolkit=None):
        super().__init__()
        self.config = config
        self.knowledge_graph = TechnicalKnowledgeGraph(config)
        self.memory_toolkit = memory_toolkit
        
        # Initialize hybrid attention router
        if config.enable_hybrid_attention:
            self.hybrid_attention = DemoHybridAttention(
            dim=256,
            num_heads=8,
            window_size=256,
            use_sliding=True,
            use_global=True,
            use_hierarchical=True
        )
        
        # Initialize smart activation router
        if config.enable_smart_routing:
            self.activation_router = SmartActivationRouter(
                vni_id=self.vni_id or "technical_vni_enhanced",  # ADD vni_id
                domain="technical",  # ADD domain
                input_dim=512,
                num_experts=4,
                expert_dim=256
            )

    def _register_technical_functions(self):
        """Register technical-specific functions with activation router"""
        if not hasattr(self, 'activation_router'):
            return
        
        # Register technical analysis function
        self.activation_router.register_function(
            function_name="analyze_technical_problem",
            function=self.reasoning_engine.analyze_technical_problem,
            domain=self.domain,
            priority=1
        )
        
        # Register solution generation function
        self.activation_router.register_function(
            function_name="generate_technical_solutions",
            function=self.reasoning_engine.generate_solutions,
            domain="technical",
            priority=2
        )
        
        # Register complexity assessment function
        self.activation_router.register_function(
            function_name="assess_technical_complexity",
            function=self.reasoning_engine._assess_enhanced_complexity,
            domain="technical",
            priority=1
        )    
        logger.info(f"Registered {len(self.activation_router.get_registered_functions())} technical functions")        

        # Enhanced neural networks with attention mechanisms
        self.technical_concept_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        
        # Multi-head attention for complex pattern recognition
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        self.problem_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.Softmax(dim=-1)
        )
        
        # Solution generator with residual connections
        self.solution_generator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(96, 64),
                nn.ReLU(),
                nn.LayerNorm(64)
            ),
            nn.Sequential(
                nn.Linear(64, 48),
                nn.ReLU(),
                nn.LayerNorm(48)
            ),
            nn.Sequential(
                nn.Linear(48, 32),
                nn.Sigmoid()
            )
        ])
        
        # Complexity assessor with multi-dimensional output
        self.complexity_assessor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # beginner, intermediate, advanced, expert
            nn.Softmax(dim=-1)
        )
        
        # Risk assessment module
        self.risk_assessor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Sigmoid()
        )
        
        # Dynamic adaptation with memory context
        self.dynamic_adapter = nn.Sequential(
            nn.Linear(256 + 64, 128),  # +64 for memory context
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Memory context encoder
        self.memory_context_encoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh()
        )
        
        # Performance tracking with memory integration
        self.performance_stats = {
            'total_issues': 0,
            'resolved_with_memory': 0,
            'avg_resolution_time': 0,
            'complexity_distribution': {},
            'technology_patterns': {},
            'learned_solutions': []
        }
        
        # Cache for frequently used patterns
        self.pattern_cache = {}
        
    def extract_technical_concepts(self, text: str, abstraction_data: Dict = None) -> Dict[str, Any]:
        """Extract technical concepts with enhanced pattern recognition"""
        technical_concepts = {
            'technical_domains': [],
            'problem_indicators': [],
            'technology_stack': [],
            'error_patterns': [],
            'complexity_signals': [],
            'urgency_signals': [],
            'query_intent': 'unknown',
            'contextual_cues': []
        }
        
        text_lower = text.lower()
        
        # Enhanced domain detection with pattern matching
        for domain, info in self.config.technical_knowledge_base['technical_domains'].items():
            domain_concepts = [c for c in info['concepts'] if c in text_lower]
            if domain_concepts:
                # Calculate domain relevance score
                relevance = len(domain_concepts) / len(info['concepts'])
                
                technical_concepts['technical_domains'].append({
                    'domain': domain,
                    'concepts_found': domain_concepts,
                    'relevance_score': min(relevance * 1.5, 1.0),  # Boost for multiple matches
                    'potential_issues': info['common_issues'],
                    'complexity_factors': info.get('complexity_factors', [])
                })
        
        # Enhanced technology stack extraction
        tech_categories = self.knowledge_graph.technology_taxonomy
        found_technologies = {}
        
        for category, technologies in tech_categories.items():
            for tech, domains in technologies.items():
                if tech.lower() in text_lower:
                    if category not in found_technologies:
                        found_technologies[category] = []
                    found_technologies[category].append({
                        'technology': tech,
                        'common_domains': domains,
                        'mentions': text_lower.count(tech.lower())
                    })
        
        technical_concepts['technology_stack'] = found_technologies
        
        # Enhanced error pattern extraction
        error_patterns = [
            (r'error(?::|\s+)([\w\s]+)', 'general_error'),
            (r'exception(?::|\s+)([\w\s\.]+)', 'exception'),
            (r'failed to ([\w\s]+)', 'failure'),
            (r'cannot ([\w\s]+)', 'inability'),
            (r'crash(?:ed)?(?:\s+on)?\s+([\w\s]+)', 'crash'),
            (r'timeout(?:\s+on)?\s+([\w\s]+)', 'timeout'),
            (r'deadlock(?:\s+in)?\s+([\w\s]+)', 'deadlock'),
            (r'memory leak(?:\s+in)?\s+([\w\s]+)', 'memory_leak')
        ]
        
        for pattern, error_type in error_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                technical_concepts['error_patterns'].append({
                    'type': error_type,
                    'description': match.strip(),
                    'severity': self._assess_error_severity(error_type)
                })
        
        # Enhanced urgency detection
        for urgency_level, info in self.config.technical_knowledge_base['urgency_matrix'].items():
            urgency_indicators = info['indicators']
            found_indicators = [ind for ind in urgency_indicators if any(word in text_lower for word in ind.split('_'))]
            
            if found_indicators:
                technical_concepts['urgency_signals'].append({
                    'level': urgency_level,
                    'indicators_found': found_indicators,
                    'response_time': info['response_time'],
                    'priority': info['priority']
                })
        
        # Complexity signal extraction
        for complexity_level, info in self.config.technical_knowledge_base['complexity_assessment'].items():
            complexity_indicators = info['indicators']
            found_indicators = [ind for ind in complexity_indicators if any(word in text_lower for word in ind.split('_'))]
            
            if found_indicators:
                technical_concepts['complexity_signals'].append({
                    'level': complexity_level,
                    'indicators_found': found_indicators,
                    'estimated_time': info['estimated_time']
                })
        
        # Query intent classification
        intent_patterns = {
            'problem_solving': ['how to fix', 'how to solve', 'troubleshoot', 'debug', 'error', 'issue'],
            'optimization': ['optimize', 'improve performance', 'speed up', 'reduce latency'],
            'architecture': ['design', 'architecture', 'system design', 'how to structure'],
            'learning': ['how does', 'what is', 'explain', 'understand', 'learn about'],
            'review': ['code review', 'review my code', 'feedback on', 'check this code'],
            'planning': ['how to implement', 'approach for', 'strategy for', 'plan to']
        }
        
        for intent, patterns in intent_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                technical_concepts['query_intent'] = intent
                break
        
        # Contextual cue extraction
        contextual_cues = [
            ('deadline', r'deadline.*?(\d+\s*(?:hour|day|week)s?)', 'time_pressure'),
            ('team_size', r'team of (\d+)', 'collaboration_context'),
            ('experience_level', r'(?:junior|mid-level|senior|expert)', 'skill_context'),
            ('project_phase', r'(?:prototype|mvp|production|legacy)', 'project_context')
        ]
        
        for cue_name, pattern, cue_type in contextual_cues:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                technical_concepts['contextual_cues'].append({
                    'type': cue_type,
                    'name': cue_name,
                    'value': matches[0] if matches else True
                })
        
        # Check memory for similar past queries
        if self.memory_toolkit and self.config.enable_memory_integration:
            similar_past_queries = self.memory_toolkit.retrieve_similar(
                category='technical_queries',
                query=text[:200],  # First 200 chars as query
                similarity_threshold=0.5
            )
            
            if similar_past_queries:
                technical_concepts['similar_past_cases'] = similar_past_queries[:3]
        
        return technical_concepts
    
    def analyze_technical_problem(self, concepts: Dict, features: torch.Tensor) -> List[Dict]:
        """Analyze technical problems with memory-enhanced reasoning"""
        problems = []
        
        # Apply hybrid attention if enabled
        if hasattr(self, 'hybrid_attention') and features is not None:
            # Get memory context
            memory_context = self._get_memory_context(concepts)
            
            if memory_context is not None:
                # Apply hybrid attention
                enhanced_features = self.hybrid_attention(
                    query=features,
                    key=features,
                    value=features,
                    memory_context=memory_context
                )
            else:
                enhanced_features = features
        else:
            enhanced_features = features
        
        # Apply smart routing if enabled
        if hasattr(self, 'smart_router') and enhanced_features is not None:
            # Route to appropriate reasoning strategy
            routed_features, routing_weights = self.smart_router(enhanced_features)
        else:
            routed_features = enhanced_features
        
        for domain_info in concepts.get('technical_domains', []):
            domain = domain_info['domain']
            relevance_score = domain_info['relevance_score']
            
            # Skip low relevance domains
            if relevance_score < 0.3:
                continue
            
            domain_data = self.config.technical_knowledge_base['technical_domains'].get(domain, {})
            
            # Check for common issues with enhanced matching
            for potential_issue in domain_data.get('common_issues', []):
                issue_keywords = potential_issue.split('_')
                
                # Calculate issue matching score
                text_representation = str(concepts).lower()
                keyword_matches = sum(1 for keyword in issue_keywords if keyword in text_representation)
                base_confidence = keyword_matches / max(len(issue_keywords), 1)
                
                # Boost confidence based on domain relevance
                base_confidence *= relevance_score
                
                # Neural analysis with enhanced features
                if routed_features is not None:
                    problem_features = self.problem_classifier(routed_features[:128])
                    neural_confidence = problem_features.mean().item()
                    base_confidence = 0.7 * base_confidence + 0.3 * neural_confidence
                
                # Apply dynamic adjustment
                if self.config.enable_dynamic_adaptation and routed_features is not None:
                    dynamic_adjustment = self.dynamic_adapter(routed_features)
                    adjustment_factor = torch.sigmoid(dynamic_adjustment.mean()).item()
                    base_confidence = min(base_confidence * (1 + adjustment_factor * 0.3), 1.0)
                
                # Only include issues above threshold
                if base_confidence < self.config.confidence_threshold:
                    continue
                
                # Assess complexity with enhanced criteria
                complexity = self._assess_enhanced_complexity(potential_issue, concepts, routed_features)
                
                # Determine urgency
                urgency = self._determine_urgency(concepts)
                
                # Check for similar past issues in memory
                similar_past_issues = []
                if self.memory_toolkit and self.config.enable_memory_integration:
                    current_issue_data = {
                        'domain': domain,
                        'issue': potential_issue,
                        'symptoms': [p['type'] for p in concepts.get('error_patterns', [])],
                        'technologies': self._extract_technologies(concepts),
                        'complexity': complexity
                    }
                    similar_past_issues = self.knowledge_graph.find_similar_issues(
                        current_issue_data, self.memory_toolkit
                    )
                
                problem_data = {
                    'domain': domain,
                    'issue': potential_issue.replace('_', ' ').title(),
                    'confidence': base_confidence,
                    'complexity': complexity,
                    'urgency': urgency,
                    'description': f'Potential {potential_issue.replace("_", " ")} in {domain}',
                    'evidence_sources': ['knowledge_base', 'neural_analysis'],
                    'domain_relevance': relevance_score,
                    'similar_past_issues': similar_past_issues[:2]  # Top 2 similar issues
                }
                
                # Add complexity factors if available
                if 'complexity_factors' in domain_data:
                    problem_data['complexity_factors'] = [
                        factor for factor in domain_data['complexity_factors']
                        if any(word in str(concepts).lower() for word in factor.split('_'))
                    ]
                
                problems.append(problem_data)
        
        # Sort by confidence and relevance
        problems.sort(key=lambda x: (x['confidence'], x['domain_relevance']), reverse=True)
        
        # Update performance stats
        self.performance_stats['total_issues'] += 1
        if problems and problems[0]['confidence'] > self.config.confidence_threshold:
            if problems[0].get('similar_past_issues'):
                self.performance_stats['resolved_with_memory'] += 1
        
        # Update complexity distribution
        for problem in problems[:3]:
            complexity = problem['complexity']
            if complexity not in self.performance_stats['complexity_distribution']:
                self.performance_stats['complexity_distribution'][complexity] = 0
            self.performance_stats['complexity_distribution'][complexity] += 1
        
        return problems[:5]  # Top 5 problems
    
    def _get_memory_context(self, concepts: Dict) -> Optional[torch.Tensor]:
        """Get memory context for current query"""
        if not self.memory_toolkit or not self.config.enable_memory_integration:
            return None
        
        # Extract key concepts for memory query
        query_concepts = []
        
        # Add technical domains
        for domain_info in concepts.get('technical_domains', []):
            query_concepts.extend(domain_info.get('concepts_found', []))
        
        # Add technologies
        for category, tech_list in concepts.get('technology_stack', {}).items():
            for tech_info in tech_list:
                query_concepts.append(tech_info['technology'])
        
        # Add error patterns
        for error in concepts.get('error_patterns', []):
            query_concepts.append(error['type'])
        
        # Query memory
        if query_concepts:
            memory_results = self.memory_toolkit.retrieve_contextual(
                query_terms=query_concepts,
                category='technical_knowledge',
                max_results=5
            )
            
            if memory_results:
                # Encode memory results into tensor
                memory_text = ' '.join([str(r) for r in memory_results[:3]])
                # Simple encoding - in production, use proper text encoding
                return torch.randn(1, 64) * 0.1
        
        return None
    
    def _assess_enhanced_complexity(self, issue: str, concepts: Dict, features: torch.Tensor) -> str:
        """Assess technical complexity with multiple factors"""
        
        # Check predefined complexity levels with weighted scoring
        complexity_scores = {}
        for level, info in self.config.technical_knowledge_base['complexity_assessment'].items():
            indicators = info['indicators']
            # Calculate match score
            match_score = sum(1 for indicator in indicators if any(word in issue for word in indicator.split('_')))
            # Adjust based on contextual cues
            if match_score > 0:
                complexity_scores[level] = match_score / len(indicators)
        
        # Neural complexity assessment
        if features is not None:
            neural_scores = self.complexity_assessor(features)
            levels = ['beginner', 'intermediate', 'advanced', 'expert']
            
            for i, level in enumerate(levels):
                if i < len(neural_scores[0]):
                    current_score = complexity_scores.get(level, 0)
                    neural_score = neural_scores[0][i].item()
                    complexity_scores[level] = 0.6 * current_score + 0.4 * neural_score
        
        # Adjust based on urgency
        urgency_signals = concepts.get('urgency_signals', [])
        if urgency_signals:
            max_urgency = max(urgency_signals, key=lambda x: x.get('priority', 4))
            if max_urgency.get('level') == 'critical':
                # Boost advanced/expert scores
                for level in ['advanced', 'expert']:
                    if level in complexity_scores:
                        complexity_scores[level] = min(complexity_scores[level] * 1.3, 1.0)
        
        # Select highest scoring complexity
        if complexity_scores:
            return max(complexity_scores.items(), key=lambda x: x[1])[0]
        
        # Fallback based on technology stack
        tech_stack_size = sum(len(techs) for techs in concepts.get('technology_stack', {}).values())
        if tech_stack_size > 5:
            return 'advanced'
        elif tech_stack_size > 2:
            return 'intermediate'
        else:
            return 'beginner'
    
    def _determine_urgency(self, concepts: Dict) -> str:
        """Determine urgency based on multiple signals"""
        urgency_signals = concepts.get('urgency_signals', [])
        
        if not urgency_signals:
            return 'medium'
        
        # Find signal with highest priority (lowest number = higher priority)
        highest_priority = min(urgency_signals, key=lambda x: x.get('priority', 4))
        return highest_priority.get('level', 'medium')
    
    def _extract_technologies(self, concepts: Dict) -> List[str]:
        """Extract technology names from concepts"""
        technologies = []
        
        # Extract from technology_stack
        for tech_category in concepts.get('technology_stack', []):
            if 'items' in tech_category:
                technologies.extend(tech_category['items'])
        
        # Extract from technical_domains
        for domain_info in concepts.get('technical_domains', []):
            if 'concepts_found' in domain_info:
                technologies.extend(domain_info['concepts_found'])
        
        # Extract from error messages
        for error_msg in concepts.get('error_messages', []):
            # Look for technology names in error messages
            tech_keywords = ['python', 'javascript', 'java', 'c++', 'react', 'django', 
                           'mysql', 'docker', 'kubernetes', 'aws', 'azure']
            for tech in tech_keywords:
                if tech.lower() in error_msg.lower() and tech not in technologies:
                    technologies.append(tech)
        
        return list(set(technologies))  # Remove duplicates
    
    def generate_solutions(self, problems: List[Dict], concepts: Dict, 
                         features: torch.Tensor) -> List[Dict]:
        """Generate technical solutions with hybrid attention routing"""
        solutions = []
        
        # Apply smart routing for solution generation
        if hasattr(self, 'smart_router') and features is not None:
            solution_features, _ = self.smart_router(features, task='solution_generation')
        else:
            solution_features = features
        
        for problem in problems[:3]:
            # Check cache first
            cache_key = f"{problem['domain']}_{problem['issue']}_{problem['complexity']}"
            if cache_key in self.pattern_cache:
                cached_solution = self.pattern_cache[cache_key].copy()
                cached_solution['confidence'] *= 0.9  # Slight decay for cached solutions
                solutions.append(cached_solution)
                continue
            
            # Generate new solution
            solution = {
                'problem': problem['issue'],
                'domain': problem['domain'],
                'complexity': problem['complexity'],
                'confidence': problem['confidence'] * 0.8,  # Solution confidence is typically lower
                'steps': [],
                'estimated_time': self._estimate_solution_time(problem),
                'resources_needed': [],
                'validation_steps': [],
                'memory_context': problem.get('similar_past_issues', [])
            }
            
            # Generate solution steps based on complexity
            if solution_features is not None:
                step_embeddings = self.solution_generator[0](solution_features[:96])
                num_steps = self._get_step_count(problem['complexity'])
                
                for i in range(num_steps):
                    step = {
                        'id': i + 1,
                        'action': f"Step {i + 1}: Implement solution component {i + 1}",
                        'details': self._get_step_details(i, problem['domain']),
                        'estimated_duration': f"{10 * (i + 1)} minutes"
                    }
                    solution['steps'].append(step)
            
            # Add resources based on technology stack
            technologies = self._extract_technologies(concepts)
            solution['resources_needed'] = self._get_solution_resources(problem['domain'], technologies)
            
            # Add validation steps
            solution['validation_steps'] = self._get_validation_steps(problem['domain'])
            
            # Cache the solution
            self.pattern_cache[cache_key] = solution.copy()
            
            solutions.append(solution)
        
        # Store successful solutions in memory
        if self.memory_toolkit and solutions:
            for solution in solutions[:2]:
                if solution['confidence'] > 0.7:
                    self.memory_toolkit.store_solution(
                        problem_type=solution['problem'],
                        domain=solution['domain'],
                        solution_steps=solution['steps'],
                        effectiveness=0.8  # Initial effectiveness score
                    )
        
        return solutions
    
    def _estimate_solution_time(self, problem: Dict) -> str:
        """Estimate solution time based on complexity"""
        time_estimates = {
            'beginner': '1-2 hours',
            'intermediate': '4-8 hours',
            'advanced': '1-3 days',
            'expert': '1-2 weeks'
        }
        return time_estimates.get(problem['complexity'], 'Unknown')
    
    def _get_step_count(self, complexity: str) -> int:
        """Get number of solution steps based on complexity"""
        step_counts = {
            'beginner': 3,
            'intermediate': 5,
            'advanced': 7,
            'expert': 10
        }
        return step_counts.get(complexity, 5)
    
    def _get_step_details(self, step_num: int, domain: str) -> str:
        """Get detailed description for each step"""
        if domain == 'software_development':
            details = [
                'Analyze the code and understand the issue',
                'Write or modify the necessary code',
                'Test the changes locally',
                'Run unit tests',
                'Submit for code review'
            ]
        elif domain == 'data_science_ai':
            details = [
                'Prepare and clean the data',
                'Select appropriate model architecture',
                'Train the model with proper parameters',
                'Validate model performance',
                'Deploy and monitor the model'
            ]
        else:
            details = [
                'Analyze the problem',
                'Design solution approach',
                'Implement the solution',
                'Test thoroughly',
                'Deploy and monitor'
            ]
        
        return details[step_num % len(details)] if details else f"Step {step_num + 1}"
    
    def _get_solution_resources(self, domain: str, technologies: List[str]) -> List[str]:
        """Get resources needed for solution"""
        resources = []
        
        # Domain-specific resources
        if domain == 'software_development':
            resources.extend(['Code editor/IDE', 'Version control system', 'Testing framework'])
        elif domain == 'data_science_ai':
            resources.extend(['Jupyter notebook', 'Data visualization tools', 'Model training environment'])
        elif domain == 'cybersecurity':
            resources.extend(['Security scanning tools', 'Log analysis tools', 'Monitoring systems'])
        
        # Technology-specific resources
        tech_resources = {
            'python': ['Python interpreter', 'pip package manager'],
            'javascript': ['Node.js', 'npm/yarn'],
            'docker': ['Docker CLI', 'Docker Compose'],
            'aws': ['AWS CLI', 'AWS Console access']
        }
        
        for tech in technologies[:3]:
            if tech.lower() in tech_resources:
                resources.extend(tech_resources[tech.lower()])
        
        return list(set(resources))  # Remove duplicates
    
    def _get_validation_steps(self, domain: str) -> List[str]:
        """Get validation steps for solution"""
        validation_steps = [
            'Test basic functionality',
            'Verify edge cases',
            'Check performance metrics',
            'Review code/implementation quality'
        ]
        
        if domain == 'software_development':
            validation_steps.extend(['Run unit tests', 'Integration testing', 'Code review'])
        elif domain == 'data_science_ai':
            validation_steps.extend(['Model validation metrics', 'Data quality checks', 
                                   'Prediction accuracy testing'])
        elif domain == 'cybersecurity':
            validation_steps.extend(['Security scanning', 'Vulnerability assessment', 
                                   'Penetration testing'])
        
        return validation_steps

from ..core.base_vni import EnhancedBaseVNI
from ..core.capabilities import VNICapabilities, VNIType

class TechnicalVNI(EnhancedBaseVNI):
    """Enhanced Technical VNI with memory integration and hybrid attention"""
    
    def __init__(self, vni_id: str = None, name: str = None, 
                 capabilities: VNICapabilities = None,
                 memory_toolkit: VNIMemory = None):
        # Use technical-specific capabilities
        tech_capabilities = VNICapabilities(
            can_process_text=True,
            can_generate_text=True,
            can_learn=True,
            has_knowledge_base=True,
            technical_expertise=True,
            has_hybrid_attention=True,  # NEW: Hybrid attention capability
            has_smart_routing=True,  # NEW: Smart routing capability
            max_context_length=8192
        )
        self.domain = "technical"
        self.vni_type = "technical"
        self.name = name or "Enhanced Technical VNI"
        self.description = "Enhanced Technical VNI with memory integration and hybrid attention"
        
        super().__init__(
            vni_id=vni_id or "technical_vni_enhanced",
            name=name or "Enhanced Technical VNI",
            capabilities=tech_capabilities
        )
        
        # Initialize technical components
        self.config = TechnicalOperActionConfig()
        self.knowledge_graph = TechnicalKnowledgeGraph(self.config)
        self.memory_toolkit = memory_toolkit or VNIMemory(retention_days=self.config.memory_retention_days)
        
        # Initialize reasoning engine with memory and hybrid components
        self.reasoning_engine = TechnicalReasoningEngine(self.config, self.memory_toolkit)
        
        # Track performance
        self.performance_metrics = {
            'total_queries': 0,
            'successful_resolutions': 0,
            'avg_confidence': 0,
            'memory_hit_rate': 0,
            'attention_patterns': []
        }

    def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process technical queries with enhanced reasoning"""
        self.performance_metrics['total_queries'] += 1
        
        # Extract technical concepts with memory context
        concepts = self.reasoning_engine.extract_technical_concepts(input_text, context)
        
        # Convert to features
        features = self._encode_to_features(input_text, concepts) if torch.is_available() else None
        
        # Analyze problems with hybrid attention
        problems = self.reasoning_engine.analyze_technical_problem(concepts, features)
        
        # Generate solutions with smart routing
        solutions = self.reasoning_engine.generate_solutions(problems, concepts, features)
        
        # Update performance metrics
        if problems and solutions:
            self.performance_metrics['successful_resolutions'] += 1
            self.performance_metrics['avg_confidence'] = (
                (self.performance_metrics['avg_confidence'] * (self.performance_metrics['successful_resolutions'] - 1) +
                 problems[0]['confidence']) / self.performance_metrics['successful_resolutions']
            )
            
            # Track memory usage
            if problems[0].get('similar_past_issues'):
                self.performance_metrics['memory_hit_rate'] += 1
        
        # Store query and solution in memory
        if self.config.enable_memory_integration and solutions:
            self._store_in_memory(input_text, concepts, problems, solutions)
        
        return {
            'query': input_text,
            'technical_domains': [d['domain'] for d in concepts.get('technical_domains', [])],
            'problems_identified': len(problems),
            'primary_problem': problems[0] if problems else None,
            'solutions': solutions,
            'complexity': problems[0]['complexity'] if problems else 'unknown',
            'confidence': problems[0]['confidence'] if problems else 0.0,
            'uses_hybrid_attention': self.config.enable_hybrid_attention,
            'uses_smart_routing': self.config.enable_smart_routing,
            'memory_context_used': bool(problems and problems[0].get('similar_past_issues')),
            'performance_metrics': {
                'query_count': self.performance_metrics['total_queries'],
                'resolution_rate': self.performance_metrics['successful_resolutions'] / max(self.performance_metrics['total_queries'], 1),
                'avg_confidence': self.performance_metrics['avg_confidence'],
                'memory_hit_rate': self.performance_metrics['memory_hit_rate'] / max(self.performance_metrics['total_queries'], 1)
            }
        }
    
    def _encode_to_features(self, text: str, concepts: Dict) -> torch.Tensor:
        """Encode text and concepts to feature tensor"""
        # Simple encoding - in production, use proper text encoder
        # Combine text length, concept count, and random features
        text_length = min(len(text), 512)
        concept_count = len(concepts.get('technical_domains', [])) + len(concepts.get('technology_stack', {}))
        
        # Create feature vector
        features = torch.randn(1, 512)
        
        # Incorporate text length and concept count
        features[0, 0] = text_length / 512
        features[0, 1] = concept_count / 20  # Normalize
        
        return features
    
    def _store_in_memory(self, query: str, concepts: Dict, 
                        problems: List[Dict], solutions: List[Dict]):
        """Store query and solution in memory for future reference"""
        if not self.memory_toolkit:
            return
        
        # Store query pattern
        query_signature = {
            'text': query[:100],  # First 100 chars
            'domains': [d['domain'] for d in concepts.get('technical_domains', [])],
            'technologies': self.reasoning_engine._extract_technologies(concepts),
            'intent': concepts.get('query_intent', 'unknown')
        }
        
        self.memory_toolkit.store_query_pattern(query_signature)
        
        # Store successful solutions
        for solution in solutions[:2]:
            if solution['confidence'] > 0.6:
                solution_data = {
                    'problem': solution['problem'],
                    'domain': solution['domain'],
                    'complexity': solution['complexity'],
                    'steps': solution['steps'],
                    'effectiveness': solution['confidence']
                }
                self.memory_toolkit.store_solution_pattern(solution_data)
        
        # Update memory performance
        self.memory_toolkit.update_performance_stats(
            category='technical',
            success_rate=solutions[0]['confidence'] if solutions else 0
        )
    
    def get_insights(self) -> Dict[str, Any]:
        """Get insights from the technical VNI"""
        return {
            'knowledge_graph_stats': {
                'domains_covered': len(self.knowledge_graph.technical_ontology),
                'solution_patterns': len(self.knowledge_graph.solution_patterns),
                'best_practices': len(self.knowledge_graph.best_practices)
            },
            'reasoning_engine_stats': self.reasoning_engine.performance_stats,
            'performance_metrics': self.performance_metrics,
            'memory_stats': self.memory_toolkit.get_stats() if self.memory_toolkit else {},
            'configuration': {
                'enable_memory': self.config.enable_memory_integration,
                'enable_hybrid_attention': self.config.enable_hybrid_attention,
                'enable_smart_routing': self.config.enable_smart_routing,
                'confidence_threshold': self.config.confidence_threshold
            }
        }
