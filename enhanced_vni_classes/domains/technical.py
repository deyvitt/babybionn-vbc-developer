# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

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
    enable_llm_generation: bool = True  # ADDED: LLM generation toggle
    
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
    
    def __init__(self, config: TechnicalOperActionConfig, memory_toolkit=None, vni_id: str = None):
        super().__init__()
        self.config = config
        self.knowledge_graph = TechnicalKnowledgeGraph(config)
        self.memory_toolkit = memory_toolkit
        self.vni_id = vni_id or "technical_vni_enhanced"  

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
                vni_id=self.vni_id,
                domain="technical",
                input_dim=512,
                num_experts=4,
                expert_dim=256
            )

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
        
        # Register functions if activation router exists
        if hasattr(self, 'activation_router') and self.activation_router:
            self._register_technical_functions()

    def _register_technical_functions(self):
        """Register technical-specific functions with activation router"""
        if not hasattr(self, 'activation_router') or self.activation_router is None:
            return
        
        try:
            # Register technical analysis function
            self.activation_router.register_function(
                function_name="analyze_technical_problem",
                function=self.analyze_technical_problem,
                domain="technical",
                priority=1
            )
            
            # Register complexity assessment function
            self.activation_router.register_function(
                function_name="assess_technical_complexity",
                function=self._assess_enhanced_complexity,
                domain="technical",
                priority=1
            )
            
            # Register analysis for aggregator function
            self.activation_router.register_function(
                function_name="analyze_for_aggregator",
                function=self._analyze_for_aggregator,
                domain="technical",
                priority=2
            )
            
            logger.info(f"Registered {len(self.activation_router.get_registered_functions())} technical functions")
        except Exception as e:
            logger.warning(f"Failed to register technical functions: {e}")
      
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

    def _assess_error_severity(self, error_type: str) -> str:
        """Assess severity of error type"""
        severity_map = {
            'crash': 'critical',
            'deadlock': 'critical',
            'memory_leak': 'high',
            'timeout': 'high',
            'exception': 'medium',
            'failure': 'medium',
            'inability': 'medium',
            'general_error': 'low'
        }
        return severity_map.get(error_type, 'medium')

    def _analyze_for_aggregator(self, problems: List[Dict], concepts: Dict, 
                               features: torch.Tensor) -> List[Dict]:
        """Analyze technical problems for aggregator - NO SOLUTION GENERATION"""
        analyses = []
        
        for problem in problems[:3]:  # Analyze top 3 problems
            analysis = {
                'problem_type': problem['issue'],
                'domain': problem['domain'],
                'complexity': problem['complexity'],
                'confidence': problem['confidence'],
                'urgency': problem.get('urgency', 'medium'),
                'similar_past_issues': problem.get('similar_past_issues', []),
                'technology_context': {
                    'technologies': self._extract_technologies(concepts),
                    'stack_patterns': self._identify_stack_patterns(concepts)
                },
                'key_technical_factors': self._extract_key_factors(problem, concepts),
                'recommended_approach_style': self._get_recommended_style(problem),
                'estimated_time_frame': self._estimate_analysis_time(problem),
                'risk_assessment': self._assess_technical_risks(problem, concepts),
                'validation_requirements': self._get_validation_requirements(problem['domain']),
                'domain_insights': self._get_domain_insights(problem['domain'], concepts),
                'complexity_guidance': self._get_complexity_guidance(problem['complexity'])
            }
            
            analyses.append(analysis)
        
        return analyses

    def _identify_stack_patterns(self, concepts: Dict) -> List[str]:
        """Identify technology stack patterns"""
        patterns = []
        found_techs = []
        
        # Extract all technologies from concepts
        tech_categories = ['programming_languages', 'frameworks', 'databases', 'cloud_platforms']
        for category in tech_categories:
            category_techs = concepts.get('technology_stack', {}).get(category, [])
            if isinstance(category_techs, list):
                for tech in category_techs:
                    if isinstance(tech, dict):
                        tech_name = tech.get('technology', '')
                    else:
                        tech_name = str(tech)
                    if tech_name:
                        found_techs.append(tech_name.lower())
        
        # Check for common stack patterns
        stack_patterns = {
            'python_django_stack': ['python', 'django'],
            'python_flask_stack': ['python', 'flask'],
            'javascript_react_stack': ['javascript', 'react'],
            'javascript_node_stack': ['javascript', 'nodejs', 'node.js'],
            'java_spring_stack': ['java', 'spring'],
            'dotnet_stack': ['c#', '.net', 'asp.net'],
            'mean_stack': ['mongodb', 'express', 'angular', 'nodejs'],
            'mern_stack': ['mongodb', 'express', 'react', 'nodejs'],
            'lamp_stack': ['linux', 'apache', 'mysql', 'php'],
            'python_data_stack': ['python', 'pandas', 'numpy', 'jupyter']
        }
        
        for pattern_name, required_techs in stack_patterns.items():
            matches = sum(1 for tech in required_techs if tech in found_techs)
            if matches >= 2:  # At least 2 technologies from the stack
                patterns.append(pattern_name)
        
        return patterns

    def _extract_key_factors(self, problem: Dict, concepts: Dict) -> List[str]:
        """Extract key technical factors affecting the problem"""
        factors = []
        
        # Complexity factor
        factors.append(f"Complexity level: {problem['complexity']}")
        
        # Technology factor
        tech_count = len(self._extract_technologies(concepts))
        if tech_count > 3:
            factors.append(f"Multi-technology integration ({tech_count} technologies)")
        elif tech_count == 0:
            factors.append("Technology context not specified")
        
        # Domain-specific factors
        domain = problem['domain']
        if domain in ['software_development', 'data_science_ai']:
            factors.append("Requires specialized algorithmic/analytical knowledge")
        elif domain in ['cybersecurity']:
            factors.append("Security, compliance, and risk management considerations")
        elif domain in ['devops_infrastructure']:
            factors.append("System reliability, scalability, and operational requirements")
        elif domain in ['cloud_computing']:
            factors.append("Cloud architecture, cost optimization, and vendor considerations")
        
        # Urgency factor
        urgency = problem.get('urgency', 'medium')
        if urgency in ['critical', 'high']:
            factors.append(f"High urgency level: requires {urgency} priority attention")
        
        # Similar past issues factor
        if problem.get('similar_past_issues'):
            factors.append("Historical patterns available for reference")
        
        # Error patterns factor
        error_patterns = concepts.get('error_patterns', [])
        if error_patterns:
            error_types = set(e['type'] for e in error_patterns)
            if len(error_types) > 1:
                factors.append(f"Multiple error types detected: {', '.join(list(error_types)[:3])}")
        
        return factors

    def _get_recommended_style(self, problem: Dict) -> Dict[str, str]:
        """Get recommended response style for aggregator"""
        styles = {
            'beginner': {
                'tone': 'educational',
                'detail_level': 'step_by_step',
                'assume_knowledge': 'basic',
                'include_examples': True,
                'include_diagrams': False,
                'technical_depth': 'introductory',
                'pacing': 'slow',
                'focus': 'fundamentals'
            },
            'intermediate': {
                'tone': 'collaborative',
                'detail_level': 'conceptual',
                'assume_knowledge': 'intermediate',
                'include_examples': True,
                'include_diagrams': True,
                'technical_depth': 'practical',
                'pacing': 'moderate',
                'focus': 'best_practices'
            },
            'advanced': {
                'tone': 'expert',
                'detail_level': 'architectural',
                'assume_knowledge': 'advanced',
                'include_examples': False,
                'include_diagrams': True,
                'technical_depth': 'detailed',
                'pacing': 'efficient',
                'focus': 'system_design'
            },
            'expert': {
                'tone': 'technical_lead',
                'detail_level': 'strategic',
                'assume_knowledge': 'expert',
                'include_examples': False,
                'include_diagrams': True,
                'technical_depth': 'comprehensive',
                'pacing': 'direct',
                'focus': 'innovation'
            }
        }
        
        return styles.get(problem['complexity'], styles['intermediate'])

    def _assess_technical_risks(self, problem: Dict, concepts: Dict) -> Dict[str, Any]:
        """Assess technical risks for the problem"""
        risks = {
            'implementation_risk': 'medium',
            'time_risk': 'medium',
            'quality_risk': 'medium',
            'security_risk': 'low',
            'dependencies': []
        }
        
        # Adjust based on complexity
        complexity = problem['complexity']
        if complexity in ['advanced', 'expert']:
            risks['implementation_risk'] = 'high'
            risks['time_risk'] = 'high'
        
        # Adjust based on urgency
        urgency = problem.get('urgency', 'medium')
        if urgency in ['critical', 'high']:
            risks['time_risk'] = 'high'
            risks['quality_risk'] = 'high'  # Rushed work may compromise quality
        
        # Adjust based on domain
        domain = problem['domain']
        if domain in ['cybersecurity', 'devops_infrastructure']:
            risks['security_risk'] = 'medium'
        if domain == 'cybersecurity' and urgency == 'critical':
            risks['security_risk'] = 'high'
        
        # Check technology dependencies
        technologies = self._extract_technologies(concepts)
        tech_count = len(technologies)
        
        if tech_count > 3:
            risks['dependencies'].append('Multiple technology integration complexity')
            risks['implementation_risk'] = 'high' if risks['implementation_risk'] == 'medium' else risks['implementation_risk']
        
        if any(tech in ['kubernetes', 'docker', 'aws', 'azure', 'gcp'] for tech in technologies):
            risks['dependencies'].append('Infrastructure/platform dependencies')
        
        # Check for similar past issues
        if problem.get('similar_past_issues'):
            similar_issues = problem['similar_past_issues']
            effectiveness_scores = [issue.get('effectiveness', 0.5) for issue in similar_issues]
            avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else 0.5
            
            if avg_effectiveness < 0.6:
                risks['dependencies'].append('Historical solutions had limited effectiveness')
                risks['quality_risk'] = 'high' if risks['quality_risk'] == 'medium' else risks['quality_risk']
        
        return risks
    
    def _get_validation_requirements(self, domain: str) -> List[str]:
        """Get validation requirements for the domain"""
        requirements = {
            'software_development': [
                'Unit testing coverage',
                'Code review process',
                'Integration testing',
                'Performance benchmarking',
                'Security scanning'
            ],
            'data_science_ai': [
                'Model validation metrics (accuracy, precision, recall)',
                'Data quality verification',
                'Cross-validation results',
                'Bias and fairness assessment',
                'Explainability analysis'
            ],
            'cybersecurity': [
                'Security vulnerability scanning',
                'Penetration testing results',
                'Compliance verification',
                'Risk assessment report',
                'Incident response testing'
            ],
            'devops_infrastructure': [
                'Performance load testing',
                'Disaster recovery testing',
                'High availability verification',
                'Monitoring and alerting validation',
                'Automation reliability testing'
            ],
            'cloud_computing': [
                'Cost optimization analysis',
                'Performance benchmarking',
                'Security compliance check',
                'Multi-region failover testing',
                'Resource utilization optimization'
            ]
        }
        
        return requirements.get(domain, [
            'Functional testing',
            'Performance validation',
            'Security assessment',
            'User acceptance verification'
        ])   

    def _get_complexity_guidance(self, complexity: str) -> Dict[str, str]:
        """Get guidance based on complexity level"""
        guidance = {
            'beginner': {
                'focus': 'Fundamental understanding, clear examples, basic troubleshooting',
                'avoid': 'Advanced optimization, premature abstraction, complex architectures',
                'suggest': 'Start with simple working solution, document each step, validate understanding',
                'resources': 'Official documentation, beginner tutorials, example projects',
                'success_indicators': 'Working solution, clear understanding, ability to explain'
            },
            'intermediate': {
                'focus': 'Best practices, patterns, optimization, maintainability',
                'avoid': 'Over-engineering, ignoring edge cases, skipping documentation',
                'suggest': 'Balance simplicity with scalability, include testing, consider maintenance',
                'resources': 'Advanced tutorials, design patterns, performance guides',
                'success_indicators': 'Efficient solution, good architecture, comprehensive testing'
            },
            'advanced': {
                'focus': 'System architecture, performance optimization, scalability, reliability',
                'avoid': 'Assumptions about constraints, ignoring failure modes, tight coupling',
                'suggest': 'Consider all edge cases, design for failure, implement monitoring',
                'resources': 'System design patterns, advanced architecture, performance tuning',
                'success_indicators': 'Scalable architecture, robust error handling, optimal performance'
            },
            'expert': {
                'focus': 'Strategic approach, innovation, research, cutting-edge solutions',
                'avoid': 'Conventional solutions without evaluation, ignoring emerging technologies',
                'suggest': 'Consider multiple innovative approaches, evaluate trade-offs thoroughly',
                'resources': 'Research papers, cutting-edge frameworks, expert communities',
                'success_indicators': 'Innovative solution, comprehensive evaluation, future-proof design'
            }
        }
        
        return guidance.get(complexity, guidance['intermediate'])

    def _get_domain_insights(self, domain: str, concepts: Dict) -> List[str]:
        """Get domain-specific insights"""
        insights = []
        
        if domain == 'software_development':
            insights.append("Code maintainability and readability are critical for long-term success")
            insights.append("Follow SOLID principles and design patterns where applicable")
            insights.append("Test-driven development improves code quality and reduces bugs")
            insights.append("Consider technical debt implications of implementation choices")
            insights.append("Documentation should explain both 'how' and 'why'")
        
        elif domain == 'data_science_ai':
            insights.append("Data quality is often more important than model complexity")
            insights.append("Consider model explainability, bias, and fairness implications")
            insights.append("Proper validation prevents overfitting and ensures generalization")
            insights.append("Feature engineering can have greater impact than algorithm choice")
            insights.append("MLOps practices ensure model reliability in production")
        
        elif domain == 'cybersecurity':
            insights.append("Security should be integrated throughout the development lifecycle")
            insights.append("Regular audits, updates, and monitoring are crucial")
            insights.append("Defense in depth approach provides multiple security layers")
            insights.append("Consider both technical and human factors in security")
            insights.append("Incident response planning is as important as prevention")
        
        elif domain == 'devops_infrastructure':
            insights.append("Automation reduces human error and improves consistency")
            insights.append("Monitoring should be proactive rather than reactive")
            insights.append("Infrastructure as code improves reproducibility and version control")
            insights.append("Consider both scalability and cost optimization")
            insights.append("Disaster recovery planning is essential for critical systems")
        
        elif domain == 'cloud_computing':
            insights.append("Cost optimization requires continuous monitoring and adjustment")
            insights.append("Multi-cloud strategies reduce vendor lock-in but increase complexity")
            insights.append("Security responsibilities are shared between cloud provider and user")
            insights.append("Consider data gravity and latency implications")
            insights.append("Serverless architectures change traditional operational models")
        
        else:
            insights.append("Apply systematic problem-solving approach")
            insights.append("Consider both technical and non-technical factors")
            insights.append("Validate assumptions through testing and measurement")
            insights.append("Document decisions and rationale for future reference")
        
        return insights[:3]  # Return top 3 most relevant insights

    def _estimate_analysis_time(self, problem: Dict) -> Dict[str, Any]:  # CHANGED NAME AND RETURN TYPE
        """Estimate analysis time needed based on complexity"""
        time_estimates = {
            'beginner': {
                'min': '15-30 minutes',
                'max': '1-2 hours',
                'typical': '45 minutes',
                'note': 'Straightforward technical analysis',
                'analysis_depth': 'basic'
            },
            'intermediate': {
                'min': '30-60 minutes',
                'max': '2-4 hours',
                'typical': '1.5 hours',
                'note': 'Moderate complexity technical analysis',
                'analysis_depth': 'detailed'
            },
            'advanced': {
                'min': '1-2 hours',
                'max': '4-8 hours',
                'typical': '3 hours',
                'note': 'Complex technical analysis with multiple factors',
                'analysis_depth': 'comprehensive'
            },
            'expert': {
                'min': '2-4 hours',
                'max': '1-2 days',
                'typical': '6 hours',
                'note': 'In-depth expert technical analysis',
                'analysis_depth': 'exhaustive'
            }
        }
        estimate = time_estimates.get(problem['complexity'], time_estimates['intermediate']).copy()
        
        # Add urgency adjustments
        urgency = problem.get('urgency', 'medium')
        if urgency == 'critical':
            estimate['priority_adjustment'] = 'critical_priority'
            estimate['adjusted_min'] = estimate['min'].split('-')[0]  # Take the lower bound
            estimate['priority_note'] = 'Critical priority - expedited analysis required'
        elif urgency == 'high':
            estimate['priority_adjustment'] = 'high_priority'
            estimate['priority_note'] = 'High priority - focused analysis'
        else:
            estimate['priority_adjustment'] = 'standard_priority'
            estimate['priority_note'] = 'Standard priority analysis'
        
        # Add complexity note
        estimate['complexity_level'] = problem['complexity']
        
        # Format for display
        if urgency == 'critical':
            estimate['display_time'] = f"{estimate['adjusted_min']} (critical)"
        elif urgency == 'high':
            estimate['display_time'] = f"{estimate['min']} to {estimate['typical']}"
        else:
            estimate['display_time'] = f"{estimate['typical']} to {estimate['max']}"
        
        return estimate

from ..core.base_vni import EnhancedBaseVNI
from ..core.capabilities import VNICapabilities, VNIType
class TechnicalVNI(EnhancedBaseVNI):
    """Enhanced Technical VNI with memory integration and hybrid attention - ANALYSIS ONLY"""
    def __init__(self, vni_id: str = None, name: str = None, 
                 capabilities: VNICapabilities = None,
                 memory_toolkit: VNIMemory = None):
        
        # Use technical-specific capabilities
        capabilities = VNICapabilities(
            can_process_text=True,
            can_generate_text=False,  # CHANGED: Cannot generate text
            can_learn=True,
            has_knowledge_base=True,
            technical_expertise=True,
            has_hybrid_attention=True,
            has_smart_routing=True,
            max_context_length=8192
        )
        
        self.domain = "technical"
        self.vni_type = "technical"
        self.name = name or "Enhanced Technical VNI"
        self.description = "Enhanced Technical VNI with memory integration and hybrid attention - Analysis Only"
        
        super().__init__(
            vni_id=vni_id or "technical_vni_enhanced",
            name=name or "Enhanced Technical VNI",
            capabilities=capabilities
        )
        
        # Initialize technical components
        self.config = TechnicalOperActionConfig()
        self.config.enable_llm_generation = False  # Ensure disabled
        
        self.knowledge_graph = TechnicalKnowledgeGraph(self.config)
        self.memory_toolkit = memory_toolkit or VNIMemory(retention_days=self.config.memory_retention_days)
        
        # Initialize reasoning engine with vni_id
        self.reasoning_engine = TechnicalReasoningEngine(
            self.config, 
            self.memory_toolkit,
            vni_id=self.vni_id  # PASS vni_id to reasoning engine
        )
        
        # Track performance
        self.performance_metrics = {
            'total_queries': 0,
            'successful_analyses': 0,
            'avg_confidence': 0,
            'memory_hit_rate': 0,
            'attention_patterns': [],
        }

    async def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Alias for process method to maintain compatibility with mesh_core.
        This allows the VNI to be called with either process or process_query."""
        logger.debug(f"TechnicalVNI.process_query called - forwarding to process()")
        return await self.process(query, context)

    def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process technical queries with enhanced reasoning"""
        self.performance_metrics['total_queries'] += 1
        
        # Extract technical concepts with memory context
        concepts = self.reasoning_engine.extract_technical_concepts(input_text, context)
        
        # Convert to features
        features = self._encode_to_features(input_text, concepts) if torch.is_available() else None
        
        # Analyze problems with hybrid attention
        problems = self.reasoning_engine.analyze_technical_problem(concepts, features)
        
        # Generate solutions using LLM Gateway if enabled
        if self.config.enable_llm_generation and hasattr(self, 'llm_gateway') and self.llm_gateway:
            try:
                solutions = self.reasoning_engine._generate_solutions_with_llm(problems, concepts, features)
                self.performance_metrics['llm_generation_count'] += 1
            except Exception as e:
                logger.error(f"LLM generation failed: {e}, using fallback")
                solutions = self.reasoning_engine._generate_fallback_solutions(problems, concepts, features)
                self.performance_metrics['fallback_generation_count'] += 1
        else:
            solutions = self.reasoning_engine._generate_fallback_solutions(problems, concepts, features)
            self.performance_metrics['fallback_generation_count'] += 1
        
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
        
        result = {
            'query': input_text,
            'technical_domains': [d['domain'] for d in concepts.get('technical_domains', [])],
            'problems_identified': len(problems),
            'primary_problem': problems[0] if problems else None,
            'solutions': solutions,
            'complexity': problems[0]['complexity'] if problems else 'unknown',
            'confidence': problems[0]['confidence'] if problems else 0.0,
            'uses_hybrid_attention': self.config.enable_hybrid_attention,
            'uses_smart_routing': self.config.enable_smart_routing,
            'uses_llm_generation': self.config.enable_llm_generation and hasattr(self, 'llm_gateway') and self.llm_gateway,
            'memory_context_used': bool(problems and problems[0].get('similar_past_issues')),
            'performance_metrics': {
                'query_count': self.performance_metrics['total_queries'],
                'resolution_rate': self.performance_metrics['successful_resolutions'] / max(self.performance_metrics['total_queries'], 1),
                'avg_confidence': self.performance_metrics['avg_confidence'],
                'memory_hit_rate': self.performance_metrics['memory_hit_rate'] / max(self.performance_metrics['total_queries'], 1),
                'llm_generation_rate': self.performance_metrics['llm_generation_count'] / max(self.performance_metrics['total_queries'], 1),
                'fallback_generation_rate': self.performance_metrics['fallback_generation_count'] / max(self.performance_metrics['total_queries'], 1)
            },
            'vni_metadata': {
                'vni_id': self.vni_id,
                'success': True,  # ← CRITICAL!
                'processing_time': 0.01,
                'timestamp': datetime.now().isoformat()
            }
        }
        result['vni_id'] = self.vni_id
        result['domain'] = 'technical'
              
        # Build a concise opinion_text from the analysis
        if problems:
            # Take the first problem as representative
            primary = problems[0]
            domain = primary.get('domain', 'technical')
            confidence = primary.get('confidence', 0.0)
            complexity = primary.get('complexity', 'unknown')
            # Summarize
            opinion_parts = [
                f"Technical problem in {domain} domain.",
                f"Complexity: {complexity}.",
                f"Confidence: {confidence:.0%}.",
                f"Generated {len(solutions)} solution(s)."
            ]
            if primary.get('similar_past_issues'):
                opinion_parts.append("Used memory of similar past issues.")
        else:
            opinion_parts = ["Technical query processed, but no specific problem identified."]
        
        result['opinion_text'] = ' '.join(opinion_parts)

        return result
    
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
                    'solution_text': solution.get('solution_text', solution.get('steps', [])),
                    'generated_by': solution.get('generated_by', 'unknown'),
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
                'enable_llm_generation': self.config.enable_llm_generation,  # ADDED
                'confidence_threshold': self.config.confidence_threshold
            },
            'llm_gateway_available': hasattr(self, 'llm_gateway') and self.llm_gateway is not None  # ADDED
        } 
