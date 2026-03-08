# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

"""Auto-generates biological configurations for ANY domain"""
import re
import logging
from typing import Dict, Any, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class BiologicalConfigGenerator:
    """Generates appropriate biological configurations for any VNI type"""
    
    # Domain classification patterns
    DOMAIN_PATTERNS = {
        'technical': {
            'keywords': ['code', 'programming', 'software', 'hardware', 'system', 
                        'algorithm', 'data', 'network', 'api', 'framework', 'database'],
            'weight': {'semantic': 0.8, 'precision': 0.7, 'hierarchical': True}
        },
        'creative': {
            'keywords': ['art', 'design', 'creative', 'music', 'write', 'story',
                        'paint', 'draw', 'photo', 'film', 'video', 'expression'],
            'weight': {'visual': 0.6, 'semantic': 0.4, 'associative': True}
        },
        'medical': {
            'keywords': ['health', 'medical', 'doctor', 'patient', 'treatment', 
                        'medicine', 'hospital', 'symptom', 'diagnosis', 'drug'],
            'weight': {'semantic': 0.7, 'contextual': 0.3, 'precision': 0.8}
        },
        'legal': {
            'keywords': ['law', 'legal', 'contract', 'rights', 'court', 'attorney',
                        'case', 'evidence', 'justice', 'liability'],
            'weight': {'semantic': 0.9, 'precision': 0.9, 'sequential': True}
        },
        'academic': {
            'keywords': ['research', 'study', 'science', 'theory', 'analysis',
                        'experiment', 'paper', 'journal', 'academic'],
            'weight': {'semantic': 0.8, 'contextual': 0.5, 'hierarchical': True}
        },
        'business': {
            'keywords': ['business', 'finance', 'market', 'strategy', 'management',
                        'corporate', 'profit', 'investment', 'economic'],
            'weight': {'semantic': 0.6, 'contextual': 0.4, 'pragmatic': True}
        }
    }
    
    @staticmethod
    def generate_for_topic(topic: str, keywords: List[str] = None) -> Dict[str, Any]:
        """Generate biological config for ANY topic"""
        logger.info(f"🔧 Generating biological config for topic: {topic}")
        
        # Analyze topic characteristics
        analysis = {
            'complexity': BiologicalConfigGenerator._analyze_complexity(topic, keywords),
            'domain_type': BiologicalConfigGenerator._classify_domain(topic, keywords),
            'semantic_density': BiologicalConfigGenerator._measure_semantic_density(keywords),
            'context_requirements': BiologicalConfigGenerator._assess_context_needs(topic)
        }
        
        logger.debug(f"Topic analysis: {analysis}")
        
        # Generate config
        config = {
            'enable_biological_systems': True,
            'attention_config': BiologicalConfigGenerator._generate_attention_config(analysis),
            'memory_config': BiologicalConfigGenerator._generate_memory_config(analysis),
            'activation_config': BiologicalConfigGenerator._generate_activation_config(analysis)
        }
        
        logger.info(f"✅ Generated biological config for {topic}: {config['domain_type']['type']}")
        return config
    
    @staticmethod
    def _analyze_complexity(topic: str, keywords: List[str] = None) -> float:
        """Analyze topic complexity (0.0-1.0 scale)"""
        # Word count complexity
        word_count = len(topic.split())
        word_complexity = min(1.0, word_count / 10.0)
        
        # Technical term complexity
        technical_terms = ['algorithm', 'implementation', 'architecture', 'paradigm',
                          'methodology', 'framework', 'protocol', 'synthesis']
        tech_count = sum(1 for term in technical_terms if term in topic.lower())
        tech_complexity = min(1.0, tech_count * 0.2)
        
        # Keyword complexity (if provided)
        keyword_complexity = 0.0
        if keywords:
            avg_keyword_len = np.mean([len(kw) for kw in keywords])
            keyword_complexity = min(1.0, avg_keyword_len / 15.0)
        
        # Combine factors
        complexity = (word_complexity * 0.3 + 
                     tech_complexity * 0.4 + 
                     keyword_complexity * 0.3)
        
        return round(complexity, 2)
    
    @staticmethod
    def _classify_domain(topic: str, keywords: List[str] = None) -> Dict[str, Any]:
        """Classify the domain type and get characteristics"""
        topic_lower = topic.lower()
        keyword_text = ' '.join(keywords) if keywords else ''
        full_text = f"{topic_lower} {keyword_text}".lower()
        
        # Calculate domain scores
        domain_scores = {}
        for domain, pattern in BiologicalConfigGenerator.DOMAIN_PATTERNS.items():
            score = 0.0
            for keyword in pattern['keywords']:
                if keyword in full_text:
                    score += 0.1
            domain_scores[domain] = min(1.0, score)
        
        # Find best matching domain
        best_domain = max(domain_scores.items(), key=lambda x: x[1])
        
        if best_domain[1] > 0.3:
            domain_type = best_domain[0]
            confidence = best_domain[1]
            weights = BiologicalConfigGenerator.DOMAIN_PATTERNS[domain_type]['weight']
        else:
            domain_type = 'general'
            confidence = 0.5
            weights = {'semantic': 0.6, 'contextual': 0.4, 'balanced': True}
        
        # Determine domain characteristics
        characteristics = {
            'stability': 0.7 if domain_type in ['medical', 'legal', 'technical'] else 0.5,
            'importance': 0.8 if domain_type in ['medical', 'legal'] else 0.6,
            'relevance': confidence,
            'needs_precision': domain_type in ['medical', 'legal', 'technical']
        }
        
        return {
            'type': domain_type,
            'confidence': confidence,
            'weights': weights,
            'characteristics': characteristics
        }
    
    @staticmethod
    def _measure_semantic_density(keywords: List[str] = None) -> float:
        """Measure semantic density of keywords (0.0-1.0)"""
        if not keywords:
            return 0.5
        
        # Calculate based on keyword characteristics
        scores = []
        for keyword in keywords:
            # Longer keywords typically have more semantic content
            length_score = min(1.0, len(keyword) / 20.0)
            
            # Multi-word keywords have higher semantic density
            word_count = len(keyword.split())
            word_count_score = min(1.0, word_count / 3.0)
            
            scores.append((length_score + word_count_score) / 2)
        
        return np.mean(scores) if scores else 0.5
    
    @staticmethod
    def _assess_context_needs(topic: str) -> float:
        """Assess how much context this topic needs (0.0-1.0)"""
        # Topics with these words need more context
        context_indicators = ['explain', 'understand', 'context', 'background',
                             'history', 'theory', 'principle', 'concept']
        
        indicator_count = sum(1 for indicator in context_indicators 
                            if indicator in topic.lower())
        
        return min(1.0, indicator_count * 0.2 + 0.3)
    
    @staticmethod
    def _generate_attention_config(analysis: Dict) -> Dict[str, Any]:
        """Generate attention configuration based on analysis"""
        domain_type = analysis['domain_type']
        
        # Base attention configuration
        config = {
            'attention_type': 'hybrid',
            'use_hierarchical': domain_type['type'] in ['technical', 'academic'],
            'use_global': True,
            'use_sliding': True,
            'multi_modal': domain_type['type'] == 'creative',
            'dim': 256,  # Default dimension
            'num_heads': 8,
            'window_size': 256
        }
        
        # Set weights based on domain analysis
        if 'weights' in domain_type:
            weights = domain_type['weights']
            
            if 'semantic' in weights:
                config['semantic_weight'] = weights['semantic']
            if 'visual' in weights:
                config['visual_weight'] = weights['visual']
            if 'contextual' in weights:
                config['contextual_weight'] = weights['contextual']
            
            # Adjust for complexity
            complexity = analysis['complexity']
            if complexity > 0.7:
                config['memory_tokens'] = 32
                config['global_token_ratio'] = 0.1
            elif complexity > 0.4:
                config['memory_tokens'] = 16
                config['global_token_ratio'] = 0.05
            else:
                config['memory_tokens'] = 8
                config['global_token_ratio'] = 0.03
        
        return config
    
    @staticmethod
    def _generate_memory_config(analysis: Dict) -> Dict[str, Any]:
        """Generate memory configuration based on analysis"""
        domain_type = analysis['domain_type']
        complexity = analysis['complexity']
        
        # Base memory configuration
        config = {
            'short_term_capacity': 100 + int(complexity * 100),
            'long_term_capacity': 1000,
            'consolidation_threshold': 0.7,
            'retention_period': 86400  # 24 hours in seconds
        }
        
        # Adjust based on domain type
        domain_adjustments = {
            'medical': {'short_term_capacity': 150, 'consolidation_threshold': 0.8},
            'legal': {'short_term_capacity': 200, 'retention_period': 86400 * 7},
            'technical': {'consolidation_threshold': 0.6, 'retention_period': 86400 * 2},
            'academic': {'long_term_capacity': 2000, 'retention_period': 86400 * 30}
        }
        
        if domain_type['type'] in domain_adjustments:
            config.update(domain_adjustments[domain_type['type']])
        
        # Adjust for semantic density
        semantic_density = analysis['semantic_density']
        if semantic_density > 0.7:
            config['long_term_capacity'] = 1500
        
        return config
    
    @staticmethod
    def _generate_activation_config(analysis: Dict) -> Dict[str, Any]:
        """Generate activation configuration based on analysis"""
        domain_type = analysis['domain_type']
        complexity = analysis['complexity']
        
        config = {
            'base_threshold': 0.3,
            'complexity_modifier': complexity * 0.2,
            'response_time_factor': 1.0
        }
        
        # Domain-specific adjustments
        if domain_type['type'] == 'medical':
            config['base_threshold'] = 0.4  # Medical needs higher certainty
            config['urgency_boost'] = 0.3
        elif domain_type['type'] == 'legal':
            config['base_threshold'] = 0.45  # Legal needs highest certainty
            config['precision_boost'] = 0.4
        elif domain_type['type'] == 'creative':
            config['base_threshold'] = 0.25  # Creative can be more exploratory
            config['creativity_boost'] = 0.3
        
        # Calculate final threshold
        config['final_threshold'] = min(0.8, config['base_threshold'] + config['complexity_modifier'])
        
        return config
    
    @staticmethod
    def generate_quick_config(topic: str) -> Dict[str, Any]:
        """Generate quick biological config (simplified version)"""
        return {
            'enable_biological_systems': True,
            'attention_config': {
                'attention_type': 'hybrid',
                'semantic_weight': 0.6,
                'memory_tokens': 16,
                'use_hierarchical': True
            },
            'memory_config': {
                'short_term_capacity': 100,
                'long_term_capacity': 1000,
                'consolidation_threshold': 0.7
            },
            'activation_config': {
                'base_threshold': 0.3
            }
        } 
