# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

"""
Dynamic VNI Factory - Creates VNIs for ANY topic using web-enhanced semantic matching
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import hashlib
import re

from ..domains.dynamic_vni import DynamicVNI, DomainConfig

logger = logging.getLogger(__name__)

class DynamicVNIFactory:
    """
    Factory that creates VNIs for ANY topic dynamically using:
    1. Web search to understand the topic (if available)
    2. Semantic analysis of search results
    3. Template application based on detected domain
    4. Smart keyword extraction and expansion
    """
    
    def __init__(self, enable_web_search: bool = True):
        self.enable_web_search = enable_web_search
        self.web_search = None
        
        # Initialize web search if enabled
        if enable_web_search:
            self.web_search = self._init_web_search()
        
        # Simple domain templates - just enough to guide the system
        self.domain_guides = {
            'technical': {
                'style': 'technical',
                'confidence': 0.6,
                'temperature': 0.7,
                'safety': 'medium',
                'template_name': '{topic} Technology Expert',
                'template_desc': 'Specialized in {topic} technologies and implementations'
            },
            'creative': {
                'style': 'creative', 
                'confidence': 0.4,
                'temperature': 0.8,
                'safety': 'low',
                'template_name': '{topic} Creative Specialist',
                'template_desc': 'Creative expert in {topic} arts and design'
            },
            'academic': {
                'style': 'academic',
                'confidence': 0.7,
                'temperature': 0.5,
                'safety': 'medium',
                'template_name': '{topic} Research Specialist',
                'template_desc': 'Expert in {topic} research and analysis'
            },
            'professional': {
                'style': 'professional',
                'confidence': 0.5,
                'temperature': 0.6,
                'safety': 'high',
                'template_name': '{topic} Professional Consultant',
                'template_desc': 'Professional consultant specializing in {topic}'
            },
            'general': {
                'style': 'informative',
                'confidence': 0.5,
                'temperature': 0.7,
                'safety': 'medium',
                'template_name': '{topic} Specialist',
                'template_desc': 'Specialized assistant for {topic} topics'
            }
        }
        
        # Cache for analyzed domains
        self.domain_cache: Dict[str, DomainConfig] = {}
    
    def create_for_topic(self, topic: str, instance_id: str = None, 
                        context: Dict[str, Any] = None) -> DynamicVNI:
        """
        Create a VNI for ANY topic dynamically by:
        1. Understanding the topic via web search
        2. Extracting key concepts and terminology
        3. Building a specialized configuration
        4. Creating the VNI
        """
        # Generate instance ID if not provided
        if instance_id is None:
            topic_hash = hashlib.md5(topic.lower().encode()).hexdigest()[:8]
            instance_id = f"dynamic-{topic_hash}-{datetime.now().strftime('%H%M%S')}"
        
        logger.info(f"🔄 Creating dynamic VNI for topic: '{topic}'")
        
        # Check cache first
        cache_key = f"{topic.lower()}:{json.dumps(context or {}, sort_keys=True)}"
        if cache_key in self.domain_cache:
            logger.debug(f"Using cached domain config for: {topic}")
            return DynamicVNI(self.domain_cache[cache_key], instance_id)
        
        # Get domain configuration dynamically
        domain_config = self._create_dynamic_config(topic, context)
        
        # Cache the result
        self.domain_cache[cache_key] = domain_config
        
        # Create and return the VNI
        vni = DynamicVNI(
            domain_config=domain_config,
            vni_id=instance_id,
            name=f"Dynamic VNI - {domain_config.name}"
        )
        logger.info(f"✅ Created dynamic VNI '{instance_id}' for topic: '{topic}'")
        
        return vni

    def _create_dynamic_config(self, topic: str, context: Dict[str, Any] = None) -> DomainConfig:
        """Create a dynamic configuration based on web research."""
        
        # Step 1: Understand the topic via web search
        web_insights = self._research_topic(topic)
        
        # Step 2: Extract key information
        domain_type = self._detect_domain_type(topic, web_insights)
        keywords = self._extract_keywords(topic, web_insights)
        concepts = self._extract_concepts(web_insights)
        
        # Step 3: Get domain guide
        guide = self.domain_guides.get(domain_type, self.domain_guides['general'])
        
        # Step 4: Build configuration
        config = {
            'name': guide['template_name'].replace('{topic}', topic.title()),
            'description': guide['template_desc'].replace('{topic}', topic),
            'keywords': keywords,
            'priority_keywords': keywords[:5] if len(keywords) > 5 else keywords,
            'confidence_threshold': guide['confidence'],
            'generation_temperature': guide['temperature'],
            'response_style': guide['style'],
            'safety_level': guide['safety'],
            'specializations': concepts[:5] if concepts else [f"{topic} fundamentals"]
        }
        
        return DomainConfig.from_dict(config)
    
    def _research_topic(self, topic: str) -> Dict[str, Any]:
        """Research a topic using web search to understand it better."""
        insights = {
            'search_results': [],
            'extracted_text': '',
            'common_terms': [],
            'related_topics': []
        }
        
        if not self.enable_web_search or not self.web_search:
            logger.debug(f"Web search not available, using basic analysis for: {topic}")
            return insights
        
        try:
            # Search for the topic to understand it
            search_queries = [
                f"what is {topic}",
                f"{topic} basics",
                f"{topic} key concepts"
            ]
            
            all_results = []
            for query in search_queries:
                results = self.web_search.search(query, max_results=2)
                all_results.extend(results)
            
            insights['search_results'] = all_results
            
            # Extract text from results
            extracted_text = []
            for result in all_results:
                extracted_text.append(result.title)
                extracted_text.append(result.snippet)
            
            insights['extracted_text'] = ' '.join(extracted_text)
            
            # Extract common terms
            if insights['extracted_text']:
                insights['common_terms'] = self._extract_common_terms(insights['extracted_text'])
                
                # Extract related topics (looking for "such as", "including", etc.)
                insights['related_topics'] = self._extract_related_topics(insights['extracted_text'])
            
            logger.debug(f"Researched '{topic}': found {len(all_results)} results, {len(insights['common_terms'])} terms")
            
        except Exception as e:
            logger.debug(f"Web research failed for '{topic}': {e}")
        
        return insights
    
    def _detect_domain_type(self, topic: str, insights: Dict[str, Any]) -> str:
        """Detect what type of domain this topic belongs to."""
        # Combine topic and extracted text for analysis
        analysis_text = f"{topic} {insights['extracted_text']}".lower()
        
        # Check for domain indicators
        tech_indicators = ['code', 'program', 'software', 'technology', 'computer', 
                          'algorithm', 'data', 'system', 'digital', 'tech', 'ai', 'ml']
        creative_indicators = ['art', 'design', 'creative', 'music', 'write', 'story',
                             'paint', 'draw', 'photo', 'film', 'video', 'expression']
        academic_indicators = ['research', 'study', 'science', 'theory', 'analysis',
                              'experiment', 'paper', 'journal', 'academic', 'university']
        professional_indicators = ['business', 'finance', 'management', 'market', 'legal',
                                  'medical', 'corporate', 'industry', 'professional', 'consulting']
        
        # Count indicators
        scores = {
            'technical': sum(1 for indicator in tech_indicators if indicator in analysis_text),
            'creative': sum(1 for indicator in creative_indicators if indicator in analysis_text),
            'academic': sum(1 for indicator in academic_indicators if indicator in analysis_text),
            'professional': sum(1 for indicator in professional_indicators if indicator in analysis_text)
        }
        
        # Find highest score
        max_score = max(scores.values())
        if max_score > 0:
            for domain, score in scores.items():
                if score == max_score:
                    logger.debug(f"Detected domain type: {domain} (score: {score})")
                    return domain
        
        return 'general'
    
    def _extract_keywords(self, topic: str, insights: Dict[str, Any]) -> List[str]:
        """Extract keywords from topic and web insights."""
        keywords = []
        
        # Always include the topic itself
        topic_lower = topic.lower()
        keywords.append(topic_lower)
        
        # Add individual words from multi-word topics
        topic_words = re.findall(r'\b\w+\b', topic_lower)
        if len(topic_words) > 1:
            keywords.extend(topic_words)
        
        # Add common terms from web research
        if insights['common_terms']:
            keywords.extend(insights['common_terms'])
        
        # Add related topics
        if insights['related_topics']:
            keywords.extend(insights['related_topics'])
        
        # Add some intelligent defaults based on topic
        defaults = self._get_topic_defaults(topic_lower)
        keywords.extend(defaults)
        
        # Clean and deduplicate
        cleaned_keywords = []
        seen = set()
        for kw in keywords:
            kw_lower = kw.lower().strip()
            if (len(kw_lower) > 2 and 
                kw_lower not in seen and 
                kw_lower not in {'the', 'and', 'for', 'with', 'that', 'this'}):
                cleaned_keywords.append(kw_lower)
                seen.add(kw_lower)
        
        return cleaned_keywords[:25]  # Limit to reasonable number
    
    def _extract_concepts(self, insights: Dict[str, Any]) -> List[str]:
        """Extract key concepts from web insights."""
        concepts = []
        
        if not insights['extracted_text']:
            return concepts
        
        # Look for patterns that indicate concepts
        text = insights['extracted_text'].lower()
        
        # Patterns to look for
        patterns = [
            r'including ([^\.]+)',  # "including X, Y, and Z"
            r'such as ([^\.]+)',     # "such as X, Y"
            r'like ([^\.]+)',        # "like X and Y"
            r'especially ([^\.]+)',  # "especially X"
            r'particularly ([^\.]+)' # "particularly X"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Split by common separators
                parts = re.split(r',|\band\b|\bor\b', match)
                for part in parts:
                    concept = part.strip()
                    if len(concept) > 3 and ' ' in concept:  # Multi-word concepts only
                        concepts.append(concept)
        
        # Also extract noun phrases (simple heuristic)
        sentences = re.split(r'[\.\!\?]', text)
        for sentence in sentences:
            words = sentence.strip().split()
            if 2 <= len(words) <= 5:  # Reasonable phrase length
                # Check if it looks like a concept (not a full sentence)
                if not sentence.startswith(('the ', 'a ', 'an ', 'it ', 'this ', 'that ')):
                    concepts.append(sentence.strip())
        
        return list(set(concepts))[:10]  # Unique concepts, max 10
    
    def _extract_common_terms(self, text: str) -> List[str]:
        """Extract common terms from text."""
        # Remove common words and get meaningful terms
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        
        # Count frequency
        from collections import Counter
        word_counts = Counter(words)
        
        # Filter out overly common English words
        common_english = {'that', 'with', 'this', 'from', 'have', 'more', 'about',
                         'they', 'what', 'when', 'where', 'which', 'there', 'their'}
        
        # Get most common meaningful words
        common_terms = []
        for word, count in word_counts.most_common(20):
            if word not in common_english and count > 1:
                common_terms.append(word)
        
        return common_terms
    
    def _extract_related_topics(self, text: str) -> List[str]:
        """Extract related topics from text."""
        # Look for capitalized phrases (potential proper nouns/topics)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Filter: at least 2 chars, not too long, not common words
        related = []
        for phrase in capitalized:
            words = phrase.split()
            if (2 <= len(words) <= 4 and 
                phrase.lower() not in {'the', 'and', 'for', 'with', 'this', 'that'}):
                related.append(phrase)
        
        return list(set(related))[:5]
    
    def _get_topic_defaults(self, topic_lower: str) -> List[str]:
        """Get intelligent default keywords based on topic analysis."""
        defaults = []
        
        # Add category-based defaults
        if any(word in topic_lower for word in ['python', 'java', 'javascript', 'code', 'programming']):
            defaults.extend(['programming', 'development', 'software', 'coding', 'algorithm'])
        elif any(word in topic_lower for word in ['art', 'design', 'creative', 'music']):
            defaults.extend(['creative', 'design', 'artistic', 'expression', 'visual'])
        elif any(word in topic_lower for word in ['business', 'finance', 'management']):
            defaults.extend(['business', 'strategy', 'management', 'finance', 'corporate'])
        elif any(word in topic_lower for word in ['science', 'research', 'academic']):
            defaults.extend(['research', 'analysis', 'study', 'scientific', 'methodology'])
        elif any(word in topic_lower for word in ['health', 'medical', 'medicine']):
            defaults.extend(['health', 'medical', 'care', 'treatment', 'wellness'])
        else:
            # General defaults for any topic
            defaults.extend(['information', 'knowledge', 'expertise', 'assistance', 'guidance'])
        
        return defaults
    
    def _init_web_search(self):
        """Initialize web search module if available."""
        try:
            from ..modules.web_search import WebSearch
            web_search = WebSearch(max_results=5)
            if web_search.is_available():
                logger.info("🌐 Web search enabled for dynamic VNI creation")
                return web_search
            else:
                logger.info("🌐 Web search dependencies not available")
                return None
        except ImportError:
            logger.debug("Web search module not available")
            return None
        except Exception as e:
            logger.debug(f"Failed to initialize web search: {e}")
            return None 
