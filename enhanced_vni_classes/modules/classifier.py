# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

# bionn-demo-chatbot/enhanced_vni_classes/modules/classifier.py
import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class Domain(Enum):
    GENERAL = "general"
    MEDICAL = "medical"
    LEGAL = "legal"
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    CREATIVE = "creative"
    PERSONAL = "personal"
    FINANCIAL = "financial"
    SCIENTIFIC = "scientific"

@dataclass
class ClassificationResult:
    """Result of domain classification."""
    primary_domain: str
    confidence: float
    secondary_domains: List[Dict[str, float]]
    keywords: List[str]
    reasoning: str
    domain_specific_confidence: Dict[str, float] = field(default_factory=dict)
    priority_keywords_matched: List[str] = field(default_factory=list)
    match_details: Dict[str, Any] = field(default_factory=dict)

class DynamicDomainClassifier:
    """Single-domain binary classifier that adapts to any domain based on keywords."""
    
    def __init__(self, 
                 domain_name: str, 
                 keywords: List[str], 
                 priority_keywords: List[str] = None,
                 confidence_threshold: float = 0.3,
                 regex_patterns: Optional[List[str]] = None):
        """
        Args:
            domain_name: Name of the domain this classifier handles
            keywords: List of keywords for this domain
            priority_keywords: Keywords that strongly indicate this domain
            confidence_threshold: Minimum confidence threshold for classification
            regex_patterns: Optional regex patterns for more sophisticated matching
        """
        self.domain_name = domain_name
        self.keywords = [k.lower() for k in keywords]
        self.priority_keywords = [k.lower() for k in priority_keywords] if priority_keywords else []
        self.confidence_threshold = confidence_threshold
        self.regex_patterns = regex_patterns or []
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.regex_patterns]
        
        # Statistics
        self.total_classifications = 0
        self.positive_classifications = 0
        
    def predict(self, texts: Union[str, List[str]]) -> Union[int, List[int]]:
        """Return 1 if text matches this domain, 0 otherwise."""
        single_input = isinstance(texts, str)
        texts_list = [texts] if single_input else texts
        
        results = []
        for text in texts_list:
            text_lower = text.lower()
            
            # Check priority keywords first (strong signal)
            priority_matches = [kw for kw in self.priority_keywords if kw in text_lower]
            if priority_matches:
                results.append(1)
                continue
            
            # Check regex patterns
            pattern_matches = 0
            for pattern in self.compiled_patterns:
                if pattern.search(text):
                    pattern_matches += 1
            
            # Check regular keywords
            keyword_matches = sum(1 for keyword in self.keywords if keyword in text_lower)
            
            # Determine if it's our domain based on match threshold
            word_count = len(text_lower.split())
            match_density = (keyword_matches + pattern_matches) / max(word_count, 1)
            
            is_domain = (keyword_matches >= 2) or (pattern_matches >= 1) or (match_density > self.confidence_threshold)
            
            # Update statistics
            self.total_classifications += 1
            if is_domain:
                self.positive_classifications += 1
            
            results.append(1 if is_domain else 0)
        
        return results[0] if single_input else results
    
    def predict_proba(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Return probability scores [not_domain, domain]."""
        single_input = isinstance(texts, str)
        texts_list = [texts] if single_input else texts
        
        results = []
        for text in texts_list:
            text_lower = text.lower()
            
            # Calculate match score from keywords
            keyword_matches = sum(1 for keyword in self.keywords if keyword in text_lower)
            
            # Check priority keywords
            priority_matches = sum(1 for keyword in self.priority_keywords if keyword in text_lower)
            
            # Check regex patterns
            pattern_matches = sum(1 for pattern in self.compiled_patterns if pattern.search(text))
            
            # Calculate probability with weights
            base_score = min(0.9, 
                (keyword_matches * 0.15) + 
                (priority_matches * 0.3) + 
                (pattern_matches * 0.25)
            )
            
            # Adjust based on text length
            word_count = len(text_lower.split())
            if word_count > 0:
                density_score = (keyword_matches + priority_matches) / word_count
                base_score = max(base_score, min(0.8, density_score * 2))
            
            # Normalize
            prob_domain = min(0.95, base_score)
            prob_not_domain = 1.0 - prob_domain
            
            results.append([prob_not_domain, prob_domain])
        
        return results[0] if single_input else results
    
    def get_keyword_matches(self, text: str) -> Dict[str, List[str]]:
        """Return which keywords matched in the text."""
        text_lower = text.lower()
        
        return {
            "priority_keywords": [kw for kw in self.priority_keywords if kw in text_lower],
            "regular_keywords": [kw for kw in self.keywords if kw in text_lower],
            "regex_matches": [
                pattern.pattern for pattern in self.compiled_patterns 
                if pattern.search(text)
            ]
        }

    def classify(self, query: str) -> Dict:
        """
        Classification interface for compatibility with DomainClassifier.
        
        Args:
            query: Input text to classify
            
        Returns:
            Dictionary with classification results in expected format
        """
        try:
            # Get prediction and probability
            prediction = self.predict(query)
            proba = self.predict_proba(query)[1]  # [not_domain, domain] -> get domain probability
            
            # Build result in DomainClassifier-compatible format
            return {
                'domain': self.domain_name if prediction == 1 else 'general',
                'confidence': float(proba),
                'reasoning': self._generate_classification_reasoning(query, prediction, proba),
                'subdomain': None,
                'metadata': {
                    'domain_name': self.domain_name,
                    'is_domain': bool(prediction == 1),
                    'probability': float(proba),
                    'keyword_matches': self.get_keyword_matches(query)
                }
            }
        except Exception as e:
            # Fallback for error cases
            import traceback
            error_msg = f"DynamicDomainClassifier.classify error: {str(e)}"
            return {
                'domain': 'general',
                'confidence': 0.1,
                'reasoning': error_msg,
                'subdomain': None,
                'metadata': {'error': str(e), 'traceback': traceback.format_exc()}
            }

    def _generate_classification_reasoning(self, query: str, prediction: int, probability: float) -> str:
        """Generate human-readable reasoning for classification."""
        if prediction == 1:
            matches = self.get_keyword_matches(query)
            priority_matches = matches.get('priority_keywords', [])
            regular_matches = matches.get('regular_keywords', [])
            
            if priority_matches:
                return f"Matched priority keywords: {', '.join(priority_matches[:3])}"
            elif regular_matches:
                return f"Matched keywords: {', '.join(regular_matches[:3])}"
            else:
                return f"Pattern-based match with {probability:.1%} confidence"
        else:
            return f"Not matching {self.domain_name} domain (confidence: {probability:.1%})"

    def get_statistics(self) -> Dict[str, Any]:
        """Get classification statistics."""
        if self.total_classifications == 0:
            return {"total": 0, "positive_rate": 0.0}
        
        return {
            "total_classifications": self.total_classifications,
            "positive_classifications": self.positive_classifications,
            "positive_rate": self.positive_classifications / self.total_classifications,
            "domain": self.domain_name
        }

class EnhancedDomainClassifier:
    """
    Hybrid classifier combining multi-domain routing with dynamic domain-specific classification.
    Uses both predefined domain patterns and dynamic domain classifiers.
    """
    
    def __init__(self, enable_context: bool = True, max_context: int = 10):
        self.enable_context = enable_context
        self.max_context = max_context
        
        # Multi-domain routing system
        self.domain_patterns = self._load_domain_patterns()
        self.keyword_lists = self._load_keyword_lists()
        
        # Dynamic domain classifiers (can be added at runtime)
        self.dynamic_classifiers: Dict[str, DynamicDomainClassifier] = {}
        
        # Context window for conversation awareness
        self.context_window = []
        
        # Statistics and learning
        self.classification_history = []
        self.domain_weights = defaultdict(float)
        
        # Initialize dynamic classifiers for predefined domains
        self._initialize_dynamic_classifiers()

    def get_domain_config(self, domain_name: str) -> Dict[str, Any]:
        """Get configuration for a specific domain."""
        if domain_name in self.keyword_lists:
            keywords_data = self.keyword_lists[domain_name]
            regex_patterns = self.domain_patterns.get(domain_name, [])
            
            return {
                "keywords": keywords_data["regular"],
                "priority_keywords": keywords_data["priority"],
                "regex_patterns": regex_patterns
            }
        
        # Default config for unknown domains
        return {
            "keywords": [],
            "priority_keywords": [],
            "regex_patterns": []
        } 
       
    def _load_domain_patterns(self) -> Dict[str, List[str]]:
        """Load domain-specific regex patterns."""
        return {
            Domain.MEDICAL.value: [
                r"(?i)\b(doctor|patient|symptom|disease|treatment|medicine|hospital|health|medical|diagnosis)\b",
                r"(?i)\b(covid|vaccine|prescription|surgery|therapy|clinical|pharmaceutical)\b",
                r"(?i)\b(pain|fever|cough|headache|infection|blood|pressure|heart|lung)\b",
                r"(?i)\b(emergency|urgent|911|ambulance|emergency room)\b"
            ],
            Domain.LEGAL.value: [
                r"(?i)\b(law|legal|attorney|lawyer|court|case|judge|trial|contract|agreement)\b",
                r"(?i)\b(rights|lawsuits|settlement|jurisdiction|evidence|testimony|verdict)\b",
                r"(?i)\b(constitution|statute|regulation|compliance|intellectual property|patent)\b",
                r"(?i)\b(urgent legal|legal emergency|arrested|lawsuit filed)\b"
            ],
            Domain.TECHNICAL.value: [
                r"(?i)\b(technical|technology|software|hardware|code|programming|algorithm|system)\b",
                r"(?i)\b(computer|server|network|database|api|interface|protocol|encryption)\b",
                r"(?i)\b(debug|optimize|deploy|configure|install|update|upgrade|maintenance)\b",
                r"(?i)\b(error|bug|crash|fix|patch|version)\b"
            ],
            Domain.ACADEMIC.value: [
                r"(?i)\b(research|study|academic|paper|thesis|dissertation|publication|journal)\b",
                r"(?i)\b(experiment|methodology|hypothesis|analysis|results|conclusion|citation)\b",
                r"(?i)\b(scholar|professor|university|college|campus|lecture|course|curriculum)\b"
            ],
            Domain.FINANCIAL.value: [
                r"(?i)\b(finance|financial|money|investment|stock|market|bank|loan|credit|debt)\b",
                r"(?i)\b(tax|income|salary|budget|expense|profit|loss|revenue|asset|liability)\b",
                r"(?i)\b(interest|rate|exchange|currency|trading|portfolio|retirement|insurance)\b"
            ],
            Domain.SCIENTIFIC.value: [
                r"(?i)\b(science|scientific|physics|chemistry|biology|mathematics|engineering)\b",
                r"(?i)\b(experiment|theory|hypothesis|observation|data|analysis|model|simulation)\b",
                r"(?i)\b(research|discovery|innovation|technology|laboratory|scientist|researcher)\b"
            ],
            Domain.CREATIVE.value: [
                r"(?i)\b(creative|art|design|music|writing|story|poem|painting|drawing|sketch)\b",
                r"(?i)\b(imagination|inspiration|idea|concept|theme|style|expression|creative process)\b",
                r"(?i)\b(compose|create|design|develop|innovate|invent|original|unique)\b"
            ],
            Domain.PERSONAL.value: [
                r"(?i)\b(personal|life|family|friend|relationship|home|personal development|goal)\b",
                r"(?i)\b(emotion|feeling|happiness|sadness|stress|anxiety|mental health|wellbeing)\b",
                r"(?i)\b(hobby|interest|passion|dream|aspiration|personal growth|self-improvement)\b"
            ],
            Domain.GENERAL.value: [
                r"(?i)\b(hello|hi|hey|how are you|what is|who is|when is|where is)\b",
                r"(?i)\b(help|assist|support|question|answer|explain|describe|tell me about)\b"
            ]
        }
    
    def _load_keyword_lists(self) -> Dict[str, List[str]]:
        """Load domain-specific keyword lists with priority indicators."""
        return {
            Domain.MEDICAL.value: {
                "priority": ["emergency", "urgent", "911", "heart attack", "stroke", "bleeding"],
                "regular": ["health", "medicine", "treatment", "diagnosis", "symptoms",
                          "patient", "doctor", "hospital", "clinical", "pharmaceutical"]
            },
            Domain.LEGAL.value: {
                "priority": ["arrest", "lawsuit", "eviction", "legal emergency", "court date"],
                "regular": ["legal", "law", "court", "contract", "agreement",
                          "rights", "case", "judge", "attorney", "jurisdiction"]
            },
            Domain.TECHNICAL.value: {
                "priority": ["error", "bug", "crash", "broken", "not working"],
                "regular": ["technical", "technology", "software", "hardware", "system",
                          "code", "programming", "algorithm", "database", "network"]
            },
            Domain.ACADEMIC.value: {
                "priority": ["deadline", "due date", "submit", "paper due"],
                "regular": ["academic", "research", "study", "paper", "thesis",
                          "publication", "journal", "scholar", "university", "campus"]
            },
            Domain.FINANCIAL.value: {
                "priority": ["debt", "bankrupt", "overdraft", "late payment"],
                "regular": ["financial", "finance", "money", "investment", "stock",
                          "market", "bank", "loan", "credit", "tax"]
            },
            Domain.SCIENTIFIC.value: {
                "priority": ["experiment failed", "data lost", "research emergency"],
                "regular": ["science", "scientific", "physics", "chemistry", "biology",
                          "mathematics", "engineering", "experiment", "theory", "data"]
            },
            Domain.CREATIVE.value: {
                "priority": ["writer's block", "creative block", "inspiration needed"],
                "regular": ["creative", "art", "design", "music", "writing",
                          "story", "poem", "painting", "imagination", "inspiration"]
            },
            Domain.PERSONAL.value: {
                "priority": ["depressed", "anxious", "stressed", "emergency help"],
                "regular": ["personal", "life", "family", "friend", "relationship",
                          "home", "emotion", "feeling", "hobby", "personal growth"]
            },
            Domain.GENERAL.value: {
                "priority": ["help", "urgent", "emergency", "quick question"],
                "regular": ["information", "knowledge", "explain", "describe", "tell",
                          "what", "how", "why", "when", "where"]
            }
        }
    
    def _initialize_dynamic_classifiers(self):
        """Initialize dynamic classifiers for each predefined domain."""
        for domain in Domain:
            domain_name = domain.value
            if domain_name in self.keyword_lists:
                keywords_data = self.keyword_lists[domain_name]
                
                # Get regex patterns for this domain
                regex_patterns = self.domain_patterns.get(domain_name, [])
                
                classifier = DynamicDomainClassifier(
                    domain_name=domain_name,
                    keywords=keywords_data["regular"],
                    priority_keywords=keywords_data["priority"],
                    regex_patterns=regex_patterns,
                    confidence_threshold=0.25
                )
                
                self.dynamic_classifiers[domain_name] = classifier
    
    def add_dynamic_domain(self, 
                          domain_name: str, 
                          keywords: List[str],
                          priority_keywords: Optional[List[str]] = None,
                          regex_patterns: Optional[List[str]] = None):
        """
        Add a new domain classifier at runtime.
        
        Args:
            domain_name: Name of the new domain
            keywords: List of keywords for this domain
            priority_keywords: Keywords that strongly indicate this domain
            regex_patterns: Optional regex patterns for matching
        """
        classifier = DynamicDomainClassifier(
            domain_name=domain_name,
            keywords=keywords,
            priority_keywords=priority_keywords or [],
            regex_patterns=regex_patterns or [],
            confidence_threshold=0.3
        )
        
        self.dynamic_classifiers[domain_name] = classifier
        logger.info(f"Added dynamic domain classifier for '{domain_name}'")
    
    def classify(self, 
                query: str, 
                context: Optional[List[str]] = None,
                use_context: bool = True,
                include_dynamic: bool = True) -> ClassificationResult:
        """
        Classify the domain of a query using hybrid approach.
        
        Args:
            query: The text to classify
            context: Optional conversation context
            use_context: Whether to use context window
            include_dynamic: Whether to use dynamic classifiers for refinement
        
        Returns:
            ClassificationResult with rich metadata
        """
        # Update context window
        if use_context and self.enable_context:
            self._update_context(query)
        
        # Step 1: Multi-domain routing using patterns and keywords
        base_domain_scores = self._analyze_with_patterns(query)
        
        # Step 2: If dynamic classifiers are enabled, refine scores
        domain_specific_confidences = {}
        if include_dynamic:
            for domain_name, classifier in self.dynamic_classifiers.items():
                proba = classifier.predict_proba(query)[1]  # Get domain probability
                domain_specific_confidences[domain_name] = proba
                
                # Blend scores: 60% from patterns, 40% from dynamic classifier
                if domain_name in base_domain_scores:
                    base_domain_scores[domain_name] = (
                        base_domain_scores[domain_name] * 0.6 + proba * 0.4
                    )
        
        # Step 3: Consider context if available
        if use_context and self.context_window:
            context_scores = self._analyze_context()
            # Combine scores (70% query, 30% context)
            for domain in base_domain_scores:
                base_domain_scores[domain] = (
                    base_domain_scores[domain] * 0.7 + 
                    context_scores.get(domain, 0) * 0.3
                )
        
        # Apply domain weights from learning
        for domain, weight in self.domain_weights.items():
            if domain in base_domain_scores and weight > 1.0:
                base_domain_scores[domain] = min(1.0, base_domain_scores[domain] * weight)
        
        # Step 4: Get primary domain
        primary_domain = max(base_domain_scores.items(), key=lambda x: x[1])
        
        # Step 5: Get secondary domains
        secondary_domains = [
            {"domain": domain, "score": score}
            for domain, score in base_domain_scores.items()
            if domain != primary_domain[0] and score > 0.2
        ]
        secondary_domains.sort(key=lambda x: x["score"], reverse=True)
        
        # Step 6: Extract keywords
        keywords = self._extract_keywords(query)
        
        # Step 7: Get match details from dynamic classifier if available
        match_details = {}
        priority_keywords_matched = []
        
        if primary_domain[0] in self.dynamic_classifiers:
            classifier = self.dynamic_classifiers[primary_domain[0]]
            match_details = classifier.get_keyword_matches(query)
            priority_keywords_matched = match_details.get("priority_keywords", [])
        
        # Step 8: Generate reasoning
        reasoning = self._generate_reasoning(
            query, 
            primary_domain[0], 
            base_domain_scores,
            priority_keywords_matched
        )
        
        # Step 9: Record classification
        self._record_classification(query, primary_domain[0], primary_domain[1])
        
        return ClassificationResult(
            primary_domain=primary_domain[0],
            confidence=primary_domain[1],
            secondary_domains=secondary_domains,
            keywords=keywords,
            reasoning=reasoning,
            domain_specific_confidence=domain_specific_confidences,
            priority_keywords_matched=priority_keywords_matched,
            match_details=match_details
        )
    
    def _analyze_with_patterns(self, query: str) -> Dict[str, float]:
        """Analyze query using regex patterns and keyword lists."""
        domain_scores = {domain.value: 0.0 for domain in Domain}
        
        # Check regex patterns
        for domain_name, patterns in self.domain_patterns.items():
            pattern_score = 0.0
            for pattern in patterns:
                try:
                    matches = re.findall(pattern, query)
                    if matches:
                        pattern_score += len(matches) * 0.25
                except re.error:
                    continue  # Skip invalid patterns
            
            # Check keywords
            keyword_score = 0.0
            if domain_name in self.keyword_lists:
                keywords_data = self.keyword_lists[domain_name]
                
                # Check priority keywords
                for keyword in keywords_data["priority"]:
                    if re.search(rf'\b{re.escape(keyword)}\b', query, re.IGNORECASE):
                        keyword_score += 0.4
                
                # Check regular keywords
                for keyword in keywords_data["regular"]:
                    if re.search(rf'\b{re.escape(keyword)}\b', query, re.IGNORECASE):
                        keyword_score += 0.2
            
            # Combine scores
            total_score = min(1.0, pattern_score + keyword_score)
            domain_scores[domain_name] = total_score
        
        # Normalize scores
        total_score = sum(domain_scores.values())
        if total_score > 0:
            for domain in domain_scores:
                domain_scores[domain] /= total_score
        
        # If no strong signals, default to general with lower confidence
        max_score = max(domain_scores.values())
        if max_score < 0.3:
            domain_scores[Domain.GENERAL.value] = 0.6
            # Reduce other domains' scores
            for domain in domain_scores:
                if domain != Domain.GENERAL.value:
                    domain_scores[domain] *= 0.3
        
        return domain_scores
    
    def _analyze_context(self) -> Dict[str, float]:
        """Analyze context window for domain classification."""
        if not self.context_window:
            return {}
        
        domain_scores = {domain.value: 0.0 for domain in Domain}
        
        # Analyze last 3-5 context queries
        recent_context = self.context_window[-min(5, len(self.context_window)):]
        
        for context_query in recent_context:
            query_scores = self._analyze_with_patterns(context_query)
            for domain, score in query_scores.items():
                domain_scores[domain] += score
        
        # Average the scores
        for domain in domain_scores:
            domain_scores[domain] /= len(recent_context)
        
        return domain_scores
    
    def _extract_keywords(self, query: str, max_keywords: int = 8) -> List[str]:
        """Extract important keywords from query."""
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", 
            "to", "for", "of", "with", "by", "is", "are", "was", "were",
            "i", "you", "he", "she", "it", "we", "they", "my", "your",
            "his", "her", "its", "our", "their", "this", "that", "these",
            "those", "what", "which", "who", "whom", "whose", "how",
            "when", "where", "why", "can", "could", "will", "would",
            "should", "may", "might", "must", "have", "has", "had",
            "do", "does", "did", "am", "are", "was", "were", "be", "been",
            "being", "about", "above", "below", "from", "into", "over",
            "under", "again", "further", "then", "once", "here", "there",
            "all", "any", "both", "each", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "just", "now"
        }
        
        words = query.lower().split()
        keywords = []
        
        for word in words:
            # Clean word and check if it's meaningful
            clean_word = re.sub(r'[^\w\s]', '', word)
            
            if (len(clean_word) > 2 and 
                clean_word not in stop_words and 
                clean_word not in keywords and
                not clean_word.isnumeric()):
                
                # Check if it's a domain keyword
                is_domain_keyword = False
                for domain_data in self.keyword_lists.values():
                    if (clean_word in domain_data.get("regular", []) or 
                        clean_word in domain_data.get("priority", [])):
                        is_domain_keyword = True
                        break
                
                if is_domain_keyword or len(keywords) < max_keywords // 2:
                    keywords.append(clean_word)
        
        return keywords[:max_keywords]
    
    def _generate_reasoning(self, 
                           query: str, 
                           primary_domain: str,
                           domain_scores: Dict[str, float],
                           priority_keywords: List[str]) -> str:
        """Generate detailed reasoning for classification."""
        
        reasons = []
        
        # Priority keywords reasoning
        if priority_keywords:
            reasons.append(f"Contains priority keywords: {', '.join(priority_keywords[:3])}")
        
        # Pattern matches
        patterns = self.domain_patterns.get(primary_domain, [])
        matched_patterns = []
        
        for pattern in patterns:
            try:
                matches = re.findall(pattern, query)
                if matches:
                    matched_patterns.extend(matches)
            except re.error:
                continue
        
        if matched_patterns:
            unique_matches = list(set(matched_patterns))[:3]
            reasons.append(f"Matches {primary_domain} patterns: {', '.join(unique_matches)}")
        
        # Dynamic classifier confidence
        if primary_domain in self.dynamic_classifiers:
            classifier = self.dynamic_classifiers[primary_domain]
            proba = classifier.predict_proba(query)[1]
            reasons.append(f"Dynamic classifier confidence: {proba:.2f}")
        
        # Compare with other domains
        other_domains = [(domain, score) for domain, score in domain_scores.items() 
                        if domain != primary_domain and score > 0.2]
        
        if other_domains:
            other_domains.sort(key=lambda x: x[1], reverse=True)
            top_other = other_domains[0]
            reasons.append(f"Also relevant to {top_other[0]} (score: {top_other[1]:.2f})")
        
        # Context awareness note
        if self.context_window and len(self.context_window) > 1:
            recent_domain = self._get_recent_context_domain()
            if recent_domain == primary_domain:
                reasons.append("Consistent with recent conversation context")
        
        if not reasons:
            reasons.append("General query with no strong domain indicators")
        
        return "; ".join(reasons)
    
    def _get_recent_context_domain(self) -> Optional[str]:
        """Get the most common domain from recent context."""
        if not self.context_window:
            return None
        
        recent_domains = []
        for query in self.context_window[-3:]:
            scores = self._analyze_with_patterns(query)
            if scores:
                domain = max(scores.items(), key=lambda x: x[1])[0]
                recent_domains.append(domain)
        
        if recent_domains:
            from collections import Counter
            domain_counts = Counter(recent_domains)
            return domain_counts.most_common(1)[0][0]
        
        return None
    
    def _update_context(self, query: str):
        """Update the context window."""
        self.context_window.append(query)
        if len(self.context_window) > self.max_context:
            self.context_window = self.context_window[-self.max_context:]
    
    def _record_classification(self, query: str, domain: str, confidence: float):
        """Record classification for learning."""
        self.classification_history.append({
            "query": query[:100],  # Store first 100 chars
            "domain": domain,
            "confidence": confidence,
            "timestamp": np.datetime64('now')
        })
        
        # Update domain weights based on confidence
        if confidence > 0.7:
            self.domain_weights[domain] = min(2.0, self.domain_weights.get(domain, 1.0) + 0.05)
        elif confidence < 0.3:
            self.domain_weights[domain] = max(0.5, self.domain_weights.get(domain, 1.0) - 0.02)
    
    def clear_context(self):
        """Clear the context window."""
        self.context_window = []
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context."""
        if not self.context_window:
            return {"context_size": 0, "recent_domains": []}
        
        # Analyze recent queries
        recent_domains = []
        for query in self.context_window[-3:]:
            classification = self.classify(query, use_context=False)
            recent_domains.append({
                "query": query[:50] + "..." if len(query) > 50 else query,
                "domain": classification.primary_domain,
                "confidence": classification.confidence
            })
        
        return {
            "context_size": len(self.context_window),
            "recent_domains": recent_domains,
            "context_window": self.context_window[-3:]  # Last 3 queries
        }
    
    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get statistics about domain classifications."""
        if not self.classification_history:
            return {"total_classifications": 0, "domain_distribution": {}}
        
        domain_counts = {}
        domain_confidences = defaultdict(list)
        
        for entry in self.classification_history[-100:]:  # Last 100 classifications
            domain = entry["domain"]
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            domain_confidences[domain].append(entry["confidence"])
        
        total = sum(domain_counts.values())
        distribution = {}
        
        for domain, count in domain_counts.items():
            confidences = domain_confidences[domain]
            distribution[domain] = {
                "count": count,
                "percentage": (count / total) * 100,
                "avg_confidence": np.mean(confidences) if confidences else 0,
                "std_confidence": np.std(confidences) if len(confidences) > 1 else 0
            }
        
        # Dynamic classifier statistics
        dynamic_stats = {}
        for domain_name, classifier in self.dynamic_classifiers.items():
            dynamic_stats[domain_name] = classifier.get_statistics()
        
        return {
            "total_classifications": len(self.classification_history),
            "recent_classifications": len(self.classification_history[-100:]),
            "domain_distribution": distribution,
            "domain_weights": dict(self.domain_weights),
            "most_common_domain": max(domain_counts.items(), key=lambda x: x[1])[0] if domain_counts else "none",
            "dynamic_classifier_stats": dynamic_stats
        }
    
    def get_dynamic_classifier(self, domain_name: str) -> Optional[DynamicDomainClassifier]:
        """Get a dynamic classifier for a specific domain."""
        return self.dynamic_classifiers.get(domain_name)
    
    def predict_single_domain(self, query: str, domain_name: str) -> Tuple[bool, float]:
        """
        Check if a query belongs to a specific domain.
        
        Args:
            query: Text to classify
            domain_name: Domain to check against
            
        Returns:
            Tuple of (is_domain, confidence)
        """
        if domain_name in self.dynamic_classifiers:
            classifier = self.dynamic_classifiers[domain_name]
            is_domain = classifier.predict(query) == 1
            confidence = classifier.predict_proba(query)[1]
            return is_domain, confidence
        
        # Fallback to pattern matching
        scores = self._analyze_with_patterns(query)
        domain_score = scores.get(domain_name, 0.0)
        return domain_score > 0.5, domain_score
    
    def export_knowledge(self) -> Dict[str, Any]:
        """Export classifier knowledge for persistence."""
        return {
            "domain_patterns": self.domain_patterns,
            "keyword_lists": self.keyword_lists,
            "domain_weights": dict(self.domain_weights),
            "classification_history_count": len(self.classification_history),
            "dynamic_domains": list(self.dynamic_classifiers.keys()),
            "export_timestamp": np.datetime64('now').astype(str)
        }
    
    def import_knowledge(self, knowledge_data: Dict[str, Any]):
        """Import classifier knowledge."""
        if "domain_weights" in knowledge_data:
            self.domain_weights.update(knowledge_data["domain_weights"])

# Alias for backward compatibility
DomainClassifier = EnhancedDomainClassifier 
