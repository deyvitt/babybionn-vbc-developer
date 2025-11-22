# enhanced_vni_classes.py - Complete Implementation
import re
import json
import hashlib
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque

# Optional PyTorch support for attention mechanisms
try:
    import torch
    import torch.nn as nn
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available - attention mechanisms disabled")

# Predictive modules with fallback
try:
    from predictive_vocabulary import PredictiveVocabulary
    from predictive_response import PredictiveResponseGenerator
    PREDICTIVE_AVAILABLE = True
except ImportError:
    PREDICTIVE_AVAILABLE = False
    logging.warning("Predictive modules not available - using fallback implementations")

# Web search capabilities
try:
    from duckduckgo_search import DDGS
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    logging.warning("DuckDuckGo search not available - install with: pip install duckduckgo-search")

logger = logging.getLogger("enhanced_vni_classes")


class NeuralPathway:
    """Represents a dynamic synaptic connection between VNIs"""
    
    def __init__(self, source_vni: str, target_vni: str, initial_strength: float = 0.5):
        self.source = source_vni
        self.target = target_vni
        self.strength = initial_strength
        self.activation_count = 0
        self.success_count = 0
        self.last_activated = None
        self.learning_rate = 0.1
        self.decay_rate = 0.01

    def activate(self, success: bool = True):
        """Activate pathway and update strength based on success"""
        self.activation_count += 1
        self.last_activated = datetime.now()

        if success:
            self.success_count += 1
            self.strength = min(1.0, self.strength + self.learning_rate)
        else:
            self.strength = max(0.1, self.strength - (self.learning_rate * 0.5))

    def decay(self):
        """Apply temporal decay to pathway strength"""
        if self.last_activated:
            time_diff = datetime.now() - self.last_activated
            if time_diff.days > 7:
                self.strength = max(0.1, self.strength - self.decay_rate)

    def get_success_rate(self) -> float:
        return self.success_count / self.activation_count if self.activation_count > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'target': self.target,
            'strength': self.strength,
            'activation_count': self.activation_count,
            'success_count': self.success_count,
            'last_activated': self.last_activated.isoformat() if self.last_activated else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuralPathway':
        pathway = cls(data['source'], data['target'], data['strength'])
        pathway.activation_count = data.get('activation_count', 0)
        pathway.success_count = data.get('success_count', 0)
        if data.get('last_activated'):
            pathway.last_activated = datetime.fromisoformat(data['last_activated'])
        return pathway


class EnhancedBaseVNI:
    """Base VNI with self-learning, attention mechanisms, and predictive capabilities"""
    
    def __init__(self, vni_type: str, instance_id: str):
        self.vni_type = vni_type
        self.instance_id = instance_id
        self.knowledge_base = self._load_knowledge_base()
        self.connection_weights = {}
        self.learning_history = []
        
        # Self-learning attributes
        self.conversation_memory = deque(maxlen=1000)
        self.learned_responses = {}
        self.context_memory = {}
        self.adaptation_rate = 0.3
        self.usage_threshold = 2
        self.memory_window = 50
        
        self.confidence_threshold = 0.7
        self.adaptive_learning_rate = 0.1

        # Attention mechanism (from File 1)
        self.attention_weights = None
        self.attention_enabled = TORCH_AVAILABLE

        # Web search
        self.web_search_enabled = WEB_SEARCH_AVAILABLE
        self.search_cache = {}
        self.max_search_cache_size = 100

        # Predictive systems
        if PREDICTIVE_AVAILABLE:
            self.predictive_vocab = PredictiveVocabulary()
            self.response_generator = PredictiveResponseGenerator()
        else:
            self.predictive_vocab = self._create_fallback_predictive_vocab()
            self.response_generator = self._create_fallback_response_generator()
            
        self.conversation_context = []
        self.neural_pathways = {}
        
        self._initialize_default_knowledge()

    # ==================== FALLBACK IMPLEMENTATIONS ====================

    def _create_fallback_predictive_vocab(self):
        """Fallback predictive vocabulary when module unavailable"""
        class FallbackPredictiveVocabulary:
            def __init__(self):
                self.vocabulary = defaultdict(int)
                self.context_patterns = defaultdict(list)
                self.domain_patterns = defaultdict(list)
                
            def update_vocabulary(self, text: str, domain: str):
                words = text.lower().split()
                for word in words:
                    if len(word) > 3:
                        self.vocabulary[word] += 1
                        self.domain_patterns[domain].append(word)
                        
            def get_predictive_suggestions(self, text: str, max_suggestions: int = 3) -> List[str]:
                words = text.lower().split()
                if not words:
                    return []
                last_word = words[-1]
                suggestions = [w for w in self.vocabulary if w.startswith(last_word) and w != last_word]
                return suggestions[:max_suggestions]
                
            def get_predictive_completions(self, text: str, max_completions: int = 3) -> List[str]:
                return self.get_predictive_suggestions(text, max_completions)
                
            def get_expansions(self, topic: str, domain: str) -> List[str]:
                related = [(w, c) for w, c in self.vocabulary.items() if w != topic and len(w) > 4]
                related.sort(key=lambda x: x[1], reverse=True)
                return [w for w, c in related[:5]]
                
            def get_guidance_patterns(self, domain: str) -> List[str]:
                patterns = {
                    'medical': ["assess symptoms carefully", "consider medical history", "consult healthcare provider"],
                    'legal': ["review relevant statutes", "document all evidence", "seek legal counsel"],
                    'general': ["analyze the situation", "consider all factors", "break down systematically"]
                }
                return patterns.get(domain, patterns['general'])
        
        return FallbackPredictiveVocabulary()

    def _create_fallback_response_generator(self):
        """Fallback response generator when module unavailable"""
        parent = self
        
        class FallbackResponseGenerator:
            def generate_response(self, query: str, vni_type: str, context=None, predictive_suggestions=None):
                base = {
                    "medical": "From a medical perspective, this requires careful symptom evaluation.",
                    "legal": "Legal matters should be reviewed by qualified professionals.",
                    "general": "I can help analyze this from multiple perspectives."
                }
                response = base.get(vni_type, "I'll help you work through this.")
                if predictive_suggestions:
                    response += f" Related: {', '.join(predictive_suggestions[:2])}."
                return {"response": response, "confidence": 0.6}
        
        return FallbackResponseGenerator()

    # ==================== ATTENTION MECHANISM (from File 1) ====================

    def integrate_attention_weights(self, attention_weights):
        """Integrate attention weights for improved response generation"""
        if TORCH_AVAILABLE and attention_weights is not None:
            self.attention_weights = attention_weights
            logger.debug(f"{self.instance_id} integrated attention weights")
        
    def process_query_with_attention(self, query: str, context: Dict, attention_scores: Dict) -> Dict[str, Any]:
        """Process query with attention-guided context"""
        self_attention_score = attention_scores.get(self.instance_id, 0.5)
        
        # Low attention = minimal participation
        if self_attention_score < 0.3:
            return {
                "response": "",
                "confidence": 0.1,
                "vni_instance": self.instance_id,
                "attention_score": self_attention_score,
                "active": False
            }
        
        # Process with attention weighting
        base_response = self.process_query(query, context)
        base_response["attention_score"] = self_attention_score
        base_response["confidence"] = base_response.get("confidence", 0.5) * self_attention_score
        base_response["active"] = True
        
        return base_response

    # ==================== KNOWLEDGE BASE MANAGEMENT ====================

    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load knowledge base from file"""
        import os
        
        knowledge_file = f"knowledge_{self.vni_type}_{self.instance_id}.json"
        default_knowledge = {
            "concepts": {},
            "patterns": {},
            "corrections": {},
            "response_templates": self.get_default_response_templates(),
            "learned_responses": {},
            "metadata": {
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "2.0"
            }
        }
        
        try:
            if os.path.exists(knowledge_file):
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    for key, value in default_knowledge.items():
                        if key not in loaded:
                            loaded[key] = value
                    self.learned_responses = loaded.get('learned_responses', {})
                    return loaded
            else:
                with open(knowledge_file, 'w', encoding='utf-8') as f:
                    json.dump(default_knowledge, f, indent=2, ensure_ascii=False)
                logger.info(f"Created knowledge base: {knowledge_file}")
                return default_knowledge
        except Exception as e:
            logger.warning(f"Failed to load knowledge base: {e}")
            return default_knowledge

    def _initialize_default_knowledge(self):
        """Initialize with domain-specific defaults"""
        if not self.knowledge_base["concepts"]:
            self.knowledge_base["concepts"].update(self.get_default_concepts())
        if not self.knowledge_base["patterns"]:
            self.knowledge_base["patterns"].update(self.get_default_patterns())
        self.save_knowledge_base()

    def save_knowledge_base(self):
        """Save knowledge base to file"""
        filename = f"knowledge_{self.vni_type}_{self.instance_id}.json"
        self.knowledge_base['learned_responses'] = self.learned_responses
        self.knowledge_base['metadata']['last_updated'] = datetime.now().isoformat()
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")

    def get_default_concepts(self) -> Dict[str, Any]:
        return {"help": {"strength": 0.8, "usage_count": 0}, "information": {"strength": 0.7, "usage_count": 0}}

    def get_default_patterns(self) -> Dict[str, Any]:
        return {"general_help": {"triggers": ["help", "assist"], "responses": ["I'm here to help."], "strength": 0.7, "usage_count": 0}}

    def get_default_response_templates(self) -> Dict[str, List[str]]:
        return {"general": ["I understand this is a {vni_type} question. Could you provide more details?"]}

    # ==================== CORE QUERY PROCESSING ====================

    def process_query(self, query: str, context: Dict = None, session_id: str = "default") -> Dict[str, Any]:
        """Main query processing with all features integrated"""
        
        # Update systems
        self.update_context_memory(session_id, query, context)
        self.predictive_vocab.update_vocabulary(query, self.vni_type)
        
        # Get context and predictions
        current_context = self.get_current_context(session_id)
        predictive_suggestions = self.predictive_vocab.get_predictive_suggestions(query)
        
        # Try learned responses first
        learned = self.get_learned_response(query, current_context)
        if learned and learned['usage_count'] >= self.usage_threshold:
            response_text = learned['response']
            confidence = 0.8
            concepts, patterns = [], ["learned_response"]
        else:
            # Generate new response using predictive system
            concepts, patterns = self.extract_concepts_and_patterns(query)
            response_text = self.generate_truly_predictive_response(query, current_context, predictive_suggestions)
            confidence = self.calculate_confidence(concepts, patterns)
            self.learn_from_conversation(query, response_text, current_context, session_id)

        # Web search enhancement
        if self.web_search_enabled and self.needs_web_search(query, confidence, concepts):
            response_text = self.enhance_with_web_search(query, response_text, concepts, confidence)

        self.remember_conversation(session_id, query, response_text, current_context)
        
        response_data = {
            'response': response_text,
            'confidence': confidence,
            'vni_instance': self.instance_id,
            'concepts_used': concepts,
            'patterns_matched': patterns,
            'response_type': f'{self.vni_type}_response',
            'context_used': bool(current_context),
            'web_search_used': self.web_search_enabled and self.needs_web_search(query, confidence, concepts),
            'timestamp': datetime.now().isoformat()
        }

        self.learning_history.append({
            'query': query, 'response': response_data,
            'timestamp': datetime.now().isoformat(),
            'context': current_context, 'session_id': session_id
        })

        if len(self.learning_history) % 10 == 0:
            self.save_knowledge_base()

        return response_data

    # ==================== PREDICTIVE RESPONSE GENERATION (from File 1) ====================

    def generate_truly_predictive_response(self, query: str, context: Dict, predictive_suggestions: List[str] = None) -> str:
        """Generate response using learned patterns and predictions"""
        
        # Get completions and analyze trend
        completions = self.predictive_vocab.get_predictive_completions(query)
        conversation_trend = self._analyze_conversation_trend(context)
        
        # Predict intent and generate
        if completions:
            predicted_intent = self._predict_user_intent(query, completions)
            return self._generate_from_predicted_intent(predicted_intent, context)
        else:
            concepts, patterns = self.extract_concepts_and_patterns(query)
            return self.generate_adaptive_response(query, concepts, patterns, context)

    def _analyze_conversation_trend(self, context: Dict) -> Dict[str, Any]:
        """Analyze conversation patterns"""
        recent = context.get('recent_conversations', [])
        if not recent:
            return {'trend': 'new_conversation', 'topic_consistency': 0.5}
        
        topics = [c.get('context_notes', {}).get('query_complexity', 'medium') for c in recent[-3:]]
        user_style = context.get('user_style', {})
        
        return {
            'trend': 'stable',
            'topic_consistency': len(set(topics)) / len(topics) if topics else 0.5,
            'conversation_depth': len(recent),
            'user_engagement': user_style.get('detail_level', 'medium')
        }

    def _predict_user_intent(self, query: str, completions: List[str]) -> str:
        """Predict user's underlying intent"""
        q = query.lower()
        
        # Check completions
        for comp in completions:
            if comp in q:
                return f"information_request_{comp}"
        
        # Intent classification
        if any(w in q for w in ['how', 'can i', 'what is']):
            return "how_to_question"
        elif any(w in q for w in ['why', 'explain']):
            return "explanation_request"
        elif any(w in q for w in ['help', 'problem', 'issue']):
            return "help_request"
        return "general_inquiry"

    def _generate_from_predicted_intent(self, intent: str, context: Dict) -> str:
        """Generate response based on predicted intent"""
        
        # Check learned responses
        intent_pattern = f"intent_{intent}"
        if intent_pattern in self.learned_responses:
            learned = self.learned_responses[intent_pattern]
            if learned['usage_count'] >= self.usage_threshold:
                return learned['response']
        
        # Generate by intent type
        if intent.startswith("information_request_"):
            topic = intent.replace("information_request_", "")
            return self._generate_informative_response(topic, context)
        elif intent == "how_to_question":
            return self._generate_guidance_response(context)
        elif intent == "explanation_request":
            return self._generate_explanation_response(context)
        elif intent == "help_request":
            return self._generate_help_response(context)
        else:
            return self._generate_general_response(context)

    def _generate_informative_response(self, topic: str, context: Dict) -> str:
        """Generate informative response using knowledge base"""
        expansions = self.predictive_vocab.get_expansions(topic, self.vni_type)
        
        if topic in self.knowledge_base.get('concepts', {}):
            concept_data = self.knowledge_base['concepts'][topic]
            strength = concept_data.get('strength', 0.5)
            
            if strength > 0.8:
                return f"I have strong knowledge about {topic}. {self._get_detailed_explanation(topic, expansions)}"
            elif strength > 0.6:
                return f"Regarding {topic}, {self._get_general_explanation(topic, expansions)}"
            else:
                return f"I'm still learning about {topic}, but: {self._get_basic_explanation(topic)}"
        else:
            if expansions:
                return f"Based on patterns, {topic} typically involves: {', '.join(expansions[:3])}."
            return f"{topic} requires specific expertise. Could you provide more context?"

    def _generate_guidance_response(self, context: Dict) -> str:
        """Generate step-by-step guidance"""
        recent_topics = context.get('topics_discussed', [])
        
        if 'help_seeking' in recent_topics:
            return "I notice you've been seeking help. Let me provide clear, step-by-step guidance."
        
        patterns = self.predictive_vocab.get_guidance_patterns(self.vni_type)
        if patterns:
            return f"Let me guide you: {patterns[0]}"
        return "I'll help you work through this systematically with careful planning."

    def _generate_explanation_response(self, context: Dict) -> str:
        """Generate explanation based on user style"""
        user_style = context.get('user_style', {})
        detail_level = user_style.get('detail_level', 'medium')
        
        if detail_level == 'high':
            return self._get_comprehensive_explanation()
        elif detail_level == 'low':
            return self._get_concise_explanation()
        return self._get_balanced_explanation()

    def _generate_help_response(self, context: Dict) -> str:
        """Generate helpful response based on context"""
        conv_length = context.get('conversation_length', 0)
        
        if conv_length > 10:
            return "Thank you for continuing our discussion. Let me provide focused assistance."
        return "I'm here to help! Please share more about your specific situation."

    def _generate_general_response(self, context: Dict) -> str:
        """Generate general response using successful patterns"""
        successful = [p for p in self.learned_responses.values() if p.get('success_rate', 0) > 0.7]
        
        if successful:
            best = max(successful, key=lambda x: x.get('success_rate', 0))
            return best['response']
        return "I'd be happy to help. Could you provide a bit more context?"

    # ==================== EXPLANATION HELPERS (from File 1) ====================

    def _get_detailed_explanation(self, topic: str, expansions: List[str]) -> str:
        if expansions:
            return f"Key aspects include {', '.join(expansions[:4])}. Want me to elaborate on any?"
        return "This is well-established with clear principles tailored to specific needs."

    def _get_general_explanation(self, topic: str, expansions: List[str]) -> str:
        if expansions:
            return f"this typically involves {expansions[0]} and requires context assessment."
        return "understanding fundamental principles and their practical applications is key."

    def _get_basic_explanation(self, topic: str) -> str:
        return "this involves important considerations that should be approached with proper understanding."

    def _get_comprehensive_explanation(self) -> str:
        return "Let me provide a comprehensive explanation covering fundamentals, applications, considerations, and variations."

    def _get_concise_explanation(self) -> str:
        return "Key points: core principles, practical application, and context-specific considerations."

    def _get_balanced_explanation(self) -> str:
        return "This involves understanding core concepts, implementation, and scenario-specific adjustments."

    def _get_topic_details(self, topic: str) -> str:
        if topic in self.knowledge_base.get('concepts', {}):
            strength = self.knowledge_base['concepts'][topic].get('strength', 0.5)
            if strength > 0.7:
                return "This is well-established with clear principles and best practices."
            return "I'm developing understanding but can share what I've learned."
        return "This involves multiple considerations worth exploring."

    # ==================== ADAPTIVE RESPONSE GENERATION ====================

    def generate_adaptive_response(self, query: str, concepts: List[str], patterns: List[str], context: Dict) -> str:
        """Generate responses that adapt based on context"""
        
        if patterns and random.random() > 0.3:
            pattern_id = patterns[0]
            if pattern_id in self.knowledge_base.get("patterns", {}):
                pattern_data = self.knowledge_base["patterns"][pattern_id]
                base = random.choice(pattern_data.get("responses", []))
                return self.personalize_response(base, query, context)
        
        return self.create_contextual_response(query, concepts, context)

    def personalize_response(self, base: str, query: str, context: Dict) -> str:
        """Personalize response based on context"""
        personalized = base
        
        # Add contextual references
        if 'help_seeking' in context.get('topics_discussed', []):
            personalized += " I'm here to help with any other questions."
        
        # Check for follow-up
        recent = context.get('recent_conversations', [])
        if len(recent) > 1 and any(w in query.lower() for w in ['more', 'else', 'another', 'also']):
            personalized = "Additionally, " + personalized.lower()
        
        # Adjust for user style
        user_style = context.get('user_style', {})
        if user_style.get('detail_level') == 'low' and len(personalized.split()) > 25:
            sentences = personalized.split('.')
            personalized = sentences[0] + '.' if sentences else personalized
        
        return personalized

    def create_contextual_response(self, query: str, concepts: List[str], context: Dict) -> str:
        """Create dynamic response from concepts and context"""
        if concepts:
            concept = concepts[0]
            if concept in self.knowledge_base["concepts"]:
                strength = self.knowledge_base["concepts"][concept].get('strength', 0.5)
                if strength > 0.7:
                    return self.create_confident_response(concept, query, context)
                return self.create_learning_response(concept, query, context)
        
        return self.create_contextual_fallback(query, context)

    def create_confident_response(self, concept: str, query: str, context: Dict) -> str:
        base = f"Based on established {self.vni_type} knowledge about {concept}, "
        if context.get('user_style', {}).get('detail_level') == 'high':
            base += f"this involves comprehensive understanding. Want more details?"
        else:
            base += "the approach focuses on practical application and key considerations."
        return base

    def create_learning_response(self, concept: str, query: str, context: Dict) -> str:
        return f"I'm developing understanding of {concept} in {self.vni_type} contexts. Based on what I've learned, this involves important considerations."

    def create_contextual_fallback(self, query: str, context: Dict) -> str:
        conv_length = context.get('conversation_length', 0)
        if conv_length > 5:
            return f"As we continue discussing {self.vni_type} topics, could you provide more specific details?"
        return f"As a {self.vni_type} AI, I'm here to help. Could you provide more details?"

    # ==================== CONTEXT MEMORY SYSTEM ====================

    def update_context_memory(self, session_id: str, query: str, context: Dict = None):
        if session_id not in self.context_memory:
            self.context_memory[session_id] = {
                'conversation_history': deque(maxlen=self.memory_window),
                'user_preferences': {},
                'topics_discussed': [],
                'last_updated': datetime.now().isoformat(),
                'session_start': datetime.now().isoformat()
            }
        
        context_notes = self.extract_context_notes(query, context)
        self.context_memory[session_id]['conversation_history'].append({
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'context_notes': context_notes
        })
        self._update_session_topics(session_id, query, context_notes)
        self.context_memory[session_id]['last_updated'] = datetime.now().isoformat()

    def get_current_context(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.context_memory:
            return {}
        
        ctx = self.context_memory[session_id]
        recent = list(ctx['conversation_history'])[-5:]
        
        return {
            'session_id': session_id,
            'recent_conversations': recent,
            'topics_discussed': ctx['topics_discussed'],
            'user_style': self.analyze_user_style(recent),
            'conversation_length': len(ctx['conversation_history']),
            'session_duration': self._get_session_duration(session_id)
        }

    def _update_session_topics(self, session_id: str, query: str, context_notes: Dict):
        new_topics = []
        if context_notes.get('query_complexity') == 'high':
            new_topics.append('complex_queries')
        if context_notes.get('emotional_tone') == 'urgent':
            new_topics.append('urgent_requests')
        if 'help' in query.lower():
            new_topics.append('help_seeking')
        
        current = set(self.context_memory[session_id]['topics_discussed'])
        current.update(new_topics)
        self.context_memory[session_id]['topics_discussed'] = list(current)

    def _get_session_duration(self, session_id: str) -> float:
        if session_id not in self.context_memory:
            return 0.0
        start = datetime.fromisoformat(self.context_memory[session_id]['session_start'])
        return (datetime.now() - start).total_seconds() / 60

    # ==================== SELF-LEARNING SYSTEM ====================

    def get_learned_response(self, query: str, context: Dict) -> Optional[Dict[str, Any]]:
        pattern = self.extract_query_pattern(query)
        
        if pattern in self.learned_responses:
            learned = self.learned_responses[pattern]
            if self.is_context_similar(learned.get('context', {}), context):
                return learned
        
        for p, learned in self.learned_responses.items():
            if self.are_patterns_similar(pattern, p) and self.is_context_similar(learned.get('context', {}), context):
                return learned
        return None

    def learn_from_conversation(self, query: str, response: str, context: Dict, session_id: str):
        pattern = self.extract_query_pattern(query)
        
        if pattern in self.learned_responses:
            self.learned_responses[pattern]['usage_count'] += 1
            self.learned_responses[pattern]['last_used'] = datetime.now().isoformat()
            self.learned_responses[pattern]['success_rate'] = min(1.0, 
                self.learned_responses[pattern].get('success_rate', 0.7) + 0.05)
            if random.random() < self.adaptation_rate:
                self.refine_response(pattern, response, context)
        else:
            self.learned_responses[pattern] = {
                'response': response,
                'usage_count': 1,
                'success_rate': 0.7,
                'last_used': datetime.now().isoformat(),
                'created': datetime.now().isoformat(),
                'context': context,
                'session_id': session_id,
                'source_type': 'conversation_learning'
            }
        
        self.learn_concepts_from_text(query + " " + response)

    def extract_query_pattern(self, query: str) -> str:
        words = query.lower().split()
        stop_words = {'i', 'you', 'the', 'a', 'an', 'is', 'are', 'what', 'how', 'my', 'me', 'can'}
        meaningful = [w for w in words if w not in stop_words and len(w) > 2]
        return " ".join(meaningful[:4]) if meaningful else query.lower()[:20]

    def are_patterns_similar(self, p1: str, p2: str, threshold: float = 0.6) -> bool:
        w1, w2 = set(p1.split()), set(p2.split())
        if not w1 or not w2:
            return False
        return len(w1 & w2) / max(len(w1), len(w2)) >= threshold

    def is_context_similar(self, c1: Dict, c2: Dict) -> bool:
        if not c1 or not c2:
            return True
        t1 = set(c1.get('topics_discussed', []))
        t2 = set(c2.get('topics_discussed', []))
        if t1 and t2 and len(t1 & t2) / len(t1 | t2) < 0.3:
            return False
        return c1.get('user_style', {}).get('style') == c2.get('user_style', {}).get('style', 'unknown')

    def refine_response(self, pattern: str, response: str, context: Dict):
        learned = self.learned_responses[pattern]
        style = context.get('user_style', {})
        if style.get('detail_level') == 'low' and len(response.split()) > 20:
            learned['response'] = self.make_concise(response)
        elif style.get('detail_level') == 'high':
            learned['response'] = self.add_detail(response, context)

    def learn_concepts_from_text(self, text: str):
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        for word in words:
            if word not in self.knowledge_base['concepts']:
                self.knowledge_base['concepts'][word] = {
                    'strength': 0.3,
                    'usage_count': 1,
                    'discovered_in_conversation': True,
                    'first_seen': datetime.now().isoformat(),
                    'last_used': datetime.now().isoformat()
                }
                logger.debug(f"Learned concept: {word}")

    def remember_conversation(self, session_id: str, query: str, response: str, context: Dict):
        self.conversation_memory.append({
            'session_id': session_id,
            'query': query,
            'response': response,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'vni_instance': self.instance_id
        })

    # ==================== WEB SEARCH ====================

    def needs_web_search(self, query: str, confidence: float, concepts: List[str]) -> bool:
        q = query.lower()
        keywords = ['current', 'recent', 'latest', 'today', 'now', 'new', 'update', 'news']
        return (confidence < 0.4 or any(k in q for k in keywords) or len(concepts) == 0) and self.web_search_enabled

    def search_web_for_information(self, query: str, max_results: int = 3) -> str:
        if not self.web_search_enabled:
            return "Web search not available"
        
        cache_key = hashlib.md5(query.lower().encode()).hexdigest()
        if cache_key in self.search_cache:
            cached_time, cached_result = self.search_cache[cache_key]
            if (datetime.now() - cached_time).seconds < 3600:
                return cached_result
        
        try:
            logger.info(f"Searching: {query}")
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                if not results:
                    return "No relevant information found."
                
                combined = []
                for i, r in enumerate(results[:max_results]):
                    combined.append(f"{i+1}. {r.get('title', '')}: {r.get('body', '')}")
                
                web_content = " ".join(combined)
                
                if len(self.search_cache) >= self.max_search_cache_size:
                    oldest = min(self.search_cache.keys(), key=lambda k: self.search_cache[k][0])
                    del self.search_cache[oldest]
                
                self.search_cache[cache_key] = (datetime.now(), web_content)
                return web_content[:1000]
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return f"Search error: {str(e)}"

    def enhance_with_web_search(self, query: str, response: str, concepts: List[str], confidence: float) -> str:
        web_info = self.search_web_for_information(query)
        if web_info and "error" not in web_info.lower() and "not available" not in web_info.lower():
            self.learn_concepts_from_text(web_info)
            return f"{response}\n\n🔍 **Latest Information:**\n{web_info}"
        return response

    # ==================== UTILITY METHODS ====================

    def extract_concepts_and_patterns(self, text: str) -> Tuple[List[str], List[str]]:
        text_lower = text.lower()
        concepts = [c for c in self.knowledge_base.get("concepts", {}) if c in text_lower]
        patterns = []
        for pid, pdata in self.knowledge_base.get("patterns", {}).items():
            if any(t in text_lower for t in pdata.get("triggers", [])):
                patterns.append(pid)
        return concepts, patterns

    def calculate_confidence(self, concepts: List[str], patterns: List[str]) -> float:
        if not concepts and not patterns:
            return 0.3
        total = len(concepts) + len(patterns)
        conf = sum(self.knowledge_base["concepts"].get(c, {}).get('strength', 0.5) for c in concepts)
        conf += sum(self.knowledge_base["patterns"].get(p, {}).get('strength', 0.7) for p in patterns)
        return min(1.0, conf / total) if total > 0 else 0.3

    def extract_context_notes(self, query: str, context: Dict = None) -> Dict[str, Any]:
        return {
            'query_complexity': self.assess_complexity(query),
            'emotional_tone': self.detect_emotional_tone(query),
            'urgency_level': self.detect_urgency(query),
            'specificity': self.assess_specificity(query),
            'user_knowledge_level': context.get('user_profile', {}).get('knowledge_level', 'unknown') if context else 'unknown'
        }

    def analyze_user_style(self, conversations: List[Dict]) -> Dict[str, str]:
        if not conversations:
            return {'style': 'unknown', 'detail_level': 'medium'}
        queries = [c.get('query', '') for c in conversations]
        avg_len = sum(len(q) for q in queries) / len(queries)
        return {
            'style': 'detailed' if avg_len > 50 else 'concise',
            'detail_level': 'high' if avg_len > 80 else 'medium' if avg_len > 30 else 'low'
        }

    def assess_complexity(self, query: str) -> str:
        wc = len(query.split())
        return 'high' if wc > 15 else 'medium' if wc > 8 else 'low'

    def detect_emotional_tone(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ['urgent', 'emergency', 'help', 'asap']):
            return 'urgent'
        if any(w in q for w in ['thank', 'appreciate', 'good', 'great']):
            return 'positive'
        if any(w in q for w in ['problem', 'issue', 'wrong', 'bad']):
            return 'negative'
        return 'neutral'

    def detect_urgency(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ['emergency', 'urgent', 'asap', 'immediately']):
            return 'high'
        if any(w in q for w in ['soon', 'quick', 'fast']):
            return 'medium'
        return 'low'

    def assess_specificity(self, query: str) -> str:
        specific = len(re.findall(r'\b(\w+ing|\w+ed|\w+ly|\d+)\b', query))
        return 'high' if specific > 3 else 'medium' if specific > 1 else 'low'

    def make_concise(self, response: str) -> str:
        sentences = response.split('.')
        return sentences[0] + '.' if sentences else response

    def add_detail(self, response: str, context: Dict) -> str:
        if 'more detail' not in response.lower():
            return response + " Would you like more specific details?"
        return response

    def learn_from_feedback(self, feedback_data: Dict[str, Any]):
        try:
            msg_id = feedback_data.get('message_id')
            fb_type = feedback_data.get('feedback_type')
            correction = feedback_data.get('correction')
            
            original = self.find_original_message(msg_id)
            if original:
                pattern = self.extract_query_pattern(original.get('query', ''))
                if pattern in self.learned_responses:
                    if fb_type == 'positive':
                        self.learned_responses[pattern]['success_rate'] = min(1.0, 
                            self.learned_responses[pattern]['success_rate'] + 0.2)
                    elif fb_type == 'negative':
                        self.learned_responses[pattern]['success_rate'] = max(0.0,
                            self.learned_responses[pattern]['success_rate'] - 0.3)
                    if correction:
                        self.learned_responses[pattern]['response'] = correction
        except Exception as e:
            logger.error(f"Feedback learning failed: {e}")

    def find_original_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        for msg in self.conversation_memory:
            if str(hash(msg.get('query', '')))[:8] == message_id:
                return msg
        return None

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            'vni_type': self.vni_type,
            'instance_id': self.instance_id,
            'web_search_enabled': self.web_search_enabled,
            'attention_enabled': self.attention_enabled,
            'predictive_capabilities': PREDICTIVE_AVAILABLE,
            'learning_enabled': True,
            'context_memory': True,
            'knowledge_base_size': len(self.knowledge_base.get('concepts', {})),
            'learned_responses_count': len(self.learned_responses)
        }


# ==================== SPECIALIZED VNI CLASSES ====================

class EnhancedMedicalVNI(EnhancedBaseVNI):
    """Medical VNI with specialized knowledge"""
    
    def __init__(self, instance_id: str):
        super().__init__("medical", instance_id)

    def get_default_concepts(self) -> Dict[str, Any]:
        return {
            'fever': {'strength': 0.8, 'usage_count': 0},
            'headache': {'strength': 0.8, 'usage_count': 0},
            'pain': {'strength': 0.9, 'usage_count': 0},
            'symptom': {'strength': 0.8, 'usage_count': 0},
            'treatment': {'strength': 0.8, 'usage_count': 0},
            'diagnosis': {'strength': 0.8, 'usage_count': 0},
            'medicine': {'strength': 0.7, 'usage_count': 0},
            'health': {'strength': 0.9, 'usage_count': 0},
            'covid': {'strength': 0.8, 'usage_count': 0},
            'vaccine': {'strength': 0.7, 'usage_count': 0},
            'patient': {'strength': 0.8, 'usage_count': 0},
            'doctor': {'strength': 0.9, 'usage_count': 0},
            'flu': {'strength': 0.8, 'usage_count': 0},
            'cold': {'strength': 0.7, 'usage_count': 0},
            'infection': {'strength': 0.8, 'usage_count': 0},
            'medication': {'strength': 0.8, 'usage_count': 0}
        }

    def get_default_patterns(self) -> Dict[str, Any]:
        return {
            "fever_advice": {
                "triggers": ["fever", "temperature", "hot", "burning up"],
                "responses": [
                    "Fever often indicates infection. Rest, hydration, and monitoring are important. Consult a doctor if it persists.",
                    "For fever, ensure proper hydration and rest. Seek medical advice for proper diagnosis.",
                    "Fever can indicate various conditions. Monitor symptoms and seek attention for breathing difficulty."
                ],
                "strength": 0.8, "usage_count": 0
            },
            "headache_help": {
                "triggers": ["headache", "head pain", "migraine"],
                "responses": [
                    "Headaches have various causes including stress and dehydration. Rest and hydration often help.",
                    "Consider triggers like stress, lack of sleep, or dehydration. Persistent headaches should be evaluated."
                ],
                "strength": 0.7, "usage_count": 0
            },
            "pain_assessment": {
                "triggers": ["pain", "hurts", "ache", "sore"],
                "responses": [
                    "Pain assessment requires understanding location, intensity, and duration. Professional evaluation is recommended.",
                    "Pain can have many causes. Note when it started, what makes it worse/better, and consult healthcare provider."
                ],
                "strength": 0.8, "usage_count": 0
            }
        }

    def get_default_response_templates(self) -> Dict[str, List[str]]:
        return {
            "general": [
                "From a medical perspective, {concept} should be evaluated by a healthcare professional.",
                "Regarding {concept}, consider individual health factors and consult a doctor.",
                "Medical advice for {concept} depends on specific symptoms and history."
            ],
            "emergency": [
                "If this is a medical emergency, please seek immediate professional help.",
                "For urgent medical concerns, contact emergency services or visit a hospital."
            ]
        }


class EnhancedLegalVNI(EnhancedBaseVNI):
    """Legal VNI with specialized knowledge"""
    
    def __init__(self, instance_id: str):
        super().__init__("legal", instance_id)

    def get_default_concepts(self) -> Dict[str, Any]:
        return {
            'contract': {'strength': 0.8, 'usage_count': 0},
            'law': {'strength': 0.9, 'usage_count': 0},
            'legal': {'strength': 0.9, 'usage_count': 0},
            'rights': {'strength': 0.8, 'usage_count': 0},
            'agreement': {'strength': 0.7, 'usage_count': 0},
            'court': {'strength': 0.8, 'usage_count': 0},
            'lawyer': {'strength': 0.9, 'usage_count': 0},
            'case': {'strength': 0.8, 'usage_count': 0},
            'evidence': {'strength': 0.7, 'usage_count': 0},
            'justice': {'strength': 0.8, 'usage_count': 0},
            'liability': {'strength': 0.8, 'usage_count': 0},
            'dispute': {'strength': 0.7, 'usage_count': 0},
            'compliance': {'strength': 0.8, 'usage_count': 0}
        }

    def get_default_patterns(self) -> Dict[str, Any]:
        return {
            "contract_help": {
                "triggers": ["contract", "agreement", "terms", "sign"],
                "responses": [
                    "Contracts should be reviewed carefully. Key elements include parties, terms, obligations, and termination clauses.",
                    "For contract matters, ensure all terms are clear. Legal review is recommended for important agreements."
                ],
                "strength": 0.8, "usage_count": 0
            },
            "rights_inquiry": {
                "triggers": ["rights", "entitled", "legal rights"],
                "responses": [
                    "Legal rights vary by jurisdiction and situation. Consulting a qualified attorney is recommended.",
                    "Understanding your rights requires specific context. Legal counsel can provide jurisdiction-specific guidance."
                ],
                "strength": 0.8, "usage_count": 0
            }
        }

    def get_default_response_templates(self) -> Dict[str, List[str]]:
        return {
            "general": [
                "From a legal standpoint, {concept} requires proper documentation and professional advice.",
                "Legal matters involving {concept} should be reviewed by a qualified attorney.",
                "Regarding {concept}, outcomes depend on specific circumstances and jurisdiction."
            ]
        }


class EnhancedGeneralVNI(EnhancedBaseVNI):
    """General VNI with multi-domain knowledge"""
    
    def __init__(self, instance_id: str):
        super().__init__("general", instance_id)

    def get_default_concepts(self) -> Dict[str, Any]:
        return {
            # Technical
            'code': {'strength': 0.8, 'usage_count': 0},
            'programming': {'strength': 0.8, 'usage_count': 0},
            'technical': {'strength': 0.9, 'usage_count': 0},
            'system': {'strength': 0.8, 'usage_count': 0},
            'software': {'strength': 0.9, 'usage_count': 0},
            'database': {'strength': 0.8, 'usage_count': 0},
            # Mathematical
            'calculate': {'strength': 0.8, 'usage_count': 0},
            'equation': {'strength': 0.9, 'usage_count': 0},
            'formula': {'strength': 0.8, 'usage_count': 0},
            'math': {'strength': 0.9, 'usage_count': 0},
            # Business
            'business': {'strength': 0.9, 'usage_count': 0},
            'strategy': {'strength': 0.8, 'usage_count': 0},
            'market': {'strength': 0.8, 'usage_count': 0},
            'profit': {'strength': 0.8, 'usage_count': 0},
            # Creative
            'write': {'strength': 0.8, 'usage_count': 0},
            'story': {'strength': 0.8, 'usage_count': 0},
            'creative': {'strength': 0.9, 'usage_count': 0},
            # Analytical
            'analyze': {'strength': 0.9, 'usage_count': 0},
            'compare': {'strength': 0.8, 'usage_count': 0},
            'evaluate': {'strength': 0.8, 'usage_count': 0}
        }

    def get_default_patterns(self) -> Dict[str, Any]:
        return {
            "technical_help": {
                "triggers": ["code", "programming", "debug", "error", "bug"],
                "responses": [
                    "For technical issues, systematic debugging helps. Check syntax, test components, review errors.",
                    "Technical challenges benefit from breaking down problems, writing tests, and incremental development."
                ],
                "strength": 0.8, "usage_count": 0
            },
            "analysis_help": {
                "triggers": ["analyze", "compare", "evaluate", "assess"],
                "responses": [
                    "Analytical thinking requires systematic evaluation and logical reasoning.",
                    "For analysis, gather relevant data, identify patterns, and draw evidence-based conclusions."
                ],
                "strength": 0.8, "usage_count": 0
            },
            "business_analysis": {
                "triggers": ["business", "strategy", "market", "profit"],
                "responses": [
                    "Business analysis involves market research, financial modeling, and strategic planning.",
                    "Consider competitive landscape, customer needs, and financial viability."
                ],
                "strength": 0.8, "usage_count": 0
            }
        }

    def get_default_response_templates(self) -> Dict[str, List[str]]:
        return {
            "technical": [
                "From a technical perspective, {concept} requires understanding implementation and architecture.",
                "Technical solutions for {concept} depend on performance and scalability."
            ],
            "analytical": [
                "From an analytical perspective, {concept} requires systematic evaluation.",
                "Analytical approaches to {concept} depend on data interpretation and deduction."
            ],
            "general": [
                "Regarding {concept}, a comprehensive approach considers multiple perspectives.",
                "Analysis of {concept} benefits from cross-domain thinking."
            ]
        }

    def process_query(self, query: str, context: Dict = None, session_id: str = "default") -> Dict[str, Any]:
        response = super().process_query(query, context, session_id)
        
        # Add domain analysis
        detected = self.detect_query_domains(query)
        response['domain_analysis'] = {
            'detected_domains': detected,
            'primary_domain': detected[0] if detected else 'general',
            'cross_domain_synthesis': self.generate_cross_domain_synthesis(detected, query)
        }
        return response

    def detect_query_domains(self, query: str) -> List[str]:
        q = query.lower()
        domain_keywords = {
            'technical': ['code', 'programming', 'system', 'software', 'database'],
            'mathematical': ['calculate', 'equation', 'formula', 'solve', 'math'],
            'business': ['business', 'strategy', 'market', 'profit', 'revenue'],
            'creative': ['write', 'story', 'creative', 'narrative', 'character'],
            'analytical': ['analyze', 'compare', 'evaluate', 'assess']
        }
        domains = [d for d, kws in domain_keywords.items() if any(k in q for k in kws)]
        return domains if domains else ['general']

    def generate_cross_domain_synthesis(self, domains: List[str], query: str) -> str:
        if len(domains) <= 1:
            return f"Analysis focused on {domains[0] if domains else 'general'} domain."
        
        synthesis = f"Multi-domain analysis: {', '.join(domains)}. "
        if 'technical' in domains and 'business' in domains:
            synthesis += "Consider technical feasibility for business requirements. "
        if 'creative' in domains and 'analytical' in domains:
            synthesis += "Balance creative expression with analytical structure. "
        return synthesis.strip()


# ==================== VNI MANAGEMENT SYSTEM ====================

class VNIManager:
    """Manager for coordinating multiple VNI instances"""
    
    def __init__(self):
        self.vni_instances: Dict[str, EnhancedBaseVNI] = {}
        self.neural_pathways: Dict[str, NeuralPathway] = {}
        self.session_manager = SessionManager()
        self.attention_scores: Dict[str, float] = {}
        
    def create_vni(self, vni_type: str, instance_id: str) -> EnhancedBaseVNI:
        vni_classes = {
            'medical': EnhancedMedicalVNI,
            'legal': EnhancedLegalVNI,
            'general': EnhancedGeneralVNI
        }
        if vni_type not in vni_classes:
            raise ValueError(f"Unknown VNI type: {vni_type}")
        
        vni = vni_classes[vni_type](instance_id)
        self.vni_instances[instance_id] = vni
        
        # Create neural pathways to other VNIs
        for existing_id in self.vni_instances:
            if existing_id != instance_id:
                pathway_id = f"{instance_id}->{existing_id}"
                self.neural_pathways[pathway_id] = NeuralPathway(instance_id, existing_id)
                
                reverse_id = f"{existing_id}->{instance_id}"
                self.neural_pathways[reverse_id] = NeuralPathway(existing_id, instance_id)
        
        return vni

    def route_query(self, query: str, context: Dict = None, session_id: str = "default") -> Dict[str, Any]:
        """Route query with attention mechanism"""
        self.session_manager.get_session(session_id)
        
        # Calculate attention scores for each VNI
        self.attention_scores = self._calculate_attention_scores(query)
        
        # Activate VNIs based on attention
        activated = []
        for vni_id, vni in self.vni_instances.items():
            score = self.attention_scores.get(vni_id, 0.5)
            if score > 0.3:  # Attention threshold
                activated.append(vni_id)
        
        # Process with activated VNIs
        responses = []
        for vni_id in activated:
            vni = self.vni_instances[vni_id]
            if vni.attention_enabled:
                response = vni.process_query_with_attention(query, context or {}, self.attention_scores)
            else:
                response = vni.process_query(query, context, session_id)
            responses.append(response)
            
            # Update neural pathways
            self._update_pathways(vni_id, response.get('confidence', 0.5) > 0.6)
        
        return self._combine_responses(responses, query)

    def _calculate_attention_scores(self, query: str) -> Dict[str, float]:
        """Calculate attention scores for each VNI based on query"""
        scores = {}
        q = query.lower()
        
        for vni_id, vni in self.vni_instances.items():
            concepts, patterns = vni.extract_concepts_and_patterns(query)
            
            # Base score from concept/pattern matching
            base_score = min(1.0, (len(concepts) * 0.2 + len(patterns) * 0.3))
            
            # Boost from neural pathway strengths
            pathway_boost = 0.0
            for pid, pathway in self.neural_pathways.items():
                if pathway.target == vni_id:
                    pathway_boost += pathway.strength * 0.1
            
            scores[vni_id] = min(1.0, base_score + pathway_boost + 0.3)  # Min 0.3 baseline
        
        return scores

    def _update_pathways(self, vni_id: str, success: bool):
        """Update neural pathways based on VNI response success"""
        for pid, pathway in self.neural_pathways.items():
            if pathway.source == vni_id:
                pathway.activate(success)
            pathway.decay()  # Apply temporal decay

    def _combine_responses(self, responses: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        if not responses:
            return {
                'response': "I'm not sure how to help. Could you provide more context?",
                'confidence': 0.1,
                'sources': [],
                'combined': True
            }
        
        # Weight by attention and confidence
        for r in responses:
            r['weighted_score'] = r.get('confidence', 0.5) * r.get('attention_score', 1.0)
        
        best = max(responses, key=lambda r: r.get('weighted_score', 0))
        
        return {
            'response': best['response'],
            'confidence': best['confidence'],
            'attention_score': best.get('attention_score', 1.0),
            'sources': [r['vni_instance'] for r in responses],
            'combined': len(responses) > 1,
            'all_responses': responses if len(responses) > 1 else None
        }

    def get_system_status(self) -> Dict[str, Any]:
        return {
            'vni_count': len(self.vni_instances),
            'pathway_count': len(self.neural_pathways),
            'active_sessions': len(self.session_manager.sessions),
            'vni_capabilities': {vid: vni.get_capabilities() for vid, vni in self.vni_instances.items()}
        }


class SessionManager:
    """Manage user sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = timedelta(hours=2)
        
    def get_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'created': datetime.now(),
                'last_activity': datetime.now(),
                'interaction_count': 0,
                'preferences': {}
            }
        else:
            self.sessions[session_id]['last_activity'] = datetime.now()
            self.sessions[session_id]['interaction_count'] += 1
        return self.sessions[session_id]
    
    def cleanup_expired_sessions(self) -> int:
        now = datetime.now()
        expired = [sid for sid, s in self.sessions.items() if now - s['last_activity'] > self.session_timeout]
        for sid in expired:
            del self.sessions[sid]
        return len(expired)


# ==================== MAIN ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create manager and VNIs
    manager = VNIManager()
    medical = manager.create_vni('medical', 'med_001')
    legal = manager.create_vni('legal', 'legal_001')
    general = manager.create_vni('general', 'gen_001')
    
    print("=== VNI System Status ===")
    print(json.dumps(manager.get_system_status(), indent=2, default=str))
    
    # Test queries
    queries = [
        "I have a headache and fever, what should I do?",
        "I need help reviewing a contract agreement",
        "How can I analyze my business data effectively?",
        "What's the latest news about AI developments?",
        "Can you help me debug this code error?",
        "What are my legal rights in this situation?"
    ]
    
    print("\n=== Processing Test Queries ===\n")
    
    for i, query in enumerate(queries):
        print(f"Query {i+1}: {query}")
        print("-" * 50)
        
        response = manager.route_query(query, session_id=f"test_session")
        
        print(f"Response: {response['response'][:200]}...")
        print(f"Confidence: {response['confidence']:.2f}")
        print(f"Attention Score: {response.get('attention_score', 'N/A')}")
        print(f"Sources: {response['sources']}")
        print(f"Combined: {response['combined']}")
        print("\n")
    
    # Test attention-based routing
    print("=== Attention Scores for Last Query ===")
    for vni_id, score in manager.attention_scores.items():
        print(f"  {vni_id}: {score:.3f}")
    
    # Test neural pathway status
    print("\n=== Neural Pathway Status ===")
    for pid, pathway in list(manager.neural_pathways.items())[:5]:
        print(f"  {pid}: strength={pathway.strength:.3f}, activations={pathway.activation_count}")
    
    # Test session cleanup
    cleaned = manager.session_manager.cleanup_expired_sessions()
    print(f"\n=== Cleaned {cleaned} expired sessions ===")
    
    # Save all knowledge bases
    for vni in manager.vni_instances.values():
        vni.save_knowledge_base()
    
    print("\n=== All knowledge bases saved ===")
    print("System ready for production use.") 
