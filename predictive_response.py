#!/usr/bin/env python3
"""
Predictive Response Generator for BabyBIONN
Uses vocabulary predictions to generate intelligent responses with autonomous word formation
"""
import datetime
import logging
import random
import re
from typing import Dict, List, Any, Tuple
from predictive_vocabulary import PredictiveVocabulary

logger = logging.getLogger("BabyBIONN-PredictiveResponse")

class MorphologicalGenerator:
    """Generates new word forms through morphological operations"""
    
    def __init__(self):
        self.common_prefixes = ['re', 'un', 'pre', 'post', 'anti', 'dis', 'mis', 'over', 'under', 'super']
        self.common_suffixes = ['ing', 'ed', 's', 'ly', 'ment', 'tion', 'ness', 'able', 'ful', 'less', 'ish']
        self.word_patterns = {}
    
    def generate_variations(self, base_word: str, context: Dict) -> List[str]:
        """Generate morphological variations of base words"""
        variations = []
        base = base_word.lower().strip()
        
        if len(base) < 3:  # Too short for meaningful variation
            return variations
        
        # Prefix variations
        for prefix in self.common_prefixes:
            prefixed = prefix + base
            if self.is_phonetically_valid(prefixed) and self.contextually_relevant(prefixed, context):
                variations.append(prefixed)
        
        # Suffix variations
        for suffix in self.common_suffixes:
            # Handle basic English morphology rules
            if base.endswith('e') and suffix.startswith(('a', 'i', 'o', 'u')):
                suffixed = base[:-1] + suffix
            elif base.endswith('y') and not suffix.startswith('i'):
                suffixed = base[:-1] + 'i' + suffix
            else:
                suffixed = base + suffix
            
            if self.is_phonetically_valid(suffixed) and self.contextually_relevant(suffixed, context):
                variations.append(suffixed)
        
        return variations
    
    def blend_words(self, word1: str, word2: str, context: Dict) -> List[str]:
        """Create portmanteau words by blending two words"""
        blends = []
        w1, w2 = word1.lower(), word2.lower()
        
        if len(w1) < 3 or len(w2) < 3:
            return blends
        
        # Different blending strategies
        blend_strategies = [
            w1[:len(w1)//2] + w2[len(w2)//2:],  # first half + second half
            w1[:len(w1)//3*2] + w2[len(w2)//3:],  # two thirds + one third
            w1 + w2[-2:],  # full first + end of second
            w1[:3] + w2[3:]  # start of first + rest of second
        ]
        
        for blend in blend_strategies:
            if (self.is_phonetically_valid(blend) and 
                self.contextually_relevant(blend, context) and
                3 <= len(blend) <= 12):
                blends.append(blend)
        
        return blends
    
    def is_phonetically_valid(self, word: str) -> bool:
        """Basic phonetic validity check"""
        # Avoid impossible consonant clusters
        invalid_clusters = ['bbb', 'ccc', 'ddd', 'fff', 'ggg', 'hhh', 'jjj', 
                           'kkk', 'lll', 'mmm', 'nnn', 'ppp', 'qqq', 'rrr', 
                           'sss', 'ttt', 'vvv', 'www', 'xxx', 'yyy', 'zzz']
        return not any(cluster in word for cluster in invalid_clusters)
    
    def contextually_relevant(self, word: str, context: Dict) -> bool:
        """Check if word fits current context"""
        domain = context.get('domain', 'general')
        keywords = context.get('keywords', [])
        
        # Simple relevance check - could be enhanced with semantic analysis
        return len(word) >= 3 and len(word) <= 15

class SemanticComposer:
    """Composes new terms and sentences based on semantic patterns"""
    
    def __init__(self):
        self.sentence_templates = [
            "The {concept} demonstrates {property} in {domain} contexts.",
            "When considering {concept1} and {concept2}, we observe {insight}.",
            "My analysis of {concept} reveals {observation} patterns.",
            "The relationship between {concept1} and {concept2} suggests {conclusion}.",
            "In {domain} domains, {concept} typically exhibits {characteristic}."
        ]
        
        self.concept_relationships = {
            'cause_effect': ['leads to', 'results in', 'causes', 'produces', 'generates'],
            'similarity': ['resembles', 'is similar to', 'parallels', 'echoes'],
            'contrast': ['contrasts with', 'differs from', 'opposes'],
            'enhancement': ['amplifies', 'strengthens', 'enhances', 'improves']
        }
    
    def compose_novel_sentence(self, concepts: List[str], predictions: List[Dict], context: Dict) -> str:
        """Compose a novel sentence using predictions and concepts"""
        if not concepts or not predictions:
            return self.fallback_sentence(concepts, context)
        
        domain = context.get('domain', 'general')
        template = random.choice(self.sentence_templates)
        
        # Fill template with contextual elements
        filled_template = template.format(
            concept=concepts[0] if concepts else 'this concept',
            concept1=concepts[0] if concepts else 'primary concept',
            concept2=concepts[1] if len(concepts) > 1 else 'secondary concept',
            domain=domain,
            property=predictions[0]['word'] if predictions else 'interesting',
            insight=self.generate_insight(concepts, predictions),
            observation=self.generate_observation(predictions),
            conclusion=self.generate_conclusion(concepts, predictions),
            characteristic=predictions[0]['word'] if predictions else 'notable'
        )
        
        return filled_template
    
    def generate_insight(self, concepts: List[str], predictions: List[Dict]) -> str:
        """Generate semantic insight from concepts and predictions"""
        if len(concepts) >= 2 and predictions:
            relationship = random.choice(list(self.concept_relationships.values()))
            return f"{concepts[0]} {random.choice(relationship)} {concepts[1]}"
        elif predictions:
            return f"emerging patterns around {predictions[0]['word']}"
        else:
            return "developing understanding"
    
    def generate_observation(self, predictions: List[Dict]) -> str:
        """Generate observational phrase from predictions"""
        if predictions:
            confidence_words = ['strong', 'consistent', 'emerging', 'developing', 'notable']
            return f"{random.choice(confidence_words)} {predictions[0]['word']}"
        return "interesting developmental"
    
    def generate_conclusion(self, concepts: List[str], predictions: List[Dict]) -> str:
        """Generate concluding phrase"""
        if concepts and predictions:
            return f"synergistic potential between {concepts[0]} and {predictions[0]['word']}"
        return "significant developmental trajectories"
    
    def fallback_sentence(self, concepts: List[str], context: Dict) -> str:
        """Fallback sentence when composition fails"""
        domain = context.get('domain', 'general')
        if concepts:
            return f"I'm analyzing {concepts[0]} in {domain} contexts to build understanding."
        return f"My predictive networks are actively learning about {domain} domains."

class PredictiveResponseGenerator:
    def __init__(self, vocabulary: PredictiveVocabulary):
        self.vocabulary = vocabulary
        self.response_evolution = {}
        self.morphological_generator = MorphologicalGenerator()
        self.semantic_composer = SemanticComposer()
        self.autonomous_terms_created = 0
        self.novel_sentences_used = 0
    
    def generate_predictive_response(self, query: str, concepts: List[str], context: Dict) -> str:
        """Generate response using word prediction and autonomous word formation"""
        query_words = self.extract_significant_words(query)
        domain = context.get('domain', self.detect_domain(concepts))
        
        # Enhance context with prediction data
        enhanced_context = {
            **context,
            'domain': domain,
            'keywords': concepts + query_words,
            'previous_words': query_words
        }
        
        # Get word predictions
        word_predictions = self.vocabulary.predict_next_words(query_words, enhanced_context)
        
        # Learn from this interaction and stimulate autonomous growth
        for word in query_words:
            self.vocabulary.learn_word(word, enhanced_context, domain)
        
        # Stimulate autonomous vocabulary growth
        new_terms = self.stimulate_vocabulary_growth(concepts, query_words, enhanced_context)
        
        # Generate appropriate response with autonomous capabilities
        if word_predictions and word_predictions[0]['probability'] > 0.7:
            response = self.high_confidence_prediction(query, word_predictions, concepts, enhanced_context, new_terms)
        elif word_predictions and word_predictions[0]['probability'] > 0.4:
            response = self.medium_confidence_prediction(query, word_predictions, concepts, enhanced_context, new_terms)
        else:
            response = self.exploratory_prediction(query, concepts, enhanced_context, new_terms)
        
        # Track response evolution
        self.track_response_evolution(query, concepts, response, word_predictions)
        
        return response
    
    def stimulate_vocabulary_growth(self, concepts: List[str], query_words: List[str], context: Dict) -> List[str]:
        """Generate and integrate new autonomous terms"""
        new_terms = []
        
        # Generate morphological variations from significant concepts
        for concept in concepts[:3]:  # Limit to top 3 concepts to avoid explosion
            variations = self.morphological_generator.generate_variations(concept, context)
            new_terms.extend(variations)
        
        # Generate word blends from concept pairs
        if len(concepts) >= 2:
            for i in range(min(2, len(concepts))):
                for j in range(i+1, min(3, len(concepts))):
                    blends = self.morphological_generator.blend_words(concepts[i], concepts[j], context)
                    new_terms.extend(blends)
        
        # Filter and integrate successful terms
        successful_terms = []
        for term in new_terms:
            if self.evaluate_term_potential(term, context):
                successful_terms.append(term)
                self.integrate_autonomous_term(term, concepts, context)
        
        logger.info(f"🧠 Generated {len(successful_terms)} new autonomous terms: {successful_terms}")
        return successful_terms
    
    def evaluate_term_potential(self, term: str, context: Dict) -> bool:
        """Evaluate if a new term has potential for integration"""
        # Basic quality checks
        if len(term) < 3 or len(term) > 15:
            return False
        
        # Phonetic plausibility
        if not self.morphological_generator.is_phonetically_valid(term):
            return False
        
        # Contextual relevance
        domain_keywords = context.get('keywords', [])
        if domain_keywords:
            # Check if term shares characteristics with domain keywords
            domain_chars = ''.join(domain_keywords)[:10]
            term_chars = term[:10]
            shared_chars = len(set(domain_chars) & set(term_chars))
            if shared_chars < 2:  # Too dissimilar
                return False
        
        return True
    
    def integrate_autonomous_term(self, term: str, base_concepts: List[str], context: Dict):
        """Integrate a new autonomous term into the vocabulary"""
        try:
            # Add to vocabulary with contextual weighting
            if hasattr(self.vocabulary, 'integrate_autonomous_term'):
                self.vocabulary.integrate_autonomous_term(term, base_concepts, context)
            else:
                # Fallback: learn the term normally
                self.vocabulary.learn_word(term, context, context.get('domain', 'general'))
            
            self.autonomous_terms_created += 1
            logger.info(f"📝 Integrated autonomous term: '{term}' from concepts {base_concepts}")
            
        except Exception as e:
            logger.warning(f"Failed to integrate autonomous term '{term}': {e}")
    
    def high_confidence_prediction(self, query: str, predictions: List[Dict], concepts: List[str], 
                                 context: Dict, new_terms: List[str]) -> str:
        """Generate response when we have high prediction confidence with autonomous enhancements"""
        primary_prediction = predictions[0]
        
        # Occasionally include novel terms in high-confidence responses
        include_novelty = random.random() < 0.2  # 20% chance
        novelty_phrase = ""
        
        if include_novelty and new_terms:
            novel_term = random.choice(new_terms)
            novelty_phrase = f" This understanding suggests potential for '{novel_term}' as an emerging concept."
            self.novel_sentences_used += 1
        
        response_parts = [
            f"I have strong predictive understanding of this {context['domain']} context.",
            f"My learning suggests '{primary_prediction['word']}' is highly relevant here,",
            f"as I've observed consistent patterns connecting these concepts.",
            f"{self.generate_prediction_insight(predictions[:2])}",
            f"{novelty_phrase}",
            f"How can I provide more specific assistance with this?"
        ]
        
        return ' '.join(filter(None, response_parts))
    
    def medium_confidence_prediction(self, query: str, predictions: List[Dict], concepts: List[str], 
                                   context: Dict, new_terms: List[str]) -> str:
        """Generate response with medium prediction confidence and autonomous learning"""
        
        # Use semantic composer for more creative responses at medium confidence
        if random.random() < 0.4:  # 40% chance to use composed sentence
            composed_sentence = self.semantic_composer.compose_novel_sentence(concepts, predictions, context)
            self.novel_sentences_used += 1
        else:
            composed_sentence = self.generate_learning_insight(predictions, concepts)
        
        response_parts = [
            f"I'm developing predictive understanding of this {context['domain']} area.",
            f"My current learning points toward '{predictions[0]['word']}' as relevant,",
            f"though I'm still strengthening these connections.",
            f"{composed_sentence}",
            f"Could you provide more context to help me learn?"
        ]
        
        return ' '.join(response_parts)
    
    def exploratory_prediction(self, query: str, concepts: List[str], context: Dict, new_terms: List[str]) -> str:
        """Generate response when predictions are low confidence with maximum autonomy"""
        
        # Use semantic composition more frequently in exploratory mode
        if random.random() < 0.6:  # 60% chance for novel sentences
            composed_sentence = self.semantic_composer.compose_novel_sentence(concepts, [], context)
            self.novel_sentences_used += 1
            learning_insight = composed_sentence
        else:
            learning_insight = "My predictive networks are actively learning from this interaction."
        
        # Include new terms if available
        term_insight = ""
        if new_terms and random.random() < 0.5:
            novel_term = random.choice(new_terms)
            term_insight = f" I'm exploring conceptual variations like '{novel_term}'."
        
        response_parts = [
            f"I'm exploring this {context['domain']} domain and building understanding.",
            f"{learning_insight}",
            f"Each conversation helps strengthen my synaptic connections",
            f"and improve my ability to anticipate relevant concepts.{term_insight}"
        ]
        
        if concepts:
            response_parts.append(f"I'm currently focusing on '{concepts[0]}' and related patterns.")
        
        return ' '.join(response_parts)
    
    def generate_prediction_insight(self, top_predictions: List[Dict]) -> str:
        """Generate insight based on prediction patterns"""
        if len(top_predictions) >= 2:
            return (f"I notice strong associations between '{top_predictions[0]['source_word']}' "
                   f"and '{top_predictions[0]['word']}', with supporting patterns from "
                   f"'{top_predictions[1]['word']}'.")
        else:
            return f"The connection between '{top_predictions[0]['source_word']}' and '{top_predictions[0]['word']}' appears well-established in my learning."
    
    def generate_learning_insight(self, predictions: List[Dict], concepts: List[str]) -> str:
        """Generate insight about current learning state"""
        if predictions:
            return (f"I'm detecting emerging patterns like '{predictions[0]['word']}' "
                   f"appearing after '{predictions[0]['source_word']}' in similar contexts.")
        elif concepts:
            return f"I'm actively learning about '{concepts[0]}' and building predictive connections."
        else:
            return "I'm forming new synaptic connections from this interaction."
    
    def extract_significant_words(self, text: str) -> List[str]:
        """Extract significant words from text (simple version)"""
        # Simple implementation - in practice, use NLP for better extraction
        words = text.lower().split()
        # Filter out common stop words and keep meaningful words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def detect_domain(self, concepts: List[str]) -> str:
        """Detect domain from concepts"""
        domain_keywords = {
            'medical': ['symptom', 'treatment', 'health', 'medical', 'doctor', 'medicine', 
                       'patient', 'clinical', 'diagnosis', 'therapy'],
            'legal': ['legal', 'law', 'contract', 'rights', 'court', 'lawyer', 'case',
                     'judge', 'statute', 'regulation', 'compliance'],
            'technical': ['technical', 'code', 'system', 'program', 'software', 'algorithm',
                         'python', 'java', 'database', 'api', 'framework', 'debug'],
            'scientific': ['research', 'study', 'experiment', 'data', 'analysis', 'hypothesis',
                          'methodology', 'results', 'conclusion', 'publication'],
            'business': ['strategy', 'market', 'revenue', 'growth', 'customer', 'product',
                        'sales', 'marketing', 'finance', 'investment']
        }
        
        domain_scores = {domain: 0 for domain in domain_keywords}
        
        for concept in concepts:
            concept_lower = concept.lower()
            for domain, keywords in domain_keywords.items():
                if any(keyword in concept_lower for keyword in keywords):
                    domain_scores[domain] += 1
        
        # Return domain with highest score, or 'general' if no strong match
        max_domain = max(domain_scores, key=domain_scores.get)
        return max_domain if domain_scores[max_domain] > 0 else 'general'
    
    def track_response_evolution(self, query: str, concepts: List[str], response: str, predictions: List[Dict]):
        """Track how responses evolve with learning"""
        evolution_key = hash(tuple(sorted(concepts)))
        
        if evolution_key not in self.response_evolution:
            self.response_evolution[evolution_key] = {
                'first_response': response,
                'response_history': [],
                'prediction_confidence_history': [],
                'learning_trajectory': [],
                'autonomous_terms_used': 0,
                'novel_sentences_used': 0
            }
        
        evolution_data = self.response_evolution[evolution_key]
        evolution_data['response_history'].append({
            'response': response,
            'timestamp': datetime.datetime.now().isoformat(),
            'predictions': predictions,
            'concepts': concepts
        })
        
        # Track autonomous creativity metrics
        evolution_data['autonomous_terms_used'] = self.autonomous_terms_created
        evolution_data['novel_sentences_used'] = self.novel_sentences_used
        
        # Track prediction confidence over time
        if predictions:
            avg_confidence = sum(p['probability'] for p in predictions) / len(predictions)
            evolution_data['prediction_confidence_history'].append(avg_confidence)
        
        logger.info(f"📈 Response evolution tracked: {len(evolution_data['response_history'])} responses for {len(concepts)} concepts")
        logger.info(f"🎨 Autonomous creativity: {self.autonomous_terms_created} terms created, {self.novel_sentences_used} novel sentences used")
    
    def get_autonomous_metrics(self) -> Dict[str, Any]:
        """Get metrics about autonomous learning performance"""
        return {
            'autonomous_terms_created': self.autonomous_terms_created,
            'novel_sentences_used': self.novel_sentences_used,
            'concept_patterns_tracked': len(self.response_evolution),
            'morphological_operations': len(self.morphological_generator.word_patterns)
        }
