# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

#!/usr/bin/env python3
"""
Predictive Vocabulary System for BabyBIONN
Enables word prediction and synaptic language learning
"""
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np

logger = logging.getLogger("BabyBIONN-Vocabulary")

class PredictiveVocabulary:
    def __init__(self, vocab_file: str = "vocabulary/predictive_vocabulary.json"):
        self.vocab_file = vocab_file
        self.vocabulary = {}  # word -> neural representation
        self.transition_probs = defaultdict(dict)  # word_A -> {word_B: probability}
        self.domain_networks = defaultdict(dict)   # domain -> {word: strength}
        self.learning_metrics = {
            'total_interactions': 0,
            'successful_predictions': 0,
            'vocabulary_growth': [],
            'domain_specialization': {}
        }
        
        self.load_vocabulary()
    
    def load_vocabulary(self):
        """Load existing vocabulary or initialize new"""
        try:
            with open(self.vocab_file, 'r') as f:
                data = json.load(f)
                self.vocabulary = data.get('vocabulary', {})
                self.transition_probs = defaultdict(dict, data.get('transition_probs', {}))
                self.domain_networks = defaultdict(dict, data.get('domain_networks', {}))
                self.learning_metrics = data.get('learning_metrics', self.learning_metrics)
            logger.info(f"📚 Loaded vocabulary with {len(self.vocabulary)} words")
        except FileNotFoundError:
            logger.info("📚 Initializing new predictive vocabulary")
            self.initialize_core_vocabulary()
    
    def initialize_core_vocabulary(self):
        """Initialize with core domain vocabulary"""
        core_words = {
            'medical': ['symptom', 'treatment', 'health', 'doctor', 'medicine', 'pain', 'care'],
            'legal': ['law', 'contract', 'rights', 'legal', 'agreement', 'court', 'lawyer'],
            'technical': ['code', 'system', 'technical', 'program', 'algorithm', 'software', 'data']
        }
        
        for domain, words in core_words.items():
            for word in words:
                self.learn_word(word, {'domain': domain, 'context': 'initialization'}, domain)
        
        self.save_vocabulary()
    
    def learn_word(self, word: str, context: Dict, domain: str = "general"):
        """Learn a new word and build synaptic connections"""
        word_lower = word.lower().strip()
        
        if word_lower not in self.vocabulary:
            # New word - initialize neural representation
            self.vocabulary[word_lower] = {
                'neural_strength': 0.1,
                'domain_associations': {},
                'context_patterns': [],
                'cooccurrence_network': {},
                'first_encountered': datetime.now().isoformat(),
                'usage_count': 0,
                'last_used': datetime.now().isoformat()
            }
            logger.info(f"🎯 Learned new word: '{word_lower}' in {domain} context")
        
        # Strengthen word representation
        self.strengthen_word_representation(word_lower, context, domain)
        
        # Update transition probabilities with previous words
        previous_words = context.get('previous_words', [])
        self.update_transition_probabilities(word_lower, previous_words, domain)
        
        # Update learning metrics
        self.learning_metrics['total_interactions'] += 1
        
        return self.vocabulary[word_lower]
    
    def strengthen_word_representation(self, word: str, context: Dict, domain: str):
        """Apply Hebbian learning to strengthen word connections"""
        word_data = self.vocabulary[word]
        word_data['usage_count'] += 1
        word_data['last_used'] = datetime.now().isoformat()
        
        # Hebbian learning: words that appear together get stronger connections
        context_words = context.get('surrounding_words', [])
        for context_word in context_words:
            if context_word in self.vocabulary:
                if context_word not in word_data['cooccurrence_network']:
                    word_data['cooccurrence_network'][context_word] = 0.1
                # Strengthen connection
                word_data['cooccurrence_network'][context_word] = min(
                    1.0, word_data['cooccurrence_network'][context_word] + 0.05
                )
        
        # Domain specialization
        if domain not in word_data['domain_associations']:
            word_data['domain_associations'][domain] = 0.1
        word_data['domain_associations'][domain] = min(
            1.0, word_data['domain_associations'][domain] + 0.03
        )
        
        # Overall neural strength
        word_data['neural_strength'] = min(1.0, word_data['neural_strength'] + 0.01)
        
        # Store context pattern
        if 'context' in context:
            word_data['context_patterns'].append({
                'context': context['context'],
                'timestamp': datetime.now().isoformat(),
                'domain': domain
            })
            # Keep only recent patterns
            word_data['context_patterns'] = word_data['context_patterns'][-10:]
    
    def update_transition_probabilities(self, current_word: str, previous_words: List[str], domain: str):
        """Update Markov-like transition probabilities between words"""
        for prev_word in previous_words[-2:]:  # Look at last 2 words
            if prev_word in self.vocabulary:
                if current_word not in self.transition_probs[prev_word]:
                    self.transition_probs[prev_word][current_word] = 0.1
                else:
                    # Strengthen transition
                    self.transition_probs[prev_word][current_word] = min(
                        1.0, self.transition_probs[prev_word][current_word] + 0.1
                    )
                
                # Normalize probabilities for this source word
                total = sum(self.transition_probs[prev_word].values())
                for word in self.transition_probs[prev_word]:
                    self.transition_probs[prev_word][word] /= total
    
    def predict_next_words(self, current_words: List[str], context: Dict, max_predictions: int = 5) -> List[Dict]:
        """Predict likely next words based on synaptic patterns"""
        predictions = []
        domain = context.get('domain', 'general')
        
        for current_word in current_words[-3:]:  # Look at last 3 words
            if current_word in self.transition_probs:
                for next_word, probability in self.transition_probs[current_word].items():
                    # Skip if next_word not in vocabulary (shouldn't happen, but safety)
                    if next_word not in self.vocabulary:
                        continue
                    
                    # Adjust probability based on context
                    context_boost = self.calculate_context_relevance(next_word, context)
                    domain_boost = self.calculate_domain_alignment(next_word, domain)
                    strength_boost = self.vocabulary[next_word]['neural_strength']
                    
                    final_probability = probability * context_boost * domain_boost * strength_boost
                    
                    predictions.append({
                        'word': next_word,
                        'probability': round(final_probability, 4),
                        'source_word': current_word,
                        'domain': domain,
                        'reasoning': self.generate_prediction_reasoning(current_word, next_word, domain),
                        'neural_strength': self.vocabulary[next_word]['neural_strength']
                    })
        
        # Remove duplicates and sort by probability
        unique_predictions = {}
        for pred in predictions:
            if pred['word'] not in unique_predictions or pred['probability'] > unique_predictions[pred['word']]['probability']:
                unique_predictions[pred['word']] = pred
        
        sorted_predictions = sorted(unique_predictions.values(), key=lambda x: x['probability'], reverse=True)
        return sorted_predictions[:max_predictions]
    
    def calculate_context_relevance(self, word: str, context: Dict) -> float:
        """Calculate how relevant a word is to the current context"""
        context_keywords = context.get('keywords', [])
        word_data = self.vocabulary.get(word, {})
        
        if not context_keywords or not word_data:
            return 0.5  # Neutral relevance
        
        # Check co-occurrence with context keywords
        relevance_scores = []
        for keyword in context_keywords:
            if keyword in word_data.get('cooccurrence_network', {}):
                relevance_scores.append(word_data['cooccurrence_network'][keyword])
        
        return np.mean(relevance_scores) if relevance_scores else 0.3
    
    def calculate_domain_alignment(self, word: str, domain: str) -> float:
        """Calculate how well a word aligns with the current domain"""
        word_data = self.vocabulary.get(word, {})
        domain_associations = word_data.get('domain_associations', {})
        
        if domain in domain_associations:
            return domain_associations[domain]
        elif domain_associations:
            # Some domain association exists, but not this one
            return 0.3
        else:
            # No domain associations yet
            return 0.5
    
    def generate_prediction_reasoning(self, source_word: str, target_word: str, domain: str) -> str:
        """Generate human-readable reasoning for predictions"""
        source_strength = self.vocabulary[source_word]['neural_strength']
        target_strength = self.vocabulary[target_word]['neural_strength']
        
        reasoning_parts = []
        
        if source_strength > 0.7:
            reasoning_parts.append(f"strong understanding of '{source_word}'")
        else:
            reasoning_parts.append(f"developing knowledge of '{source_word}'")
        
        if target_strength > 0.7:
            reasoning_parts.append(f"established expertise with '{target_word}'")
        
        reasoning_parts.append(f"frequent co-occurrence in {domain} contexts")
        
        return f"Based on {', '.join(reasoning_parts)}"

    def get_expansions(self, topic: str, vni_type: str) -> List[str]:
        """Get predictive expansions for a topic"""
        key = f"{vni_type}_{topic}"
        return self.vocabulary.get(key, {}).get('expansions', [])

    def get_guidance_patterns(self, vni_type: str) -> List[str]:
        """Get guidance patterns for step-by-step instructions"""
        guidance_key = f"{vni_type}_guidance"
        return self.vocabulary.get(guidance_key, {}).get('patterns', [])

    def get_vocabulary_metrics(self) -> Dict[str, Any]:
        """Get comprehensive vocabulary learning metrics"""
        total_words = len(self.vocabulary)
        strong_words = sum(1 for data in self.vocabulary.values() if data['neural_strength'] > 0.7)
        avg_strength = np.mean([data['neural_strength'] for data in self.vocabulary.values()]) if self.vocabulary else 0
        
        return {
            'total_words': total_words,
            'strong_words': strong_words,
            'average_neural_strength': round(avg_strength, 3),
            'domains_covered': len(self.domain_networks),
            'total_transitions': sum(len(transitions) for transitions in self.transition_probs.values()),
            'learning_progress': self.learning_metrics
        }
    def get_predictive_completions(self, text: str, max_completions: int = 3) -> List[str]:
        """Get predictive completions for the given text"""
        try:
            words = text.lower().split()
            if not words:
                return []
        
            last_word = words[-1]
            completions = []
        
            # Simple prefix matching from vocabulary
            for word in self.vocabulary.keys():
                if word.startswith(last_word) and word != last_word:
                    completions.append(word)
        
            return completions[:max_completions]
    
        except Exception as e:
            logger.warning(f"Predictive completions failed: {e}")
            return []

    def update_vocabulary(self, text: str, domain: str = "general"):
        """Update vocabulary with text from a specific domain"""
        try:
            words = text.lower().split()
            for word in words:
                if len(word) > 2:  # Only consider words with more than 2 characters
                    context = {
                        'domain': domain,
                        'surrounding_words': words,
                        'previous_words': words[:-1] if len(words) > 1 else []
                    }
                    self.learn_word(word, context, domain)
        
            # Update learning metrics
            self.learning_metrics['total_interactions'] += 1
            logger.debug(f"Updated vocabulary with {len(words)} words from {domain} domain")
            return True
        
        except Exception as e:
            logger.error(f"Vocabulary update failed: {e}")
            return False

    def get_predictive_suggestions(self, text: str, max_suggestions: int = 3) -> List[str]:
        """Get predictive suggestions for the given text - alias for get_predictive_completions"""
        return self.get_predictive_completions(text, max_suggestions)

    def integrate_autonomous_term(self, term: str, base_concepts: List[str], context: Dict):
        """Integrate autonomously created terms into vocabulary"""
        # If this term doesn't exist, add it
        if term not in self.vocabulary:
            self.vocabulary[term] = {
                'domain': context.get('domain', 'general'),
                'frequency': 1,
                'contexts': [context],
                'autonomous': True,
                'base_concepts': base_concepts,
                'created_at': datetime.datetime.now().isoformat()
            }
            logger.info(f"🧠 Integrated autonomous term: '{term}'"
        )
        else:
            # Update existing term
            self.vocabulary[term]['frequency'] += 1
            self.vocabulary[term]['contexts'].append(context)

    def save_vocabulary(self):
        """Save vocabulary state to file"""
        import os
        os.makedirs(os.path.dirname(self.vocab_file), exist_ok=True)
        
        data = {
            'vocabulary': self.vocabulary,
            'transition_probs': dict(self.transition_probs),
            'domain_networks': dict(self.domain_networks),
            'learning_metrics': self.learning_metrics,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.vocab_file, 'w') as f:
            json.dump(data, f, indent=2)
    
        logger.info(f"💾 Vocabulary saved: {len(self.vocabulary)} words, {sum(len(t) for t in self.transition_probs.values())} transitions")
