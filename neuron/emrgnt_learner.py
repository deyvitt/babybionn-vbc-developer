# /neuron/shared/emergent_semantic_learner.py
"""
True emergent semantic learning - learns meaning from scratch like a baby
No pre-trained models, no hardcoded rules - pure learning from interaction
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict
import re
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import logging

from neuron.shared.types import LearningMetrics, PerformanceTrend
from neuron.shared.synaptic_config import SynapticConfig
from neuron.shared.constants import METRIC_NAMES, TREND_THRESHOLDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("emergent_learner")

@dataclass
class WordConcept:
    """A concept learned from ground up"""
    word: str
    contexts: Set[str] = field(default_factory=set)  # Contexts where word appears
    associations: Dict[str, float] = field(default_factory=dict)  # Other words that co-occur
    emotional_valence: float = 0.0  # Positive/negative from outcomes
    usage_count: int = 0
    first_seen: datetime = field(default_factory=datetime.now)
    
    def strength(self) -> float:
        """Concept strength based on usage and consistency"""
        return np.log1p(self.usage_count) * (1 + len(self.associations) / 10)

@dataclass
class PhrasePattern:
    """Pattern of words that co-occur meaningfully"""
    words: List[str]
    contexts: Set[str] = field(default_factory=set)
    outcome_correlation: float = 0.0  # How well this pattern predicts good outcomes
    usage_count: int = 0
    variance: float = 0.0  # Contextual variance (low = specific meaning)
    
    def specificity(self) -> float:
        """How specific this pattern is to certain contexts"""
        return 1.0 / (1.0 + len(self.contexts))

class EmergentSemanticLearner:
    """
    Learns semantics from scratch through VNI interactions
    Like a baby learning language from conversation
    """
    
    def __init__(self):
        # Core semantic memory - built from scratch
        self.word_concepts: Dict[str, WordConcept] = {}
        self.phrase_patterns: List[PhrasePattern] = []
        
        # Context tracking
        self.context_word_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Association matrices (learned, not pre-trained)
        self.word_association_matrix: Dict[Tuple[str, str], float] = {}
        
        # Emotional valence tracking (from outcomes)
        self.word_valence: Dict[str, List[float]] = defaultdict(list)
        
        # Concept clusters (emergent categories)
        self.concept_clusters: Dict[str, Set[str]] = defaultdict(set)
        
        logger.info("🧒 EmergentSemanticLearner initialized - learning from scratch")
    
    def learn_from_interaction(self, 
                              vni_outputs: Dict[str, str],
                              context: Dict[str, Any],
                              outcome_quality: float):
        """
        Learn semantics from a complete VNI interaction
        Like a baby learning from a conversation
        """
        context_id = self._hash_context(context)
        
        # Extract and learn from each VNI's output
        for vni_id, output in vni_outputs.items():
            self._process_sentence(output, context_id, outcome_quality)
        
        # Find patterns across VNI outputs
        self._find_cross_vni_patterns(vni_outputs, context_id, outcome_quality)
        
        # Update emotional valence based on outcome
        self._update_valence_from_outcome(vni_outputs, outcome_quality)
        
        # Form concept clusters
        self._form_concept_clusters()
    
    def _process_sentence(self, text: str, context_id: str, outcome: float):
        """Process a single sentence, learning words and patterns"""
        # Simple tokenization (baby doesn't know punctuation rules)
        words = self._simple_tokenize(text)
        
        # Learn individual words
        for word in words:
            self._learn_word(word, context_id, outcome)
        
        # Learn word pairs (n-grams starting simple)
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]
            self._learn_word_pair(word1, word2, context_id, outcome)
        
        # Learn short phrases (3-4 words)
        if len(words) >= 3:
            for i in range(len(words) - 2):
                phrase = words[i:i+3]
                self._learn_phrase(phrase, context_id, outcome)
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """Baby-like tokenization - no complex NLP"""
        # Lowercase
        text = text.lower()
        
        # Very basic splitting (baby doesn't know punctuation well)
        # Replace punctuation with spaces
        for punct in ',.!?;:()[]{}\'"':
            text = text.replace(punct, ' ')
        
        # Split and filter
        words = [w for w in text.split() if len(w) > 1 and w not in ['the', 'and', 'or', 'but']]
        
        return words
    
    def _learn_word(self, word: str, context_id: str, outcome: float):
        """Learn a single word concept"""
        if word not in self.word_concepts:
            self.word_concepts[word] = WordConcept(word=word)
        
        concept = self.word_concepts[word]
        concept.contexts.add(context_id)
        concept.usage_count += 1
        
        # Update emotional valence
        self.word_valence[word].append(outcome)
        concept.emotional_valence = np.mean(self.word_valence[word][-10:])  # Recent average
        
        # Track context usage
        self.context_word_counts[context_id][word] += 1
    
    def _learn_word_pair(self, word1: str, word2: str, context_id: str, outcome: float):
        """Learn association between two words"""
        key = tuple(sorted([word1, word2]))
        
        if key not in self.word_association_matrix:
            self.word_association_matrix[key] = 0.5  # Start with neutral
        
        # Strengthen association based on co-occurrence and outcome
        current_strength = self.word_association_matrix[key]
        
        # Outcome influences learning (positive outcomes strengthen associations)
        learning_rate = 0.1 * outcome
        
        new_strength = current_strength * (1 - learning_rate) + 1.0 * learning_rate
        self.word_association_matrix[key] = min(new_strength, 1.0)
        
        # Update concept associations
        for word in [word1, word2]:
            if word in self.word_concepts:
                other_word = word2 if word == word1 else word1
                self.word_concepts[word].associations[other_word] = new_strength
    
    def _learn_phrase(self, phrase: List[str], context_id: str, outcome: float):
        """Learn a phrase pattern"""
        phrase_str = ' '.join(phrase)
        
        # Check if similar phrase exists
        existing_pattern = None
        for pattern in self.phrase_patterns:
            if self._phrase_similarity(phrase, pattern.words) > 0.7:
                existing_pattern = pattern
                break
        
        if existing_pattern:
            # Update existing pattern
            existing_pattern.contexts.add(context_id)
            existing_pattern.usage_count += 1
            
            # Update outcome correlation
            prev_corr = existing_pattern.outcome_correlation
            usage = existing_pattern.usage_count
            existing_pattern.outcome_correlation = (prev_corr * (usage - 1) + outcome) / usage
            
        else:
            # Create new pattern
            pattern = PhrasePattern(
                words=phrase,
                contexts={context_id},
                outcome_correlation=outcome,
                usage_count=1
            )
            self.phrase_patterns.append(pattern)
        
        # Limit patterns to strongest ones
        if len(self.phrase_patterns) > 100:
            self.phrase_patterns.sort(key=lambda p: p.outcome_correlation * p.usage_count, reverse=True)
            self.phrase_patterns = self.phrase_patterns[:80]
    
    def _phrase_similarity(self, phrase1: List[str], phrase2: List[str]) -> float:
        """Simple phrase similarity (word overlap)"""
        set1 = set(phrase1)
        set2 = set(phrase2)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union
    
    def _find_cross_vni_patterns(self, vni_outputs: Dict[str, str], context_id: str, outcome: float):
        """Find patterns that appear across multiple VNI outputs"""
        all_words = []
        for output in vni_outputs.values():
            words = self._simple_tokenize(output)
            all_words.extend(words)
        
        # Find words that appear in multiple VNI outputs
        word_counts = defaultdict(int)
        for word in all_words:
            word_counts[word] += 1
        
        # Words that appear in at least 2 VNI outputs are "consensus words"
        consensus_words = [word for word, count in word_counts.items() if count >= 2]
        
        # Learn that consensus words in this context led to this outcome
        for word in consensus_words:
            if word in self.word_concepts:
                # Consensus words get valence boost
                self.word_concepts[word].emotional_valence = (
                    self.word_concepts[word].emotional_valence * 0.7 + outcome * 0.3
                )
    
    def _update_valence_from_outcome(self, vni_outputs: Dict[str, str], outcome: float):
        """Update emotional valence of words based on outcome"""
        # All words in this interaction get valence update
        all_words = set()
        for output in vni_outputs.values():
            words = self._simple_tokenize(output)
            all_words.update(words)
        
        for word in all_words:
            if word in self.word_valence:
                # Recent outcomes weighted more heavily
                self.word_valence[word].append(outcome)
                if len(self.word_valence[word]) > 20:
                    self.word_valence[word] = self.word_valence[word][-20:]
    
    def _form_concept_clusters(self):
        """Form clusters of related words (emergent categories)"""
        # Clear old clusters
        self.concept_clusters.clear()
        
        # Group words by strong associations
        processed_words = set()
        
        for word, concept in self.word_concepts.items():
            if word in processed_words:
                continue
            
            # Find strongly associated words
            cluster = {word}
            processed_words.add(word)
            
            # Look for strong associations
            for other_word, strength in concept.associations.items():
                if strength > 0.7 and other_word in self.word_concepts:
                    cluster.add(other_word)
                    processed_words.add(other_word)
            
            if len(cluster) > 1:
                # Create cluster
                cluster_id = f"cluster_{hashlib.md5(''.join(sorted(cluster)).encode()).hexdigest()[:8]}"
                self.concept_clusters[cluster_id] = cluster
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Create context identifier"""
        # Use simple features that a baby might notice
        features = []
        
        if 'detected_domains' in context:
            features.append(f"domains:{sorted(context['detected_domains'])}")
        
        if 'query_complexity' in context:
            # Baby-like complexity: simple vs complex
            complexity = "simple" if context['query_complexity'] < 0.5 else "complex"
            features.append(f"complexity:{complexity}")
        
        if 'session_id' in context:
            features.append(f"session:{context['session_id'][:8]}")
        
        context_str = "|".join(features) if features else "default"
        return hashlib.md5(context_str.encode()).hexdigest()[:12]
    
    def understand_connection(self, vni1_output: str, vni2_output: str) -> Dict[str, Any]:
        """
        Understand the relationship between two VNI outputs
        Based on learned semantics, not pre-trained models
        """
        words1 = set(self._simple_tokenize(vni1_output))
        words2 = set(self._simple_tokenize(vni2_output))
        
        # Find overlapping concepts
        overlapping = words1.intersection(words2)
        
        # Calculate semantic similarity based on learned associations
        similarity = self._calculate_semantic_similarity(words1, words2)
        
        # Find complementary concepts (different but related)
        complementary = self._find_complementary_concepts(words1, words2)
        
        # Detect potential contradictions
        contradiction_score = self._detect_contradictions(words1, words2)
        
        # Emotional tone matching
        emotional_alignment = self._calculate_emotional_alignment(words1, words2)
        
        # Generate baby-like understanding
        understanding = self._generate_understanding(
            words1, words2, similarity, complementary, contradiction_score
        )
        
        return {
            'semantic_similarity': similarity,
            'shared_concepts': list(overlapping),
            'complementary_concepts': complementary,
            'contradiction_risk': contradiction_score,
            'emotional_alignment': emotional_alignment,
            'understanding': understanding,
            'concept_clusters_involved': self._get_involved_clusters(words1.union(words2))
        }
    
    def _calculate_semantic_similarity(self, words1: Set[str], words2: Set[str]) -> float:
        """Calculate semantic similarity based on learned associations"""
        if not words1 or not words2:
            return 0.0
        
        similarities = []
        
        for w1 in words1:
            if w1 not in self.word_concepts:
                continue
            
            for w2 in words2:
                if w2 not in self.word_concepts:
                    continue
                
                # Direct association
                key = tuple(sorted([w1, w2]))
                if key in self.word_association_matrix:
                    similarities.append(self.word_association_matrix[key])
                
                # Indirect through shared associations
                concept1 = self.word_concepts[w1]
                concept2 = self.word_concepts[w2]
                
                shared_associations = set(concept1.associations.keys()).intersection(
                    set(concept2.associations.keys())
                )
                
                if shared_associations:
                    # Average strength of shared associations
                    shared_strengths = []
                    for shared in shared_associations:
                        s1 = concept1.associations.get(shared, 0)
                        s2 = concept2.associations.get(shared, 0)
                        shared_strengths.append((s1 + s2) / 2)
                    
                    if shared_strengths:
                        similarities.append(np.mean(shared_strengths))
        
        return np.mean(similarities) if similarities else 0.0
    
    def _find_complementary_concepts(self, words1: Set[str], words2: Set[str]) -> List[Tuple[str, str]]:
        """Find concepts that are different but related"""
        complementary = []
        
        for w1 in words1:
            if w1 not in self.word_concepts:
                continue
            
            for w2 in words2:
                if w2 not in self.word_concepts or w1 == w2:
                    continue
                
                # Check if they're in the same concept cluster
                in_same_cluster = False
                for cluster in self.concept_clusters.values():
                    if w1 in cluster and w2 in cluster:
                        in_same_cluster = True
                        break
                
                if in_same_cluster:
                    # Same category but different words = complementary
                    complementary.append((w1, w2))
        
        return complementary
    
    def _detect_contradictions(self, words1: Set[str], words2: Set[str]) -> float:
        """Detect potential contradictions based on emotional valence"""
        if not words1 or not words2:
            return 0.0
        
        # Check for opposite emotional valence words
        valence_pairs = []
        
        for w1 in words1:
            if w1 in self.word_concepts:
                for w2 in words2:
                    if w2 in self.word_concepts:
                        v1 = self.word_concepts[w1].emotional_valence
                        v2 = self.word_concepts[w2].emotional_valence
                        
                        # Large valence difference suggests contradiction
                        valence_diff = abs(v1 - v2)
                        if valence_diff > 0.6:  # Strong opposite emotions
                            valence_pairs.append(valence_diff)
        
        # Also check for known contradiction patterns
        contradiction_patterns = [
            (['yes', 'no'], 1.0),
            (['good', 'bad'], 0.8),
            (['should', 'shouldnt'], 0.9),
            (['recommend', 'avoid'], 0.7)
        ]
        
        pattern_contradictions = 0
        for pattern1, pattern2, strength in contradiction_patterns:
            has_pattern1 = any(p in words1 for p in pattern1) or any(p in words2 for p in pattern1)
            has_pattern2 = any(p in words1 for p in pattern2) or any(p in words2 for p in pattern2)
            
            if has_pattern1 and has_pattern2:
                pattern_contradictions += strength
        
        # Combine valence and pattern contradictions
        valence_score = np.mean(valence_pairs) if valence_pairs else 0.0
        pattern_score = min(pattern_contradictions, 1.0)
        
        return max(valence_score, pattern_score)
    
    def _calculate_emotional_alignment(self, words1: Set[str], words2: Set[str]) -> float:
        """Calculate emotional alignment between word sets"""
        valences1 = []
        valences2 = []
        
        for w in words1:
            if w in self.word_concepts:
                valences1.append(self.word_concepts[w].emotional_valence)
        
        for w in words2:
            if w in self.word_concepts:
                valences2.append(self.word_concepts[w].emotional_valence)
        
        if not valences1 or not valences2:
            return 0.5  # Neutral
        
        avg1 = np.mean(valences1)
        avg2 = np.mean(valences2)
        
        # Alignment: 1.0 = perfect match, 0.0 = opposite
        alignment = 1.0 - abs(avg1 - avg2)
        
        return max(0.0, min(1.0, alignment))
    
    def _generate_understanding(self, 
                              words1: Set[str], 
                              words2: Set[str],
                              similarity: float,
                              complementary: List[Tuple[str, str]],
                              contradiction: float) -> str:
        """Generate baby-like understanding of the relationship"""
        
        if similarity > 0.7:
            return "Both saying similar things about the same topics."
        
        elif complementary:
            comp_pairs = [f"{w1} and {w2}" for w1, w2 in complementary[:2]]
            comp_str = ", ".join(comp_pairs)
            return f"Talking about related things: {comp_str}."
        
        elif contradiction > 0.6:
            return "Seems to have different opinions or feelings."
        
        elif 0.3 < similarity <= 0.6:
            return "Some overlap in what they're talking about."
        
        else:
            return "Talking about different things."
    
    def _get_involved_clusters(self, words: Set[str]) -> List[Dict[str, Any]]:
        """Get concept clusters involved"""
        involved = []
        
        for cluster_id, cluster_words in self.concept_clusters.items():
            overlap = cluster_words.intersection(words)
            if overlap:
                involved.append({
                    'cluster_id': cluster_id,
                    'words_in_cluster': list(cluster_words),
                    'overlap_words': list(overlap),
                    'overlap_ratio': len(overlap) / len(cluster_words)
                })
        
        return involved
    
    def get_semantic_statistics(self) -> Dict[str, Any]:
        """Get statistics about learned semantics"""
        total_words = len(self.word_concepts)
        total_associations = len(self.word_association_matrix)
        total_patterns = len(self.phrase_patterns)
        
        # Average word strength
        avg_strength = np.mean([c.strength() for c in self.word_concepts.values()]) if self.word_concepts else 0.0
        
        # Most learned concepts
        strongest_concepts = sorted(
            self.word_concepts.items(),
            key=lambda x: x[1].strength(),
            reverse=True
        )[:10]
        
        strongest = [
            {
                'word': word,
                'strength': concept.strength(),
                'usage_count': concept.usage_count,
                'valence': concept.emotional_valence,
                'associations': len(concept.associations)
            }
            for word, concept in strongest_concepts
        ]
        
        return {
            'total_words_learned': total_words,
            'total_associations': total_associations,
            'total_phrase_patterns': total_patterns,
            'average_concept_strength': avg_strength,
            'concept_clusters': len(self.concept_clusters),
            'strongest_concepts': strongest,
            'learning_progress': {
                'early_stage': total_words < 100,
                'intermediate': 100 <= total_words < 500,
                'advanced': total_words >= 500
            }
        }

# ==================== DEMONSTRATION ====================

def demo_emergent_learning():
    """Demonstrate emergent semantic learning like a baby"""
    print("=" * 80)
    print("🧒 EMERGENT SEMANTIC LEARNING DEMONSTRATION")
    print("=" * 80)
    print("Learning from scratch like a baby learns language...")
    print()
    
    learner = EmergentSemanticLearner()
    
    # Simulate baby's first interactions
    interactions = [
        {
            'vni_outputs': {
                'vni1': 'Patient has fever and headache',
                'vni2': 'Recommend rest and fluids'
            },
            'context': {'detected_domains': ['medical'], 'query_complexity': 0.3},
            'outcome': 0.9  # Good outcome
        },
        {
            'vni_outputs': {
                'vni1': 'Legal contract requires review',
                'vni2': 'Important to check liability clauses'
            },
            'context': {'detected_domains': ['legal'], 'query_complexity': 0.6},
            'outcome': 0.8
        },
        {
            'vni_outputs': {
                'vni1': 'Code has bug in function',
                'vni2': 'Need to debug the algorithm'
            },
            'context': {'detected_domains': ['technical'], 'query_complexity': 0.7},
            'outcome': 0.7
        }
    ]
    
    print("📚 Learning from interactions...")
    for i, interaction in enumerate(interactions, 1):
        print(f"\nInteraction {i}:")
        print(f"  VNI1: {interaction['vni_outputs']['vni1']}")
        print(f"  VNI2: {interaction['vni_outputs']['vni2']}")
        print(f"  Outcome: {interaction['outcome']:.1f}")
        
        learner.learn_from_interaction(
            interaction['vni_outputs'],
            interaction['context'],
            interaction['outcome']
        )
    
    print("\n" + "=" * 80)
    print("🧠 WHAT THE LEARNER UNDERSTANDS NOW:")
    print("=" * 80)
    
    stats = learner.get_semantic_statistics()
    print(f"\n📊 Learning Statistics:")
    print(f"  • Words Learned: {stats['total_words_learned']}")
    print(f"  • Associations Formed: {stats['total_associations']}")
    print(f"  • Phrase Patterns: {stats['total_phrase_patterns']}")
    print(f"  • Concept Clusters: {stats['concept_clusters']}")
    
    print(f"\n💪 Strongest Concepts Learned:")
    for i, concept in enumerate(stats['strongest_concepts'][:5], 1):
        print(f"  {i}. '{concept['word']}' - Strength: {concept['strength']:.2f}")
        print(f"     Used {concept['usage_count']} times, Valence: {concept['valence']:.2f}")
    
    print("\n" + "=" * 80)
    print("🔗 TESTING UNDERSTANDING BETWEEN VNIS:")
    print("=" * 80)
    
    test_pairs = [
        ("Patient needs medication", "Doctor prescribed antibiotics"),
        ("Contract has problem", "Legal review required"),
        ("System has error", "Code needs debugging")
    ]
    
    for vni1_says, vni2_says in test_pairs:
        print(f"\nVNI1: {vni1_says}")
        print(f"VNI2: {vni2_says}")
        
        understanding = learner.understand_connection(vni1_says, vni2_says)
        
        print(f"🤔 Understanding: {understanding['understanding']}")
        print(f"   Similarity: {understanding['semantic_similarity']:.2f}")
        print(f"   Shared concepts: {understanding['shared_concepts'][:3]}")
        print(f"   Emotional alignment: {understanding['emotional_alignment']:.2f}")
    
    print("\n" + "=" * 80)
    print("✅ EMERGENT LEARNING DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Insights:")
    print("• Learned semantics from scratch, no pre-trained models")
    print("• Formed word associations through co-occurrence")
    print("• Developed emotional valence through outcomes")
    print("• Created concept clusters naturally")
    print("• Understands relationships based on learned patterns")
    
    return learner

if __name__ == "__main__":
    learner = demo_emergent_learning() 
