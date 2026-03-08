# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

# enhanced_vni_classes/modules/attention.py
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
from datetime import datetime

class AttentionType(Enum):
    CONTENT_BASED = "content_based"
    LOCATION_BASED = "location_based"
    MULTI_HEAD = "multi_head"
    SELF_ATTENTION = "self_attention"

@dataclass
class AttentionWeight:
    """Represents attention weight for a specific component."""
    component: str
    weight: float
    relevance: float
    confidence: float

class AttentionMechanism:
    """Attention mechanism for focusing on relevant information."""
    
    def __init__(self, vni_id: str):
        self.vni_id = vni_id
        self.attention_weights = {}
        self.attention_history = []
        self.max_history = 100
        
    def compute_attention(self,
                         query: str,
                         context: Dict[str, Any],
                         attention_type: str = AttentionType.CONTENT_BASED.value) -> Dict[str, Any]:
        """Compute attention weights for different context components."""
        
        attention_weights = {}
        
        # Process different context components
        if context.get("knowledge"):
            attention_weights["knowledge"] = self._compute_knowledge_attention(
                query, context["knowledge"], attention_type
            )
        
        if context.get("web_results"):
            attention_weights["web"] = self._compute_web_attention(
                query, context["web_results"], attention_type
            )
        
        if context.get("collaboration_results"):
            attention_weights["collaboration"] = self._compute_collaboration_attention(
                query, context["collaboration_results"], attention_type
            )
        
        if context.get("previous_responses"):
            attention_weights["history"] = self._compute_history_attention(
                query, context["previous_responses"], attention_type
            )
        
        # Normalize weights
        total_weight = sum(w.weight for w in attention_weights.values())
        if total_weight > 0:
            for key in attention_weights:
                attention_weights[key].weight /= total_weight
        
        # Store in history
        self._store_attention_history(query, attention_weights, context.get("domain", "general"))
        
        return {
            "attention_weights": attention_weights,
            "primary_focus": self._get_primary_focus(attention_weights),
            "attention_type": attention_type,
            "query_relevance": self._compute_query_relevance(query, context),
            "computation_timestamp": datetime.now().isoformat()
        }
    
    def _compute_knowledge_attention(self, 
                                    query: str, 
                                    knowledge: Dict[str, Any],
                                    attention_type: str) -> AttentionWeight:
        """Compute attention for knowledge base."""
        
        relevance = self._compute_text_relevance(query, knowledge.get("content", ""))
        confidence = knowledge.get("confidence", 0.5)
        
        # Adjust weight based on attention type
        if attention_type == AttentionType.CONTENT_BASED.value:
            weight = relevance * confidence
        elif attention_type == AttentionType.SELF_ATTENTION.value:
            weight = relevance * 0.8 + confidence * 0.2
        else:
            weight = relevance
        
        return AttentionWeight(
            component="knowledge",
            weight=weight,
            relevance=relevance,
            confidence=confidence
        )
    
    def _compute_web_attention(self,
                              query: str,
                              web_results: List[Dict[str, Any]],
                              attention_type: str) -> AttentionWeight:
        """Compute attention for web search results."""
        
        if not web_results:
            return AttentionWeight(
                component="web",
                weight=0.0,
                relevance=0.0,
                confidence=0.0
            )
        
        # Aggregate relevance from top results
        relevances = []
        confidences = []
        
        for result in web_results[:3]:  # Top 3 results
            snippet = result.get("snippet", "")
            relevance = self._compute_text_relevance(query, snippet)
            confidence = result.get("relevance_score", 0.5) * result.get("verification_level", 0.5)
            
            relevances.append(relevance)
            confidences.append(confidence)
        
        avg_relevance = np.mean(relevances) if relevances else 0.0
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        if attention_type == AttentionType.CONTENT_BASED.value:
            weight = avg_relevance * avg_confidence
        elif attention_type == AttentionType.MULTI_HEAD.value:
            weight = avg_relevance * 0.6 + avg_confidence * 0.4
        else:
            weight = avg_relevance
        
        return AttentionWeight(
            component="web",
            weight=weight,
            relevance=avg_relevance,
            confidence=avg_confidence
        )
    
    def _compute_collaboration_attention(self,
                                        query: str,
                                        collaboration_results: List[Dict[str, Any]],
                                        attention_type: str) -> AttentionWeight:
        """Compute attention for collaboration results."""
        
        if not collaboration_results:
            return AttentionWeight(
                component="collaboration",
                weight=0.0,
                relevance=0.0,
                confidence=0.0
            )
        
        # Consider responses from other VNIs
        relevances = []
        confidences = []
        
        for collab in collaboration_results:
            response = collab.get("response", "")
            relevance = self._compute_text_relevance(query, response)
            confidence = collab.get("confidence", 0.5) * collab.get("expertise_level", 0.5)
            
            relevances.append(relevance)
            confidences.append(confidence)
        
        avg_relevance = np.mean(relevances) if relevances else 0.0
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        weight = avg_relevance * avg_confidence
        
        return AttentionWeight(
            component="collaboration",
            weight=weight,
            relevance=avg_relevance,
            confidence=avg_confidence
        )
    
    def _compute_history_attention(self,
                                  query: str,
                                  previous_responses: List[Dict[str, Any]],
                                  attention_type: str) -> AttentionWeight:
        """Compute attention for response history."""
        
        if not previous_responses:
            return AttentionWeight(
                component="history",
                weight=0.0,
                relevance=0.0,
                confidence=0.0
            )
        
        # Look for similar previous queries
        relevances = []
        
        for prev in previous_responses[-5:]:  # Last 5 responses
            prev_query = prev.get("query", "")
            relevance = self._compute_text_relevance(query, prev_query)
            relevances.append(relevance)
        
        avg_relevance = np.mean(relevances) if relevances else 0.0
        weight = avg_relevance * 0.7  # History gets less weight
        
        return AttentionWeight(
            component="history",
            weight=weight,
            relevance=avg_relevance,
            confidence=0.6  # Moderate confidence in history
        )
    
    def _compute_text_relevance(self, query: str, text: str) -> float:
        """Compute relevance between query and text."""
        if not query or not text:
            return 0.0
        
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words or not text_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(query_words.intersection(text_words))
        union = len(query_words.union(text_words))
        
        if union == 0:
            return 0.0
        
        similarity = intersection / union
        
        # Boost if query is substring of text
        if query.lower() in text.lower():
            similarity = min(1.0, similarity + 0.3)
        
        return similarity
    
    def _compute_query_relevance(self, query: str, context: Dict[str, Any]) -> float:
        """Compute overall query relevance to context."""
        relevance_scores = []
        
        if context.get("knowledge"):
            relevance_scores.append(
                self._compute_text_relevance(query, context["knowledge"].get("content", ""))
            )
        
        if context.get("web_results"):
            for result in context["web_results"][:2]:
                relevance_scores.append(
                    self._compute_text_relevance(query, result.get("snippet", ""))
                )
        
        if relevance_scores:
            return np.mean(relevance_scores)
        
        return 0.5  # Default moderate relevance
    
    def _get_primary_focus(self, attention_weights: Dict[str, AttentionWeight]) -> str:
        """Get the primary focus component."""
        if not attention_weights:
            return "none"
        
        primary = max(attention_weights.items(), key=lambda x: x[1].weight)
        return primary[0]
    
    def _store_attention_history(self, 
                                query: str, 
                                attention_weights: Dict[str, AttentionWeight],
                                domain: str):
        """Store attention computation in history."""
        
        history_entry = {
            "query": query,
            "domain": domain,
            "attention_distribution": {
                component: {
                    "weight": aw.weight,
                    "relevance": aw.relevance,
                    "confidence": aw.confidence
                }
                for component, aw in attention_weights.items()
            },
            "timestamp": datetime.now().isoformat(),
            "query_hash": hashlib.md5(query.encode()).hexdigest()[:8]
        }
        
        self.attention_history.append(history_entry)
        
        # Limit history size
        if len(self.attention_history) > self.max_history:
            self.attention_history = self.attention_history[-self.max_history:]
    
    def get_attention_patterns(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """Get attention patterns for a domain."""
        
        if domain:
            domain_history = [entry for entry in self.attention_history 
                            if entry["domain"] == domain]
        else:
            domain_history = self.attention_history
        
        if not domain_history:
            return {"patterns": [], "insights": "No history available"}
        
        # Analyze patterns
        component_usage = {}
        avg_weights = {}
        
        for entry in domain_history:
            for component, weights in entry["attention_distribution"].items():
                if component not in component_usage:
                    component_usage[component] = 0
                    avg_weights[component] = []
                
                component_usage[component] += 1
                avg_weights[component].append(weights["weight"])
        
        # Calculate statistics
        patterns = []
        for component in component_usage:
            weights = avg_weights[component]
            patterns.append({
                "component": component,
                "usage_frequency": component_usage[component] / len(domain_history),
                "average_weight": np.mean(weights) if weights else 0,
                "weight_std": np.std(weights) if len(weights) > 1 else 0,
                "dominance_score": component_usage[component] * np.mean(weights) if weights else 0
            })
        
        # Sort by dominance
        patterns.sort(key=lambda x: x["dominance_score"], reverse=True)
        
        insights = []
        if patterns and patterns[0]["usage_frequency"] > 0.7:
            insights.append(f"Primary reliance on {patterns[0]['component']}")
        
        return {
            "patterns": patterns,
            "insights": insights,
            "total_entries": len(domain_history),
            "time_range": {
                "oldest": domain_history[0]["timestamp"],
                "newest": domain_history[-1]["timestamp"]
            } if domain_history else {}
        }
    
    def adjust_attention_bias(self, component: str, bias_factor: float):
        """Adjust attention bias for a specific component."""
        if 0 <= bias_factor <= 2:  # Allow up to 2x bias
            self.attention_weights[component] = bias_factor
    
    def get_attention_summary(self) -> Dict[str, Any]:
        """Get summary of attention mechanism."""
        return {
            "vni_id": self.vni_id,
            "history_size": len(self.attention_history),
            "active_biases": self.attention_weights,
            "domains_analyzed": list(set(entry["domain"] for entry in self.attention_history)),
            "most_common_focus": self._get_most_common_focus(),
            "attention_efficiency": self._compute_efficiency()
        }
    
    def _get_most_common_focus(self) -> str:
        """Get the most common focus component."""
        if not self.attention_history:
            return "none"
        
        focus_counts = {}
        for entry in self.attention_history:
            focus = entry.get("attention_distribution", {})
            if focus:
                primary = max(focus.items(), key=lambda x: x[1]["weight"])
                component = primary[0]
                focus_counts[component] = focus_counts.get(component, 0) + 1
        
        if focus_counts:
            return max(focus_counts.items(), key=lambda x: x[1])[0]
        
        return "none"
    
    def _compute_efficiency(self) -> float:
        """Compute attention efficiency score."""
        if len(self.attention_history) < 10:
            return 0.5
        
        # Efficiency based on how quickly primary focus is determined
        focus_changes = 0
        last_focus = None
        
        for entry in self.attention_history[-20:]:  # Last 20 entries
            focus = entry.get("attention_distribution", {})
            if focus:
                primary = max(focus.items(), key=lambda x: x[1]["weight"])[0]
                if last_focus and primary != last_focus:
                    focus_changes += 1
                last_focus = primary
        
        change_rate = focus_changes / min(19, len(self.attention_history) - 1)
        efficiency = 1.0 - min(1.0, change_rate * 2)  # Lower change rate is better
        
        return efficiency 
