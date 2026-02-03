# enhanced_vni_classes/modules/learning_system.py
import json
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib
from collections import defaultdict

@dataclass
class LearningExperience:
    """Represents a single learning experience."""
    id: str
    timestamp: str
    interaction_id: str
    vni_id: str
    domain: str
    content: str
    outcome: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningExperience':
        return cls(**data)

class LearningSystem:
    """Self-learning system for VNIs."""
    
    def __init__(self, vni_id: str):
        self.vni_id = vni_id
        self.learning_data = []
        self.patterns = {}
        self.knowledge_graph = defaultdict(list)
        self.experience_cache = {}
        self.mistakes_log = []
        self.learning_rate = 0.1
        
    def record_interaction(self, 
                          interaction_id: str,
                          prompt: str,
                          response: str,
                          domain: str,
                          feedback: Optional[Dict] = None,
                          metadata: Optional[Dict] = None) -> str:
        """Record an interaction for learning."""
        experience_id = hashlib.md5(
            f"{interaction_id}_{self.vni_id}_{datetime.now().timestamp()}".encode()
        ).hexdigest()[:16]
        
        experience = LearningExperience(
            id=experience_id,
            timestamp=datetime.now().isoformat(),
            interaction_id=interaction_id,
            vni_id=self.vni_id,
            domain=domain,
            content=json.dumps({"prompt": prompt, "response": response}),
            outcome="success" if not feedback else feedback.get("status", "unknown"),
            metadata=metadata or {}
        )
        
        self.learning_data.append(experience)
        self._analyze_pattern(prompt, response, domain)
        self._update_knowledge_graph(prompt, response, domain)
        
        return experience_id
    
    def _analyze_pattern(self, prompt: str, response: str, domain: str):
        """Analyze patterns in the interaction."""
        key = f"{domain}_{hashlib.md5(prompt.lower().encode()).hexdigest()[:8]}"
        
        if key not in self.patterns:
            self.patterns[key] = {
                "count": 0,
                "domain": domain,
                "prompt_pattern": prompt,
                "responses": [],
                "last_used": datetime.now().isoformat()
            }
        
        self.patterns[key]["count"] += 1
        self.patterns[key]["responses"].append(response)
        self.patterns[key]["last_used"] = datetime.now().isoformat()
        
        # Keep only last 5 responses
        if len(self.patterns[key]["responses"]) > 5:
            self.patterns[key]["responses"] = self.patterns[key]["responses"][-5:]
    
    def _update_knowledge_graph(self, prompt: str, response: str, domain: str):
        """Update the knowledge graph with new information."""
        # Simple keyword extraction
        keywords = self._extract_keywords(prompt)
        
        for keyword in keywords:
            self.knowledge_graph[keyword].append({
                "response": response,
                "domain": domain,
                "timestamp": datetime.now().isoformat(),
                "confidence": 0.8
            })
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple implementation - can be enhanced with NLP
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        words = text.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        return keywords[:5]  # Limit to 5 keywords
    
    def learn_from_feedback(self, 
                           experience_id: str, 
                           feedback: Dict[str, Any]):
        """Learn from user feedback."""
        for exp in self.learning_data:
            if exp.id == experience_id:
                exp.outcome = feedback.get("status", "unknown")
                exp.metadata["feedback"] = feedback
                
                if feedback.get("status") == "correction":
                    self.mistakes_log.append({
                        "experience_id": experience_id,
                        "original_response": json.loads(exp.content)["response"],
                        "correction": feedback.get("correction"),
                        "timestamp": datetime.now().isoformat()
                    })
                break
    
    def get_recommendations(self, prompt: str, domain: str) -> List[Dict[str, Any]]:
        """Get recommendations based on learned patterns."""
        recommendations = []
        
        # Search patterns
        for key, pattern in self.patterns.items():
            if domain in key and pattern["count"] > 1:
                if any(word in prompt.lower() for word in pattern["prompt_pattern"].lower().split()):
                    recommendations.append({
                        "type": "pattern_match",
                        "pattern": pattern["prompt_pattern"],
                        "suggested_responses": pattern["responses"][-2:],
                        "confidence": min(0.9, pattern["count"] * 0.1),
                        "source": "learning_pattern"
                    })
        
        # Search knowledge graph
        keywords = self._extract_keywords(prompt)
        for keyword in keywords:
            if keyword in self.knowledge_graph:
                for item in self.knowledge_graph[keyword][-3:]:  # Last 3 items
                    if item["domain"] == domain:
                        recommendations.append({
                            "type": "knowledge_match",
                            "keyword": keyword,
                            "response": item["response"],
                            "confidence": item["confidence"],
                            "source": "knowledge_graph"
                        })
        
        return sorted(recommendations, key=lambda x: x["confidence"], reverse=True)[:3]
    
    def adjust_learning_rate(self, success_rate: float):
        """Adjust learning rate based on success rate."""
        if success_rate > 0.8:
            self.learning_rate = min(0.3, self.learning_rate * 1.1)
        elif success_rate < 0.5:
            self.learning_rate = max(0.05, self.learning_rate * 0.9)
    
    def export_knowledge(self) -> Dict[str, Any]:
        """Export learned knowledge."""
        return {
            "vni_id": self.vni_id,
            "patterns": self.patterns,
            "knowledge_graph": dict(self.knowledge_graph),
            "learning_data_count": len(self.learning_data),
            "mistakes_count": len(self.mistakes_log),
            "learning_rate": self.learning_rate,
            "export_timestamp": datetime.now().isoformat()
        }
    
    def import_knowledge(self, knowledge_data: Dict[str, Any]):
        """Import previously learned knowledge."""
        if knowledge_data.get("vni_id") == self.vni_id:
            self.patterns = knowledge_data.get("patterns", {})
            self.knowledge_graph = defaultdict(list, knowledge_data.get("knowledge_graph", {}))
            self.learning_rate = knowledge_data.get("learning_rate", 0.1)
    
    def save(self, filepath: str):
        """Save learning system to file."""
        data = {
            "vni_id": self.vni_id,
            "learning_data": [exp.to_dict() for exp in self.learning_data],
            "patterns": self.patterns,
            "knowledge_graph": dict(self.knowledge_graph),
            "mistakes_log": self.mistakes_log,
            "learning_rate": self.learning_rate
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load learning system from file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            if data.get("vni_id") == self.vni_id:
                self.learning_data = [LearningExperience.from_dict(exp) 
                                     for exp in data.get("learning_data", [])]
                self.patterns = data.get("patterns", {})
                self.knowledge_graph = defaultdict(list, data.get("knowledge_graph", {}))
                self.mistakes_log = data.get("mistakes_log", [])
                self.learning_rate = data.get("learning_rate", 0.1)
        except FileNotFoundError:
            pass  # Start fresh if no file exists 
