# enhanced_vni_classes/modules/knowledge_base.py
"""
Knowledge base management for VNIs
"""
import json
import os
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class KnowledgeConcept:
    """Represents a knowledge concept"""
    name: str
    domain: str
    description: str
    metadata: Dict[str, Any]
    embeddings: Optional[List[float]] = None
    frequency: int = 1
    last_accessed: datetime = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = datetime.now()
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        # Handle datetime serialization
        data['last_accessed'] = self.last_accessed.isoformat()
        data['created_at'] = self.created_at.isoformat()
        return data

@dataclass
class KnowledgeEntry:
    """Represents a general knowledge entry (for backward compatibility and flexibility)"""
    id: str
    content: str
    category: str = "general"
    tags: List[str] = None
    source: str = "manual"
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None
    accessed_count: int = 0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "tags": self.tags,
            "source": self.source,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "accessed_count": self.accessed_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEntry':
        """Create from dictionary"""
        # Handle datetime conversion
        created_at = datetime.fromisoformat(data['created_at']) if 'created_at' in data else None
        updated_at = datetime.fromisoformat(data['updated_at']) if 'updated_at' in data else None
        
        return cls(
            id=data['id'],
            content=data['content'],
            category=data.get('category', 'general'),
            tags=data.get('tags', []),
            source=data.get('source', 'manual'),
            confidence=data.get('confidence', 1.0),
            metadata=data.get('metadata', {}),
            created_at=created_at,
            updated_at=updated_at,
            accessed_count=data.get('accessed_count', 0)
        )
    
    def mark_accessed(self):
        """Mark the entry as accessed"""
        self.accessed_count += 1
        self.updated_at = datetime.now()

@dataclass
class KnowledgePattern:
    """Pattern for knowledge matching"""
    pattern: str
    response_template: str
    confidence: float = 0.8
    usage_count: int = 0

class KnowledgeBase:
    """Knowledge base for a VNI"""
    
    def __init__(self, domain: str = "general", embedding_model: str = "all-MiniLM-L6-v2"):
        self.domain = domain
        self.concepts: Dict[str, KnowledgeConcept] = {}
        self.patterns: List[KnowledgePattern] = []
        self.learned_responses: Dict[str, str] = {}
        
        # Embedding model for semantic search
        self.embedding_model = None
        self.embedding_model_name = embedding_model
        self.concept_embeddings: Dict[str, np.ndarray] = {}
        
        logger.info(f"Initialized KnowledgeBase for domain: {domain}")
    
    def load_multiple(self, filepaths: List[str]) -> Dict[str, Any]:
        """
        Load knowledge from multiple files
        """
        loaded_files = []
        total_concepts = 0
        total_patterns = 0
        
        for filepath in filepaths:
            try:
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    # Load concepts
                    if 'concepts' in data:
                        for concept_name, concept_data in data['concepts'].items():
                            self.add_concept(concept_name, concept_data)
                    
                    # Load patterns
                    if 'patterns' in data:
                        for pattern_data in data['patterns']:
                            self.add_pattern(
                                pattern_data['pattern'],
                                pattern_data.get('response_template', '')
                            )
                    
                    # Load learned responses
                    if 'learned_responses' in data:
                        self.learned_responses.update(data['learned_responses'])
                    
                    loaded_files.append(os.path.basename(filepath))
                    logger.info(f"Loaded knowledge from: {filepath}")
                    
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
        
        return {
            "loaded_files": loaded_files,
            "concepts_loaded": len(self.concepts),
            "patterns_loaded": len(self.patterns),
            "learned_responses": len(self.learned_responses)
        }
    
    def add_concept(self, name: str, data: Dict[str, Any]) -> bool:
        """Add a new concept"""
        try:
            concept = KnowledgeConcept(
                name=name,
                domain=self.domain,
                description=data.get('description', ''),
                metadata=data
            )
            
            # Generate embedding if description available
            if concept.description and self._get_embedding_model():
                embedding = self.embedding_model.encode(concept.description)
                concept.embeddings = embedding.tolist()
                self.concept_embeddings[name] = embedding
            
            self.concepts[name] = concept
            return True
            
        except Exception as e:
            logger.error(f"Error adding concept {name}: {e}")
            return False
    
    def add_pattern(self, pattern: str, response_template: str) -> bool:
        """Add a pattern"""
        self.patterns.append(
            KnowledgePattern(pattern=pattern, response_template=response_template)
        )
        return True
    
    def query(self, query: str, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Query knowledge base
        Returns matching concepts sorted by relevance
        """
        results = []
        
        # 1. Check learned responses first (exact match)
        if query in self.learned_responses:
            results.append({
                "type": "learned_response",
                "content": self.learned_responses[query],
                "confidence": 0.9,
                "source": "learned_responses"
            })
        
        # 2. Check patterns
        for pattern in self.patterns:
            if re.search(pattern.pattern, query, re.IGNORECASE):
                response = pattern.response_template
                # Replace placeholders
                response = response.replace("{query}", query)
                results.append({
                    "type": "pattern_match",
                    "content": response,
                    "confidence": pattern.confidence,
                    "source": "patterns"
                })
        
        # 3. Check concepts (keyword matching)
        query_lower = query.lower()
        for concept_name, concept in self.concepts.items():
            if concept_name.lower() in query_lower:
                results.append({
                    "type": "concept_match",
                    "concept": concept_name,
                    "content": concept.description,
                    "confidence": 0.7,
                    "source": "concepts"
                })
        
        # 4. Semantic search if embeddings available
        if self._get_embedding_model() and self.concept_embeddings:
            query_embedding = self.embedding_model.encode(query)
            for concept_name, embedding in self.concept_embeddings.items():
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                if similarity > threshold:
                    concept = self.concepts[concept_name]
                    results.append({
                        "type": "semantic_match",
                        "concept": concept_name,
                        "content": concept.description,
                        "confidence": float(similarity),
                        "source": "embeddings"
                    })
        
        # Sort by confidence
        return sorted(results, key=lambda x: x["confidence"], reverse=True)
    
    def learn_response(self, query: str, response: str) -> bool:
        """Learn a new response"""
        self.learned_responses[query] = response
        return True
    
    def save(self, filepath: str) -> bool:
        """Save knowledge base to file"""
        try:
            data = {
                "domain": self.domain,
                "timestamp": datetime.now().isoformat(),
                "concepts": {},
                "patterns": [],
                "learned_responses": self.learned_responses
            }
            
            # Save concepts
            for name, concept in self.concepts.items():
                data["concepts"][name] = concept.to_dict()
            
            # Save patterns
            for pattern in self.patterns:
                data["patterns"].append({
                    "pattern": pattern.pattern,
                    "response_template": pattern.response_template,
                    "confidence": pattern.confidence,
                    "usage_count": pattern.usage_count
                })
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved knowledge to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving knowledge: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about knowledge base"""
        return {
            "domain": self.domain,
            "concepts_count": len(self.concepts),
            "patterns_count": len(self.patterns),
            "learned_responses_count": len(self.learned_responses),
            "has_embeddings": self.embedding_model is not None
        }
    
    def _get_embedding_model(self):
        """Lazy load embedding model"""
        if self.embedding_model is None:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
        return self.embedding_model 
