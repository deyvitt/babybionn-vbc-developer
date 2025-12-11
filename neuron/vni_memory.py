"""
Memory system with thumbdrive persistence
"""
import numpy as np
import torch
import hashlib
import time
import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

from vni_storage import StorageManager

logger = logging.getLogger("vni_memory")

@dataclass
class MemoryEntry:
    """Single memory entry"""
    vni_id: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    success_score: float
    timestamp: float
    input_hash: str
    input_embedding: np.ndarray  # Vector representation
    metadata: Dict[str, Any]
    
    def to_dict(self):
        return {
            "vni_id": self.vni_id,
            "success_score": self.success_score,
            "timestamp": self.timestamp,
            "input_hash": self.input_hash,
            "metadata": self.metadata,
            "output_summary": str(self.output_data)[:200]
        }
    
    def to_storage_dict(self):
        """Convert to storage-friendly format"""
        return {
            "vni_id": self.vni_id,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "success_score": self.success_score,
            "timestamp": self.timestamp,
            "input_hash": self.input_hash,
            "input_embedding": self.input_embedding.tolist(),  # Convert numpy to list
            "metadata": self.metadata
        }
    
    @classmethod
    def from_storage_dict(cls, data: Dict):
        """Create from storage dictionary"""
        return cls(
            vni_id=data["vni_id"],
            input_data=data["input_data"],
            output_data=data["output_data"],
            success_score=data["success_score"],
            timestamp=data["timestamp"],
            input_hash=data["input_hash"],
            input_embedding=np.array(data["input_embedding"], dtype=np.float32),
            metadata=data.get("metadata", {})
        )

class VniMemory:
    """
    Memory system with thumbdrive persistence
    """
    
    def __init__(self, 
                 vni_id: str, 
                 storage_manager: StorageManager,
                 embedding_dim: int = 256,
                 auto_save: bool = True):
        self.vni_id = vni_id
        self.storage = storage_manager
        self.embedding_dim = embedding_dim
        self.auto_save = auto_save
        
        # Load existing memories from thumbdrive
        self.memories: List[MemoryEntry] = []
        self.pattern_embeddings: List[np.ndarray] = []
        
        self._load_from_storage()
        
        # Statistics
        self.hit_count = 0
        self.miss_count = 0
        self.save_counter = 0
        self.save_threshold = 10  # Save every 10 new memories
        
        logger.info(f"Memory initialized for VNI: {vni_id} (loaded {len(self.memories)} memories)")
    
    def _load_from_storage(self):
        """Load memories from thumbdrive"""
        memories_data, vectors_data = self.storage.load_memory(self.vni_id)
        
        if memories_data:
            # Convert storage data back to MemoryEntry objects
            for mem_dict in memories_data:
                try:
                    entry = MemoryEntry.from_storage_dict(mem_dict)
                    self.memories.append(entry)
                    self.pattern_embeddings.append(entry.input_embedding)
                except Exception as e:
                    logger.error(f"Failed to load memory entry: {e}")
    
    def _save_to_storage(self):
        """Save memories to thumbdrive"""
        if not self.auto_save:
            return
        
        try:
            # Convert to storage format
            memories_data = [mem.to_storage_dict() for mem in self.memories]
            
            # Save using storage manager
            self.storage.save_memory(self.vni_id, memories_data, self.pattern_embeddings)
            
            self.save_counter = 0
            logger.debug(f"Saved {len(self.memories)} memories for {self.vni_id}")
            
        except Exception as e:
            logger.error(f"Failed to save memory for {self.vni_id}: {e}")
    
    def compute_input_hash(self, input_data: Dict[str, Any]) -> str:
        """Create deterministic hash of input data"""
        input_str = json.dumps(input_data, sort_keys=True, default=str)
        return hashlib.md5(input_str.encode()).hexdigest()
    
    def create_embedding(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Create vector embedding from input data"""
        features = []
        
        # Extract numeric features from tensor if present
        if isinstance(input_data, dict) and 'cognitive' in input_data:
            cognitive = input_data['cognitive']
            if 'tensor' in cognitive and isinstance(cognitive['tensor'], torch.Tensor):
                tensor_data = cognitive['tensor'].detach().cpu().numpy()
                features.extend(tensor_data.flatten()[:128])
        
        # Add concept count
        if isinstance(input_data, dict) and 'cognitive' in input_data:
            concepts = input_data['cognitive'].get('concepts', [])
            features.append(len(concepts))
        
        # Pad or truncate to embedding_dim
        if len(features) > self.embedding_dim:
            features = features[:self.embedding_dim]
        else:
            features.extend([0.0] * (self.embedding_dim - len(features)))
        
        return np.array(features, dtype=np.float32)
    
    def remember(self, 
                input_data: Dict[str, Any], 
                output_data: Dict[str, Any], 
                success_score: float = 1.0,
                metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a successful pattern in memory"""
        
        input_hash = self.compute_input_hash(input_data)
        
        # Check if similar pattern already exists
        similar_memory = self.find_similar_memory(input_data, threshold=0.9)
        if similar_memory:
            # Update existing memory with better success score
            if success_score > similar_memory.success_score:
                similar_memory.output_data = output_data
                similar_memory.success_score = success_score
                similar_memory.timestamp = time.time()
                
                # Trigger save if auto-save enabled
                self.save_counter += 1
                if self.save_counter >= self.save_threshold:
                    self._save_to_storage()
                
                logger.debug(f"Updated memory for VNI {self.vni_id}")
            return input_hash
        
        # Create new memory entry
        embedding = self.create_embedding(input_data)
        entry = MemoryEntry(
            vni_id=self.vni_id,
            input_data=input_data,
            output_data=output_data,
            success_score=success_score,
            timestamp=time.time(),
            input_hash=input_hash,
            input_embedding=embedding,
            metadata=metadata or {}
        )
        
        self.memories.append(entry)
        self.pattern_embeddings.append(embedding)
        
        # Trigger save
        self.save_counter += 1
        if self.save_counter >= self.save_threshold:
            self._save_to_storage()
        
        logger.debug(f"Stored new memory for VNI {self.vni_id} (total: {len(self.memories)})")
        return input_hash
    
    def find_similar_memory(self, 
                           input_data: Dict[str, Any], 
                           threshold: float = 0.7,
                           top_k: int = 3) -> Optional[MemoryEntry]:
        """Find similar past patterns using vector similarity"""
        
        if not self.memories:
            self.miss_count += 1
            return None
        
        # Create embedding for current input
        query_embedding = self.create_embedding(input_data)
        
        # Basic cosine similarity (we'll add FAISS later if needed)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return None
        
        best_similarity = 0
        best_memory = None
        
        for i, memory_embedding in enumerate(self.pattern_embeddings):
            memory_norm = np.linalg.norm(memory_embedding)
            if memory_norm == 0:
                continue
                
            similarity = np.dot(query_embedding, memory_embedding) / (query_norm * memory_norm)
            
            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_memory = self.memories[i]
        
        if best_memory:
            self.hit_count += 1
            return best_memory
        
        self.miss_count += 1
        return None
    
    def should_self_activate(self, 
                           incoming_data: Dict[str, Any], 
                           activation_threshold: float = 0.8) -> Tuple[bool, Optional[MemoryEntry]]:
        """
        Decide if this VNI should activate spontaneously
        based on similarity to successful past patterns
        """
        
        similar_memory = self.find_similar_memory(incoming_data, threshold=activation_threshold)
        
        if similar_memory:
            logger.info(f"VNI {self.vni_id} self-activating (similarity: >{activation_threshold})")
            return True, similar_memory
        
        return False, None
    
    def force_save(self):
        """Force immediate save to thumbdrive"""
        self._save_to_storage()
        logger.info(f"Force saved memory for {self.vni_id}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "vni_id": self.vni_id,
            "memory_count": len(self.memories),
            "hit_rate": self.hit_count / max(1, self.hit_count + self.miss_count),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "storage_path": str(self.storage.base_path),
            "recent_memories": [
                entry.to_dict() 
                for entry in self.memories[-5:]  # Last 5 memories
            ]
        }
    
    def clear_memory(self, also_clear_storage: bool = False):
        """Clear memories (optionally from storage too)"""
        self.memories.clear()
        self.pattern_embeddings.clear()
        
        if also_clear_storage:
            # This would delete the file
            pass
        
        logger.info(f"Cleared memory for VNI {self.vni_id}") 
