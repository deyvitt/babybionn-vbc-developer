# /bionn-demo-chatbot$ nano neuron/vni_memory.py
"""
Enhanced Memory system with thumbdrive persistence
"""
import numpy as np
import torch
import hashlib
import time
import json
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
import logging
from collections import defaultdict
from datetime import datetime

from neuron.vni_storage import StorageManager

logger = logging.getLogger("vni_memory")

# Try to import FAISS for fast similarity search
try:
    import faiss
    HAS_FAISS = True
    logger.info("FAISS imported successfully for fast vector search")
except ImportError:
    HAS_FAISS = False
    logger.warning("FAISS not installed, using basic similarity search")

@dataclass
class MemoryVersion:
    """Version of a memory entry"""
    timestamp: float
    success_score: float
    output_data: Dict[str, Any]
    
    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "success_score": self.success_score,
            "output_summary": str(self.output_data)[:100]
        }

@dataclass
class MemoryEntry:
    """Single memory entry with versioning"""
    vni_id: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    success_score: float
    timestamp: float
    input_hash: str
    input_embedding: np.ndarray  # Vector representation
    metadata: Dict[str, Any]
    versions: List[MemoryVersion] = field(default_factory=list)
    access_count: int = 0
    last_accessed: float = 0.0
    
    def __post_init__(self):
        # Initialize versions if empty
        if not self.versions:
            self.versions = [
                MemoryVersion(
                    timestamp=self.timestamp,
                    success_score=self.success_score,
                    output_data=self.output_data.copy()
                )
            ]
        # Initialize last_accessed if not set
        if self.last_accessed == 0.0:
            self.last_accessed = self.timestamp
    
    def to_dict(self):
        return {
            "vni_id": self.vni_id,
            "success_score": self.success_score,
            "timestamp": self.timestamp,
            "input_hash": self.input_hash,
            "metadata": self.metadata,
            "output_summary": str(self.output_data)[:200],
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "version_count": len(self.versions)
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
            "input_embedding": self.input_embedding.tolist(),
            "metadata": self.metadata,
            "versions": [{"timestamp": v.timestamp, 
                         "success_score": v.success_score, 
                         "output_data": v.output_data} 
                        for v in self.versions],
            "access_count": self.access_count,
            "last_accessed": self.last_accessed
        }
    
    def add_version(self, new_score: float, new_output: Dict[str, Any]):
        """Add a new version of this memory"""
        self.versions.append(
            MemoryVersion(
                timestamp=time.time(),
                success_score=new_score,
                output_data=new_output.copy()
            )
        )
        self.success_score = new_score
        self.output_data = new_output
        self.timestamp = time.time()
    
    def mark_accessed(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = time.time()

@dataclass
class MemoryCluster:
    """Cluster of related memories"""
    cluster_id: str
    centroid: np.ndarray
    memories: List[MemoryEntry]
    topic: str
    created_at: float
    updated_at: float
    size: int = 0
    
    def __post_init__(self):
        self.size = len(self.memories)
    
    def update_centroid(self):
        """Recalculate cluster centroid"""
        if not self.memories:
            return
        embeddings = [mem.input_embedding for mem in self.memories]
        self.centroid = np.mean(embeddings, axis=0)
        self.updated_at = time.time()
    
    def add_memory(self, memory: MemoryEntry):
        """Add memory to cluster"""
        self.memories.append(memory)
        self.size += 1
        self.updated_at = time.time()
    
    def to_dict(self):
        return {
            "cluster_id": self.cluster_id,
            "topic": self.topic,
            "size": self.size,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sample_memories": [mem.to_dict() for mem in self.memories[:3]]  # First 3
        }

class VniMemory:
    """Enhanced Memory system with thumbdrive persistence and advanced features"""
    def __init__(self, 
                 vni_id: str = None, 
                 storage_manager: StorageManager = None,
                 domain: str = "general",
                 memory_type: str = "episodic",
                 embedding_dim: int = 256,
                 auto_save: bool = True,
                 max_memories: int = 1000,
                 prune_frequency: int = 50,
                 config: Any = None):
        # Handle case where config object is passed as vni_id
        if isinstance(vni_id, (dict, object)) and not isinstance(vni_id, str):
            # vni_id is actually a config object
            config = vni_id
            vni_id = None
        
        # If vni_id is still None, try to get it from config
        if vni_id is None:
            if config is not None:
                if hasattr(config, 'aggregator_id'):
                    vni_id = config.aggregator_id
                elif hasattr(config, 'vni_id'):
                    vni_id = config.vni_id
                elif hasattr(config, 'router_id'):
                    vni_id = config.router_id
                elif isinstance(config, str):
                    vni_id = config
                else:
                    # Generate a default ID
                    vni_id = f"memory_{id(config)}"
            else:
                vni_id = "unknown_vni"

        self.vni_id = vni_id
        self.domain = domain
        self.memory_type = memory_type
        self.storage = storage_manager
        if self.storage is None:
            # Create simple in-memory storage
            class SimpleStorage:
                def __init__(self):
                    self.data = {}
                def save(self, key, value):
                    self.data[key] = value
                def load(self, key):
                    return self.data.get(key)
                
                def save_memory(self, vni_id, memories_data, vectors_data):
                    """Save memory data"""
                    self.save(vni_id, {"memories": memories_data, "vectors": vectors_data})
                
                def load_memory(self, vni_id):
                    """Load memory data"""
                    data = self.load(vni_id)
                    if data:
                        return data.get("memories", []), data.get("vectors", [])
                    return [], []
            self.storage = SimpleStorage()
        self.embedding_dim = embedding_dim
        self.auto_save = auto_save
        self.max_memories = max_memories
        self.prune_frequency = prune_frequency
                
        # Memory storage
        self.memories: List[MemoryEntry] = []
        self.pattern_embeddings: List[np.ndarray] = []
        
        # FAISS index for fast similarity search
        self.faiss_index = None
        self.faiss_needs_rebuild = False
        
        # Clustering
        self.clusters: List[MemoryCluster] = []
        self.cluster_threshold = 0.8  # Similarity threshold for clustering
        
        # Statistics and counters
        self.hit_count = 0
        self.miss_count = 0
        self.save_counter = 0
        self.save_threshold = 10  # Save every 10 new memories
        self.prune_counter = 0
        
        # Context tracking
        self.current_context: List[str] = []
        
        # Load existing memories from thumbdrive
        self._load_from_storage()
        
        # Initialize FAISS index if available
        if HAS_FAISS and self.pattern_embeddings:
            self._build_faiss_index()
        
        # Initial clustering if we have memories
        if self.memories:
            self._cluster_memories()
        
        logger.info(f"Memory initialized for VNI: {vni_id} (loaded {len(self.memories)} memories)")

    def record_interaction_pattern(self, vni_ids=None, outputs=None, biological_states=None, **kwargs):
        """Record an interaction pattern for learning
        Args:
            vni_ids: List of VNI IDs that were activated
            outputs: Dictionary of VNI outputs
            biological_states: Biological states of the VNIs
            **kwargs: Any additional parameters"""
        logger = logging.getLogger(__name__)
        logger.debug(f"Recording interaction pattern with {len(vni_ids) if vni_ids else 0} VNIs")
        
        # Initialize patterns storage if it doesn't exist
        if not hasattr(self, 'interaction_patterns'):
            self.interaction_patterns = {}
        
        # Create a key from the vni_ids (this is the pattern)
        if vni_ids and isinstance(vni_ids, list):
            pattern_key = "->".join(str(p) for p in vni_ids)
        else:
            pattern_key = str(vni_ids) if vni_ids else "unknown"
        
        # Calculate a success metric from biological states if available
        success_metric = 0.5  # default
        if biological_states and isinstance(biological_states, dict):
            # Average the activation levels as a simple success metric
            activations = [state.get('current', 0.5) for state in biological_states.values() if isinstance(state, dict)]
            if activations:
                success_metric = sum(activations) / len(activations)
        
        # Store the pattern with its success metric
        self.interaction_patterns[pattern_key] = {
            'success': success_metric,
            'timestamp': time.time(),
            'count': self.interaction_patterns.get(pattern_key, {}).get('count', 0) + 1,
            'vni_count': len(vni_ids) if vni_ids else 0
        }
        
        # Keep only recent patterns (last 1000)
        if len(self.interaction_patterns) > 1000:
            oldest_keys = sorted(
                self.interaction_patterns.keys(),
                key=lambda k: self.interaction_patterns[k]['timestamp']
            )[:len(self.interaction_patterns) - 1000]
            for key in oldest_keys:
                del self.interaction_patterns[key]
        
        return True

    def retrieve_relevant_memory(self, query: str, limit: int = 5) -> List[Dict]:
        try:
            if not self.memories or not query.strip():
                return []
            
            query_lower = query.lower()
            relevant_memories = []
            
            for memory_entry in self.memories[-100:]:  # memory_entry is MemoryEntry object
                # Convert MemoryEntry to dict for the helper method
                memory_dict = {
                    'content': str(memory_entry.input_data),
                    'summary': str(memory_entry.output_data) if memory_entry.output_data else "",
                    'domain': memory_entry.metadata.get('domain', '') if memory_entry.metadata else '',
                    'metadata': memory_entry.metadata
                }
                
                if self._is_relevant_memory(memory_dict, query_lower):
                    # Return the full MemoryEntry or convert to dict
                    relevant_memories.append({
                        'content': memory_entry.input_data,
                        'output': memory_entry.output_data,
                        'timestamp': memory_entry.timestamp,
                        'success_score': memory_entry.success_score,
                        'metadata': memory_entry.metadata,
                        'vni_id': memory_entry.vni_id
                    })
                    
                if len(relevant_memories) >= limit:
                    break
                    
            return relevant_memories
            
        except Exception as e:
            self.logger.error(f"Error retrieving relevant memory: {str(e)}")
            return []
        
    def _is_relevant_memory(self, memory: Dict, query: str) -> bool:
        """Helper to determine if a memory is relevant to the query."""
        try:
            # Check memory content
            memory_text = ""
            if 'content' in memory:
                memory_text += str(memory['content']).lower()
            if 'summary' in memory:
                memory_text += " " + str(memory['summary']).lower()
            if 'domain' in memory:
                memory_text += " " + str(memory['domain']).lower()
            
            # Check for keyword matches
            common_terms = ['patient', 'diagnosis', 'treatment', 'symptom', 
                           'medical', 'health', 'doctor', 'hospital']
            
            query_terms = query.split()
            for term in query_terms:
                if len(term) > 3 and term in memory_text:
                    return True
            
            # Check for common medical terms
            for term in common_terms:
                if term in query and term in memory_text:
                    return True        
            return False
            
        except Exception:
            return False
            
    def _load_from_storage(self):
        """Load memories from thumbdrive"""
        memories_data, vectors_data = self.storage.load_memory(self.vni_id)
        
        if memories_data:
            for mem_dict in memories_data:
                try:
                    # Convert storage data back to MemoryEntry
                    versions_data = mem_dict.get("versions", [])
                    versions = [
                        MemoryVersion(
                            timestamp=v["timestamp"],
                            success_score=v["success_score"],
                            output_data=v["output_data"]
                        ) for v in versions_data
                    ]
                    
                    entry = MemoryEntry(
                        vni_id=mem_dict["vni_id"],
                        input_data=mem_dict["input_data"],
                        output_data=mem_dict["output_data"],
                        success_score=mem_dict["success_score"],
                        timestamp=mem_dict["timestamp"],
                        input_hash=mem_dict["input_hash"],
                        input_embedding=np.array(mem_dict["input_embedding"], dtype=np.float32),
                        metadata=mem_dict.get("metadata", {}),
                        versions=versions,
                        access_count=mem_dict.get("access_count", 0),
                        last_accessed=mem_dict.get("last_accessed", mem_dict["timestamp"])
                    )
                    
                    self.memories.append(entry)
                    self.pattern_embeddings.append(entry.input_embedding)
                    
                except Exception as e:
                    logger.error(f"Failed to load memory entry: {e}")
    
    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        if not HAS_FAISS or not self.pattern_embeddings:
            return
        
        try:
            embeddings = np.array(self.pattern_embeddings, dtype=np.float32)
            if embeddings.shape[0] > 0:
                # Use IndexFlatIP for cosine similarity (after normalization)
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings)
                self.faiss_index.add(embeddings)
                self.faiss_needs_rebuild = False
                logger.debug(f"Built FAISS index with {embeddings.shape[0]} vectors")
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            self.faiss_index = None
    
    def _save_to_storage(self):
        """Save memories to thumbdrive"""
        if not self.auto_save or not self.memories:
            return
        
        try:
            memories_data = [mem.to_storage_dict() for mem in self.memories]
            self.storage.save_memory(self.vni_id, memories_data, self.pattern_embeddings)
            self.save_counter = 0
            logger.debug(f"Saved {len(self.memories)} memories for {self.vni_id}")
        except Exception as e:
            logger.error(f"Failed to save memory for {self.vni_id}: {e}")
    
    def compute_input_hash(self, input_data: Dict[str, Any]) -> str:
        """Create deterministic hash of input data"""
        input_str = json.dumps(input_data, sort_keys=True, default=str)
        return hashlib.sha256(input_str.encode()).hexdigest()[:32]
    
    def create_embedding(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Create vector embedding from input data"""
        features = []
        
        # Extract numeric features from tensor if present
        if isinstance(input_data, dict) and 'cognitive' in input_data:
            cognitive = input_data['cognitive']
            
            # Extract tensor features
            if 'tensor' in cognitive and isinstance(cognitive['tensor'], torch.Tensor):
                tensor_data = cognitive['tensor'].detach().cpu().numpy()
                # Take first 128 values or flatten if smaller
                flat_tensor = tensor_data.flatten()
                features.extend(flat_tensor[:128].tolist())
            
            # Add concept-based features
            concepts = cognitive.get('concepts', [])
            features.append(len(concepts))
            
            # Add concept length statistics
            if concepts:
                concept_lengths = [len(str(c)) for c in concepts]
                features.append(np.mean(concept_lengths))
                features.append(np.std(concept_lengths))
        
        # Add metadata features if present
        if 'metadata' in input_data:
            metadata = input_data['metadata']
            # Add simple numeric metadata
            for key, value in metadata.items():
                if isinstance(value, (int, float)):
                    features.append(float(value))
        
        # Pad or truncate to embedding_dim
        if len(features) > self.embedding_dim:
            features = features[:self.embedding_dim]
        else:
            # Use random values for padding to avoid zero vectors
            padding_needed = self.embedding_dim - len(features)
            if padding_needed > 0:
                # Small random values to avoid zero vectors
                features.extend(np.random.normal(0, 0.01, padding_needed).tolist())
        
        embedding = np.array(features, dtype=np.float32)
        
        # Normalize to unit length for better cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return np.dot(a, b) / (norm_a * norm_b)
    
    def _find_similar_memory_basic(self, query_embedding: np.ndarray, 
                                   threshold: float = 0.7) -> Optional[Tuple[int, float]]:
        """Basic linear search for similar memories"""
        best_idx = -1
        best_similarity = 0
        
        for i, memory_embedding in enumerate(self.pattern_embeddings):
            similarity = self._cosine_similarity(query_embedding, memory_embedding)
            
            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_idx = i
        
        if best_idx >= 0:
            return best_idx, best_similarity
        
        return None
    
    def _find_similar_memory_faiss(self, query_embedding: np.ndarray,
                                   threshold: float = 0.7, top_k: int = 1) -> Optional[Tuple[int, float]]:
        """FAISS-accelerated similarity search"""
        if not self.faiss_index or not self.memories:
            return None
        
        try:
            # Prepare query vector
            query = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query)  # Normalize for cosine similarity
            
            # Search
            distances, indices = self.faiss_index.search(
                query, min(top_k, len(self.memories))
            )
            
            if distances[0][0] > threshold:
                return int(indices[0][0]), float(distances[0][0])
            
        except Exception as e:
            logger.warning(f"FAISS search failed: {e}, falling back to basic search")
        
        return None
    
    def find_similar_memory(self, 
                           input_data: Dict[str, Any], 
                           threshold: float = 0.7,
                           use_context: bool = True) -> Optional[MemoryEntry]:
        """Find similar past patterns using vector similarity"""
        
        if not self.memories:
            self.miss_count += 1
            return None
        
        # Create embedding for current input
        query_embedding = self.create_embedding(input_data)
        
        # Try FAISS first if available
        result = None
        if HAS_FAISS and self.faiss_index:
            result = self._find_similar_memory_faiss(query_embedding, threshold)
        
        # Fallback to basic search
        if result is None:
            result = self._find_similar_memory_basic(query_embedding, threshold)
        
        if result:
            idx, similarity = result
            memory = self.memories[idx]
            
            # Context filtering if enabled
            if use_context and self.current_context:
                mem_context = memory.metadata.get('context', [])
                context_overlap = len(set(self.current_context) & set(mem_context))
                
                if context_overlap > 0:
                    # Boost confidence with context match
                    logger.debug(f"Context match: {context_overlap} shared topics")
                    memory.mark_accessed()
                    self.hit_count += 1
                    return memory
            
            # Return even without context if threshold is high
            memory.mark_accessed()
            self.hit_count += 1
            return memory
        
        self.miss_count += 1
        return None
    
    def _cluster_memories(self, similarity_threshold: float = 0.8):
        """Cluster memories based on similarity"""
        if len(self.memories) < 5:  # Need minimum memories for clustering
            return
        
        self.clusters = []
        clustered_indices = set()
        
        # Sort memories by success score (higher first)
        scored_memories = [(i, mem.success_score) for i, mem in enumerate(self.memories)]
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        for i, score in scored_memories:
            if i in clustered_indices:
                continue
            
            # Find similar memories
            cluster_indices = [i]
            query_embedding = self.pattern_embeddings[i]
            
            for j, _ in scored_memories:
                if j == i or j in clustered_indices:
                    continue
                
                similarity = self._cosine_similarity(query_embedding, self.pattern_embeddings[j])
                
                if similarity > similarity_threshold:
                    cluster_indices.append(j)
            
            # Create cluster if we have enough similar memories
            if len(cluster_indices) >= 2:
                cluster_memories = [self.memories[idx] for idx in cluster_indices]
                
                # Calculate centroid
                cluster_embeddings = [self.pattern_embeddings[idx] for idx in cluster_indices]
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # Extract common topic from metadata
                topics = []
                for mem in cluster_memories:
                    topic = mem.metadata.get('topic', '')
                    if topic:
                        topics.append(topic)
                
                common_topic = max(set(topics), key=topics.count) if topics else "general"
                
                cluster = MemoryCluster(
                    cluster_id=f"cluster_{len(self.clusters)}_{int(time.time())}",
                    centroid=centroid,
                    memories=cluster_memories,
                    topic=common_topic,
                    created_at=time.time(),
                    updated_at=time.time()
                )
                
                self.clusters.append(cluster)
                clustered_indices.update(cluster_indices)
        
        logger.debug(f"Created {len(self.clusters)} memory clusters")
    
    def _prune_memories(self):
        """Remove least useful memories if we exceed capacity"""
        if len(self.memories) <= self.max_memories:
            return
        
        logger.info(f"Pruning memories for VNI {self.vni_id} ({len(self.memories)} > {self.max_memories})")
        
        # Calculate utility score for each memory
        # Combines: success_score, recency, access frequency
        memory_scores = []
        current_time = time.time()
        
        for idx, mem in enumerate(self.memories):
            # Recency factor (exponential decay)
            recency = 1.0 / (1.0 + np.log1p(current_time - mem.last_accessed))
            
            # Access frequency factor
            frequency = np.log1p(mem.access_count)
            
            # Combined utility score
            utility = (
                0.5 * mem.success_score +  # Success importance
                0.3 * recency +            # Recency importance
                0.2 * frequency            # Frequency importance
            )
            
            memory_scores.append((idx, utility))
        
        # Sort by utility (lowest first)
        memory_scores.sort(key=lambda x: x[1])
        
        # Determine how many to remove
        to_remove = len(self.memories) - self.max_memories
        indices_to_remove = [idx for idx, _ in memory_scores[:to_remove]]
        indices_to_remove.sort(reverse=True)  # Remove from end to preserve indices
        
        # Remove memories
        removed_count = 0
        for idx in indices_to_remove:
            if idx < len(self.memories):
                self.memories.pop(idx)
                self.pattern_embeddings.pop(idx)
                removed_count += 1
        
        # Mark FAISS for rebuild
        self.faiss_needs_rebuild = True
        
        # Re-cluster if we have clusters
        if self.clusters:
            self._cluster_memories()
        
        logger.info(f"Pruned {removed_count} memories for VNI {self.vni_id}")
    
    def remember(self, 
                input_data: Dict[str, Any], 
                output_data: Dict[str, Any], 
                success_score: float = 1.0,
                metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a successful pattern in memory"""
        
        # Check pruning
        self.prune_counter += 1
        if self.prune_counter >= self.prune_frequency:
            self._prune_memories()
            self.prune_counter = 0
        
        input_hash = self.compute_input_hash(input_data)
        metadata = metadata or {}
        
        # Add context to metadata if available
        if self.current_context:
            metadata['context'] = self.current_context.copy()
        
        # Check if similar pattern already exists
        similar_memory = self.find_similar_memory(input_data, threshold=0.85, use_context=False)
        
        if similar_memory:
            # Update existing memory if new success score is better
            if success_score > similar_memory.success_score:
                similar_memory.add_version(success_score, output_data)
                similar_memory.metadata.update(metadata)
                
                # Update embedding in pattern_embeddings list
                idx = self.memories.index(similar_memory)
                self.pattern_embeddings[idx] = similar_memory.input_embedding
                self.faiss_needs_rebuild = True
                
                # Check if it should be added to a cluster
                self._assign_to_cluster(similar_memory)
                
                # Trigger save
                self.save_counter += 1
                if self.save_counter >= self.save_threshold:
                    self._save_to_storage()
                
                logger.debug(f"Updated memory for VNI {self.vni_id} with better score")
            
            return input_hash
        
        # Create new memory entry
        embedding = self.create_embedding(input_data)
        entry = MemoryEntry(
            vni_id=self.vni_id,
            input_data=input_data.copy(),
            output_data=output_data.copy(),
            success_score=success_score,
            timestamp=time.time(),
            input_hash=input_hash,
            input_embedding=embedding,
            metadata=metadata.copy()
        )
        
        self.memories.append(entry)
        self.pattern_embeddings.append(embedding)
        
        # Mark FAISS for rebuild
        self.faiss_needs_rebuild = True
        
        # Assign to cluster
        self._assign_to_cluster(entry)
        
        # Trigger save
        self.save_counter += 1
        if self.save_counter >= self.save_threshold:
            self._save_to_storage()
        
        # Rebuild FAISS if needed and auto-save
        if self.faiss_needs_rebuild and HAS_FAISS and self.auto_save:
            self._build_faiss_index()
        
        logger.debug(f"Stored new memory for VNI {self.vni_id} (total: {len(self.memories)})")
        return input_hash
    
    def _assign_to_cluster(self, memory: MemoryEntry):
        """Assign memory to appropriate cluster"""
        if not self.clusters:
            return
        
        best_cluster = None
        best_similarity = 0
        
        for cluster in self.clusters:
            similarity = self._cosine_similarity(memory.input_embedding, cluster.centroid)
            
            if similarity > best_similarity and similarity > self.cluster_threshold:
                best_similarity = similarity
                best_cluster = cluster
        
        if best_cluster:
            best_cluster.add_memory(memory)
            best_cluster.update_centroid()
    
    def should_self_activate(self, 
                           incoming_data: Dict[str, Any], 
                           activation_threshold: float = 0.8) -> Tuple[bool, Optional[MemoryEntry]]:
        """
        Decide if this VNI should activate spontaneously
        based on similarity to successful past patterns
        """
        similar_memory = self.find_similar_memory(incoming_data, threshold=activation_threshold)
        
        if similar_memory:
            logger.info(f"VNI {self.vni_id} self-activating (success score: {similar_memory.success_score})")
            return True, similar_memory
        
        return False, None
    
    def retrieve_with_context(self, input_data: Dict[str, Any], 
                              threshold: float = 0.6,
                              limit: int = 5) -> List[MemoryEntry]:
        """
        Retrieve multiple similar memories with context awareness
        """
        if not self.memories:
            return []
        
        query_embedding = self.create_embedding(input_data)
        results = []
        
        for i, memory in enumerate(self.memories):
            similarity = self._cosine_similarity(query_embedding, self.pattern_embeddings[i])
            
            if similarity < threshold:
                continue
            
            # Context boosting
            boosted_similarity = similarity
            if self.current_context:
                mem_context = memory.metadata.get('context', [])
                context_overlap = len(set(self.current_context) & set(mem_context))
                if context_overlap > 0:
                    boosted_similarity *= (1.0 + 0.1 * context_overlap)
            
            results.append((boosted_similarity, memory))
        
        # Sort by boosted similarity
        results.sort(key=lambda x: x[0], reverse=True)
        
        # Mark as accessed
        for _, memory in results[:limit]:
            memory.mark_accessed()
        
        return [memory for _, memory in results[:limit]]
    
    def set_context(self, context: List[str]):
        """Set current context for memory retrieval"""
        self.current_context = context.copy()
        logger.debug(f"Memory context set: {context}")
    
    def get_memory_by_hash(self, input_hash: str) -> Optional[MemoryEntry]:
        """Retrieve memory by its input hash"""
        for memory in self.memories:
            if memory.input_hash == input_hash:
                memory.mark_accessed()
                return memory
        return None
    
    def get_cluster_memories(self, cluster_topic: str) -> List[MemoryEntry]:
        """Get all memories from a specific cluster"""
        for cluster in self.clusters:
            if cluster.topic == cluster_topic:
                return cluster.memories.copy()
        return []
    
    def force_save(self):
        """Force immediate save to thumbdrive and rebuild FAISS index"""
        self._save_to_storage()
        
        if self.faiss_needs_rebuild and HAS_FAISS:
            self._build_faiss_index()
        
        logger.info(f"Force saved memory for {self.vni_id} (FAISS rebuilt: {not self.faiss_needs_rebuild})")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        try:
            # Safely get storage path
            storage_path = "unknown"
            if hasattr(self, 'storage'):
                if hasattr(self.storage, 'base_path'):
                    storage_path = str(self.storage.base_path)
                elif hasattr(self.storage, 'path'):
                    storage_path = str(self.storage.path)
                elif hasattr(self.storage, 'storage_path'):
                    storage_path = str(self.storage.storage_path)
    
            # Safely get memory categories
            memory_categories = {}
            if hasattr(self, 'memory_categories'):
                memory_categories = dict(self.memory_categories)
            elif hasattr(self, 'categories'):
                memory_categories = dict(self.categories)

            return {
                "total_memories": len(self.memories) if hasattr(self, 'memories') else 0,
                "memory_by_category": memory_categories,
                "storage_path": storage_path,
                "memory_limit": getattr(self, 'max_memories', 1000),
                "memory_usage_percent": (len(self.memories) / self.max_memories * 100) if hasattr(self, 'memories') and hasattr(self, 'max_memories') and self.max_memories > 0 else 0,
                "avg_retention": self._calculate_avg_retention() if hasattr(self, '_calculate_avg_retention') else 0.0
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {
                "total_memories": len(self.memories) if hasattr(self, 'memories') else 0,
                "memory_by_category": {},
                "storage_path": "error",
                "memory_limit": getattr(self, 'max_memories', 1000),
                "memory_usage_percent": 0,
                "avg_retention": 0.0,
                "error": str(e)
            }
    
    def clear_memory(self, also_clear_storage: bool = False):
        """Clear all memories (optionally from storage too)"""
        self.memories.clear()
        self.pattern_embeddings.clear()
        self.clusters.clear()
        self.faiss_index = None
        self.faiss_needs_rebuild = False
        
        if also_clear_storage:
            try:
                self.storage.delete_memory(self.vni_id)
                logger.info(f"Cleared memory from storage for VNI {self.vni_id}")
            except Exception as e:
                logger.error(f"Failed to clear storage for VNI {self.vni_id}: {e}")
        
        logger.info(f"Cleared all memory for VNI {self.vni_id}")
    
    def export_memories(self, filepath: str):
        """Export memories to JSON file"""
        try:
            data = {
                "vni_id": self.vni_id,
                "export_timestamp": time.time(),
                "memory_count": len(self.memories),
                "memories": [mem.to_storage_dict() for mem in self.memories],
                "clusters": [cluster.to_dict() for cluster in self.clusters],
                "statistics": self.get_memory_stats()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Exported {len(self.memories)} memories to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export memories: {e}")
    
    def import_memories(self, filepath: str, merge: bool = True):
        """Import memories from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            imported_count = 0
            
            for mem_dict in data.get("memories", []):
                try:
                    # Check if memory already exists
                    existing = self.get_memory_by_hash(mem_dict.get("input_hash", ""))
                    
                    if existing and not merge:
                        continue
                    
                    # Convert to MemoryEntry
                    versions_data = mem_dict.get("versions", [])
                    versions = [
                        MemoryVersion(
                            timestamp=v["timestamp"],
                            success_score=v["success_score"],
                            output_data=v["output_data"]
                        ) for v in versions_data
                    ]
                    
                    entry = MemoryEntry(
                        vni_id=self.vni_id,  # Use current VNI ID
                        input_data=mem_dict["input_data"],
                        output_data=mem_dict["output_data"],
                        success_score=mem_dict["success_score"],
                        timestamp=mem_dict["timestamp"],
                        input_hash=mem_dict["input_hash"],
                        input_embedding=np.array(mem_dict["input_embedding"], dtype=np.float32),
                        metadata=mem_dict.get("metadata", {}),
                        versions=versions,
                        access_count=mem_dict.get("access_count", 0),
                        last_accessed=mem_dict.get("last_accessed", mem_dict["timestamp"])
                    )
                    
                    if existing and merge:
                        # Merge versions
                        for version in versions:
                            existing.add_version(version.success_score, version.output_data)
                        existing.metadata.update(entry.metadata)
                    else:
                        self.memories.append(entry)
                        self.pattern_embeddings.append(entry.input_embedding)
                    
                    imported_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to import memory entry: {e}")
            
            # Rebuild indices
            self.faiss_needs_rebuild = True
            self._cluster_memories()
            
            logger.info(f"Imported {imported_count} memories from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to import memories: {e}") 

# For Backward Compatibility
VNIMemory = VniMemory
