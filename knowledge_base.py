# enhanced_vni_classes/modules/knowledge_base.py
"""
Knowledge base management for VNIs with automatic drive discovery AND auto-mount
"""
import json
import os
import re
import glob
import platform
import shutil
import subprocess
import hashlib
from typing import Dict, List, Any, Optional, Tuple
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


class DriveDiscovery:
    """Handles automatic discovery AND mounting of drives and knowledge files"""
    
    def __init__(self, auto_mount: bool = False):
        self.system = platform.system()
        self.discovered_drives: Dict[str, Dict] = {}
        self.discovered_files: Dict[str, List[str]] = {}
        self.last_discovery = None
        self.auto_mount = auto_mount
        
        # Default search locations
        self.search_paths = self._get_default_search_paths()
        
        # Knowledge file patterns
        self.knowledge_patterns = [
            "knowledge_*.json",
            "*_knowledge*.json",
            "learned_*.json",
            "*.knowledge.json",
            "*.kb.json"
        ]
    
    def _get_default_search_paths(self) -> List[str]:
        """Get OS-specific default search paths"""
        paths = [
            ".",  # Current directory
            "./knowledge",
            "./knowledge_bases",
            os.path.expanduser("~/.vni_knowledge"),
        ]
        
        if self.system == "Windows":
            # Windows paths including all drive letters
            for drive in range(ord('C'), ord('Z')+1):
                drive_letter = f"{chr(drive)}:/"
                paths.append(drive_letter)
            
            # Add common Windows locations
            if 'USERPROFILE' in os.environ:
                user_path = os.environ['USERPROFILE']
                paths.extend([
                    os.path.join(user_path, "Documents", "VNI_Knowledge"),
                    os.path.join(user_path, "Desktop"),
                    os.path.join(user_path, "Downloads")
                ])
                
        elif self.system == "Linux":
            # Linux paths for mounted drives
            paths.extend([
                "/media",  # Auto-mounted drives
                "/mnt",    # Manually mounted drives
                "/run/media",
                "/media/$USER",
                "/home/$USER/Desktop",
                "/home/$USER/Downloads"
            ])
            
        elif self.system == "Darwin":  # macOS
            paths.extend([
                "/Volumes",  # Mounted drives
                "/Users/Shared",
                os.path.expanduser("~/Desktop"),
                os.path.expanduser("~/Documents/VNI_Knowledge")
            ])
        
        # Expand environment variables and user home
        expanded_paths = []
        for path in paths:
            try:
                expanded = os.path.expanduser(os.path.expandvars(path))
                if os.path.exists(expanded):
                    expanded_paths.append(expanded)
            except:
                continue
        
        return list(set(expanded_paths))  # Remove duplicates
    
    def _try_mount_drive(self, drive_path: str) -> bool:
        """Attempt to mount a drive if not already mounted"""
        # Check if already mounted
        if os.path.ismount(drive_path):
            return True
        
        # Only auto-mount if enabled
        if not self.auto_mount:
            return False
        
        # Different mount logic for different systems
        try:
            if self.system == "Windows":
                # WSL: Mount Windows drives
                # Extract drive letter from path like "/mnt/e"
                if drive_path.startswith("/mnt/"):
                    drive_letter = drive_path.split("/")[-1].upper()
                    cmd = ["sudo", "mount", "-t", "drvfs", f"{drive_letter}:", drive_path]
                    
                    # Try without sudo first (might already have permissions)
                    result = subprocess.run(
                        ["mount", "-t", "drvfs", f"{drive_letter}:", drive_path],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode != 0:
                        # Try with sudo
                        result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        logger.info(f"Auto-mounted {drive_letter}: to {drive_path}")
                        return True
                    else:
                        logger.warning(f"Failed to auto-mount {drive_path}: {result.stderr}")
            
            elif self.system in ["Linux", "Darwin"]:
                # Linux/macOS: Check if it's a common mount point
                if any(pattern in drive_path for pattern in ['/media/', '/mnt/', '/Volumes/']):
                    # Check if device exists
                    cmd = ["lsblk", "-o", "MOUNTPOINT,NAME", "-n"]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        for line in lines:
                            if line and drive_path in line:
                                # Already mounted or mountable
                                return True
        
        except Exception as e:
            logger.debug(f"Mount attempt failed for {drive_path}: {e}")
        
        return False
    
    def discover_drives(self, force_remount: bool = False) -> Dict[str, Dict]:
        """Discover all available drives on the system, optionally auto-mounting"""
        self.discovered_drives = {}
        
        # Check all search paths for valid drives/directories
        for path in self.search_paths:
            try:
                if os.path.exists(path):
                    # Check if it's likely a drive/mount point
                    if self._is_drive_or_mount_point(path):
                        # Try to mount if not already mounted
                        is_mounted = os.path.ismount(path)
                        if not is_mounted and self.auto_mount:
                            is_mounted = self._try_mount_drive(path)
                        
                        free_space = self._get_free_space(path) if is_mounted else 0
                        is_removable = self._is_removable_drive(path)
                        
                        self.discovered_drives[path] = {
                            "type": "removable" if is_removable else "fixed",
                            "free_space_gb": free_space,
                            "is_removable": is_removable,
                            "path": path,
                            "exists": True,
                            "mounted": is_mounted,
                            "writable": os.access(path, os.W_OK) if is_mounted else False
                        }
            except Exception as e:
                logger.debug(f"Error checking path {path}: {e}")
        
        logger.info(f"Discovered {len(self.discovered_drives)} drives/mount points")
        return self.discovered_drives
    
    def _is_drive_or_mount_point(self, path: str) -> bool:
        """Check if path is likely a drive or mount point"""
        if self.system == "Windows":
            return len(path) == 3 and path[1:3] == ":/"  # C:/, D:/, etc.
        else:
            # Check common mount point patterns
            mount_patterns = ['/media/', '/mnt/', '/Volumes/', '/run/media/']
            return any(pattern in path for pattern in mount_patterns) or path == "/"
    
    def _is_removable_drive(self, path: str) -> bool:
        """Check if drive is removable (thumbdrive, USB, etc.)"""
        if self.system == "Windows":
            # On Windows, removable drives are usually D:, E:, F:, etc.
            if len(path) == 3 and path[1:3] == ":/":
                drive_letter = path[0]
                return drive_letter not in ['C', 'c']  # Assume C: is not removable
        else:
            # On Linux/macOS, check mount point patterns
            removable_patterns = ['/media/', '/Volumes/', '/run/media/']
            return any(pattern in path for pattern in removable_patterns)
        return False
    
    def _get_free_space(self, path: str) -> float:
        """Get free space in GB"""
        try:
            stat = shutil.disk_usage(path)
            return round(stat.free / (1024**3), 2)  # Convert to GB
        except:
            return 0.0
    
    def discover_knowledge_files(self, domain: str = None) -> Dict[str, List[str]]:
        """
        Discover knowledge files across all drives
        
        Args:
            domain: Optional domain to filter files
        
        Returns:
            Dictionary mapping domains to file lists
        """
        logger.info("Discovering knowledge files...")
        
        # First discover drives (with auto-mount if enabled)
        self.discover_drives()
        
        # Reset discovered files
        self.discovered_files = {}
        
        # Search in all discovered drives plus default paths
        all_search_paths = list(self.discovered_drives.keys()) + self.search_paths
        
        # Remove duplicates
        unique_paths = []
        for path in all_search_paths:
            path_norm = os.path.normpath(path)
            if path_norm not in unique_paths and os.path.exists(path_norm):
                # Only search mounted drives or directories we can access
                drive_info = self.discovered_drives.get(path, {})
                if drive_info.get("mounted", True) or os.path.isdir(path_norm):
                    unique_paths.append(path_norm)
        
        files_found = 0
        for search_path in unique_paths:
            try:
                for pattern in self.knowledge_patterns:
                    full_pattern = os.path.join(search_path, "**", pattern) if search_path != "." else pattern
                    found_files = glob.glob(full_pattern, recursive=True)
                    
                    for filepath in found_files:
                        # Extract domain from filename
                        file_domain = self._extract_domain_from_file(filepath)
                        
                        # If specific domain requested, filter
                        if domain and file_domain != domain and not self._is_domain_match(file_domain, domain):
                            continue
                        
                        if file_domain not in self.discovered_files:
                            self.discovered_files[file_domain] = []
                        
                        if filepath not in self.discovered_files[file_domain]:
                            self.discovered_files[file_domain].append(filepath)
                            files_found += 1
                            
            except Exception as e:
                logger.debug(f"Error searching {search_path}: {e}")
        
        # Sort files by size (largest first) and modification time (newest first)
        for file_domain, files in self.discovered_files.items():
            sorted_files = sorted(files, 
                key=lambda f: (
                    os.path.getsize(f) if os.path.exists(f) else 0,
                    os.path.getmtime(f) if os.path.exists(f) else 0
                ), 
                reverse=True
            )
            self.discovered_files[file_domain] = sorted_files
        
        self.last_discovery = datetime.now()
        
        logger.info(f"Discovery complete: Found {files_found} knowledge files across {len(self.discovered_files)} domains")
        return self.discovered_files
    
    def _extract_domain_from_file(self, filepath: str) -> str:
        """Extract domain from filename"""
        filename = os.path.basename(filepath).lower()
        
        # Check common patterns
        if filename.startswith("knowledge_"):
            # knowledge_medical.json or knowledge_medical_detailed.json
            parts = filename.replace(".json", "").split("_")
            if len(parts) >= 2:
                return parts[1]
        
        elif "_knowledge" in filename:
            # medical_knowledge.json
            domain = filename.split("_knowledge")[0]
            return domain
        
        # Try to read domain from file content
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("domain", "general").lower()
        except:
            return "general"
    
    def _is_domain_match(self, file_domain: str, target_domain: str) -> bool:
        """Check if file domain matches target domain (with aliases)"""
        if file_domain == target_domain:
            return True
        
        # Domain aliases
        domain_aliases = {
            "medical": ["med", "health", "healthcare", "medicine"],
            "legal": ["law", "legal"],
            "technical": ["tech", "technology", "it"],
            "general": ["common", "basic"]
        }
        
        if target_domain in domain_aliases:
            return file_domain in domain_aliases[target_domain]
        
        return False
    
    def get_files_for_domain(self, domain: str, max_files: int = 10) -> List[str]:
        """Get knowledge files for a specific domain"""
        if not self.discovered_files:
            self.discover_knowledge_files(domain)
        
        files = []
        
        # Get exact domain matches
        if domain in self.discovered_files:
            files.extend(self.discovered_files[domain][:max_files])
        
        # Get domain alias matches
        for file_domain, file_list in self.discovered_files.items():
            if self._is_domain_match(file_domain, domain) and file_domain != domain:
                files.extend(file_list[:max_files - len(files)])
                if len(files) >= max_files:
                    break
        
        return files[:max_files]
    
    def print_discovery_summary(self):
        """Print a summary of discovered drives and files"""
        print("\n" + "="*60)
        print("KNOWLEDGE DISCOVERY SUMMARY")
        print("="*60)
        
        # Print drives
        print(f"\n📁 DISCOVERED DRIVES ({len(self.discovered_drives)}):")
        for drive_path, drive_info in self.discovered_drives.items():
            removable = " (Removable)" if drive_info.get("is_removable", False) else ""
            mounted = " ✅" if drive_info.get("mounted", False) else " ❌"
            free_gb = drive_info.get("free_space_gb", 0)
            print(f"  {drive_path} - {free_gb} GB free{removable}{mounted}")
        
        # Print files
        print(f"\n📚 KNOWLEDGE FILES ({sum(len(f) for f in self.discovered_files.values())}):")
        for domain, files in sorted(self.discovered_files.items()):
            print(f"  {domain.upper()}: {len(files)} files")
            
            # Show top files
            for i, filepath in enumerate(files[:3]):
                filename = os.path.basename(filepath)
                size_mb = os.path.getsize(filepath) / (1024 * 1024) if os.path.exists(filepath) else 0
                print(f"    {i+1}. {filename} ({size_mb:.1f} MB)")
            
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more")
        
        print("="*60)
    
    def check_disk_space(self, min_space_gb: float = 1.0) -> Dict[str, Any]:
        """Check disk space on all mounted drives"""
        space_info = {}
        
        for drive_path, drive_info in self.discovered_drives.items():
            if drive_info.get("mounted", False):
                free_gb = drive_info.get("free_space_gb", 0)
                is_removable = drive_info.get("is_removable", False)
                
                space_info[drive_path] = {
                    "free_gb": free_gb,
                    "is_low": free_gb < min_space_gb,
                    "is_removable": is_removable,
                    "suggest_migration": is_removable and not free_gb < min_space_gb
                }
        
        return {
            "all_drives": space_info,
            "has_low_space": any(info["is_low"] for info in space_info.values()),
            "suggested_drives": [
                path for path, info in space_info.items()
                if info.get("suggest_migration", False)
            ]
        }


class KnowledgeBase:
    """Knowledge base for a VNI with automatic drive discovery and auto-mount"""
    
    def __init__(self, domain: str = "general", embedding_model: str = "all-MiniLM-L6-v2",
                 auto_mount: bool = False, max_retries: int = 2):
        self.domain = domain
        self.concepts: Dict[str, KnowledgeConcept] = {}
        self.patterns: List[KnowledgePattern] = []
        self.learned_responses: Dict[str, str] = {}
        self.entries: Dict[str, KnowledgeEntry] = {}  # Added for backward compatibility
        
        # Embedding model for semantic search
        self.embedding_model = None
        self.embedding_model_name = embedding_model
        self.concept_embeddings: Dict[str, np.ndarray] = {}
        
        # Drive discovery with auto-mount option
        self.drive_discovery = DriveDiscovery(auto_mount=auto_mount)
        self.max_retries = max_retries
        
        # Auto-load knowledge on initialization
        self._auto_load_knowledge_with_retry()
        
        logger.info(f"Initialized KnowledgeBase for domain: {domain} (auto-mount: {auto_mount})")
    
    def _auto_load_knowledge_with_retry(self):
        """Automatically load knowledge from discovered sources with retry logic"""
        logger.info(f"Auto-discovering knowledge for domain: {self.domain}")
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Attempt {attempt + 1}/{self.max_retries + 1}")
                
                # Discover and load knowledge files
                files = self.drive_discovery.get_files_for_domain(self.domain, max_files=5)
                
                if files:
                    logger.info(f"Found {len(files)} knowledge files for domain '{self.domain}':")
                    for filepath in files:
                        logger.info(f"  - {os.path.basename(filepath)}")
                    
                    # Load the files
                    load_result = self.load_multiple(files)
                    
                    if load_result.get("concepts_loaded", 0) > 0 or load_result.get("patterns_loaded", 0) > 0:
                        logger.info(f"✅ Successfully loaded {load_result.get('concepts_loaded', 0)} concepts and {load_result.get('patterns_loaded', 0)} patterns")
                        return
                    else:
                        logger.warning("No concepts or patterns loaded from discovered files")
                
                # If no files found or no concepts loaded
                if attempt < self.max_retries:
                    logger.info("Retrying with forced re-scan...")
                    # Force rescan on next attempt
                    self.drive_discovery.last_discovery = None
                else:
                    # Last attempt failed, use defaults
                    logger.warning("All attempts failed, using default knowledge")
                    self._add_default_knowledge()
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries:
                    logger.warning("Using default knowledge due to errors")
                    self._add_default_knowledge()
    
    def _add_default_knowledge(self):
        """Add default knowledge if no files are found"""
        logger.info("Adding default knowledge")
        
        default_concepts = self._get_default_concepts_for_domain()
        
        for concept_name, concept_data in default_concepts.items():
            self.add_concept(concept_name, concept_data)
    
    def _get_default_concepts_for_domain(self) -> Dict[str, Dict[str, Any]]:
        """Get default concepts based on domain"""
        if self.domain == "medical":
            return {
                "headache": {
                    "description": "A headache is pain or discomfort in the head, scalp, or neck area. Common types include tension headaches, migraines, and cluster headaches.",
                    "metadata": {"type": "symptom", "severity": "common"}
                },
                "fever": {
                    "description": "Fever is an elevated body temperature, often a sign that your body is fighting an infection. Normal body temperature is around 98.6°F (37°C).",
                    "metadata": {"type": "symptom", "severity": "common"}
                },
                "doctor": {
                    "description": "A medical professional who diagnoses and treats illnesses and injuries. You should see a doctor for serious or persistent symptoms.",
                    "metadata": {"type": "professional", "category": "healthcare"}
                }
            }
        elif self.domain == "legal":
            return {
                "contract": {
                    "description": "A legally binding agreement between two or more parties that creates obligations enforceable by law.",
                    "metadata": {"type": "legal_document", "category": "agreements"}
                }
            }
        else:  # general
            return {
                "help": {
                    "description": "I'm an AI assistant that can help with various topics. You can ask me about medical, legal, technical, or general information.",
                    "metadata": {"type": "introduction", "category": "general"}
                }
            }
    
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
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Load concepts
                    if 'concepts' in data:
                        for concept_name, concept_data in data['concepts'].items():
                            self.add_concept(concept_name, concept_data)
                            total_concepts += 1
                    
                    # Load patterns
                    if 'patterns' in data:
                        for pattern_data in data['patterns']:
                            self.add_pattern(
                                pattern_data['pattern'],
                                pattern_data.get('response_template', '')
                            )
                            total_patterns += 1
                    
                    # Load learned responses
                    if 'learned_responses' in data:
                        self.learned_responses.update(data['learned_responses'])
                    
                    loaded_files.append(os.path.basename(filepath))
                    logger.info(f"Loaded knowledge from: {filepath}")
                    
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
        
        return {
            "loaded_files": loaded_files,
            "concepts_loaded": total_concepts,
            "patterns_loaded": total_patterns,
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
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
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
    
    def refresh_knowledge(self, force_remount: bool = False):
        """Refresh knowledge by rediscovering and reloading files"""
        logger.info("Refreshing knowledge from all sources...")
        
        # Clear current knowledge
        self.concepts.clear()
        self.patterns.clear()
        self.concept_embeddings.clear()
        self.learned_responses.clear()
        
        # Rediscover and load
        self._auto_load_knowledge_with_retry()
        
        logger.info("Knowledge refresh complete")
    
    def discover_and_print_summary(self):
        """Discover drives and files, then print summary"""
        self.drive_discovery.discover_knowledge_files(self.domain)
        self.drive_discovery.print_discovery_summary()
    
    def check_disk_space(self, min_space_gb: float = 1.0) -> Dict[str, Any]:
        """Check disk space and suggest thumbdrive if hard drive is full"""
        return self.drive_discovery.check_disk_space(min_space_gb)
    
    def migrate_to_thumbdrive(self, target_drive: str = None) -> Dict[str, Any]:
        """
        Save current knowledge to thumbdrive when hard drive is low on space
        
        Returns:
            Dictionary with migration status and details
        """
        # Check disk space
        space_info = self.check_disk_space(min_space_gb=0.5)
        
        if not space_info["has_low_space"]:
            return {
                "status": "no_action",
                "reason": "sufficient_disk_space",
                "space_info": space_info
            }
        
        # Find a suitable thumbdrive
        suggested_drives = space_info.get("suggested_drives", [])
        
        if not suggested_drives and not target_drive:
            return {
                "status": "failed",
                "reason": "no_thumbdrive_found",
                "space_info": space_info
            }
        
        # Use target drive or first suggested drive
        target = target_drive or suggested_drives[0]
        
        # Create target directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = os.path.join(target, f"BabyBIONN_Knowledge_{timestamp}")
        
        try:
            os.makedirs(target_dir, exist_ok=True)
            
            # Save current knowledge to thumbdrive
            backup_file = os.path.join(target_dir, f"knowledge_{self.domain}_backup.json")
            if self.save(backup_file):
                return {
                    "status": "success",
                    "reason": "migrated_to_thumbdrive",
                    "target_drive": target,
                    "target_directory": target_dir,
                    "backup_file": backup_file,
                    "space_info": space_info
                }
            else:
                return {
                    "status": "failed",
                    "reason": "failed_to_save_backup",
                    "target_drive": target,
                    "space_info": space_info
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "reason": f"exception: {str(e)}",
                "target_drive": target,
                "space_info": space_info
            } 
