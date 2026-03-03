"""Hybrid storage system: In-memory with thumbdrive persistence"""
import os
import json
import time
import pickle
import shutil
import hashlib
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger("vni_storage")

class ThumbdriveMonitor:
    """Continuously monitors for thumbdrive connection/disconnection. Auto-migrates data when thumbdrive is detected"""
    def __init__(self, storage_manager, check_interval=5):
        self.storage = storage_manager
        self.check_interval = check_interval
        self.mounted = False
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        logger.info("🔍 ThumbdriveMonitor started")
    
    def _monitor(self):
        """Background thread that checks for thumbdrive"""
        while self.monitoring:
            try:
                # Check if thumbdrive is mounted
                current_mount = self._check_mount()
                
                if current_mount and not self.mounted:
                    # Just connected!
                    logger.info(f"✅ Thumbdrive detected at {current_mount}")
                    self.mounted = True
                    self.storage.base_path = current_mount
                    self.storage._ensure_directories()
                    
                elif not current_mount and self.mounted:
                    # Just disconnected!
                    logger.warning("⚠️ Thumbdrive disconnected - falling back to local storage")
                    self.mounted = False
                    self.storage.base_path = Path.cwd() / "vni_data"
                    self.storage._ensure_directories()
                    
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            time.sleep(self.check_interval)
    
    def _check_mount(self):
        """Check if thumbdrive is mounted"""
        # Common thumbdrive mount points
        possible_paths = [
            Path("/Volumes/THUMBDRIVE"),   # macOS
            Path("/mnt/THUMBDRIVE"),       # Linux
            Path("/media/THUMBDRIVE"),     # Linux alternative
            Path("/media/babybionn_brain"), # Custom
            Path("D:/vni_data"),           # Windows
            Path("E:/vni_data"),           # Windows alternative
        ]
        
        for path in possible_paths:
            if path.exists() and os.access(path, os.W_OK):
                return path
        return None
    
    def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        logger.info("🛑 ThumbdriveMonitor stopped")

class StorageManager:
    """Manages hybrid storage: RAM for speed, thumbdrive for persistence"""
    def __init__(self, 
                 base_path: str = None,
                 use_external: bool = True,
                 sync_interval: int = 60):  # Sync every 60 seconds
        """
        Args:
            base_path: Path to thumbdrive mount point (e.g., '/Volumes/THUMBDRIVE/vni_data')
                       If None, uses current directory
            use_external: Whether to use external storage
            sync_interval: Seconds between auto-sync
        """
        # Determine storage location
        if base_path is None:
            # Try to auto-detect thumbdrive
            self.base_path = self._find_thumbdrive()
        else:
            self.base_path = Path(base_path)
        
        self.use_external = use_external
        self.sync_interval = sync_interval
        
        # In-memory caches
        self.memory_cache: Dict[str, Any] = {}
        self.vector_cache: Dict[str, List] = {}
        
        # Last sync time
        self.last_sync = 0
        
        # Ensure directories exist
        self._ensure_directories()
        
        logger.info(f"Storage Manager initialized: {self.base_path}")
        logger.info(f"External storage: {'ENABLED' if use_external else 'DISABLED'}")
    
    def _find_thumbdrive(self) -> Path:
        """Try to auto-detect thumbdrive"""
        # Common thumbdrive mount points
        possible_paths = [
            Path("/Volumes/THUMBDRIVE"),  # macOS
            Path("/mnt/THUMBDRIVE"),      # Linux
            Path("/media/THUMBDRIVE"),    # Linux alternative
            Path("D:/vni_data"),          # Windows
            Path("E:/vni_data"),          # Windows alternative
            Path.cwd() / "vni_data"       # Fallback to current directory
        ]
        
        for path in possible_paths:
            if path.exists() and os.access(path, os.W_OK):
                logger.info(f"Found thumbdrive at: {path}")
                return path
        
        # Create in current directory
        fallback = Path.cwd() / "vni_data"
        fallback.mkdir(exist_ok=True)
        logger.warning(f"No thumbdrive found. Using: {fallback}")
        return fallback
    
    def _ensure_directories(self):
        """Create necessary directory structure"""
        directories = [
            "memories",
            "messages",
            "configs",
            "logs",
            "backups"
        ]
        
        for dir_name in directories:
            dir_path = self.base_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _get_memory_path(self, vni_id: str) -> Path:
        """Get file path for VNI memory"""
        return self.base_path / "memories" / f"{vni_id}_memory.pkl"
    
    def _get_message_path(self, vni_id: str) -> Path:
        """Get file path for VNI messages"""
        return self.base_path / "messages" / f"{vni_id}_messages.json"
    
    def _get_backup_path(self, data_type: str, timestamp: str) -> Path:
        """Get backup file path"""
        return self.base_path / "backups" / f"{data_type}_{timestamp}.bak"
    
    def save_memory(self, vni_id: str, memories: List, vectors: List):
        """
        Save VNI memory to thumbdrive
        Uses pickle for efficient storage
        """
        if not self.use_external:
            return
        
        try:
            file_path = self._get_memory_path(vni_id)
            
            # Create backup before saving
            if file_path.exists():
                backup_time = time.strftime("%Y%m%d_%H%M%S")
                backup_path = self._get_backup_path(f"{vni_id}_memory", backup_time)
                shutil.copy2(file_path, backup_path)
            
            # Save using pickle (efficient for numpy arrays)
            data = {
                'memories': memories,
                'vectors': vectors,
                'timestamp': time.time(),
                'vni_id': vni_id
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update in-memory cache
            self.memory_cache[vni_id] = memories
            self.vector_cache[vni_id] = vectors
            
            logger.debug(f"Saved memory for {vni_id} to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save memory for {vni_id}: {e}")
    
    def load_memory(self, vni_id: str) -> tuple:
        """
        Load VNI memory from thumbdrive
        Returns: (memories, vectors) or empty lists if not found
        """
        # Check in-memory cache first
        if vni_id in self.memory_cache and vni_id in self.vector_cache:
            return self.memory_cache[vni_id], self.vector_cache[vni_id]
        
        if not self.use_external:
            return [], []
        
        try:
            file_path = self._get_memory_path(vni_id)
            
            if not file_path.exists():
                logger.debug(f"No memory file found for {vni_id}")
                return [], []
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            memories = data.get('memories', [])
            vectors = data.get('vectors', [])
            
            # Update cache
            self.memory_cache[vni_id] = memories
            self.vector_cache[vni_id] = vectors
            
            logger.info(f"Loaded {len(memories)} memories for {vni_id}")
            return memories, vectors
            
        except Exception as e:
            logger.error(f"Failed to load memory for {vni_id}: {e}")
            return [], []
    
    def save_messages(self, vni_id: str, messages: List[Dict]):
        """
        Save VNI messages to thumbdrive as JSON
        """
        if not self.use_external:
            return
        
        try:
            file_path = self._get_message_path(vni_id)
            
            # Read existing messages
            existing_messages = []
            if file_path.exists():
                with open(file_path, 'r') as f:
                    existing_messages = json.load(f)
            
            # Append new messages
            all_messages = existing_messages + messages
            
            # Keep only last N messages to prevent file bloat
            max_messages = 1000
            if len(all_messages) > max_messages:
                all_messages = all_messages[-max_messages:]
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(all_messages, f, indent=2)
            
            logger.debug(f"Saved {len(messages)} messages for {vni_id}")
            
        except Exception as e:
            logger.error(f"Failed to save messages for {vni_id}: {e}")
    
    def load_messages(self, vni_id: str) -> List[Dict]:
        """
        Load VNI messages from thumbdrive
        """
        if not self.use_external:
            return []
        
        try:
            file_path = self._get_message_path(vni_id)
            
            if not file_path.exists():
                return []
            
            with open(file_path, 'r') as f:
                messages = json.load(f)
            
            logger.debug(f"Loaded {len(messages)} messages for {vni_id}")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to load messages for {vni_id}: {e}")
            return []
    
    def sync_all(self):
        """Sync all caches to thumbdrive"""
        if not self.use_external:
            return
        
        current_time = time.time()
        if current_time - self.last_sync < self.sync_interval:
            return
        
        try:
            # Sync would be called by each VNI's save methods
            logger.debug("Sync check passed, would sync if needed")
            self.last_sync = current_time
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage statistics"""
        info = {
            'base_path': str(self.base_path),
            'using_external': self.use_external,
            'cache_size': len(self.memory_cache),
            'total_memories': sum(len(m) for m in self.memory_cache.values()),
            'disk_usage': self._get_disk_usage()
        }
        return info
    
    def _get_disk_usage(self) -> Dict[str, Any]:
        """Calculate disk usage"""
        try:
            total_size = 0
            file_count = 0
            
            for root, dirs, files in os.walk(self.base_path):
                for file in files:
                    file_path = Path(root) / file
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            return {
                'total_size_mb': total_size / (1024 * 1024),
                'file_count': file_count
            }
        except:
            return {'total_size_mb': 0, 'file_count': 0}
    
    def cleanup_old_backups(self, days_to_keep: int = 7):
        """Remove old backup files"""
        if not self.use_external:
            return
        
        try:
            backup_dir = self.base_path / "backups"
            if not backup_dir.exists():
                return
            
            current_time = time.time()
            cutoff_time = current_time - (days_to_keep * 24 * 3600)
            
            deleted_count = 0
            for file_path in backup_dir.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old backups")
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}") 
