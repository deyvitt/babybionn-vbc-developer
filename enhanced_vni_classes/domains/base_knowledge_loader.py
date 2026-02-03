# enhanced_vni_classes/domains/base_knowledge_loader.py
import os
import json
from typing import Dict, Any, List, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)

class BaseKnowledgeLoader:
    """Base class for loading knowledge from JSON files."""
    
    def load_domain_knowledge(self, domain: str) -> bool:
        """
        Load knowledge for a specific domain.
        
        Args:
            domain: Domain name (medical, legal, general, technical)
            
        Returns:
            True if knowledge was loaded successfully
        """
        # Determine the right file pattern for this domain
        file_patterns = self._get_domain_file_patterns(domain)
        
        # Look for files in root and knowledge_bases directory
        search_paths = [".", "knowledge_bases"]
        
        loaded_files = []
        
        for search_path in search_paths:
            for file_pattern in file_patterns:
                # Check for exact filename
                file_path = os.path.join(search_path, file_pattern)
                if os.path.exists(file_path):
                    if self._load_knowledge_file(file_path, domain):
                        loaded_files.append(file_path)
                
                # Also check for variations
                elif "_001.json" in file_pattern:
                    # Try without the _001
                    alt_pattern = file_pattern.replace("_001.json", ".json")
                    alt_path = os.path.join(search_path, alt_pattern)
                    if os.path.exists(alt_path):
                        if self._load_knowledge_file(alt_path, domain):
                            loaded_files.append(alt_path)
        
        if loaded_files:
            logger.info(f"Loaded {len(loaded_files)} knowledge files for {domain}: {loaded_files}")
            return True
        else:
            logger.warning(f"No knowledge files found for {domain} domain")
            return False
    
    def _get_domain_file_patterns(self, domain: str) -> List[str]:
        """Get file patterns for a specific domain."""
        patterns = {
            "medical": [
                "knowledge_medical_med_001.json",
                "knowledge_medical_medical_001.json",
                "knowledge_medical_medical_0.json",
                "medical_knowledge.json"
            ],
            "legal": [
                "knowledge_legal_legal_001.json",
                "knowledge_legal_legal_0.json",
                "legal_knowledge.json"
            ],
            "general": [
                "knowledge_general_gen_001.json",
                "knowledge_general_general_001.json",
                "knowledge_general_general_0.json",
                "general_knowledge.json"
            ],
            "technical": [
                "knowledge_technical_technical_001.json",
                "knowledge_technical_technical_0.json",
                "technical_knowledge.json"
            ]
        }
        
        return patterns.get(domain, [f"knowledge_{domain}_{domain}_001.json"])
    
    def _load_knowledge_file(self, file_path: str, domain: str) -> bool:
        """
        Load knowledge from a specific JSON file.
        
        Args:
            file_path: Path to JSON file
            domain: Domain name
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
            
            # Check if we have a domain-specific attribute to merge into
            domain_knowledge_attr = f"{domain}_knowledge"
            if hasattr(self, domain_knowledge_attr):
                self._merge_knowledge_data(getattr(self, domain_knowledge_attr), knowledge_data)
            elif hasattr(self, 'knowledge'):
                # Generic knowledge attribute
                self._merge_knowledge_data(self.knowledge, knowledge_data)
            
            logger.debug(f"Loaded knowledge from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return False
    
    def _merge_knowledge_data(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Merge source knowledge data into target dictionary."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Merge dictionaries
                target[key].update(value)
            elif key in target and isinstance(target[key], list) and isinstance(value, list):
                # Merge lists (avoid duplicates)
                for item in value:
                    if item not in target[key]:
                        target[key].append(item)
            else:
                # Replace or add new key
                target[key] = value 
