# enhanced_vni_classes/utils/vni_config.py
import os
import json
import yaml
import logging
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def to_logging_level(self) -> int:
        """Convert LogLevel enum to logging module level."""
        mapping = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        return mapping[self.value]
    
    @classmethod
    def from_logging_level(cls, level_int: int):
        """Convert logging module level to LogLevel enum."""
        mapping = {
            logging.DEBUG: cls.DEBUG,
            logging.INFO: cls.INFO,
            logging.WARNING: cls.WARNING,
            logging.ERROR: cls.ERROR,
            logging.CRITICAL: cls.CRITICAL
        }
        return mapping.get(level_int, cls.INFO)
    
@dataclass
class VNIConfig:
    """Configuration for VNI instances."""
    
    # Basic settings
    vni_id: str
    name: str = "VNI Assistant"
    description: str = "Virtual Networked Intelligence"
    
    # Performance settings
    max_context_length: int = 3000
    response_timeout: int = 30  # seconds
    cache_enabled: bool = True
    cache_size: int = 100
    
    # Learning settings
    learning_enabled: bool = True
    learning_rate: float = 0.1
    max_learning_examples: int = 1000
    
    # Collaboration settings
    collaboration_enabled: bool = True
    max_collaborators: int = 5
    collaboration_timeout: int = 10
    
    # Search settings
    web_search_enabled: bool = True
    max_search_results: int = 5
    search_cache_expiry: int = 3600
    
    # Logging settings
    log_level: LogLevel = LogLevel.INFO
    log_file: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Domain-specific settings
    domain_settings: Dict[str, Any] = field(default_factory=dict)
    
    # Advanced settings
    enable_attention: bool = True
    enable_classification: bool = True
    enable_generation_styles: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = asdict(self)
        result["log_level"] = self.log_level.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VNIConfig':
        """Create config from dictionary."""
        if "log_level" in data and isinstance(data["log_level"], str):
            data["log_level"] = LogLevel(data["log_level"])
        return cls(**data)

class ConfigManager:
    """Manages configuration for VNI system."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.configs = {}
        
        # Create config directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
    
    def load_config(self, 
                    vni_id: str, 
                    config_file: Optional[str] = None) -> VNIConfig:
        """Load configuration for a VNI."""
        
        if vni_id in self.configs:
            return self.configs[vni_id]
        
        if config_file is None:
            config_file = os.path.join(self.config_dir, f"{vni_id}_config.json")
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                        data = yaml.safe_load(f)
                    else:
                        data = json.load(f)
                
                config = VNIConfig.from_dict(data)
                config.vni_id = vni_id  # Ensure vni_id matches
            else:
                # Create default config
                config = VNIConfig(vni_id=vni_id)
                self.save_config(config, config_file)
            
            self.configs[vni_id] = config
            return config
            
        except Exception as e:
            print(f"Error loading config for {vni_id}: {e}")
            # Return default config
            return VNIConfig(vni_id=vni_id)
    
    def save_config(self, 
                    config: VNIConfig, 
                    config_file: Optional[str] = None):
        """Save configuration to file."""
        
        if config_file is None:
            config_file = os.path.join(self.config_dir, f"{config.vni_id}_config.json")
        
        try:
            data = config.to_dict()
            
            with open(config_file, 'w') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    yaml.dump(data, f, default_flow_style=False)
                else:
                    json.dump(data, f, indent=2)
            
            self.configs[config.vni_id] = config
            
        except Exception as e:
            print(f"Error saving config for {config.vni_id}: {e}")
    
    def get_default_config(self, domain: str) -> Dict[str, Any]:
        """Get default configuration for a domain."""
        
        defaults = {
            "general": {
                "max_context_length": 3000,
                "learning_enabled": True,
                "collaboration_enabled": True,
                "enable_attention": True
            },
            "medical": {
                "max_context_length": 4000,
                "learning_enabled": True,
                "collaboration_enabled": True,
                "enable_attention": True,
                "response_timeout": 45,
                "domain_settings": {
                    "safety_checks": True,
                    "emergency_keywords": ["emergency", "urgent", "911"],
                    "require_disclaimers": True
                }
            },
            "legal": {
                "max_context_length": 3500,
                "learning_enabled": True,
                "collaboration_enabled": True,
                "enable_attention": True,
                "response_timeout": 40,
                "domain_settings": {
                    "jurisdiction_aware": True,
                    "require_disclaimers": True,
                    "emergency_topics": ["arrest", "lawsuit", "eviction"]
                }
            },
            "technical": {
                "max_context_length": 5000,
                "learning_enabled": True,
                "collaboration_enabled": False,
                "enable_attention": True,
                "domain_settings": {
                    "code_formatting": True,
                    "technical_depth": "intermediate"
                }
            }
        }
        
        return defaults.get(domain, defaults["general"])
    
    def update_config(self, 
                      vni_id: str, 
                      updates: Dict[str, Any],
                      save: bool = True) -> VNIConfig:
        """Update configuration for a VNI."""
        
        config = self.load_config(vni_id)
        
        # Update config fields
        for key, value in updates.items():
            if hasattr(config, key):
                if key == "log_level" and isinstance(value, str):
                    value = LogLevel(value)
                setattr(config, key, value)
            elif key in config.domain_settings:
                config.domain_settings[key] = value
        
        if save:
            self.save_config(config)
        
        return config
    
    def list_configs(self) -> Dict[str, VNIConfig]:
        """List all loaded configurations."""
        return self.configs.copy()
    
    def create_domain_config(self, 
                            vni_id: str, 
                            domain: str,
                            custom_settings: Optional[Dict[str, Any]] = None) -> VNIConfig:
        """Create a domain-specific configuration."""
        
        defaults = self.get_default_config(domain)
        config_data = {
            "vni_id": vni_id,
            "name": f"{domain.title()} VNI",
            "description": f"Specialized VNI for {domain} domain"
        }
        
        # Apply defaults
        for key, value in defaults.items():
            if key not in config_data:
                config_data[key] = value
        
        # Apply custom settings
        if custom_settings:
            config_data.update(custom_settings)
        
        config = VNIConfig.from_dict(config_data)
        self.save_config(config)
        
        return config 
