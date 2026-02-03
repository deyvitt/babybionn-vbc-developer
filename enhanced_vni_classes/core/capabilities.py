# enhanced_vni_classes/core/capabilities.py
"""
VNI capabilities definitions
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set
from enum import Enum


class VNIType(Enum):
    """Types of VNIs"""
    BASE = "base"
    SPECIALIZED = "specialized"
    DYNAMIC = "dynamic"
    HYBRID = "hybrid"


@dataclass
class VNICapabilities:
    # Core fields (required)
    domains: List[str] = field(default_factory=lambda: ["general"])
    
    # Basic capabilities
    can_search: bool = True
    can_learn: bool = True
    can_collaborate: bool = True
    max_context_length: int = 2000
    special_abilities: List[str] = field(default_factory=list)
    
    # Additional fields for backward compatibility
    domain: str = "general"  # For single domain
    vni_type: str = "specialized"
    learning_enabled: bool = True
    collaboration_enabled: bool = True
    supported_modalities: List[str] = field(default_factory=lambda: ["text"])
    domain_knowledge: List[str] = field(default_factory=list)
    abstraction_levels: List[str] = field(default_factory=list)
    processing_speed: float = 1.0
    collaboration_score: float = 0.7
    specializations: Set[str] = field(default_factory=set)
    generation_enabled: bool = False
    web_search_enabled: bool = False
    memory_enabled: bool = True

    def __post_init__(self):
        # Ensure domains list is populated
        if not self.domains and self.domain:
            self.domains = [self.domain]
        elif self.domains and not self.domain:
            self.domain = self.domains[0]

        # Convert vni_type to string if it's an Enum
        if hasattr(self.vni_type, 'value'):
            self.vni_type = self.vni_type.value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "domains": self.domains,
            "can_search": self.can_search,
            "can_learn": self.can_learn,
            "can_collaborate": self.can_collaborate,
            "max_context_length": self.max_context_length,
            "special_abilities": self.special_abilities,
            "domain": self.domain,
            "vni_type": self.vni_type,  # Now it's a string
            "learning_enabled": self.learning_enabled,
            "collaboration_enabled": self.collaboration_enabled,
            "supported_modalities": self.supported_modalities,
            "domain_knowledge": self.domain_knowledge,
            "abstraction_levels": self.abstraction_levels,
            "processing_speed": self.processing_speed,
            "collaboration_score": self.collaboration_score,
            "specializations": list(self.specializations),
            "generation_enabled": self.generation_enabled,
            "web_search_enabled": self.web_search_enabled,
            "memory_enabled": self.memory_enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VNICapabilities':
        """Create from dictionary"""
        # Extract all fields that match the dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
