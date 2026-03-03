from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class VNIOpinion:
    vni_id: str
    domain: str
    confidence: float
    opinion_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReasoningOutput:
    query: str
    primary_domain: str
    vni_opinions: List[VNIOpinion]
    consensus_level: str          # "strong", "moderate", "weak", "mixed", "none"
    consensus_summary: str
    memory_snippets: List[str] = field(default_factory=list)
    overall_confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict) 
