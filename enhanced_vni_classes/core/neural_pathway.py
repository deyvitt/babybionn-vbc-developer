# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

# enhanced_vni_classes/core/neural_pathway.py
"""
Neural pathway between VNIs
"""
from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime


@dataclass
class NeuralPathway:
    """
    Represents a neural pathway/connection between two VNIs
    """
    source_id: str
    target_id: str
    pathway_type: str  # "bidirectional", "feedforward", "feedback"
    strength: float = 0.1
    created_at: datetime = None
    last_activated: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_activated is None:
            self.last_activated = self.created_at
    
    def update_strength(self, delta: float) -> float:
        """
        Update pathway strength
        """
        self.strength = max(0.0, min(1.0, self.strength + delta))
        self.last_activated = datetime.now()
        return self.strength
    
    def activate(self, signal_strength: float = 1.0) -> float:
        """
        Activate pathway with signal
        """
        self.update_strength(0.01 * signal_strength)
        return self.strength * signal_strength
    
    def decay(self, decay_rate: float = 0.01) -> float:
        """
        Apply decay to pathway strength
        """
        self.strength = max(0.0, self.strength - decay_rate)
        return self.strength
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "pathway_type": self.pathway_type,
            "strength": self.strength,
            "created_at": self.created_at.isoformat(),
            "last_activated": self.last_activated.isoformat()
        } 
