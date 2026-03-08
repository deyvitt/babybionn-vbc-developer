# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

# neuron/reinforcement_learning/vni_core.py
"""
Minimal VNI core implementation for RL integration
Provides the necessary classes and interfaces for the reinforcement learning system
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum

class VNIType(Enum):
    """Types of Virtual Neural Interfaces"""
    GENERAL = "general"
    MEDICAL = "medical"
    LEGAL = "legal"
    TECHNICAL = "technical"
    HEALTH = "health"
    COMPLIANCE = "compliance"
    SOFTWARE = "software"
    BIO = "bio"
    ETHICS = "ethics"
    MATH = "math"

@dataclass
class VNIStimulus:
    """Stimulus input for VNIs"""
    content: Any
    stimulus_type: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0:
            import time
            self.timestamp = time.time()

@dataclass
class VNIResponse:
    """Response output from VNIs"""
    content: Any
    confidence: float
    response_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0

class VNIManager:
    """
    Minimal VNI Manager implementation
    Manages multiple Virtual Neural Interfaces and routes stimuli appropriately
    """
    
    def __init__(self):
        self.vnis = {}
        self.response_history = []
        
    def register_vni(self, vni_id: str, vni_type: VNIType, processor_func=None):
        """Register a VNI with the manager"""
        self.vnis[vni_id] = {
            'type': vni_type,
            'processor': processor_func or self._default_processor,
            'activation_count': 0
        }
    
    def process_stimulus(self, vni_id: str, stimulus: VNIStimulus) -> VNIResponse:
        """Process a stimulus through a specific VNI"""
        import time
        
        if vni_id not in self.vnis:
            raise ValueError(f"VNI {vni_id} not registered")
        
        start_time = time.time()
        vni = self.vnis[vni_id]
        vni['activation_count'] += 1
        
        # Process the stimulus
        response_content = vni['processor'](stimulus, vni_id)
        
        # Calculate confidence based on various factors
        confidence = self._calculate_confidence(stimulus, response_content, vni_id)
        
        processing_time = time.time() - start_time
        
        response = VNIResponse(
            content=response_content,
            confidence=confidence,
            response_type=vni['type'].value,
            processing_time=processing_time
        )
        
        # Store in history
        self.response_history.append({
            'vni_id': vni_id,
            'stimulus': stimulus,
            'response': response,
            'timestamp': start_time
        })
        
        return response
    
    def _default_processor(self, stimulus: VNIStimulus, vni_id: str) -> Any:
        """Default VNI processing function"""
        vni_type = self.vnis[vni_id]['type']
        
        # Simple type-based processing
        if vni_type == VNIType.MEDICAL:
            return f"Medical analysis of: {stimulus.content}"
        elif vni_type == VNIType.LEGAL:
            return f"Legal review of: {stimulus.content}"
        elif vni_type == VNIType.TECHNICAL:
            return f"Technical assessment of: {stimulus.content}"
        else:
            return f"Processed by {vni_id}: {stimulus.content}"
    
    def _calculate_confidence(self, stimulus: VNIStimulus, response_content: Any, vni_id: str) -> float:
        """Calculate confidence score for VNI response"""
        base_confidence = 0.7
        
        # Adjust based on stimulus type matching VNI type
        vni_type = self.vnis[vni_id]['type'].value
        if vni_type in stimulus.stimulus_type.lower():
            base_confidence += 0.2
        
        # Adjust based on content length (longer responses might be more thoughtful)
        content_str = str(response_content)
        length_factor = min(len(content_str) / 500, 0.3)  # Cap at 0.3 bonus
        
        # Adjust based on VNI activation history (experienced VNIs get confidence boost)
        experience_factor = min(self.vnis[vni_id]['activation_count'] / 100, 0.2)
        
        return min(base_confidence + length_factor + experience_factor, 1.0)
    
    def get_vni_stats(self) -> Dict[str, Any]:
        """Get statistics for all VNIs"""
        return {
            vni_id: {
                'type': info['type'].value,
                'activations': info['activation_count'],
                'average_confidence': self._get_average_confidence(vni_id)
            }
            for vni_id, info in self.vnis.items()
        }
    
    def _get_average_confidence(self, vni_id: str) -> float:
        """Calculate average confidence for a VNI"""
        vni_responses = [r for r in self.response_history if r['vni_id'] == vni_id]
        if not vni_responses:
            return 0.0
        return sum(r['response'].confidence for r in vni_responses) / len(vni_responses)

# Example usage and VNI setup
def create_sample_vni_manager() -> VNIManager:
    """Create a VNI manager with sample VNIs for testing"""
    manager = VNIManager()
    
    # Register various types of VNIs
    manager.register_vni("medical_0", VNIType.MEDICAL)
    manager.register_vni("legal_0", VNIType.LEGAL)  
    manager.register_vni("general_0", VNIType.GENERAL)
    
    return manager

if __name__ == "__main__":
    # Test the VNI manager
    manager = create_sample_vni_manager()
    
    # Create a test stimulus
    stimulus = VNIStimulus(
        content="Patient with fever needs medical attention and legal consent forms",
        stimulus_type="medical_legal",
        metadata={"urgency": "medium"}
    )
    
    # Process through different VNIs
    responses = {}
    for vni_id in ["medical_0", "legal_0", "general_0"]:
        try:
            response = manager.process_stimulus(vni_id, stimulus)
            responses[vni_id] = response
            print(f"{vni_id}: {response.content} (confidence: {response.confidence:.2f})")
        except ValueError as e:
            print(f"Error with {vni_id}: {e}")
    
    print(f"\nVNI Statistics: {manager.get_vni_stats()}") 
