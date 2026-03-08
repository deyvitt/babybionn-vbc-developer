# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

# safety_check.py
"""
Safety and Ethical Enforcement Module for BabyBIONN
Constitutional safety rules that cannot be overridden by learning
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger("BabyBIONN-Safety")

@dataclass
class SafetyCheckResult:
    """Result of safety validation"""
    safe: bool
    category: str = "safe"
    reason: str = ""
    corrected_response: str = ""
    confidence: float = 1.0

class ConstitutionalSafetyVNI:
    """Hard-coded safety VNI that cannot learn unsafe patterns"""
    
    def __init__(self, instance_id: str = "safety_0"):
        self.instance_id = instance_id
        self.vni_type = "safety"
        self.safety_rules = self._load_constitutional_rules()
        self.unsafe_patterns = self._load_unsafe_patterns()
        self.safety_log = []
        
    def _load_constitutional_rules(self) -> Dict:
        """Immutable safety rules - cannot be modified by learning"""
        return {
            "ethical_principles": [
                "Always prioritize human wellbeing and safety",
                "Never provide harmful, dangerous, or illegal information",
                "Respect privacy and confidentiality",
                "Avoid bias and treat all people equally",
                "Be honest about limitations and capabilities"
            ],
            "content_restrictions": [
                "No hate speech, harassment, or discrimination",
                "No instructions for violence, self-harm, or illegal activities", 
                "No sexually explicit content",
                "No personal information or privacy violations",
                "No medical advice beyond general information",
                "No legal advice beyond general information"
            ],
            "response_requirements": [
                "Always be helpful, respectful, and honest",
                "Decline inappropriate requests politely but firmly",
                "Provide clear disclaimers for sensitive topics",
                "Redirect to appropriate resources when needed"
            ]
        }
    
    def _load_unsafe_patterns(self) -> Dict:
        """Patterns that automatically trigger safety responses"""
        return {
            "violence_keywords": ["kill", "hurt", "harm", "attack", "weapon", "bomb", "murder", "assault"],
            "hate_speech": ["hate", "racist", "sexist", "discriminate", "superior", "inferior", "racial"],
            "illegal_activities": ["illegal", "steal", "cheat", "hack", "fraud", "scam", "pirate"],
            "dangerous_advice": ["suicide", "self-harm", "dangerous", "unsafe", "poison", "overdose"],
            "explicit_content": ["porn", "explicit", "adult", "nude", "sexual", "xxx", "nsfw"],
            "medical_emergency": ["heart attack", "stroke", "severe bleeding", "unconscious", "emergency"],
            "legal_emergency": ["arrest", "lawsuit", "court case", "legal emergency", "attorney needed"]
        }
    
    def validate_query(self, query: str) -> SafetyCheckResult:
        """Check if query violates safety rules"""
        query_lower = query.lower()
        
        for category, patterns in self.unsafe_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return SafetyCheckResult(
                        safe=False,
                        category=category,
                        reason=f"Query contains {category} content",
                        corrected_response=self._generate_safety_response(category)
                    )
        
        return SafetyCheckResult(safe=True, reason="Query passed safety check")
    
    def validate_response(self, response: str, source_vni: str) -> SafetyCheckResult:
        """Validate responses from other VNIs before they reach the user"""
        response_lower = response.lower()
        
        for category, patterns in self.unsafe_patterns.items():
            for pattern in patterns:
                if pattern in response_lower:
                    return SafetyCheckResult(
                        safe=False,
                        category=category, 
                        reason=f"Response from {source_vni} contains {category} content",
                        corrected_response=self._generate_safety_response(category)
                    )
        
        return SafetyCheckResult(safe=True, reason="Response passed safety check")
    
    def _generate_safety_response(self, category: str) -> str:
        """Generate appropriate safety responses"""
        safety_responses = {
            "violence_keywords": "I cannot provide information about violence or harm. If you're in danger, please contact emergency services immediately.",
            "hate_speech": "I'm designed to be respectful and inclusive. I cannot engage with hateful or discriminatory content.",
            "illegal_activities": "I cannot provide information about illegal activities. Please respect the law and community guidelines.",
            "dangerous_advice": "Your safety is important. If you're experiencing distress, please contact a mental health professional or emergency services.",
            "explicit_content": "I cannot engage with explicit or adult content. Please keep our conversation appropriate.",
            "medical_emergency": "This sounds like a medical emergency. Please call emergency services or go to the nearest hospital immediately.",
            "legal_emergency": "This sounds like a legal emergency. Please contact a qualified attorney or legal aid service immediately.",
            "default": "I'm sorry, but I cannot assist with this request. Is there something else I can help you with?"
        }
        
        return safety_responses.get(category, safety_responses["default"])
    
    def log_safety_event(self, event_type: str, content: str, details: Dict):
        """Log safety events for monitoring"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "content": content[:500],  # Limit log size
            "details": details
        }
        self.safety_log.append(event)
        logger.info(f"🛡️ Safety Event: {event_type} - {details.get('reason', 'Unknown')}")

class EthicalFrameworkVNI:
    """VNI that ensures ethical reasoning in responses"""
    
    def __init__(self, instance_id: str = "ethics_0"):
        self.instance_id = instance_id
        self.vni_type = "ethics"
        self.ethical_frameworks = self._load_ethical_frameworks()
        
    def _load_ethical_frameworks(self) -> Dict:
        return {
            "beneficence": "Promote wellbeing and prevent harm",
            "non_maleficence": "Do no harm", 
            "autonomy": "Respect individual choice and consent",
            "justice": "Be fair and avoid discrimination",
            "transparency": "Be clear about capabilities and limitations",
            "accountability": "Take responsibility for outputs"
        }
    
    def evaluate_ethical_implications(self, query: str, proposed_response: str) -> Dict:
        """Evaluate ethical implications of responses"""
        implications = []
        
        # Check for potential harms
        if self._could_cause_harm(proposed_response):
            implications.append("potential_harm")
            
        # Check for bias
        if self._contains_bias(proposed_response):
            implications.append("potential_bias")
            
        # Check for privacy issues
        if self._violates_privacy(proposed_response):
            implications.append("privacy_concern")
            
        # Check for overconfidence
        if self._overconfident(proposed_response):
            implications.append("overconfidence")
            
        return {
            "ethically_sound": len(implications) == 0,
            "implications": implications,
            "adjusted_response": self._apply_ethical_corrections(proposed_response, implications)
        }
    
    def _could_cause_harm(self, response: str) -> bool:
        harm_indicators = ["definitely safe", "completely harmless", "no risk", "guaranteed"]
        return any(indicator in response.lower() for indicator in harm_indicators)
    
    def _contains_bias(self, response: str) -> bool:
        bias_indicators = ["all people", "everyone knows", "obviously", "of course", "naturally"]
        return any(indicator in response.lower() for indicator in bias_indicators)
    
    def _violates_privacy(self, response: str) -> bool:
        privacy_indicators = ["personal information", "private data", "confidential", "should not share"]
        return any(indicator in response.lower() for indicator in privacy_indicators)
    
    def _overconfident(self, response: str) -> bool:
        overconfidence_indicators = ["definitely", "certainly", "absolutely", "without doubt"]
        return any(indicator in response.lower() for indicator in overconfidence_indicators)
    
    def _apply_ethical_corrections(self, response: str, implications: List[str]) -> str:
        """Apply ethical corrections to response"""
        corrected = response
        
        if "potential_harm" in implications:
            corrected += " Please note: Safety should always be your primary concern."
            
        if "potential_bias" in implications:
            corrected += " This is a general perspective; individual experiences may vary."
            
        if "privacy_concern" in implications:
            corrected += " Remember to protect personal and private information."
            
        if "overconfidence" in implications:
            corrected += " This is based on general knowledge; specific situations may differ."
            
        return corrected

class SafetyManager:
    """Main safety management class that orchestrates all safety checks"""
    
    def __init__(self):
        self.safety_vni = ConstitutionalSafetyVNI()
        self.ethics_vni = EthicalFrameworkVNI()
        self.safety_log = []
        
    def validate_input(self, user_message: str, session_id: str) -> Optional[SafetyCheckResult]:
        """Validate user input before processing"""
        safety_check = self.safety_vni.validate_query(user_message)
        
        if not safety_check.safe:
            self.safety_vni.log_safety_event(
                "input_rejection", 
                user_message, 
                {"session_id": session_id, "category": safety_check.category}
            )
            
        return safety_check if not safety_check.safe else None
    
    def validate_responses(self, vni_responses: List[Dict]) -> List[Dict]:
        """Validate all VNI responses for safety and ethics"""
        safe_responses = []
        
        for response in vni_responses:
            response_text = response.get('response', '')
            vni_id = response.get('vni_instance', 'unknown')
            
            # Safety validation
            safety_check = self.safety_vni.validate_response(response_text, vni_id)
            if not safety_check.safe:
                response['response'] = safety_check.corrected_response
                self.safety_vni.log_safety_event(
                    "response_correction",
                    response_text,
                    {"vni_id": vni_id, "category": safety_check.category}
                )
            
            # Ethical validation (only if safety passed)
            if safety_check.safe:
                ethical_check = self.ethics_vni.evaluate_ethical_implications(
                    response.get('original_query', ''), 
                    response_text
                )
                if not ethical_check["ethically_sound"]:
                    response['response'] = ethical_check["adjusted_response"]
                    self.safety_vni.log_safety_event(
                        "ethical_adjustment",
                        response_text,
                        {"vni_id": vni_id, "implications": ethical_check["implications"]}
                    )
            
            safe_responses.append(response)
        
        return safe_responses
    
    def validate_final_output(self, final_response: str) -> SafetyCheckResult:
        """Final safety check before sending to user"""
        return self.safety_vni.validate_response(final_response, "final_output")
    
    def get_safety_report(self) -> Dict:
        """Get safety activity report"""
        return {
            "total_safety_events": len(self.safety_vni.safety_log),
            "recent_events": self.safety_vni.safety_log[-10:],  # Last 10 events
            "safety_config": {
                "rules_categories": list(self.safety_vni.safety_rules.keys()),
                "pattern_categories": list(self.safety_vni.unsafe_patterns.keys())
            }
        } 
