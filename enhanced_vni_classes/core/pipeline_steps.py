# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class PipelineStep(ABC):
    """Base class for all pipeline steps"""
    
    @abstractmethod
    def execute(self, query: str, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute this pipeline step"""
        pass
    
    @property
    def name(self) -> str:
        """Name of this step"""
        return self.__class__.__name__.lower()

# Common pipeline steps
class ClassifyStep(PipelineStep):
    def execute(self, query: str, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        data['classification'] = {
            'domain': 'general',
            'confidence': 0.5,
            'keywords': []
        }
        return data

class SafetyCheckStep(PipelineStep):
    def execute(self, query: str, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        data['safety_check'] = {
            'passed': True,
            'warnings': []
        }
        return data

class KnowledgeLookupStep(PipelineStep):
    def execute(self, query: str, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        data['knowledge_results'] = []
        return data

class GenerateResponseStep(PipelineStep):
    def execute(self, query: str, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        data['response'] = f"Response to: {query}"
        data['confidence'] = 0.7
        return data

# Domain-specific steps
class MedicalSafetyStep(PipelineStep):
    def execute(self, query: str, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        # Check for emergency keywords
        emergency_keywords = ['emergency', '911', 'heart attack', 'stroke', 'bleeding']
        is_emergency = any(keyword in query.lower() for keyword in emergency_keywords)
        
        data['medical_safety'] = {
            'is_emergency': is_emergency,
            'requires_caution': is_emergency,
            'recommendation': 'Seek professional help' if is_emergency else 'Proceed'
        }
        return data

class LegalDisclaimerStep(PipelineStep):
    def execute(self, query: str, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        data['legal_disclaimer'] = "I am an AI assistant and not a licensed attorney."
        return data
