# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

# /bionn-demo-chatbot/enhanced_vni_classes/domains/medical.py
"""
Medical domain VNI implementation.
Specialized Virtual Networked Intelligence for medical information and health advice.
"""
import asyncio
import hashlib
import logging
from datetime import datetime
from types import SimpleNamespace
from bionn_synaptic import VNIMemory
from ..modules.web_search import WebSearch
from ..core.base_vni import EnhancedBaseVNI
from bionn_attention import DemoHybridAttention
from ..utils.logger import get_logger, VNILogger
from typing import Dict, List, Any, Optional, Tuple
from .base_knowledge_loader import BaseKnowledgeLoader
from ..core.capabilities import VNICapabilities, VNIType
from ..modules.knowledge_base import KnowledgeBase, KnowledgeEntry
from bionn_activation import SmartActivationRouter, FunctionRegistry
from ..modules.learning_system import LearningSystem, LearningExperience
from ..modules.classifier import DomainClassifier, ClassificationResult, Domain

logger = logging.getLogger(__name__)

class MedicalVNI(EnhancedBaseVNI, BaseKnowledgeLoader):
    """Specialized VNI for medical domain with enhanced safety protocols and biological systems integration."""
    def __init__(self, 
                 vni_id: str,
                 name: str = "Medical Assistant",
                 description: str = "Specialized in medical information and health advice",
                 vni_config: Optional[Dict[str, Any]] = None,
                 auto_load_knowledge: bool = True,
                 enable_biological_systems: bool = True):  # ADDED: biological systems flag
        self.vni_id = vni_id  # Store the VNI ID        
        # Set up capabilities
        capabilities = VNICapabilities(
            domains=["medical", "health", "wellness", "biology"],
            can_search=True,
            can_learn=True,
            can_collaborate=True,
            max_context_length=4000,
            special_abilities=[
                "medical_terminology", 
                "symptom_analysis", 
                "treatment_info",
                "safety_protocols",
                "emergency_detection",
                "biological_systems_integration"  # ADDED: New special ability
            ],
            vni_type='specialized'            
        )
        
        # Keep name and description for MedicalVNI's own use if needed
        self.vni_type = "medical"
        self.domain = "medical"
        self.name = name
        self.description = description
        
        # ============ BIOLOGICAL SYSTEMS CONFIGURATION ============ ADDED
        self.enable_biological_systems = enable_biological_systems
        self.medical_attention_config = {
            'dim': 256,
            'num_heads': 8,
            'window_size': 256,
            'use_sliding': True,
            'use_global': True,
            'use_hierarchical': True,
            'global_token_ratio': 0.1,  # Higher for medical precision
            'memory_tokens': 20,
            'multi_modal': False,
            'semantic_weight': 0.7,  # High semantic focus for medical terms
            'precision_weight': 0.8  # High precision required
        }
        
        self.medical_memory_config = {
            'short_term_capacity': 150,  # Higher for patient details
            'long_term_capacity': 2000,
            'consolidation_threshold': 0.8,  # Higher threshold for medical accuracy
            'retention_period': 86400 * 7,  # 7 days for patient history
            'priority_retention': True
        }
        
        # Medical-specific setup
        self.medical_knowledge = self._initialize_medical_knowledge()
        self.safety_protocols = self._initialize_safety_protocols()
        self.medical_keywords = self._setup_medical_keywords()
        self.patient_contexts: Dict[str, Dict] = {}  # ADDED: Biological systems patient tracking
        self.medical_guidelines = self._load_medical_guidelines()  # ADDED: Biological systems guidelines
        
        # Initialize VNIMemory for medical domain
        self.memory = VNIMemory(
            domain="medical",
            vni_id=vni_id,
            memory_type="episodic"
        )
        
        # Initialize Hybrid Attention Mechanism from neuron directory
        self.attention = DemoHybridAttention(
            dim=256,
            num_heads=8,
            window_size=256,
            use_sliding=True,
            use_global=True,
            use_hierarchical=True
        )
        
        # Initialize Smart Activation Router from neuron directory
        self.activation_router = SmartActivationRouter(
            vni_id=vni_id,           # Keep for identification
            domain="medical",        # Keep for domain
            input_dim=512,           # ADD: Neural network dimension
            num_experts=3,           # ADD: Medical might need 3 experts (diagnosis, treatment, prevention)
            expert_dim=256           # ADD: Expert layer dimension
        )
        self._register_medical_functions()  # Register medical-specific functions
                        
        # Load knowledge from files if enabled
        if auto_load_knowledge:
            self.load_domain_knowledge("medical")
        logger.info(f"Initialized MedicalVNI: {vni_id} ({name}) with biological systems: {enable_biological_systems}")

        # Initialize base VNI - FIXED VERSION
        super().__init__(
            instance_id=vni_id,  # Pass vni_id as instance_id
            domain="medical",
            capabilities=capabilities,
            vni_config=vni_config or {}
        )

    def _check_safety(self, query: str) -> Dict[str, Any]:
        """Check medical query safety"""
        import re
        from typing import Dict, Any
        
        query_lower = query.lower()
        
        # Medical emergency keywords - highest priority
        emergency_keywords = [
            "heart attack", "stroke", "bleeding", "unconscious", 
            "can't breathe", "chest pain", "severe pain", "emergency",
            "911", "ambulance", "urgent", "critical", "dying", 
            "not breathing", "choking", "overdose", "suicide",
            "allergic reaction", "anaphylaxis", "seizure"
        ]
        
        for keyword in emergency_keywords:
            if keyword in query_lower:
                return {
                    'requires_caution': True,  # ← ADD THIS
                    'is_safe': False,
                    'message': f"🚨 MEDICAL EMERGENCY DETECTED: '{keyword}'. " +
                              "PLEASE CALL EMERGENCY SERVICES IMMEDIATELY: 911 (US) or your local emergency number. " +
                              "Do not wait for a response from this system.",
                    'confidence': 1.0,
                    'emergency': True,
                    'risk_level': 'critical'
                }
        
        # High-risk conditions
        high_risk_keywords = [
            "pregnant", "child", "baby", "infant", "elderly", 
            "diabetes", "cancer", "hiv", "aids", "transplant",
            "pregnancy", "newborn", "toddler"
        ]
        
        for keyword in high_risk_keywords:
            if keyword in query_lower:
                return {
                    'is_safe': True,  # Still safe to answer, but with strong warning
                    'message': f"⚠️ HIGH-RISK CONTEXT DETECTED: '{keyword}'. " +
                              "This information is for educational purposes only. " +
                              "CONSULT A HEALTHCARE PROFESSIONAL IMMEDIATELY for medical advice. " +
                              "Do not delay seeking professional medical care.",
                    'confidence': 0.9,
                    'emergency': False,
                    'risk_level': 'high'
                }
        
        # Medication/dosage questions
        medication_patterns = [
            r"how much.*(take|dosage|dose|mg|ml)",
            r"(take|use).*\d+.*(mg|ml|pill|tablet)",
            r"prescription.*(drug|medicine|medication)",
            r"side effect.*(of|from)",
            r"should i take.*(medicine|medication|drug)",
            r"what.*dose.*of.*",
            r"how many.*(pills|tablets|capsules)"
        ]
        
        for pattern in medication_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return {
                    'is_safe': True,
                    'message': "⚠️ MEDICATION/DOSAGE QUESTION DETECTED. " +
                              "IMPORTANT: Do not change or start medication without consulting a doctor. " +
                              "This information is for educational purposes only. " +
                              "Always follow your healthcare provider's instructions.",
                    'confidence': 0.8,
                    'emergency': False,
                    'risk_level': 'medium'
                }
        
        # Self-diagnosis/treatment questions
        diagnosis_patterns = [
            r"do i have.*",
            r"what.*disease.*i have",
            r"self.*treat.*",
            r"home remedy.*",
            r"should i.*(go to|see).*doctor"
        ]
        
        for pattern in diagnosis_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return {
                    'requires_caution': True,
                    'is_safe': True,
                    'message': "⚠️ MEDICAL DIAGNOSIS/TREATMENT QUESTION. " +
                              "This system cannot diagnose medical conditions. " +
                              "Please consult a qualified healthcare professional for diagnosis and treatment.",
                    'confidence': 0.7,
                    'emergency': False,
                    'risk_level': 'medium'
                }
        
        # General medical question - standard disclaimer
        medical_terms = ["pain", "symptom", "fever", "headache", "nausea", "virus", "infection"]
        is_medical = any(term in query_lower for term in medical_terms)
        
        if is_medical:
            return {
                'requires_caution': True,
                'is_safe': True,
                'message': "General medical information request. " +
                          "DISCLAIMER: This is not medical advice. For diagnosis and treatment, " +
                          "consult a qualified healthcare professional.",
                'confidence': 0.6,
                'emergency': False,
                'risk_level': 'low'
            }
        
        # Non-medical or very general question
        return {
            'requires_caution': False,
            'is_safe': True,
            'message': "General information request. Always consult professionals for medical concerns.",
            'confidence': 0.5,
            'emergency': False,
            'risk_level': 'none'
        }

    def _generate_safety_response(self, query: str, safety_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response when safety check requires caution."""
        # If safety_result already has a message, use it
        if 'message' in safety_result:
            response_text = safety_result['message']
        else:
            # Fallback response
            response_text = (
                "⚠️ **SAFETY NOTICE** ⚠️\n\n"
                "This medical query requires professional consultation. "
                "Please consult with a qualified healthcare professional."
            )
        
        return {
            "response": response_text,
            "domain": "medical",
            "confidence": safety_result.get('confidence', 0.5),
            "safety_check": safety_result,
        }

    def _is_relevant_medical_query(self, query: str) -> bool:
        """Determine if a query is relevant to the medical domain.
        Args:
            query: User query string
        Returns:
            Boolean indicating if query is medical-related"""
        try:
            if not query or not isinstance(query, str):
                return False
            
            query_lower = query.lower().strip()
           
            # Common medical keywords
            medical_keywords = [
                # Symptoms
                'pain', 'hurt', 'ache', 'fever', 'headache', 'cough', 'cold', 
                'flu', 'nausea', 'dizzy', 'tired', 'fatigue', 'swollen',
                'sore', 'infection', 'rash', 'bleed', 'bleeding',
                
                # Conditions
                'diabetes', 'cancer', 'heart', 'stroke', 'asthma', 'allergy',
                'arthritis', 'depression', 'anxiety', 'hypertension',
                'high blood pressure', 'cholesterol', 'migraine',
                
                # Medical terms
                'diagnosis', 'treatment', 'therapy', 'medication', 'prescription',
                'surgery', 'operation', 'appointment', 'checkup', 'vaccine',
                'vaccination', 'test results', 'x-ray', 'mri', 'scan', 'lab',
                
                # Body parts
                'stomach', 'chest', 'back', 'head', 'neck', 'shoulder', 'arm',
                'leg', 'knee', 'foot', 'hand', 'eye', 'ear', 'nose', 'throat',
                'skin', 'bone', 'muscle', 'nerve', 'lung', 'liver', 'kidney',
                
                # Healthcare
                'doctor', 'physician', 'nurse', 'surgeon', 'dentist', 'pharmacist',
                'hospital', 'clinic', 'emergency', 'er', 'urgent care', 'pharmacy',
                'insurance', 'medicare', 'medicaid', 'health plan',
                
                # General health
                'health', 'medical', 'patient', 'sick', 'ill', 'unwell', 'disease',
                'condition', 'symptom', 'treatment', 'recovery', 'heal'
            ]
        
            # Check for exact keyword matches
            for keyword in medical_keywords:
                if keyword in query_lower:
                    return True
        
            # Check for medical question patterns
            medical_patterns = [
                r'(my|i have a|i\'ve got|feeling)\s+\w+\s+(pain|ache|hurt)',
                r'(should|can)\s+i\s+(take|use)',
                r'medication\s+(for|to)',
                r'(doctor|physician)\s+(said|told|recommended)',
                r'treatment\s+(for|options)',
                r'symptoms?\s+(of|for)',
                r'diagnosed\s+(with|as)',
                r'what.*(medicine|medication|treatment)',
                r'how.*(treat|handle|manage)',
                r'is.*(safe|dangerous|serious)'
            ]
            
            import re
            for pattern in medical_patterns:
                if re.search(pattern, query_lower):
                   return True
            
            # Check for health-related phrases
            health_phrases = [
                'feel sick', 'not feeling well', 'under the weather',
                'come down with', 'running a fever', 'high temperature',
                'blood pressure', 'heart rate', 'taking medication',
                'see a doctor', 'go to hospital', 'emergency room',
                'side effects', 'allergic reaction', 'follow up'
            ]
            
            for phrase in health_phrases:
                if phrase in query_lower:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking medical relevance: {str(e)}")
            return False
    
    # ============ MEDICAL FUNCTION REGISTRATION ============
    def _register_medical_functions(self) -> None:
        """Register medical-specific functions with the SmartActivationRouter"""
        try:
            logger.info(f"Registering medical functions with SmartActivationRouter")
            
            # Register basic medical functions - you can expand these later
            self.activation_router.register_function(
                function_name="medical_diagnosis_assistant",
                function=self._simple_medical_helper,                
                domain=self.domain,
                priority=2
            )
            
            # Add more functions as needed...

            logger.info(f"✅ Registered medical functions with SmartActivationRouter")
            
        except Exception as e:
            logger.error(f"❌ Failed to register medical functions: {e}")
            # Don't raise, just log - allow VNI to continue without function registration
    
    def _simple_medical_helper(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Simple helper for medical functions"""
        return {
            "function": "medical_helper",
            "query": query[:50],
            "note": "Medical function placeholder - implement specific logic as needed",
            "status": "available"
        }
    
    # ============ NEW BIOLOGICAL SYSTEMS METHODS ============ ADDED
    def _load_medical_guidelines(self) -> Dict[str, Any]:
        """Load medical guidelines for biological systems"""
        return {
            "emergency_protocols": {
                "activation_threshold": 0.7,
                "response_time_limit": 2.0,  # seconds
                "immediate_actions": ["call_emergency", "provide_first_aid_instructions"]
            },
            "symptom_categories": {
                "cardiovascular": ["chest pain", "shortness of breath", "palpitations"],
                "neurological": ["headache", "dizziness", "confusion"],
                "respiratory": ["cough", "difficulty breathing", "wheezing"],
                "gastrointestinal": ["nausea", "vomiting", "abdominal pain"]
            },
            "biological_monitoring": {
                "vital_signs": ["heart_rate", "blood_pressure", "temperature", "respiratory_rate"],
                "alert_levels": {
                    "normal": 0.0,
                    "caution": 0.4,
                    "warning": 0.7,
                    "critical": 0.9
                }
            }
        }
    
    def _extract_symptoms(self, query: str) -> List[str]:
        """Extract symptoms from query using biological systems analysis"""
        query_lower = query.lower()
        extracted_symptoms = []
        
        # Check each symptom category
        for category, symptoms in self.medical_guidelines["symptom_categories"].items():
            for symptom in symptoms:
                if symptom in query_lower:
                    extracted_symptoms.append({
                        "symptom": symptom,
                        "category": category,
                        "confidence": 0.8
                    })
        
        # Also check generic symptom keywords
        generic_symptoms = ["pain", "fever", "fatigue", "weakness", "swelling", "rash"]
        for symptom in generic_symptoms:
            if symptom in query_lower:
                extracted_symptoms.append({
                    "symptom": symptom,
                    "category": "general",
                    "confidence": 0.6
                })
        
        return extracted_symptoms
    
    def _get_patient_history(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get patient history from context for biological systems"""
        if not context or "patient_id" not in context:
            return {"available": False, "history": {}}
        
        patient_id = context.get("patient_id")
        history = self.patient_contexts.get(patient_id, {})
        
        return {
            "available": bool(history),
            "history": history,
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat()
        }

    def _initialize_medical_knowledge(self):
        """Initialize medical-specific knowledge base"""
        return {
            "medical_concepts": {},
            "drug_database": {},
            "symptom_patterns": {}
        }

    def _initialize_safety_protocols(self):
        """Initialize safety protocols for medical responses"""
        return {
            "safety_check": True,
            "disclaimer_required": True,
            "emergency_keywords": ["heart attack", "stroke", "bleeding"]
        }

    def _setup_medical_keywords(self):
        """Setup medical domain keywords"""
        return [
            "health", "medicine", "treatment", "diagnosis", "symptoms",
            "patient", "doctor", "hospital", "clinical", "pharmaceutical"
        ]
        
    def process_with_biological_systems(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process query with medical biological systems integration"""
        if not self.enable_biological_systems:
            return {
                "biological_processing": False,
                "reason": "Biological systems disabled"
            }
        
        logger.debug(f"🏥 Medical biological processing: {query[:50]}...")
        
        try:
            # Extract biological insights
            symptoms = self._extract_symptoms(query)
            patient_history = self._get_patient_history(context)
            
            # Check for emergency indicators
            emergency_keywords = self.safety_protocols["emergency_keywords"]
            is_emergency = any(keyword in query.lower() for keyword in emergency_keywords)
            
            # Calculate activation level based on biological factors
            activation_level = self._calculate_biological_activation(query, symptoms, patient_history)
            
            # Determine urgency level
            urgency_level = self._determine_urgency_level(activation_level, symptoms, is_emergency)
            
            # Generate biological insights
            biological_insights = {
                "symptoms_detected": len(symptoms),
                "symptom_categories": list(set([s["category"] for s in symptoms])),
                "activation_level": activation_level,
                "urgency_level": urgency_level,
                "requires_immediate_attention": activation_level > self.medical_guidelines["emergency_protocols"]["activation_threshold"],
                "patient_history_available": patient_history["available"],
                "biological_timestamp": datetime.now().isoformat()
            }
            
            # Update patient context if patient ID exists
            if context and "patient_id" in context:
                self._update_patient_context(context["patient_id"], query, symptoms, biological_insights)
            
            return {
                "biological_processing": True,
                "biological_insights": biological_insights,
                "activation_result": {
                    "activation_level": activation_level,
                    "urgency": urgency_level,
                    "recommended_actions": self._get_recommended_actions(urgency_level)
                },
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Medical biological processing failed: {e}")
            return {
                "biological_processing": False,
                "error": str(e),
                "success": False
            }

    def process(self, query: str, pipeline: Optional[List[str]] = None, 
                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Implement abstract method from EnhancedBaseVNI."""
        # Use the existing process_query method
        return self.process_query(query, context)       
    
    def _calculate_biological_activation(self, query: str, symptoms: List[Dict], 
                                        patient_history: Dict[str, Any]) -> float:
        """Calculate biological activation level based on query analysis"""
        base_activation = 0.3  # Base level for any medical query
        
        # Increase based on symptom count
        symptom_factor = min(len(symptoms) * 0.1, 0.3)
        base_activation += symptom_factor
        
        # Increase based on symptom severity
        severe_symptoms = ["chest pain", "difficulty breathing", "unconscious", "severe bleeding"]
        for symptom in symptoms:
            if symptom["symptom"] in severe_symptoms:
                base_activation += 0.2
        
        # Increase based on emergency keywords
        emergency_keywords = self.safety_protocols["emergency_keywords"]
        for keyword in emergency_keywords:
            if keyword in query.lower():
                base_activation += 0.3
                break
        
        # Consider patient history
        if patient_history.get("available"):
            historical_issues = patient_history.get("history", {}).get("previous_issues", [])
            if historical_issues:
                base_activation += 0.1
        
        # Cap at 1.0
        return min(base_activation, 1.0)
    
    def _determine_urgency_level(self, activation_level: float, symptoms: List[Dict], 
                                is_emergency: bool) -> str:
        """Determine urgency level based on biological analysis"""
        if is_emergency or activation_level >= 0.9:
            return "critical"
        elif activation_level >= 0.7:
            return "high"
        elif activation_level >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _get_recommended_actions(self, urgency_level: str) -> List[str]:
        """Get recommended actions based on urgency level"""
        actions = {
            "critical": [
                "Call emergency services immediately",
                "Do not move the patient if injury is suspected",
                "Monitor vital signs if possible",
                "Prepare for ambulance arrival"
            ],
            "high": [
                "Contact healthcare provider urgently",
                "Monitor symptoms closely",
                "Keep patient comfortable",
                "Prepare for possible emergency room visit"
            ],
            "medium": [
                "Schedule appointment with doctor",
                "Monitor symptoms",
                "Note any changes in condition",
                "Follow up if symptoms worsen"
            ],
            "low": [
                "Monitor for any changes",
                "Consider telehealth consultation",
                "Practice preventive measures",
                "Follow general health guidelines"
            ]
        }
        return actions.get(urgency_level, ["Consult healthcare provider"])
    
    def _update_patient_context(self, patient_id: str, query: str, 
                               symptoms: List[Dict], insights: Dict[str, Any]):
        """Update patient context for biological tracking"""
        if patient_id not in self.patient_contexts:
            self.patient_contexts[patient_id] = {
                "first_interaction": datetime.now().isoformat(),
                "symptoms_history": [],
                "interaction_count": 0,
                "previous_issues": []
            }
        
        patient_data = self.patient_contexts[patient_id]
        patient_data["interaction_count"] += 1
        patient_data["last_interaction"] = datetime.now().isoformat()
        
        # Add symptoms to history
        for symptom in symptoms:
            patient_data["symptoms_history"].append({
                "symptom": symptom["symptom"],
                "category": symptom["category"],
                "timestamp": datetime.now().isoformat(),
                "query_context": query[:100]
            })
        
        # Keep only recent history (last 100 entries)
        if len(patient_data["symptoms_history"]) > 100:
            patient_data["symptoms_history"] = patient_data["symptoms_history"][-100:]
        
        # Update previous issues if high urgency
        if insights.get("urgency_level") in ["high", "critical"]:
            patient_data["previous_issues"].append({
                "timestamp": datetime.now().isoformat(),
                "urgency": insights["urgency_level"],
                "symptoms": [s["symptom"] for s in symptoms]
            })
    
    def _enhance_medical_response(self, result: Dict[str, Any], query: str, 
                                 context: Optional[Dict[str, Any]] = None,
                                 biological_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhance response with medical biological insights"""
        enhanced_result = result.copy()
        
        if biological_result and biological_result.get("biological_processing"):
            # Add biological insights to response
            insights = biological_result.get("biological_insights", {})
            activation = biological_result.get("activation_result", {})
            
            enhanced_result["biological_analysis"] = {
                "performed": True,
                "symptoms_detected": insights.get("symptoms_detected", 0),
                "urgency_level": insights.get("urgency_level", "unknown"),
                "activation_level": activation.get("activation_level", 0.0),
                "requires_professional_review": activation.get("activation_level", 0) > 0.7
            }
            
            # Add urgency-based guidance
            if enhanced_result["biological_analysis"]["requires_professional_review"]:
                guidance = "⚠️ **URGENT MEDICAL ATTENTION RECOMMENDED** ⚠️\n"
                guidance += "Based on symptom analysis, this may require immediate medical evaluation.\n"
                guidance += "Please contact a healthcare provider or emergency services.\n\n"
                
                # Prepend guidance to response
                enhanced_result["response"] = guidance + enhanced_result.get("response", "")
                
                # Add disclaimer
                if "disclaimers" not in enhanced_result:
                    enhanced_result["disclaimers"] = []
                enhanced_result["disclaimers"].append(
                    "Biological systems analysis indicates potential medical urgency."
                )
        
        return enhanced_result
    # ============ END OF BIOLOGICAL SYSTEMS METHODS ============

    # ============ UPDATED METHODS FOR BIOLOGICAL INTEGRATION ============

    def process_query(self, 
                      query: str, 
                      context: Optional[Dict[str, Any]] = None,
                      patient_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process medical query with comprehensive safety checks and biological system"""
        logger.info(f"Processing medical query: {query[:100]}...")
        
        # Check relevance first (from V1)
        if not self._is_relevant_medical_query(query):
            result = {
                "response": "This appears to be outside my medical expertise.",
                "confidence": 0.1,
                "vni_instance": self.vni_id,
                "suggestion": "Try asking about symptoms, treatments, health advice, or medical conditions.",
                "response_type": "medical_out_of_scope",
                'vni_metadata': {
                    'vni_id': self.vni_id,
                    'success': True,
                    'processing_time': 0.01,
                    'timestamp': datetime.now().isoformat()
                }
            }
            result['vni_id'] = self.vni_id
            result['domain'] = 'medical'
            result['opinion_text'] = result['response']
            return result
        
        # Safety check (from V2)
        safety_result = self._check_safety(query)
        if safety_result["requires_caution"]:
            safety_response = self._generate_safety_response(query, safety_result)
            safety_response['vni_metadata'] = {
                'vni_id': self.vni_id,
                'success': True,
                'processing_time': 0.01,
                'timestamp': datetime.now().isoformat()
            }
            safety_response['vni_id'] = self.vni_id
            safety_response['domain'] = 'medical'
            safety_response['opinion_text'] = safety_response.get('response', 'Safety warning')           
            return safety_response
        
        # Classify query type
        classification = self._safe_classify(query)
        query_type = self._classify_medical_query_type(query)
        
        # Query knowledge base (from V1, enhanced)
        knowledge_results = self._query_medical_knowledge(query)
        
        # If knowledge found with high confidence, use it
        if knowledge_results and knowledge_results.get("confidence", 0) > 0.7:
            result = {
                "response": knowledge_results["content"],
                "confidence": knowledge_results["confidence"],
                "vni_instance": self.vni_id,
                "source": knowledge_results.get("source", "medical_knowledge_base"),
                "query_type": query_type,
                "response_type": "medical_knowledge",
                "safety_checked": True,
                'vni_metadata': {
                    'vni_id': self.vni_id,
                    'success': True,
                    'processing_time': 0.01,
                    'timestamp': datetime.now().isoformat()
                }                
            }
            
            # Enhance with biological systems if enabled
            if self.enable_biological_systems:
                biological_result = self.process_with_biological_systems(query, context)
                result = self._enhance_medical_response(result, query, context, biological_result)
            result['vni_id'] = self.vni_id
            result['domain'] = 'medical'
            result['opinion_text'] = result['response']
            return result
        
        # Use enhanced processing for complex queries
        complex_result = self._process_complex_medical_query(query, classification, patient_context)
        # === ADD THIS to the complex result if it doesn't have vni_metadata ===
        if 'vni_metadata' not in complex_result:
            complex_result['vni_metadata'] = {
                'vni_id': self.vni_id,
                'success': True,
                'processing_time': 0.01,
                'timestamp': datetime.now().isoformat()
            }
        complex_result['vni_id'] = self.vni_id
        complex_result['domain'] = 'medical'
        complex_result['opinion_text'] = complex_result.get('response', 'Complex medical analysis performed')
        return complex_result
        
    async def process_medical_query_async(self,
                                     query: str,
                                     patient_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Async version - returns comprehensive medical insights for aggregator to process."""
        logger.info(f"Async medical analysis for aggregator: {query[:100]}...")
        
        # 1. SAFETY CHECK (Critical - must run first)
        safety_result = self._check_safety(query)
        if safety_result["requires_caution"]:
            # Return safety insights for aggregator (not final response)
            return {
                "status": "medical_safety_check_triggered",
                "processing_complete": True,
                "medical_insights": {
                    "safety_assessment": safety_result,
                    "is_emergency": safety_result.get("emergency", False),
                    "risk_level": safety_result.get("risk_level", "none"),
                    "immediate_action_required": safety_result.get("emergency", False),
                    "query_blocked": True,
                    "block_reason": "safety_protocols"
                },
                "aggregator_instructions": {
                    "handle_emergency": safety_result.get("emergency", False),
                    "priority": "critical" if safety_result.get("emergency") else "high",
                    "suggested_response_template": "safety_warning"
                },
                "raw_data": {
                    "query": query,
                    "safety_result": safety_result
                }
            }
        
        # 2. CLASSIFY QUERY
        classification = self._safe_classify(query)
        
        # 3. BUILD MEDICAL CONTEXT (async)
        context = await self._build_medical_context(query, patient_context)
        
        # 4. BIOLOGICAL SYSTEMS ANALYSIS
        biological_result = None
        if self.enable_biological_systems:
            biological_result = self.process_with_biological_systems(query, context)
        
        # 5. RETRIEVE RELEVANT MEMORY (async)
        memory_insights = {}
        if hasattr(self, 'memory'):
            try:
                memory_context = await self.memory.retrieve_relevant_memory(
                    query=query,
                    patient_context=patient_context
                )
                if memory_context:
                    memory_insights = {
                        "has_relevant_memory": True,
                        "memory_context": memory_context,
                        "memory_retrieval_success": True
                    }
            except Exception as e:
                logger.warning(f"Memory retrieval failed: {e}")
                memory_insights = {"has_relevant_memory": False, "error": str(e)}
        
        # 6. COMPUTE ATTENTION FOCUS
        attention_focus = self.attention.compute_attention(query, context)
        
        # 7. EXTRACT MEDICAL COMPONENTS
        query_type = self._classify_medical_query_type(query)
        symptoms = self._extract_symptoms(query)
        urgency_level = self._determine_urgency_level(
            query, 
            symptoms,
            biological_result.get("biological_insights", {}) if biological_result else {}
        )
        
        # 8. BUILD COMPREHENSIVE MEDICAL INSIGHTS
        medical_insights = {
            # Core analysis
            "safety_assessment": safety_result,
            "classification": {
                "primary_domain": classification.primary_domain,
                "confidence": classification.confidence,
                "subdomains": classification.subdomains if hasattr(classification, 'subdomains') else []
            },
            "query_type": query_type,
            
            # Biological systems
            "biological_processing": biological_result.get("biological_processing") if biological_result else False,
            "biological_insights": biological_result.get("biological_insights", {}) if biological_result else {},
            "activation_result": biological_result.get("activation_result", {}) if biological_result else {},
            
            # Symptoms & urgency
            "symptoms_detected": symptoms,
            "symptom_count": len(symptoms),
            "urgency_level": urgency_level,
            "requires_professional_review": urgency_level in ["high", "critical"],
            
            # Patient context
            "patient_context_used": bool(patient_context),
            "patient_info": {
                "has_patient_id": "patient_id" in (patient_context or {}),
                "age": patient_context.get("age") if patient_context else None,
                "gender": patient_context.get("gender") if patient_context else None
            },
            
            # Attention & focus
            "attention_focus": attention_focus.get("primary_focus", "general"),
            "attention_weights": attention_focus.get("weights", {}),
            
            # Memory insights
            "memory_insights": memory_insights,
            
            # Metadata
            "processing_timestamp": datetime.now().isoformat(),
            "vni_instance": self.vni_id,
            "confidence_score": self._calculate_medical_confidence(query, symptoms, biological_result)
        }
        
        # 9. PREPARE INSTRUCTIONS FOR AGGREGATOR
        aggregator_instructions = {
            "style_suggestions": {
                "primary_style": "technical_precise",
                "alternative_styles": ["formal", "empathetic"],
                "temperature": 0.3,  # Low for medical accuracy
                "max_tokens": 500
            },
            "content_requirements": {
                "include_safety_disclaimers": True,
                "include_urgency_notice": medical_insights["requires_professional_review"],
                "reference_biological_insights": medical_insights["biological_processing"],
                "prioritize_accuracy": True
            },
            "formatting": {
                "include_bullet_points": len(symptoms) > 1,
                "include_urgency_banner": medical_insights["requires_professional_review"],
                "structure": ["summary", "analysis", "recommendations", "disclaimers"]
            },
            "priority_level": "high" if medical_insights["requires_professional_review"] else "normal"
        }
        
        # 10. BUILD FINAL OUTPUT FOR AGGREGATOR
        return {
            "status": "medical_analysis_complete",
            "processing_method": "async",
            "processing_time": datetime.now().isoformat(),
            
            # The main payload for aggregator
            "medical_insights": medical_insights,
            
            # How aggregator should use these insights
            "aggregator_instructions": aggregator_instructions,
            
            # Raw data if aggregator needs it
            "raw_context": {
                "query": query,
                "built_context": context,
                "patient_context": patient_context or {}
            },
            
            # Metadata
            "vni_metadata": {
                "vni_id": self.vni_id,
                "vni_type": self.vni_type,
                "name": self.name,
                "biological_systems_enabled": self.enable_biological_systems,
                "capabilities": [cap for cap in self.capabilities.special_abilities] if hasattr(self, 'capabilities') else [],
                "success": True,
                "processing_time": 0.01,
                "timestamp": datetime.now().isoformat()        
            },
            
            # Next steps
            "next_step": "aggregator_combine_and_generate",
            "expected_output_format": "final_response_with_medical_context"
        }

    def _record_medical_interaction_insights(self,
                                            query: str,
                                            medical_insights: Dict[str, Any],
                                            classification,
                                            patient_context: Optional[Dict[str, Any]] = None):
        """Record medical interaction insights for learning (without final response)."""
        if hasattr(self, 'learning'):
            try:
                self.learning.record_interaction(
                    interaction_id=hashlib.md5(query.encode()).hexdigest()[:16],
                    prompt=query,
                    response="[Medical insights for aggregator]",
                    domain="medical",
                    metadata={
                        "query_type": medical_insights.get("query_type", "general_inquiry"),
                        "patient_context_used": bool(patient_context),
                        "confidence": classification.confidence if hasattr(classification, 'confidence') else 0.7,
                        "safety_checked": True,
                        "biological_processing": medical_insights.get("biological_processing", False),
                        "urgency_level": medical_insights.get("urgency_level", "unknown"),
                        "processed_for_aggregator": True
                    }
                )
            except Exception as e:
                logger.error(f"Failed to record medical interaction insights: {e}")

    def get_medical_stats(self) -> Dict[str, Any]:
        """
        Get medical-specific statistics including memory stats and biological systems.
        
        Returns:
            Dictionary with medical VNI statistics
        """
        stats = {
            "vni_id": self.vni_id,
            "name": self.name,
            "medical_conditions_known": len(self.medical_knowledge["common_conditions"]),
            "medical_terms_known": len(self.medical_knowledge["medical_terms"]),
            "safety_protocols": len(self.safety_protocols["disclaimers"]),
            "emergency_keywords": len(self.safety_protocols["emergency_keywords"]),
            "keyword_categories": {
                "symptoms": len(self.medical_keywords["symptoms"]),
                "conditions": len(self.medical_keywords["conditions"]),
                "treatments": len(self.medical_keywords["treatments"]),
                "professionals": len(self.medical_keywords["professionals"]),
                "facilities": len(self.medical_keywords["facilities"])
            },
            "biological_systems": {  # ADDED: Biological systems stats
                "enabled": self.enable_biological_systems,
                "patient_contexts_tracked": len(self.patient_contexts),
                "medical_guidelines": len(self.medical_guidelines),
                "attention_config": self.medical_attention_config,
                "memory_config": self.medical_memory_config
            }
        }
        
        # Add learning stats if available
        if hasattr(self, 'learning'):
            learning_stats = self.learning.export_knowledge()
            stats["learning"] = {
                "experiences": learning_stats.get("learning_data_count", 0),
                "patterns": len(learning_stats.get("patterns", {})),
                "learning_rate": learning_stats.get("learning_rate", 0.1)
            }
        
        # Add memory stats if available
        if hasattr(self, 'memory'):
            try:
                memory_stats = self.memory.get_stats()
                stats["memory"] = {
                    "total_interactions": memory_stats.get("total_interactions", 0),
                    "memory_type": memory_stats.get("memory_type", "unknown"),
                    "retrieval_success_rate": memory_stats.get("retrieval_success_rate", 0.0)
                }
            except Exception as e:
                logger.warning(f"Could not get memory stats: {e}")
                stats["memory"] = {"error": "unavailable"}
        
        return stats

    # ============ NEW METHOD NEEDED ============ ADDED
    def _classify_medical_query_type(self, query: str) -> str:
        """Classify the type of medical query (existing method that's referenced but not defined)"""
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ["emergency", "urgent", "911", "immediate"]):
            return "emergency"
        elif any(keyword in query_lower for keyword in ["symptom", "pain", "fever", "headache"]):
            return "symptom_analysis"
        elif any(keyword in query_lower for keyword in ["treatment", "medicine", "prescription", "therapy"]):
            return "treatment_info"
        elif any(keyword in query_lower for keyword in ["diagnosis", "what do i have", "condition"]):
            return "diagnosis_inquiry"
        elif any(keyword in query_lower for keyword in ["prevention", "prevent", "avoid"]):
            return "prevention_advice"
        else:
            return "general_inquiry"
        
    def _safe_classify(self, query: str) -> Any:
        """Safely classify the medical query with fallback.
        Returns a classification result (could be a dict or SimpleNamespace)."""
        try:
            # Try to use the classifier if available
            if hasattr(self, 'classifier') and self.classifier:
                result = self.classifier.classify(query)
                if result:
                    return result
            
            # Fallback to simple classification
            from ..modules.classifier import ClassificationResult, Domain
            
            # Simple classification based on keywords
            query_lower = query.lower()
            
            # Determine primary domain
            if any(word in query_lower for word in ['emergency', 'urgent', '911', 'immediate']):
                primary_domain = Domain.EMERGENCY
                confidence = 0.9
            elif any(word in query_lower for word in ['symptom', 'pain', 'fever', 'headache', 'cough']):
                primary_domain = Domain.SYMPTOM
                confidence = 0.8
            elif any(word in query_lower for word in ['treatment', 'medicine', 'drug', 'prescription']):
                primary_domain = Domain.TREATMENT
                confidence = 0.8
            elif any(word in query_lower for word in ['diagnosis', 'condition', 'disease', 'illness']):
                primary_domain = Domain.DIAGNOSIS
                confidence = 0.7
            else:
                primary_domain = Domain.GENERAL
                confidence = 0.6
            
            # Create a simple result object
            result = SimpleNamespace(
                primary_domain=primary_domain,
                confidence=confidence,
                subdomains=[]
            )
            return result
        except Exception as e:
            logger.error(f"Error in _safe_classify: {e}")
            # Return a minimal default
            from ..modules.classifier import Domain
            return SimpleNamespace(
                primary_domain=Domain.GENERAL,
                confidence=0.5,
                subdomains=[]
            )
       
# Backward compatibility alias
EnhancedMedicalVNI = MedicalVNI
