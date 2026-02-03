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
from typing import Dict, List, Any, Optional, Tuple

from .base_knowledge_loader import BaseKnowledgeLoader
from ..core.base_vni import EnhancedBaseVNI
from ..core.capabilities import VNICapabilities, VNIType
from ..modules.knowledge_base import KnowledgeBase, KnowledgeEntry
from ..modules.learning_system import LearningSystem, LearningExperience
from ..modules.generation import EnhancedGenerationModule, GenerationStyle # TextGenerator
from ..modules.web_search import WebSearch
# from ..modules.attention import AttentionMechanism, AttentionType, AttentionWeight
from ..modules.classifier import DomainClassifier, ClassificationResult, Domain
from ..utils.logger import get_logger, VNILogger
from neuron.vni_memory import VniMemory as VNIMemory
from neuron.demoHybridAttention import DemoHybridAttention
from neuron.smart_activation_router import SmartActivationRouter, FunctionRegistry

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
        """
        Implement abstract method from EnhancedBaseVNI.
        Required because EnhancedBaseVNI has @abstractmethod process()
        
        Args:
            query: Input query
            pipeline: Optional pipeline steps (not used here, using default)
            context: Additional context
            
        Returns:
            Processing result
        """
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
        """
        Process medical query with comprehensive safety checks and biological systems.
        
        Args:
            query: The medical query to process
            context: Optional conversation context
            patient_context: Optional patient-specific information (age, symptoms, etc.)
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"Processing medical query: {query[:100]}...")
        
        # Check relevance first (from V1)
        if not self._is_relevant_medical_query(query):
            return {
                "response": "This appears to be outside my medical expertise.",
                "confidence": 0.1,
                "vni_instance": self.vni_id,
                "suggestion": "Try asking about symptoms, treatments, health advice, or medical conditions.",
                "response_type": "medical_out_of_scope"
            }
        
        # Safety check (from V2)
        safety_result = self._check_safety(query)
        if safety_result["requires_caution"]:
            return self._generate_safety_response(query, safety_result)
        
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
                "safety_checked": True
            }
            
            # Enhance with biological systems if enabled
            if self.enable_biological_systems:
                biological_result = self.process_with_biological_systems(query, context)
                result = self._enhance_medical_response(result, query, context, biological_result)
            
            return result
        
        # Use enhanced processing for complex queries
        return self._process_complex_medical_query(query, classification, patient_context)
    
    async def process_medical_query_async(self,
                                         query: str,
                                         patient_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Async version of medical query processing with web search, context building, and biological systems."""
        # Safety check
        safety_result = self._check_safety(query)
        if safety_result["requires_caution"]:
            return self._generate_safety_response(query, safety_result)
        
        # Classify query
        classification = self._safe_classify(query)
        
        # Build medical context
        context = await self._build_medical_context(query, patient_context)
        
        # BIOLOGICAL SYSTEMS INTEGRATION ADDED
        biological_result = None
        if self.enable_biological_systems:
            biological_result = self.process_with_biological_systems(query, context)
            if biological_result.get("biological_processing"):
                # Add biological insights to context
                context["biological_insights"] = biological_result.get("biological_insights", {})
                context["activation_result"] = biological_result.get("activation_result", {})

        # Retrieve relevant memory before computing attention
        if hasattr(self, 'memory'):
            memory_context = await self.memory.retrieve_relevant_memory(
                query=query,
                patient_context=patient_context
            )
            if memory_context:
                context["memory_context"] = memory_context

        # Compute attention (from V2)
        attention = self.attention.compute_attention(query, context)
        
        # Generate response with medical style
        response = self.generator.generate_response(
            query=query,
            context=context,
            style=GenerationStyle.TECHNICAL.value,
            #temperature=0.6  # Lower temperature for medical accuracy
        )
        
        # Add medical disclaimers
        response["response"] = self._add_medical_disclaimers(response["response"])
        
        # Enhance with biological systems if enabled
        if self.enable_biological_systems and biological_result:
            response = self._enhance_medical_response(response, query, context, biological_result)
        
        # Record learning
        self._record_medical_interaction(query, response, classification)
        
        # Store interaction in memory
        if hasattr(self, 'memory'):
            memory_id = await self.memory.store_interaction(
                query=query,
                response=response["response"],
                context=context,
                metadata={
                    "query_type": self._classify_medical_query_type(query),
                    "safety_checked": True,
                    "patient_info_used": bool(patient_context),
                    "biological_processing": biological_result.get("biological_processing", False) if biological_result else False
                }
            )
            response["memory_id"] = memory_id
        
        # Build final response with biological data
        final_response = {
            "response": response["response"],
            "domain": classification.primary_domain,
            "confidence": classification.confidence,
            "safety_check": safety_result,
            "attention_focus": attention["primary_focus"],
            "medical_context": {
                "has_patient_info": bool(patient_context),
                "query_type": self._classify_medical_query_type(query),
                "patient_age": patient_context.get("age") if patient_context else None
            },
            "disclaimers": self.safety_protocols["disclaimers"][:2],  # First 2 disclaimers
            "quality_score": response.get("quality_score", 0.7),
            "generation_metadata": response.get("generation_metadata", {})
        }
        
        # Add biological data if available
        if biological_result and biological_result.get("biological_processing"):
            final_response["biological_analysis"] = response.get("biological_analysis", {})
            final_response["requires_professional_review"] = response.get("biological_analysis", {}).get(
                "requires_professional_review", False
            )
        
        return final_response
    
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

# Backward compatibility alias
EnhancedMedicalVNI = MedicalVNI
