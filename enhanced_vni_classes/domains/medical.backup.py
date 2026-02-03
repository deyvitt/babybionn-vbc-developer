# enhanced_vni_classes/domains/medical.py
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
from ..modules.attention import AttentionMechanism, AttentionType, AttentionWeight
from ..modules.classifier import DomainClassifier, ClassificationResult, Domain
from ..utils.logger import get_logger, VNILogger

logger = logging.getLogger(__name__)

class MedicalVNI(EnhancedBaseVNI, BaseKnowledgeLoader):
    """Specialized VNI for medical domain with enhanced safety protocols."""
    def __init__(self, 
                 vni_id: str,
                 name: str = "Medical Assistant",
                 description: str = "Specialized in medical information and health advice",
                 vni_config: Optional[Dict[str, Any]] = None,
                 auto_load_knowledge: bool = True):        
        """Initialize Medical VNI.
        Args:
            vni_id: Unique identifier for this VNI instance
            name: Display name for the VNI
            description: Description of the VNI's capabilities
            vni_config: Optional configuration dictionary"""
        
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
                "emergency_detection"
            ],
            vni_type='specialized'            
        )
        # Keep name and description for MedicalVNI's own use if needed
        self.vni_type = "medical" 
        self.name = name
        self.description = description
        
        # Medical-specific setup
        self.medical_knowledge = self._initialize_medical_knowledge()
        self.safety_protocols = self._initialize_safety_protocols()
        self.medical_keywords = self._setup_medical_keywords()
        # Load knowledge from files if enabled
        if auto_load_knowledge:
            self.load_domain_knowledge("medical")
        logger.info(f"Initialized MedicalVNI: {vni_id} ({name})")

        # Initialize base VNI - FIXED VERSION
        super().__init__(
            instance_id=vni_id,  # Pass vni_id as instance_id
            domain="medical",
            capabilities=capabilities,
            vni_config=vni_config or {}
        )

    def _safe_classify(self, query: str, use_context: bool = True):
        """
        Safely classify query, handling missing or misconfigured classifier.
        
        Args:
            query: Query to classify
            use_context: Whether to use context (not used in fallback)
            
        Returns:
            SimpleNamespace with classification results
        """
        try:
            # Try to use classifier if available and has classify method
            if hasattr(self, 'classifier') and self.classifier is not None:
                if hasattr(self.classifier, 'classify'):
                    return self.classifier.classify(query, use_context=use_context)
                elif hasattr(self.classifier, 'predict'):
                    # Handle sklearn-style classifier
                    prediction = self.classifier.predict([query])[0]
                    confidence = 0.7
                    if hasattr(self.classifier, 'predict_proba'):
                        proba = self.classifier.predict_proba([query])
                        confidence = float(proba.max())
                    
                    from types import SimpleNamespace
                    return SimpleNamespace(
                        primary_domain=str(prediction),
                        confidence=confidence,
                        subdomains=[str(prediction)]
                    )
        except Exception as e:
            logger.warning(f"Classifier error: {e}, using fallback classification")
    
        # Fallback classification based on keywords
        from types import SimpleNamespace
        query_lower = query.lower()
        
        # Check for medical keywords
        medical_keywords = ["pain", "fever", "cough", "headache", "doctor", "hospital", 
                           "symptom", "treatment", "medicine", "health", "medical",
                           "disease", "illness", "patient", "diagnosis", "prescription"]
        
        is_medical = any(keyword in query_lower for keyword in medical_keywords)
        
        return SimpleNamespace(
            primary_domain="medical" if is_medical else "general",
            confidence=0.7 if is_medical else 0.3,
            subdomains=["medical"] if is_medical else ["general"]
        )

    def _init_generation(self):
        """Initialize generation module for medical domain"""
        try:
            from ..modules.generation import EnhancedGenerationModule
        
            # Create generation module
            self.generator = EnhancedGenerationModule(
                domain="medical",
                enable_llm=True,
                model_name="microsoft/DialoGPT-medium"  # Default model
            )
            
            # Setup the generator
            success = self.generator.setup()
            if not success:
                logger.warning(f"Generation setup failed for medical domain")
                self.generation_enabled = False
            
        except Exception as e:
            logger.error(f"Medical generation initialization failed: {e}")
            self.generation_enabled = False
        
        # ENABLE GENERATION
        self.generation_enabled = True
                
        # Configure generation for medical domain
        if self.generation_enabled:
            self.configure_generation(
                #temperature=0.5,  # Low-medium for medical accuracy
                top_p=0.9,
                max_length=600
            )        

    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Implement abstract process method from EnhancedBaseVNI."""
        return self.process_query(query, context)
        
    def _initialize_medical_knowledge(self) -> Dict[str, Any]:
        """Initialize medical-specific knowledge base."""
        return {
            "common_conditions": {
                "hypertension": "High blood pressure, often asymptomatic but can lead to serious complications.",
                "diabetes": "Chronic condition affecting blood sugar regulation.",
                "influenza": "Viral respiratory infection with fever, cough, and body aches.",
                "asthma": "Chronic respiratory condition causing breathing difficulties.",
                "arthritis": "Inflammation of joints causing pain and stiffness.",
                "migraine": "Severe headache often accompanied by nausea and sensitivity to light.",
                "allergies": "Immune system reactions to substances like pollen, food, or medications."
            },
            "medical_terms": [
                "diagnosis", "prognosis", "etiology", "pathology", "symptomatology",
                "therapeutics", "pharmacology", "epidemiology", "immunology", "neurology"
            ],
            "safety_guidelines": [
                "Always consult with healthcare professionals for medical advice",
                "Information provided is for educational purposes only",
                "In emergencies, contact local emergency services immediately",
                "Do not discontinue prescribed medications without medical advice",
                "Report adverse reactions to healthcare providers promptly"
            ],
            "emergency_procedures": {
                "heart_attack": "Call emergency services immediately, have person sit down, give aspirin if not allergic",
                "stroke": "Remember FAST: Face drooping, Arm weakness, Speech difficulty, Time to call emergency",
                "allergic_reaction": "Use epinephrine auto-injector if available, call emergency services",
                "bleeding": "Apply direct pressure to wound, elevate if possible, call for help if severe"
            }
        }
    
    def _initialize_safety_protocols(self) -> Dict[str, Any]:
        """Initialize comprehensive medical safety protocols."""
        return {
            "disclaimers": [
                "I am an AI assistant and not a licensed medical professional.",
                "My responses should not be considered medical advice.",
                "Always consult with qualified healthcare providers for medical decisions.",
                "For emergencies, contact local emergency services immediately."
            ],
            "emergency_keywords": [
                "emergency", "urgent", "911", "heart attack", "stroke", "bleeding",
                "can't breathe", "unconscious", "severe pain", "overdose", "poison"
            ],
            "sensitive_topics": [
                "self-harm", "suicide", "abuse", "overdose", "eating disorder",
                "mental crisis", "trauma", "addiction"
            ],
            "high_risk_indicators": [
                "immediate danger", "life threatening", "need help now",
                "emergency room", "ambulance", "paramedic"
            ],
            "response_templates": {
                "emergency": "This appears to be an emergency. Please call emergency services immediately.",
                "sensitive": "This topic requires professional support. Please contact a healthcare provider.",
                "general": "I can provide general information, but please consult a doctor for medical advice."
            }
        }
    
    def _setup_medical_keywords(self) -> Dict[str, List[str]]:
        """Setup medical-specific keywords for classification."""
        return {
            "symptoms": [
                "pain", "fever", "cough", "headache", "nausea", "fatigue",
                "dizziness", "rash", "swelling", "bleeding", "shortness of breath"
            ],
            "conditions": [
                "cancer", "diabetes", "asthma", "arthritis", "hypertension",
                "migraine", "allergy", "infection", "virus", "bacteria"
            ],
            "treatments": [
                "medication", "surgery", "therapy", "vaccine", "antibiotic",
                "treatment", "cure", "remedy", "prescription", "dosage"
            ],
            "professionals": [
                "doctor", "nurse", "surgeon", "physician", "specialist",
                "therapist", "psychiatrist", "paramedic", "pharmacist"
            ],
            "facilities": [
                "hospital", "clinic", "emergency room", "pharmacy",
                "laboratory", "medical center", "health center"
            ]
        }
    
    def process_query(self, 
                      query: str, 
                      context: Optional[Dict[str, Any]] = None,
                      patient_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process medical query with comprehensive safety checks.
        
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
            return {
                "response": knowledge_results["content"],
                "confidence": knowledge_results["confidence"],
                "vni_instance": self.vni_id,
                "source": knowledge_results.get("source", "medical_knowledge_base"),
                "query_type": query_type,
                "response_type": "medical_knowledge",
                "safety_checked": True
            }
        
        # Use enhanced processing for complex queries
        return self._process_complex_medical_query(query, classification, patient_context)
    
    async def process_medical_query_async(self,
                                         query: str,
                                         patient_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Async version of medical query processing with web search and context building.
        
        Args:
            query: The medical query to process
            patient_context: Optional patient-specific information
            
        Returns:
            Dictionary with comprehensive medical response
        """
        # Safety check
        safety_result = self._check_safety(query)
        if safety_result["requires_caution"]:
            return self._generate_safety_response(query, safety_result)
        
        # Classify query
        classification = self._safe_classify(query)
        
        # Build medical context
        context = await self._build_medical_context(query, patient_context)
        
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
        
        # Record learning
        self._record_medical_interaction(query, response, classification)
        
        return {
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
    
    def _is_relevant_medical_query(self, query: str, threshold: float = 0.3) -> bool:
        """
        Check if query is relevant to medical domain.
        
        Args:
            query: Query to check
            threshold: Minimum relevance threshold
            
        Returns:
            True if query is medically relevant
        """
        query_lower = query.lower()
        
        # Check for medical keywords (from V1, enhanced)
        medical_indicators = 0
        
        # Check symptoms
        for symptom in self.medical_keywords["symptoms"]:
            if symptom in query_lower:
                medical_indicators += 1
        
        # Check conditions
        for condition in self.medical_keywords["conditions"]:
            if condition in query_lower:
                medical_indicators += 1
        
        # Check medical terms
        for term in self.medical_knowledge["medical_terms"]:
            if term in query_lower:
                medical_indicators += 1
        
        # Check classifier relevance
        classification = self._safe_classify(query, use_context=False)
        # Check if classification has the expected attributes
        if hasattr(classification, 'primary_domain') and hasattr(classification, 'confidence'):
            classifier_score = classification.confidence if classification.primary_domain == "medical" else 0
        else:
            # Handle case where classification doesn't have expected structure
            classifier_score = 0.5 if any(keyword in query_lower for keyword in self.medical_keywords["symptoms"] + self.medical_keywords["conditions"]) else 0
        
        # Combine scores
        keyword_score = min(1.0, medical_indicators * 0.2)
        total_score = (keyword_score + classifier_score) / 2
        
        return total_score >= threshold
    
    def _check_safety(self, query: str) -> Dict[str, Any]:
        """
        Check query for medical safety concerns.
        
        Args:
            query: Query to check
            
        Returns:
            Dictionary with safety assessment
        """
        requires_caution = False
        flags = []
        recommendations = []
        safety_level = "normal"
        
        query_lower = query.lower()
        
        # Check for emergency keywords (from V2)
        for keyword in self.safety_protocols["emergency_keywords"]:
            if keyword in query_lower:
                requires_caution = True
                flags.append("emergency_keyword")
                recommendations.append(f"Query contains '{keyword}' - recommend contacting emergency services")
                safety_level = "high"
        
        # Check for sensitive topics
        for topic in self.safety_protocols["sensitive_topics"]:
            if topic in query_lower:
                requires_caution = True
                flags.append("sensitive_topic")
                recommendations.append(f"Query relates to '{topic}' - handle with extreme caution")
                safety_level = "high"
        
        # Check for high-risk indicators
        for indicator in self.safety_protocols["high_risk_indicators"]:
            if indicator in query_lower:
                requires_caution = True
                flags.append("high_risk_indicator")
                recommendations.append("Query indicates high-risk situation")
                safety_level = "high"
        
        # Check for self-diagnosis requests
        if any(phrase in query_lower for phrase in ["what do i have", "do i have", "self diagnose", "am i sick"]):
            requires_caution = True
            flags.append("self_diagnosis")
            recommendations.append("Query appears to request self-diagnosis - advise professional consultation")
            safety_level = "medium"
        
        return {
            "requires_caution": requires_caution,
            "flags": flags,
            "recommendations": recommendations,
            "safety_level": safety_level,
            "query_length": len(query),
            "checked_at": datetime.now().isoformat()
        }
    
    def _generate_safety_response(self, 
                                 query: str, 
                                 safety_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a safety-focused medical response.
        
        Args:
            query: Original query
            safety_result: Safety assessment from _check_safety
            
        Returns:
            Safety-focused response
        """
        if safety_result["safety_level"] == "high":
            if "emergency_keyword" in safety_result["flags"]:
                response = (
                    "⚠️ **EMERGENCY ALERT** ⚠️\n\n"
                    "This sounds like it could be an emergency medical situation. "
                    "**Please call emergency services (911 or local emergency number) immediately** "
                    "or go to the nearest emergency room.\n\n"
                    "I cannot provide emergency medical advice. Your safety is the top priority."
                )
            else:
                response = (
                    "I understand you're asking about a serious medical concern. "
                    "This topic requires immediate professional support.\n\n"
                    "**Please reach out to:**\n"
                    "• A qualified healthcare provider\n"
                    "• Emergency services if urgent\n"
                    "• Crisis helpline for immediate support\n\n"
                    "I can provide general information, but this requires professional medical attention."
                )
        else:
            response = (
                "For accurate medical advice about this topic, please consult with a qualified healthcare professional.\n\n"
                "I can provide general information but cannot:\n"
                "• Diagnose medical conditions\n"
                "• Provide treatment advice\n"
                "• Replace professional medical consultation\n\n"
                "Always consult with your doctor for medical decisions."
            )
        
        return {
            "response": response,
            "domain": "medical",
            "confidence": 0.95,  # High confidence for safety responses
            "safety_check": safety_result,
            "attention_focus": "safety",
            "is_safety_response": True,
            "recommended_actions": safety_result["recommendations"],
            "response_type": "medical_safety"
        }
    
    def _query_medical_knowledge(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Query medical knowledge base for information.
        
        Args:
            query: Query to search for
            
        Returns:
            Knowledge entry if found, None otherwise
        """
        query_lower = query.lower()
        
        # Check common conditions
        for condition, info in self.medical_knowledge["common_conditions"].items():
            if condition in query_lower:
                return {
                    "content": f"**{condition.title()}**: {info}",
                    "confidence": 0.8,
                    "source": "medical_knowledge_base",
                    "concept": condition
                }
        
        # Check medical terms
        for term in self.medical_knowledge["medical_terms"]:
            if term in query_lower:
                return {
                    "content": f"The medical term **'{term}'** refers to a concept that should be explained by a healthcare professional.",
                    "confidence": 0.7,
                    "source": "medical_terminology",
                    "concept": term
                }
        
        # Query the VNI's knowledge base if available
        if hasattr(self, 'knowledge_base'):
            try:
                results = self.knowledge_base.search(query, domain="medical", limit=3)
                if results:
                    best_result = results[0]
                    return {
                        "content": best_result.get("content", ""),
                        "confidence": best_result.get("confidence", 0.6),
                        "source": best_result.get("source", "knowledge_base"),
                        "concept": best_result.get("concept", "medical_information")
                    }
            except Exception as e:
                logger.warning(f"Knowledge base query failed: {e}")
        
        return None
    
    async def _build_medical_context(self,
                                    query: str,
                                    patient_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build context for medical query processing.
        
        Args:
            query: Medical query
            patient_context: Optional patient information
            
        Returns:
            Context dictionary for response generation
        """
        context = {
            "knowledge": {"content": "", "confidence": 0.0},
            "web_results": [],
            "collaboration_results": [],
            "patient_info": {}
        }
        
        # Add medical knowledge from local database
        medical_info = self._get_medical_information(query)
        if medical_info:
            context["knowledge"] = {
                "content": medical_info,
                "confidence": 0.8,
                "source": "medical_knowledge_base"
            }
        
        # Web search for current medical information (if available)
        if hasattr(self, 'web_search'):
            try:
                web_results = await self.web_search.search(query, domain="medical", num_results=3)
                context["web_results"] = web_results.get("results", [])
            except Exception as e:
                logger.warning(f"Web search failed: {e}")
        
        # Add patient context if available
        if patient_context:
            context["patient_info"] = {
                "age": patient_context.get("age"),
                "gender": patient_context.get("gender"),
                "symptoms": patient_context.get("symptoms", []),
                "duration": patient_context.get("duration"),
                "existing_conditions": patient_context.get("existing_conditions", []),
                "medications": patient_context.get("medications", []),
                "allergies": patient_context.get("allergies", [])
            }
        
        return context
    
    def _get_medical_information(self, query: str) -> Optional[str]:
        """
        Get medical information from knowledge base.
        
        Args:
            query: Query to search for
            
        Returns:
            Medical information string if found
        """
        # Simple implementation - can be enhanced with actual knowledge base
        query_lower = query.lower()
        
        for condition, info in self.medical_knowledge["common_conditions"].items():
            if condition in query_lower:
                return f"Information about {condition}: {info}"
        
        for term in self.medical_knowledge["medical_terms"]:
            if term in query_lower:
                return f"The medical term '{term}' refers to a concept that should be explained by a healthcare professional."
        
        return None
    
    def _classify_medical_query_type(self, query: str) -> str:
        """
        Classify the type of medical query.
        
        Args:
            query: Medical query
            
        Returns:
            Query type classification
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["symptom", "pain", "hurt", "ache", "feel", "feeling"]):
            return "symptom_inquiry"
        elif any(word in query_lower for word in ["treatment", "cure", "medicine", "drug", "medication", "therapy"]):
            return "treatment_inquiry"
        elif any(word in query_lower for word in ["diagnosis", "what is", "condition", "disease", "illness"]):
            return "diagnosis_inquiry"
        elif any(word in query_lower for word in ["prevent", "avoid", "risk", "prevention", "protection"]):
            return "prevention_inquiry"
        elif any(word in query_lower for word in ["side effect", "adverse", "reaction", "interaction"]):
            return "safety_inquiry"
        elif any(word in query_lower for word in ["test", "exam", "scan", "diagnostic", "lab"]):
            return "testing_inquiry"
        else:
            return "general_inquiry"
    
    def _process_complex_medical_query(self,
                                      query: str,
                                      classification: ClassificationResult,
                                      patient_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process complex medical queries with enhanced features.
        
        Args:
            query: Medical query
            classification: Domain classification result
            patient_context: Optional patient information
            
        Returns:
            Comprehensive medical response
        """
        # Generate response using the text generator
        context = {
            "knowledge": {"content": self._get_medical_information(query) or "", "confidence": 0.6},
            "query_type": self._classify_medical_query_type(query),
            "domain": "medical"
        }
        
        response = self.generator.generate_response(
            query=query,
            context=context,
            style=GenerationStyle.DETAILED.value,
            #temperature=0.5,
            max_length=600
        )
        
        # Add medical disclaimers
        response_text = self._add_medical_disclaimers(response["response"])
        
        # Record interaction for learning
        self._record_medical_interaction_simple(query, response_text, classification)
        
        return {
            "response": response_text,
            "confidence": response.get("quality_score", 0.7) * classification.confidence,
            "vni_instance": self.vni_id,
            "domain": classification.primary_domain,
            "query_type": context["query_type"],
            "response_type": "medical_generated",
            "generation_metadata": response.get("generation_metadata", {}),
            "attention_used": "generation_focus",
            "disclaimers_included": True
        }
    
    def _add_medical_disclaimers(self, response: str) -> str:
        """
        Add medical disclaimers to response.
        
        Args:
            response: Original response text
            
        Returns:
            Response text with disclaimers added
        """
        disclaimer = (
            "\n\n---\n"
            "**Medical Disclaimer**: I am an AI assistant and not a licensed medical professional. "
            "This information is for educational purposes only and does not constitute medical advice. "
            "Always consult with a qualified healthcare provider for medical decisions. "
            "In emergencies, contact local emergency services immediately."
        )
        
        return response + disclaimer
    
    def _record_medical_interaction(self, 
                                   query: str, 
                                   response: Dict[str, Any],
                                   classification: ClassificationResult):
        """
        Record medical interaction for learning system.
        
        Args:
            query: Original query
            response: Response dictionary
            classification: Domain classification result
        """
        if hasattr(self, 'learning'):
            self.learning.record_interaction(
                interaction_id=hashlib.md5(f"{query}_{datetime.now().timestamp()}".encode()).hexdigest()[:16],
                prompt=query,
                response=response.get("response", ""),
                domain="medical",
                metadata={
                    "query_type": self._classify_medical_query_type(query),
                    "safety_checked": True,
                    "confidence": classification.confidence,
                    "response_type": response.get("response_type", "unknown"),
                    "patient_context_used": "patient_info" in response
                }
            )
    
    def _record_medical_interaction_simple(self,
                                          query: str,
                                          response: str,
                                          classification: ClassificationResult):
        """
        Simplified version for recording interactions.
        
        Args:
            query: Original query
            response: Response text
            classification: Domain classification result
        """
        if hasattr(self, 'learning'):
            self.learning.record_interaction(
                interaction_id=hashlib.md5(query.encode()).hexdigest()[:16],
                prompt=query,
                response=response,
                domain="medical",
                metadata={
                    "query_type": self._classify_medical_query_type(query),
                    "confidence": classification.confidence
                }
            )
    
    def get_medical_stats(self) -> Dict[str, Any]:
        """
        Get medical-specific statistics.
        
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
        
        return stats


# Backward compatibility alias
EnhancedMedicalVNI = MedicalVNI
