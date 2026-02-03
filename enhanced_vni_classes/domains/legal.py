"""Legal domain VNI implementation"""
from datetime import datetime
from .base_knowledge_loader import BaseKnowledgeLoader
from typing import Dict, Any, Optional, List
from ..core.base_vni import EnhancedBaseVNI
from ..core.capabilities import VNICapabilities, VNIType
from ..modules.generation import EnhancedGenerationModule, GenerationStyle # TextGenerator, GenerationModule
from ..modules.classifier import DomainClassifier
from ..modules.web_search import WebSearch
# from ..modules.attention import AttentionMechanism
from ..modules.learning_system import LearningSystem
from ..modules.knowledge_base import KnowledgeBase
from ..utils.logger import get_logger
import hashlib
# Import VNIMemory
from neuron.vni_memory import VNIMemory
from neuron.demoHybridAttention import DemoHybridAttention
from neuron.smart_activation_router import SmartActivationRouter, FunctionRegistry

logger = get_logger(__name__)


class LegalVNI(EnhancedBaseVNI, BaseKnowledgeLoader):
    """Legal domain VNI with biological systems integration"""
    def __init__(self, instance_id: str = "legal_001", vni_config: Dict[str, Any] = None, 
                 auto_load_knowledge: bool = True, enable_biological_systems: bool = True):
        # Extract domain from config if present
        self.domain = "legal"    
        config = vni_config or {}
        config.pop('domain', None)  # Remove domain to prevent conflicts
        
        # Set up capabilities
        capabilities = VNICapabilities(
            domains=["legal", "law", "regulatory", "compliance"],
            can_search=True,
            can_learn=True,
            can_collaborate=True,
            max_context_length=3500,
            special_abilities=["legal_terminology", "document_analysis", "regulation_info", "biological_systems_integration"],  # ADDED
            vni_type='specialized'
        ) 
        
        # ============ BIOLOGICAL SYSTEMS CONFIGURATION ============ ADDED
        self.enable_biological_systems = enable_biological_systems
        self.legal_attention_config = {
            'dim': 256,
            'num_heads': 8,
            'window_size': 256,
            'use_sliding': True,
            'use_global': True,
            'use_hierarchical': True,
            'global_token_ratio': 0.15,  # Higher for legal precision
            'memory_tokens': 25,
            'multi_modal': False,
            'semantic_weight': 0.8,  # Very high semantic focus for legal terms
            'precision_weight': 0.9  # Very high precision required for legal
        }
        
        self.legal_memory_config = {
            'short_term_capacity': 200,  # Higher for case details
            'long_term_capacity': 3000,
            'consolidation_threshold': 0.85,  # Higher threshold for legal accuracy
            'retention_period': 86400 * 30,  # 30 days for case history
            'priority_retention': True,
            'jurisdiction_aware': True
        }
        
        # Store the instance_id for reference if needed
        self.vni_id = instance_id
        
        # Biological systems case tracking
        self.legal_cases: Dict[str, Dict] = {}
        self.legal_guidelines = self._load_legal_guidelines()
        
        # ============ INITIALIZE BIOLOGICAL TOOLKITS ============ ADDED
        # Initialize Hybrid Attention Mechanism
        self.attention = DemoHybridAttention(
            dim=256,
            num_heads=8,
            window_size=256,
            use_sliding=True,
            use_global=True,
            use_hierarchical=True
        )
        
        # Initialize Smart Activation Router
        self.activation_router = SmartActivationRouter(
            vni_id=instance_id,
            domain="legal",
            input_dim=512,
            num_experts=3,           # Legal experts: contract, litigation, compliance
            expert_dim=256        )

        # Add LEGAL-SPECIFIC disclaimers (this is what's missing!)
        self.legal_disclaimers = [
            "I am an AI assistant and not a licensed attorney or legal professional.",
            "This information is for educational purposes only and does not constitute legal advice.",
            "Always consult with a qualified attorney for legal matters specific to your situation.",
            "Laws vary by jurisdiction - what applies in one area may not apply in another.",
            "This is not a substitute for professional legal consultation.",
            "For legal emergencies, contact appropriate legal authorities or legal aid services."
        ]
        
        # Initialize VNIMemory for legal domain
        self.memory = VNIMemory(
            domain="legal",
            vni_id=instance_id,
            memory_type="semantic"
        )        
        # Register legal-specific functions with activation router
        self._register_legal_functions() 

        # Load knowledge from files if enabled
        if auto_load_knowledge:
            self.load_domain_knowledge("legal")

        logger.info(f"LegalVNI instance {instance_id} initialized with capabilities: {capabilities}")

        # NOW set legal-specific attributes
        self.vni_id = instance_id
        self.vni_type = "legal"
        self.domain = "legal"
                
        # Initialize base VNI - FIXED VERSION  
        super().__init__(
            instance_id=instance_id,
            domain="legal",
            capabilities=capabilities,
            vni_config=vni_config
        )
        
        # Legal-specific setup
        self.legal_knowledge = self._setup_legal_knowledge()
        self.legal_disclaimers = self._setup_legal_disclaimers()
        self.web_search = WebSearch(vni_id=instance_id)

        # Initialize VNIMemory for legal domain
        self.memory = VNIMemory(
            domain="legal",  # Hardcoded here
            vni_id=instance_id,
            memory_type="semantic"
        )
        
        # Load knowledge from files if enabled
        if auto_load_knowledge:
            self.load_domain_knowledge("legal")
    
        logger.info(f"LegalVNI instance {instance_id} initialized with biological systems: {enable_biological_systems}")

    # ============ NEW: REGISTER LEGAL FUNCTIONS ============ ADDED
    def _register_legal_functions(self):
        """Register legal-specific functions with activation router"""
        if not hasattr(self, 'activation_router'):
            return
            
        # Register document analysis function
        try:
            self.activation_router.register_function(
                function_name="analyze_legal_document",
                function=self.analyze_legal_document,
                domain=self.domain,
                priority=1
            )
        except Exception as e:
            logger.error(f"Failed to register analyze_legal_document: {e}")
        
        # Register jurisdiction lookup function
        try:
            self.activation_router.register_function(
                function_name="lookup_jurisdiction_laws",
                function=self._lookup_jurisdiction_laws,
                domain="legal",
                priority=1
            )
        except Exception as e:
            logger.error(f"Failed to register lookup_jurisdiction_laws: {e}")
        
        # Register legal precedent search function
        try:
            self.activation_router.register_function(
                function_name="search_legal_precedents",
                function=self._search_legal_precedents,
                domain="legal",
                priority=1
            )
        except Exception as e:
            logger.error(f"Failed to register search_legal_precedents: {e}")
        
        # Register emergency response function
        try:
            self.activation_router.register_function(
                function_name="handle_legal_emergency",
                function=self._handle_legal_emergency,
                domain="legal",
                priority=2
            )
        except Exception as e:
            logger.error(f"Failed to register handle_legal_emergency: {e}")
        
        # Register case history analysis function
        try:
            self.activation_router.register_function(
                function_name="analyze_case_history",
                function=self._analyze_case_history,
                domain="legal",
                priority=1
            )
        except Exception as e:
            logger.error(f"Failed to register analyze_case_history: {e}")
        
        logger.info(f"Registered legal functions with activation router")

    
    def _lookup_jurisdiction_laws(self, jurisdiction: str, legal_topic: str) -> Dict[str, Any]:
        """Look up jurisdiction-specific laws"""
        # Implementation would query legal database
        return {
            "jurisdiction": jurisdiction,
            "topic": legal_topic,
            "laws_found": [],
            "notes": f"Laws in {jurisdiction} may vary. Consult local statutes."
        }
    
    def _search_legal_precedents(self, legal_issue: str, jurisdiction: str) -> Dict[str, Any]:
        """Search for legal precedents"""
        return {
            "issue": legal_issue,
            "jurisdiction": jurisdiction,
            "precedents_found": [],
            "recommendation": "Search legal databases or case law repositories"
        }
    
    def _handle_legal_emergency(self, emergency_type: str, urgency_level: str) -> Dict[str, Any]:
        """Handle legal emergencies"""
        return {
            "emergency_type": emergency_type,
            "urgency": urgency_level,
            "actions": [
                "Contact legal counsel immediately",
                "Document all details",
                "Preserve evidence",
                "Know your rights"
            ],
            "timestamp": datetime.now().isoformat()
        }
    
    def _analyze_case_history(self, case_id: str, analysis_type: str = "patterns") -> Dict[str, Any]:
        """Analyze case history"""
        case_data = self.legal_cases.get(case_id, {})
        return {
            "case_id": case_id,
            "analysis_type": analysis_type,
            "interaction_count": case_data.get("interaction_count", 0),
            "legal_factors": case_data.get("legal_factors_history", []),
            "previous_issues": case_data.get("previous_issues", [])
        }
    # ============ END OF NEW FUNCTIONS ============

    # ============ NEW BIOLOGICAL SYSTEMS METHODS ============ ADDED
    def _load_legal_guidelines(self) -> Dict[str, Any]:
        """Load legal guidelines for biological systems"""
        return {
            "emergency_protocols": {
                "activation_threshold": 0.8,
                "response_time_limit": 1.5,  # seconds - faster for legal emergencies
                "immediate_actions": ["refer_to_attorney", "contact_legal_aid", "suggest_emergency_services"]
            },
            "legal_categories": {
                "criminal": ["arrest", "charges", "bail", "trial", "sentencing"],
                "civil": ["lawsuit", "contract", "property", "damages", "injunction"],
                "family": ["divorce", "custody", "adoption", "support", "guardianship"],
                "business": ["incorporation", "contracts", "liability", "intellectual_property"],
                "administrative": ["regulations", "licenses", "permits", "compliance", "appeals"]
            },
            "biological_monitoring": {
                "risk_factors": ["immediate_deadline", "court_date", "statute_expiration", "financial_impact"],
                "alert_levels": {
                    "normal": 0.0,
                    "caution": 0.5,
                    "warning": 0.8,
                    "critical": 0.95
                }
            }
        }
    
    def _extract_legal_factors(self, query: str) -> List[Dict]:
        """Extract legal factors from query using biological systems analysis"""
        query_lower = query.lower()
        extracted_factors = []
        
        # Check each legal category
        for category, factors in self.legal_guidelines["legal_categories"].items():
            for factor in factors:
                if factor in query_lower:
                    extracted_factors.append({
                        "factor": factor,
                        "category": category,
                        "urgency": "high" if factor in ["arrest", "eviction", "deadline"] else "medium"
                    })
        
        # Check for legal emergencies
        emergency_terms = ["arrested", "eviction", "lawsuit filed", "court tomorrow", "deadline today"]
        for term in emergency_terms:
            if term in query_lower:
                extracted_factors.append({
                    "factor": "legal_emergency",
                    "category": "emergency",
                    "urgency": "critical"
                })
        
        return extracted_factors
    
    def _get_case_history(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get case history from context for biological systems"""
        if not context or "case_id" not in context:
            return {"available": False, "history": {}}
        
        case_id = context.get("case_id")
        history = self.legal_cases.get(case_id, {})
        
        return {
            "available": bool(history),
            "history": history,
            "case_id": case_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def process_with_biological_systems(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process query with legal biological systems integration"""
        if not self.enable_biological_systems:
            return {
                "biological_processing": False,
                "reason": "Biological systems disabled"
            }
        
        logger.debug(f"⚖️ Legal biological processing: {query[:50]}...")
        
        try:
            # Extract legal insights
            legal_factors = self._extract_legal_factors(query)
            case_history = self._get_case_history(context)
            
            # Check for legal emergencies
            emergency_keywords = self.legal_disclaimers.get("emergency_topics", [])
            is_emergency = any(keyword in query.lower() for keyword in emergency_keywords)
            
            # Calculate activation level based on legal factors
            activation_level = self._calculate_legal_activation(query, legal_factors, case_history)
            
            # Determine urgency level
            urgency_level = self._determine_legal_urgency(activation_level, legal_factors, is_emergency)
            
            # Generate legal insights
            legal_insights = {
                "legal_factors_detected": len(legal_factors),
                "legal_categories": list(set([f["category"] for f in legal_factors])),
                "activation_level": activation_level,
                "urgency_level": urgency_level,
                "requires_legal_counsel": activation_level > self.legal_guidelines["emergency_protocols"]["activation_threshold"],
                "case_history_available": case_history["available"],
                "biological_timestamp": datetime.now().isoformat()
            }
            
            # Update case context if case ID exists
            if context and "case_id" in context:
                self._update_case_context(context["case_id"], query, legal_factors, legal_insights)
            
            return {
                "biological_processing": True,
                "biological_insights": legal_insights,
                "activation_result": {
                    "activation_level": activation_level,
                    "urgency": urgency_level,
                    "recommended_actions": self._get_legal_recommendations(urgency_level)
                },
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Legal biological processing failed: {e}")
            return {
                "biological_processing": False,
                "error": str(e),
                "success": False
            }
    
    def _calculate_legal_activation(self, query: str, legal_factors: List[Dict], 
                                  case_history: Dict[str, Any]) -> float:
        """Calculate legal activation level based on query analysis"""
        base_activation = 0.4  # Higher base for legal queries
        
        # Increase based on legal factor count
        factor_bonus = min(len(legal_factors) * 0.15, 0.3)
        base_activation += factor_bonus
        
        # Increase based on factor severity
        critical_factors = ["arrest", "eviction", "court_tomorrow", "deadline_today"]
        for factor in legal_factors:
            if factor["factor"] in critical_factors:
                base_activation += 0.3
        
        # Increase based on emergency keywords
        emergency_keywords = self.legal_disclaimers.get("emergency_topics", [])
        for keyword in emergency_keywords:
            if keyword in query.lower():
                base_activation += 0.4
                break
        
        # Consider case history
        if case_history.get("available"):
            historical_issues = case_history.get("history", {}).get("previous_issues", [])
            if historical_issues:
                base_activation += 0.2
        
        # Cap at 1.0
        return min(base_activation, 1.0)
    
    def _determine_legal_urgency(self, activation_level: float, legal_factors: List[Dict], 
                               is_emergency: bool) -> str:
        """Determine urgency level based on legal analysis"""
        if is_emergency or activation_level >= 0.95:
            return "critical"
        elif activation_level >= 0.8:
            return "high"
        elif activation_level >= 0.5:
            return "medium"
        else:
            return "low"
    
    def _get_legal_recommendations(self, urgency_level: str) -> List[str]:
        """Get recommended actions based on legal urgency"""
        actions = {
            "critical": [
                "Contact an attorney immediately",
                "If arrested, exercise your right to remain silent",
                "Document everything related to the situation",
                "Contact legal aid services if unable to afford attorney"
            ],
            "high": [
                "Schedule consultation with attorney as soon as possible",
                "Gather all relevant documents and evidence",
                "Review applicable laws and regulations",
                "Avoid making statements without legal counsel"
            ],
            "medium": [
                "Research legal requirements in your jurisdiction",
                "Consider initial consultation with attorney",
                "Document your situation thoroughly",
                "Review similar case outcomes"
            ],
            "low": [
                "General legal education and research",
                "Monitor deadlines and requirements",
                "Consult legal self-help resources",
                "Consider online legal services for simple matters"
            ]
        }
        return actions.get(urgency_level, ["Consult legal professional"])
    
    def _update_case_context(self, case_id: str, query: str, 
                            legal_factors: List[Dict], insights: Dict[str, Any]):
        """Update case context for biological tracking"""
        if case_id not in self.legal_cases:
            self.legal_cases[case_id] = {
                "first_interaction": datetime.now().isoformat(),
                "legal_factors_history": [],
                "interaction_count": 0,
                "previous_issues": [],
                "jurisdiction": None
            }
        
        case_data = self.legal_cases[case_id]
        case_data["interaction_count"] += 1
        case_data["last_interaction"] = datetime.now().isoformat()
        
        # Add legal factors to history
        for factor in legal_factors:
            case_data["legal_factors_history"].append({
                "factor": factor["factor"],
                "category": factor["category"],
                "timestamp": datetime.now().isoformat(),
                "query_context": query[:100]
            })
        
        # Keep only recent history (last 50 entries)
        if len(case_data["legal_factors_history"]) > 50:
            case_data["legal_factors_history"] = case_data["legal_factors_history"][-50:]
        
        # Update previous issues if high urgency
        if insights.get("urgency_level") in ["high", "critical"]:
            case_data["previous_issues"].append({
                "timestamp": datetime.now().isoformat(),
                "urgency": insights["urgency_level"],
                "legal_factors": [f["factor"] for f in legal_factors]
            })
    
    def _enhance_legal_response(self, result: Dict[str, Any], query: str, 
                               context: Optional[Dict[str, Any]] = None,
                               biological_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhance response with legal biological insights"""
        enhanced_result = result.copy()
        
        if biological_result and biological_result.get("biological_processing"):
            # Add biological insights to response
            insights = biological_result.get("biological_insights", {})
            activation = biological_result.get("activation_result", {})
            
            enhanced_result["biological_analysis"] = {
                "performed": True,
                "legal_factors_detected": insights.get("legal_factors_detected", 0),
                "urgency_level": insights.get("urgency_level", "unknown"),
                "activation_level": activation.get("activation_level", 0.0),
                "requires_legal_counsel": activation.get("activation_level", 0) > 0.8
            }
            
            # Add urgency-based guidance
            if enhanced_result["biological_analysis"]["requires_legal_counsel"]:
                guidance = "⚖️ **LEGAL COUNSEL STRONGLY RECOMMENDED** ⚖️\n"
                guidance += "Based on legal factor analysis, this situation may require professional legal advice.\n"
                guidance += "Please consult with a licensed attorney for proper legal guidance.\n\n"
                
                # Prepend guidance to response
                enhanced_result["response"] = guidance + enhanced_result.get("response", "")
                
                # Add disclaimer
                if "disclaimers" not in enhanced_result:
                    enhanced_result["disclaimers"] = []
                enhanced_result["disclaimers"].append(
                    "Biological systems analysis indicates potential legal complexity requiring professional counsel."
                )
        
        return enhanced_result
    # ============ END OF BIOLOGICAL SYSTEMS METHODS ============

    async def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Implement abstract process method from EnhancedBaseVNI."""
        return await self.process_query(query, **({'context': context} if context else {}))
        
    def _setup_legal_knowledge(self) -> Dict[str, Any]:
        """Initialize legal-specific knowledge."""
        return {
            "legal_areas": {
                "contract_law": "Governs agreements between parties.",
                "criminal_law": "Deals with crimes and punishments.",
                "civil_law": "Addresses disputes between individuals/organizations.",
                "intellectual_property": "Protects creations of the mind.",
                "family_law": "Deals with family relationships and domestic matters.",
                "employment_law": "Governs employer-employee relationships.",
                "property_law": "Concerns real property and personal property rights.",
                "tax_law": "Governs taxation at various levels."
            },
            "legal_terms": [
                "jurisdiction", "precedent", "liability", "litigation",
                "arbitration", "compliance", "regulation", "statute",
                "tort", "negligence", "consideration", "due_diligence"
            ],
            "common_documents": [
                "contract", "agreement", "will", "power_of_attorney",
                "incorporation", "lease", "license", "affidavit",
                "complaint", "summons", "discovery", "deposition"
            ],
            "legal_principles": {
                "presumption_of_innocence": "One is considered innocent until proven guilty.",
                "due_process": "Fair treatment through the judicial system.",
                "statute_of_limitations": "Time limit for initiating legal proceedings.",
                "burden_of_proof": "Obligation to prove allegations in court."
            }
        }
    
    def _setup_legal_disclaimers(self) -> Dict[str, Any]:
        """Initialize legal disclaimers."""
        return {
            "disclaimers": [
                "I am an AI assistant and not a licensed attorney.",
                "My responses do not constitute legal advice.",
                "Always consult with a qualified legal professional for legal matters.",
                "Laws vary by jurisdiction and are subject to change.",
                "This information is for educational purposes only."
            ],
            "jurisdiction_keywords": ["state", "federal", "local", "international", "jurisdiction", "county", "municipal"],
            "emergency_topics": ["arrest", "lawsuit", "eviction", "legal emergency", "detained", "court date tomorrow", "urgent legal help"],
            "specific_legal_advice_indicators": [
                "should I", "what should I do", "can I sue", "is it legal to",
                "am I required to", "do I have to", "must I", "am I liable for"
            ]
        }
    
    async def process_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Main method to process legal queries with biological systems integration."""
        jurisdiction = kwargs.get("jurisdiction", None)
        return await self.process_legal_query(query, jurisdiction)
    
    async def process_legal_query(self, 
                                 query: str,
                                 jurisdiction: Optional[str] = None) -> Dict[str, Any]:
        """Process a legal query with appropriate disclaimers and biological systems."""
        
        # Check for emergency topics
        emergency_check = self._check_emergency_topics(query)
        if emergency_check["is_emergency"]:
            response = self._generate_emergency_response(query, emergency_check)
            # Enhance with biological systems
            if self.enable_biological_systems:
                biological_result = self.process_with_biological_systems(query, {"jurisdiction": jurisdiction})
                response = self._enhance_legal_response(response, query, {"jurisdiction": jurisdiction}, biological_result)
            return response
        
        # Check if query asks for specific legal advice
        advice_check = self._check_specific_legal_advice(query)
        if advice_check["requires_caution"]:
            response = self._generate_caution_response(query, advice_check)
            # Enhance with biological systems
            if self.enable_biological_systems:
                biological_result = self.process_with_biological_systems(query, {"jurisdiction": jurisdiction})
                response = self._enhance_legal_response(response, query, {"jurisdiction": jurisdiction}, biological_result)
            return response
        
        # Classify query
        classification = self.classifier.classify(query)
        
        # Build context
        context = await self._build_legal_context(query, jurisdiction)
        
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
                additional_context={"jurisdiction": jurisdiction} if jurisdiction else None
            )
            if memory_context:
                context["memory_context"] = memory_context
                
        # Compute attention
        attention = self.attention.compute_attention(query, context)
        
        # Generate response using generation module
        response_text = self.generation_module.generate(
            query=query,
            context=context,
            style=GenerationStyle.FORMAL.value,
            #temperature=0.3,
            max_length=400
        )
        
        # Add legal disclaimers
        response_text = self._add_legal_disclaimers(response_text, jurisdiction)
        
        # Record learning
        self._record_legal_interaction(query, response_text, classification, jurisdiction)
        
        # Store interaction in memory
        if hasattr(self, 'memory'):
            memory_id = await self.memory.store_interaction(
                query=query,
                response=response_text,
                context=context,
                metadata={
                    "jurisdiction": jurisdiction,
                    "query_type": self._classify_legal_query_type(query),
                    "legal_area": self._identify_legal_area(query),
                    "emergency": emergency_check["is_emergency"],
                    "biological_processing": biological_result.get("biological_processing", False) if biological_result else False
                }
            )

        # Build response
        response = {
            "response": response_text,
            "domain": classification.primary_domain,
            "confidence": classification.confidence,
            "emergency_check": emergency_check,
            "advice_check": advice_check,
            "attention_focus": attention.get("primary_focus", "general"),
            "legal_context": {
                "jurisdiction_specified": bool(jurisdiction),
                "query_type": self._classify_legal_query_type(query),
                "legal_area": self._identify_legal_area(query)
            },
            "disclaimers": self.legal_disclaimers["disclaimers"]
        }
        
        # Enhance with biological systems if enabled
        if self.enable_biological_systems and biological_result:
            response = self._enhance_legal_response(response, query, {"jurisdiction": jurisdiction}, biological_result)
        
        return response
    
    async def _build_legal_context(self, 
                                  query: str,
                                  jurisdiction: Optional[str] = None) -> Dict[str, Any]:
        """Build context for legal query processing with memory integration."""
        context = {
            "knowledge": {"content": "", "confidence": 0.0},
            "web_results": [],
            "collaboration_results": [],
            "query_analysis": {},
            "memory_context": {}
        }
        
        # Analyze query
        context["query_analysis"] = {
            "query_type": self._classify_legal_query_type(query),
            "legal_area": self._identify_legal_area(query),
            "contains_jurisdiction_reference": bool(jurisdiction) or any(
                kw in query.lower() for kw in self.legal_disclaimers["jurisdiction_keywords"]
            )
        }
        
        # Retrieve relevant memory if available
        if hasattr(self, 'memory'):
            try:
                memory_context = await self.memory.retrieve_relevant_memory(
                    query=query,
                    additional_context={"jurisdiction": jurisdiction} if jurisdiction else None
                )
                if memory_context:
                    context["memory_context"] = memory_context
            except Exception as e:
                logger.warning(f"Memory retrieval failed: {e}")
        
        # Add legal knowledge
        legal_info = self._get_legal_information(query)
        if legal_info:
            context["knowledge"] = {
                "content": legal_info,
                "confidence": 0.7,
                "source": "legal_knowledge_base"
            }
        
        # Web search for current legal information (if configured)
        if hasattr(self, 'web_search'):
            try:
                search_query = query
                if jurisdiction:
                    search_query = f"{query} {jurisdiction}"
                
                web_results = await self.web_search.search(search_query, domain="legal", num_results=3)
                context["web_results"] = web_results.get("results", [])
            except Exception as e:
                logger.warning(f"Web search failed: {e}")
        
        # Add jurisdiction context if specified
        if jurisdiction:
            context["jurisdiction"] = {
                "specified": jurisdiction,
                "note": f"Legal information may vary by jurisdiction. This response considers {jurisdiction} context where applicable."
            }
        
        return context
    
    def _record_legal_interaction(self, 
                                 query: str, 
                                 response: str,
                                 classification,
                                 jurisdiction: Optional[str] = None):
        """Record legal interaction for learning with memory integration."""
        if hasattr(self, 'learning'):
            try:
                self.learning.record_interaction(
                    interaction_id=hashlib.md5(query.encode()).hexdigest()[:16],
                    prompt=query,
                    response=response,
                    domain="legal",
                    metadata={
                        "query_type": self._classify_legal_query_type(query),
                        "jurisdiction": jurisdiction,
                        "confidence": classification.confidence,
                        "legal_area": self._identify_legal_area(query)
                    }
                )
            except Exception as e:
                logger.error(f"Failed to record legal interaction: {e}")
        
        # Also store in VNIMemory
        if hasattr(self, 'memory'):
            try:
                import asyncio
                asyncio.create_task(
                    self.memory.store_interaction(
                        query=query,
                        response=response,
                        context={
                            "jurisdiction": jurisdiction,
                            "classification": classification.primary_domain,
                            "confidence": classification.confidence
                        },
                        metadata={
                            "vni_id": self.vni_id,
                            "domain": "legal",
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to store interaction in memory: {e}")

    async def analyze_legal_document(self, document_text: str) -> Dict[str, Any]:
        """Analyze a legal document (basic analysis) with memory integration."""
        
        # Store document analysis in memory
        if hasattr(self, 'memory'):
            await self.memory.store_interaction(
                query="document_analysis",
                response="Legal document analyzed",
                context={"document_text": document_text[:500]},  # Store first 500 chars
                metadata={
                    "document_type": self._identify_document_type(document_text),
                    "word_count": len(document_text.split()),
                    "analysis_type": "document_analysis"
                }
            )
        
        return {
            "analysis": "Basic document analysis performed",
            "document_type": self._identify_document_type(document_text),
            "key_terms_found": self._extract_legal_terms(document_text),
            "recommendation": "Have this document reviewed by a qualified attorney",
            "word_count": len(document_text.split()),
            "memory_stored": True if hasattr(self, 'memory') else False
        }
    
    def get_legal_stats(self) -> Dict[str, Any]:
        """Get legal-specific statistics including memory stats and biological systems."""
        stats = {
            "vni_id": self.vni_id,
            "legal_areas": len(self.legal_knowledge["legal_areas"]),
            "legal_terms": len(self.legal_knowledge["legal_terms"]),
            "common_documents": len(self.legal_knowledge["common_documents"]),
            "legal_principles": len(self.legal_knowledge["legal_principles"]),
            "biological_systems": {  # ADDED: Biological systems stats
                "enabled": self.enable_biological_systems,
                "cases_tracked": len(self.legal_cases),
                "legal_guidelines": len(self.legal_guidelines),
                "attention_config": self.legal_attention_config,
                "memory_config": self.legal_memory_config
            }
        }
        
        # Add memory stats if available
        if hasattr(self, 'memory'):
            try:
                memory_stats = self.memory.get_stats()
                stats["memory"] = {
                    "total_interactions": memory_stats.get("total_interactions", 0),
                    "memory_type": memory_stats.get("memory_type", "unknown"),
                    "jurisdiction_memories": memory_stats.get("jurisdiction_count", 0)
                }
            except Exception as e:
                logger.warning(f"Could not get memory stats: {e}")
                stats["memory"] = {"error": "unavailable"}
        
        return stats

    def _check_emergency_topics(self, query: str) -> Dict[str, Any]:
        """Check for legal emergency topics."""
        is_emergency = False
        flags = []
        
        query_lower = query.lower()
        
        for topic in self.legal_disclaimers["emergency_topics"]:
            if topic in query_lower:
                is_emergency = True
                flags.append(topic)
        
        return {
            "is_emergency": is_emergency,
            "flags": flags,
            "recommendation": "Consult an attorney immediately" if is_emergency else "Proceed with general guidance"
        }
    
    def _check_specific_legal_advice(self, query: str) -> Dict[str, Any]:
        """Check if query asks for specific legal advice."""
        requires_caution = False
        indicators_found = []
        
        query_lower = query.lower()
        
        for indicator in self.legal_disclaimers["specific_legal_advice_indicators"]:
            if indicator in query_lower:
                requires_caution = True
                indicators_found.append(indicator)
        
        return {
            "requires_caution": requires_caution,
            "indicators": indicators_found,
            "note": "This query appears to request specific legal advice which I cannot provide."
        }
    
    def _generate_emergency_response(self, 
                                    query: str, 
                                    emergency_check: Dict[str, Any]) -> Dict[str, Any]:
        """Generate emergency response for legal queries."""
        
        base_response = "I understand you're asking about a legal matter that may be time-sensitive. "
        
        if emergency_check["is_emergency"]:
            base_response += (
                "This appears to be a legal emergency. "
                "You should consult with a qualified attorney immediately. "
                "If you cannot afford an attorney, you may be eligible for legal aid services. "
                "I cannot provide emergency legal advice.\n\n"
            )
            
            # Add specific recommendations based on flags
            if "arrest" in query.lower() or "detained" in query.lower():
                base_response += "If you or someone you know has been arrested:\n"
                base_response += "1. Exercise your right to remain silent\n"
                base_response += "2. Request an attorney immediately\n"
                base_response += "3. Do not discuss the case without your attorney present\n"
            
            if "eviction" in query.lower():
                base_response += "For eviction matters:\n"
                base_response += "1. Review your local tenant rights\n"
                base_response += "2. Contact local housing authority\n"
                base_response += "3. Seek legal aid for housing issues\n"
        
        return {
            "response": base_response,
            "domain": "legal",
            "confidence": 0.9,
            "emergency_check": emergency_check,
            "attention_focus": "emergency",
            "is_emergency_response": True,
            "recommended_actions": [
                "Contact a licensed attorney immediately",
                "Check with local bar association for referrals",
                "Consider legal aid if financially eligible",
                "For emergencies involving safety, contact local authorities"
            ]
        }
    
    def _generate_caution_response(self, query: str, advice_check: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cautious response for queries asking specific legal advice."""
        
        response = (
            "I understand you're asking about a legal situation. "
            "I need to emphasize that I cannot provide specific legal advice. "
            "What I can do is explain general legal concepts and principles.\n\n"
        )
        
        # Try to provide general information without giving advice
        legal_info = self._get_legal_information(query)
        if legal_info:
            response += f"Here's some general information that might be helpful:\n{legal_info}\n\n"
        
        response += (
            "For advice specific to your situation, you should:\n"
            "1. Consult with a licensed attorney in your jurisdiction\n"
            "2. Provide them with all relevant details\n"
            "3. Follow their professional guidance\n\n"
        )
        
        return {
            "response": response,
            "domain": "legal",
            "confidence": 0.8,
            "advice_check": advice_check,
            "attention_focus": "caution",
            "is_caution_response": True
        }
 
    def _get_legal_information(self, query: str) -> Optional[str]:
        """Get legal information from knowledge base."""
        query_lower = query.lower()
        
        # Check legal areas
        for area, info in self.legal_knowledge["legal_areas"].items():
            area_clean = area.replace('_', ' ')
            if area_clean in query_lower or area in query_lower:
                return f"Information about {area_clean}: {info}"
        
        # Check legal principles
        for principle, explanation in self.legal_knowledge["legal_principles"].items():
            principle_clean = principle.replace('_', ' ')
            if principle_clean in query_lower:
                return f"Legal principle: {principle_clean}\nExplanation: {explanation}"
        
        # Check legal terms
        for term in self.legal_knowledge["legal_terms"]:
            if term in query_lower:
                return f"The legal term '{term}' refers to a concept in law. For a precise definition applicable to your situation, consult legal resources or an attorney."
        
        # Check common documents
        for doc in self.legal_knowledge["common_documents"]:
            if doc in query_lower:
                return f"A {doc} is a legal document that typically requires professional preparation. General information about {doc}s can be found in legal form books or through legal professionals."
        
        return None
