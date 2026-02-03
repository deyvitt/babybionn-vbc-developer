"""
Legal domain VNI implementation
"""
from .base_knowledge_loader import BaseKnowledgeLoader
from typing import Dict, Any, Optional, List
from ..core.base_vni import EnhancedBaseVNI
from ..core.capabilities import VNICapabilities, VNIType
from ..modules.generation import EnhancedGenerationModule, GenerationStyle # TextGenerator, GenerationModule
from ..modules.classifier import DomainClassifier
from ..modules.web_search import WebSearch
from ..modules.attention import AttentionMechanism
from ..modules.learning_system import LearningSystem
from ..modules.knowledge_base import KnowledgeBase
from ..utils.logger import get_logger
import hashlib

logger = get_logger(__name__)


class LegalVNI(EnhancedBaseVNI, BaseKnowledgeLoader):
    """Legal domain VNI"""
    def __init__(self, instance_id: str = "legal_001", vni_config: Dict[str, Any] = None, auto_load_knowledge: bool = True):
        # Set up capabilities
        capabilities = VNICapabilities(
            domains=["legal", "law", "regulatory", "compliance"],
            can_search=True,
            can_learn=True,
            can_collaborate=True,
            max_context_length=3500,
            special_abilities=["legal_terminology", "document_analysis", "regulation_info"],
            vni_type='specialized'
        )   
        
        # Add LEGAL-SPECIFIC disclaimers (this is what's missing!)
        self.legal_disclaimers = [
            "I am an AI assistant and not a licensed attorney or legal professional.",
            "This information is for educational purposes only and does not constitute legal advice.",
            "Always consult with a qualified attorney for legal matters specific to your situation.",
            "Laws vary by jurisdiction - what applies in one area may not apply in another.",
            "This is not a substitute for professional legal consultation.",
            "For legal emergencies, contact appropriate legal authorities or legal aid services."
        ]
       
        # Load knowledge from files if enabled
        if auto_load_knowledge:
            self.load_domain_knowledge("legal")

        logger.info(f"LegalVNI instance {instance_id} initialized with capabilities: {capabilities}")
        
        # Initialize base VNI - FIXED VERSION  
        super().__init__(
            instance_id=instance_id,
            domain="legal",
            capabilities=capabilities,
            vni_config=vni_config or {}
        )
    def _init_generation(self):
        """Initialize generation module for legal domain"""
        try:
            from ..modules.generation import EnhancedGenerationModule
            
            # Create generation module
            self.generator = EnhancedGenerationModule(
                domain="legal",
                enable_llm=True,
                model_name="microsoft/DialoGPT-medium"  # Default model
            )
            
            # Setup the generator
            success = self.generator.setup()
            if not success:
                logger.warning(f"Generation setup failed for legal domain")
                self.generation_enabled = False
            
        except Exception as e:
            logger.error(f"Legal generation initialization failed: {e}")
            self.generation_enabled = False

        # ENABLE GENERATION
        self.generation_enabled = True
                
        # Configure generation for legal domain
        if self.generation_enabled:
            self.configure_generation(
                #temperature=0.3,  # Very low for legal precision
                top_p=0.9,
                max_length=400
            )
    
        # Legal-specific setup
        self.vni_type = "legal"
        self.legal_knowledge = self._setup_legal_knowledge()
        self.legal_disclaimers = self._setup_legal_disclaimers()
        self.generation_module = EnhancedGenerationModule(domain="legal")
        
        # Initialize modules - no need. Already initialized in base class
        # self.classifier = DomainClassifier()
        # self.generator = EnhancedGenerationModule()
        self.web_search = WebSearch(vni_id=self.instance_id)
        # self.attention = AttentionMechanism()
        # self.learning = LearningSystem()

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
        """Main method to process legal queries."""
        jurisdiction = kwargs.get("jurisdiction", None)
        return await self.process_legal_query(query, jurisdiction)
    
    async def process_legal_query(self, 
                                 query: str,
                                 jurisdiction: Optional[str] = None) -> Dict[str, Any]:
        """Process a legal query with appropriate disclaimers."""
        
        # Check for emergency topics
        emergency_check = self._check_emergency_topics(query)
        if emergency_check["is_emergency"]:
            return self._generate_emergency_response(query, emergency_check)
        
        # Check if query asks for specific legal advice
        advice_check = self._check_specific_legal_advice(query)
        if advice_check["requires_caution"]:
            return self._generate_caution_response(query, advice_check)
        
        # Classify query
        classification = self.classifier.classify(query)
        
        # Build context
        context = await self._build_legal_context(query, jurisdiction)
        
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
        
        return {
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
    
    async def _build_legal_context(self, 
                                  query: str,
                                  jurisdiction: Optional[str] = None) -> Dict[str, Any]:
        """Build context for legal query processing."""
        context = {
            "knowledge": {"content": "", "confidence": 0.0},
            "web_results": [],
            "collaboration_results": [],
            "query_analysis": {}
        }
        
        # Analyze query
        context["query_analysis"] = {
            "query_type": self._classify_legal_query_type(query),
            "legal_area": self._identify_legal_area(query),
            "contains_jurisdiction_reference": bool(jurisdiction) or any(
                kw in query.lower() for kw in self.legal_disclaimers["jurisdiction_keywords"]
            )
        }
        
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
    
    def _identify_legal_area(self, query: str) -> str:
        """Identify the primary legal area of the query."""
        query_lower = query.lower()
        
        area_mapping = {
            "contract": "contract_law",
            "agreement": "contract_law",
            "criminal": "criminal_law",
            "crime": "criminal_law",
            "civil": "civil_law",
            "lawsuit": "civil_law",
            "intellectual property": "intellectual_property",
            "copyright": "intellectual_property",
            "trademark": "intellectual_property",
            "patent": "intellectual_property",
            "family": "family_law",
            "divorce": "family_law",
            "custody": "family_law",
            "employment": "employment_law",
            "workplace": "employment_law",
            "property": "property_law",
            "real estate": "property_law",
            "tax": "tax_law",
            "irs": "tax_law"
        }
        
        for keyword, area in area_mapping.items():
            if keyword in query_lower:
                return area
        
        return "general_law"
    
    def _classify_legal_query_type(self, query: str) -> str:
        """Classify the type of legal query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["contract", "agreement", "document", "form"]):
            return "document_inquiry"
        elif any(word in query_lower for word in ["right", "law", "regulation", "statute", "legal requirement"]):
            return "law_inquiry"
        elif any(word in query_lower for word in ["sue", "lawsuit", "case", "court", "litigation", "settlement"]):
            return "litigation_inquiry"
        elif any(word in query_lower for word in ["company", "business", "incorporation", "llc", "corporation"]):
            return "business_law_inquiry"
        elif any(word in query_lower for word in ["definition", "meaning of", "what is"]):
            return "definition_inquiry"
        elif any(word in query_lower for word in ["procedure", "process", "how to", "steps"]):
            return "procedure_inquiry"
        else:
            return "general_inquiry"
    
    def _add_legal_disclaimers(self, response: str, jurisdiction: Optional[str] = None) -> str:
        """Add legal disclaimers to response."""
        disclaimer = (
            "\n\n**Legal Disclaimer**: I am an AI assistant and not a licensed attorney. "
            "This information is for general guidance only and does not constitute legal advice. "
            "Laws vary by jurisdiction and are subject to change. "
            "Always consult with a qualified legal professional for specific legal matters."
        )
        
        if jurisdiction:
            disclaimer += f" This response considers {jurisdiction} context where applicable."
        
        return response + disclaimer
    
    def _record_legal_interaction(self, 
                                 query: str, 
                                 response: str,
                                 classification,
                                 jurisdiction: Optional[str] = None):
        """Record legal interaction for learning."""
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

    async def analyze_legal_document(self, document_text: str) -> Dict[str, Any]:
        """Analyze a legal document (basic analysis)."""
        return {
            "analysis": "Basic document analysis performed",
            "document_type": self._identify_document_type(document_text),
            "key_terms_found": self._extract_legal_terms(document_text),
            "recommendation": "Have this document reviewed by a qualified attorney",
            "word_count": len(document_text.split())
        }
    
    def _identify_document_type(self, document_text: str) -> str:
        """Identify the type of legal document."""
        text_lower = document_text.lower()
        
        if any(term in text_lower for term in ["party a", "party b", "hereinafter", "agreement"]):
            return "contract"
        elif "last will and testament" in text_lower:
            return "will"
        elif "power of attorney" in text_lower:
            return "power_of_attorney"
        elif any(term in text_lower for term in ["lease", "tenant", "landlord"]):
            return "lease_agreement"
        else:
            return "unknown_legal_document"
    
    def _extract_legal_terms(self, document_text: str) -> List[str]:
        """Extract legal terms from document text."""
        found_terms = []
        text_lower = document_text.lower()
        
        for term in self.legal_knowledge["legal_terms"]:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms
