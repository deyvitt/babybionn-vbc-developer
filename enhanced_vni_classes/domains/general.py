"""General domain VNI implementation with biological systems integration"""
from .base_knowledge_loader import BaseKnowledgeLoader
from typing import Dict, Any, Optional, List
from ..core.base_vni import EnhancedBaseVNI
from ..core.capabilities import VNICapabilities, VNIType
from ..modules.generation import EnhancedGenerationModule, GenerationStyle
from ..modules.classifier import DomainClassifier
from ..modules.web_search import WebSearch
from ..modules.learning_system import LearningSystem
from ..utils.logger import get_logger
import hashlib
from datetime import datetime

# Import ALL required biological toolkit components
from neuron.vni_memory import VNIMemory
from neuron.demoHybridAttention import DemoHybridAttention
from neuron.smart_activation_router import SmartActivationRouter, FunctionRegistry  # ADDED

logger = get_logger(__name__)

class GeneralVNI(BaseKnowledgeLoader, EnhancedBaseVNI):
    """General domain VNI with biological systems integration"""
    def __init__(self,                 
                 instance_id: str = "general_001", 
                 vni_config: Dict[str, Any] = None, 
                 auto_load_knowledge: bool = True, 
                 enable_biological_systems: bool = True,
                 attention_config: Dict[str, Any] = None,
                 memory_config: Dict[str, Any] = None,
                 auto_validate: bool = True,
                 strict_validation: bool = False,
                 validation_timeout: float = 5.0):
        
        capabilities = VNICapabilities(
            domains=["general", "conversation", "information", "assistance", "multi_domain"],
            can_search=True,
            can_learn=True,
            can_collaborate=True,
            max_context_length=3000,
            special_abilities=[
                "multi_domain", 
                "conversational", 
                "task_assistance", 
                "knowledge_synthesis",
                "biological_systems_integration",  # ADDED
                "general_attention_focus",  # ADDED
                "dynamic_function_routing"  # ADDED
            ],
            vni_type='specialized'
        )
        
        # ============ BIOLOGICAL SYSTEMS CONFIGURATION ============ ADDED
        self.enable_biological_systems = enable_biological_systems
        # Use provided configs or defaults
        if attention_config is None:
            attention_config = {
                'dim': 256,
                'num_heads': 8,
                'window_size': 512,
                'use_sliding': True,
                'use_global': True,
                'use_hierarchical': True,
                'global_token_ratio': 0.08,
                'memory_tokens': 30,
                'multi_modal': False,
                'semantic_weight': 0.6,
                'precision_weight': 0.7
            }
        
        if memory_config is None:
            memory_config = {
                'short_term_capacity': 200,
                'long_term_capacity': 5000,
                'consolidation_threshold': 0.7,
                'retention_period': 86400 * 14,
                'priority_retention': True,
                'conversation_aware': True
            }
        
        # Biological systems tracking
        self.conversation_patterns: Dict[str, Dict] = {}
        self.user_profiles: Dict[str, Dict] = {}
        self.general_guidelines = self._load_general_guidelines()
        
        # General-specific setup (AFTER super().__init__)
        self.vni_type = "general" 
        self.domain = "general"
        self.name = f"GeneralVNI-{instance_id}"
        self.description = "General domain VNI with biological systems integration"        
        
        # Initialize EnhancedBaseVNI with ALL required parameters
        super().__init__(
            instance_id=instance_id,
            domain="general", 
            capabilities=capabilities,
            vni_config=vni_config or {},
            name=f"GeneralVNI-{instance_id}",
            description="General domain VNI with biological systems integration",
            vni_type="general",
            enable_biological_systems=enable_biological_systems,
            attention_config=attention_config,
            memory_config=memory_config,
            auto_validate=auto_validate,
            strict_validation=strict_validation,
            validation_timeout=validation_timeout
        )
        
        # General-specific setup (AFTER super().__init__)
        self.vni_type = "general" 
        self.generation_module = EnhancedGenerationModule(domain="general")
        self.general_knowledge = self._initialize_general_knowledge()
        
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
            vni_id=self.instance_id,
            domain="general",
            input_dim=512,
            num_experts=4,           # General experts: information, explanation, guidance, synthesis
            expert_dim=256
        )
        
        # Initialize VNIMemory
        self.memory = VNIMemory(
            domain="general",
            vni_id=instance_id,
            memory_type="conversational"
        )
        
        # Register general-specific functions with activation router
        self._register_general_functions()  # ADDED
        
        # Initialize enhanced modules
        self.web_search = WebSearch(vni_id=self.instance_id)

        # Load knowledge from files if enabled
        if auto_load_knowledge:
            self.load_domain_knowledge("general")        
        
        logger.info(f"Initialized GeneralVNI: {instance_id} with full biological toolkit")

    # ============ NEW BIOLOGICAL SYSTEMS METHODS ============ ADDED
    def _load_general_guidelines(self) -> Dict[str, Any]:
        """Load general guidelines for biological systems"""
        return {
            "interaction_protocols": {
                "activation_threshold": 0.6,
                "response_time_limit": 3.0,  # More time for thoughtful responses
                "adaptive_behaviors": ["topic_switching", "detail_adjustment", "tone_matching"]
            },
            "conversation_categories": {
                "informational": ["what is", "how to", "explain", "define", "information about"],
                "assistive": ["help with", "can you", "could you", "assist with", "guide me"],
                "conversational": ["hello", "hi", "how are you", "tell me about", "chat"],
                "creative": ["ideas for", "suggest", "recommend", "brainstorm", "creative"],
                "analytical": ["analyze", "compare", "evaluate", "pros and cons", "advantages"]
            },
            "biological_monitoring": {
                "engagement_factors": ["response_length", "question_complexity", "topic_depth", "emotional_tone"],
                "alert_levels": {
                    "normal": 0.0,
                    "engaged": 0.4,
                    "deep": 0.7,
                    "exceptional": 0.9
                }
            }
        }
    
    def _extract_conversation_factors(self, query: str) -> List[Dict]:
        """Extract conversation factors using biological systems analysis"""
        query_lower = query.lower()
        extracted_factors = []
        
        # Check conversation categories
        for category, patterns in self.general_guidelines["conversation_categories"].items():
            for pattern in patterns:
                if pattern in query_lower:
                    extracted_factors.append({
                        "factor": pattern,
                        "category": category,
                        "engagement_level": "high" if category in ["analytical", "creative"] else "medium"
                    })
        
        # Check for emotional tone indicators
        emotional_indicators = {
            "positive": ["great", "wonderful", "excited", "happy", "thank you"],
            "negative": ["problem", "issue", "difficult", "struggling", "confused"],
            "urgent": ["urgent", "asap", "quick", "immediate", "right now"],
            "casual": ["just curious", "by the way", "when you have time", "no rush"]
        }
        
        for tone, indicators in emotional_indicators.items():
            for indicator in indicators:
                if indicator in query_lower:
                    extracted_factors.append({
                        "factor": "emotional_tone",
                        "tone": tone,
                        "indicator": indicator
                    })
        
        return extracted_factors
    
    def _get_user_profile(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get user profile for biological systems"""
        user_id = context.get("user_id", "anonymous") if context else "anonymous"
        profile = self.user_profiles.get(user_id, {})
        
        return {
            "available": bool(profile),
            "profile": profile,
            "user_id": user_id,
            "interaction_count": profile.get("interaction_count", 0),
            "preferred_topics": profile.get("preferred_topics", []),
            "timestamp": datetime.now().isoformat()
        }
    
    def process_with_biological_systems(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process query with general biological systems integration"""
        if not self.enable_biological_systems:
            return {
                "biological_processing": False,
                "reason": "Biological systems disabled"
            }
        
        logger.debug(f"🌐 General biological processing: {query[:50]}...")
        
        try:
            # Extract conversation insights
            conversation_factors = self._extract_conversation_factors(query)
            user_profile = self._get_user_profile(context)
            
            # Calculate engagement level
            engagement_level = self._calculate_general_engagement(query, conversation_factors, user_profile)
            
            # Determine conversation style
            conversation_style = self._determine_conversation_style(engagement_level, conversation_factors)
            
            # Generate biological insights
            biological_insights = {
                "conversation_factors_detected": len(conversation_factors),
                "conversation_categories": list(set([f["category"] for f in conversation_factors])),
                "engagement_level": engagement_level,
                "conversation_style": conversation_style,
                "user_profile_available": user_profile["available"],
                "requires_detailed_response": engagement_level > self.general_guidelines["interaction_protocols"]["activation_threshold"],
                "biological_timestamp": datetime.now().isoformat()
            }
            
            # Update user profile
            if context and "user_id" in context:
                self._update_user_profile(context["user_id"], query, conversation_factors, biological_insights)
            
            return {
                "biological_processing": True,
                "biological_insights": biological_insights,
                "activation_result": {
                    "engagement_level": engagement_level,
                    "conversation_style": conversation_style,
                    "recommended_approaches": self._get_general_approaches(conversation_style)
                },
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ General biological processing failed: {e}")
            return {
                "biological_processing": False,
                "error": str(e),
                "success": False
            }
    
    def _calculate_general_engagement(self, query: str, conversation_factors: List[Dict], 
                                    user_profile: Dict[str, Any]) -> float:
        """Calculate engagement level based on query analysis"""
        base_engagement = 0.3
        
        # Increase based on conversation factor count
        factor_bonus = min(len(conversation_factors) * 0.1, 0.2)
        base_engagement += factor_bonus
        
        # Increase based on factor complexity
        complex_categories = ["analytical", "creative", "assistive"]
        for factor in conversation_factors:
            if factor["category"] in complex_categories:
                base_engagement += 0.15
        
        # Increase based on query length and complexity
        query_length = len(query.split())
        if query_length > 20:
            base_engagement += 0.1
        elif query_length > 10:
            base_engagement += 0.05
        
        # Consider user profile history
        if user_profile.get("available"):
            interaction_count = user_profile.get("interaction_count", 0)
            if interaction_count > 5:
                base_engagement += 0.1  # Regular users get more engaged responses
        
        # Cap at 1.0
        return min(base_engagement, 1.0)
    
    def _determine_conversation_style(self, engagement_level: float, 
                                     conversation_factors: List[Dict]) -> str:
        """Determine conversation style based on biological analysis"""
        if engagement_level >= 0.8:
            return "detailed_analytical"
        elif engagement_level >= 0.6:
            return "informative_assistive"
        elif engagement_level >= 0.4:
            return "balanced_conversational"
        else:
            return "friendly_casual"
    
    def _get_general_approaches(self, conversation_style: str) -> List[str]:
        """Get recommended approaches based on conversation style"""
        approaches = {
            "detailed_analytical": [
                "Provide comprehensive information",
                "Include examples and evidence",
                "Offer multiple perspectives",
                "Suggest further reading"
            ],
            "informative_assistive": [
                "Clear, structured explanations",
                "Step-by-step guidance",
                "Practical applications",
                "Resource recommendations"
            ],
            "balanced_conversational": [
                "Friendly but informative tone",
                "Engage with follow-up questions",
                "Balance depth with accessibility",
                "Show genuine interest"
            ],
            "friendly_casual": [
                "Warm, approachable tone",
                "Keep responses concise",
                "Use conversational language",
                "Offer to help further"
            ]
        }
        return approaches.get(conversation_style, ["Adapt to user's tone and needs"])
    
    def _update_user_profile(self, user_id: str, query: str, 
                            conversation_factors: List[Dict], insights: Dict[str, Any]):
        """Update user profile for biological tracking"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "first_interaction": datetime.now().isoformat(),
                "interaction_count": 0,
                "preferred_topics": [],
                "conversation_history": [],
                "engagement_patterns": []
            }
        
        user_data = self.user_profiles[user_id]
        user_data["interaction_count"] += 1
        user_data["last_interaction"] = datetime.now().isoformat()
        
        # Add conversation to history
        user_data["conversation_history"].append({
            "query": query[:100],
            "timestamp": datetime.now().isoformat(),
            "engagement_level": insights.get("engagement_level", 0.5),
            "conversation_style": insights.get("conversation_style", "unknown")
        })
        
        # Update preferred topics
        for factor in conversation_factors:
            if factor.get("category"):
                if factor["category"] not in user_data["preferred_topics"]:
                    user_data["preferred_topics"].append(factor["category"])
        
        # Keep history manageable
        if len(user_data["conversation_history"]) > 100:
            user_data["conversation_history"] = user_data["conversation_history"][-100:]
        if len(user_data["preferred_topics"]) > 10:
            user_data["preferred_topics"] = user_data["preferred_topics"][-10:]
    
    def _enhance_general_response(self, result: Dict[str, Any], query: str, 
                                 context: Optional[Dict[str, Any]] = None,
                                 biological_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhance response with general biological insights"""
        enhanced_result = result.copy()
        
        if biological_result and biological_result.get("biological_processing"):
            # Add biological insights to response
            insights = biological_result.get("biological_insights", {})
            activation = biological_result.get("activation_result", {})
            
            enhanced_result["biological_analysis"] = {
                "performed": True,
                "conversation_factors": insights.get("conversation_factors_detected", 0),
                "conversation_style": insights.get("conversation_style", "unknown"),
                "engagement_level": activation.get("engagement_level", 0.0),
                "user_profile_used": insights.get("user_profile_available", False)
            }
            
            # Apply conversation style adjustments
            conversation_style = insights.get("conversation_style", "balanced_conversational")
            if conversation_style == "detailed_analytical":
                if "response" in enhanced_result:
                    # Ensure response is comprehensive
                    if len(enhanced_result["response"]) < 300:
                        enhanced_result["response"] += "\n\nWould you like more detailed information on any aspect?"
            elif conversation_style == "friendly_casual":
                if "response" in enhanced_result:
                    # Ensure response is warm and concise
                    if not enhanced_result["response"].startswith(("Hi", "Hello", "Hey")):
                        enhanced_result["response"] = "Hi! " + enhanced_result["response"]
        
        return enhanced_result
    # ============ END OF BIOLOGICAL SYSTEMS METHODS ============

    # ============ NEW: REGISTER GENERAL FUNCTIONS ============ ADDED
    def _register_general_functions(self):
        """Register general-specific functions with activation router"""
        if not hasattr(self, 'activation_router'):
            return
            
        # Register conversation analysis function
        self.activation_router.register_function(
            function_name="analyze_conversation_pattern",
            function=self._analyze_conversation_pattern,
            domain=self.domain,
            priority=1
        )
        
        # Register topic switching function
        self.activation_router.register_function(
            function_name="switch_conversation_topic",
            function=self._switch_conversation_topic,
            domain="general",
            priority=2
        )
        
        # Register multi-domain assistance function
        self.activation_router.register_function(
            function_name="provide_multi_domain_assistance",
            function=self._provide_multi_domain_assistance,
            domain="general",
            priority=2
        )
        
        # Register engagement boosting function
        self.activation_router.register_function(
            function_name="boost_conversation_engagement",
            function=self._boost_conversation_engagement,
            domain="general",
            priority=3
        )
        
        # Register user profile management function
        self.activation_router.register_function(
            function_name="manage_user_profile",
            function=self._manage_user_profile,
            domain="general",
            priority=3
        )
        
        # Safely count registered functions
        try:
            # Try different possible attribute names where functions might be stored
            if hasattr(self.activation_router, 'registered_functions'):
                func_count = len(self.activation_router.registered_functions)
            elif hasattr(self.activation_router, 'functions'):
                func_count = len(self.activation_router.functions)
            elif hasattr(self.activation_router, '_registered_functions'):
                func_count = len(self.activation_router._registered_functions)
            elif hasattr(self.activation_router, '_functions'):
                func_count = len(self.activation_router._functions)
            else:
                # Try to inspect the object to find where functions are stored
                import inspect
                attributes = inspect.getmembers(self.activation_router)
                func_dicts = [attr for name, attr in attributes 
                             if isinstance(attr, dict) and 'function' in name.lower()]
                func_count = len(func_dicts[0]) if func_dicts else 0
        
            logger.info(f"Registered {func_count} general functions")
            
        except Exception as e:
            logger.warning(f"Could not count registered functions: {e}")
            logger.info("Registered general functions with activation router")
    
    def _analyze_conversation_pattern(self, conversation_history: List[Dict], 
                                    analysis_type: str = "engagement") -> Dict[str, Any]:
        """Analyze conversation patterns"""
        return {
            "analysis_type": analysis_type,
            "total_interactions": len(conversation_history),
            "average_engagement": sum(h.get("engagement_level", 0.5) for h in conversation_history) / max(len(conversation_history), 1),
            "topic_distribution": {},
            "recommendations": ["Maintain consistent engagement", "Vary topic depth based on interest"]
        }
    
    def _switch_conversation_topic(self, current_topic: str, 
                                 user_preferences: List[str]) -> Dict[str, Any]:
        """Switch conversation topics"""
        # Logic to suggest new topics based on preferences
        suggested_topics = []
        for topic in self.general_knowledge.get("conversation_topics", []):
            if topic not in current_topic and any(pref in topic for pref in user_preferences):
                suggested_topics.append(topic)
        
        return {
            "current_topic": current_topic,
            "user_preferences": user_preferences,
            "suggested_topics": suggested_topics[:5],
            "transition_phrases": [
                f"Speaking of {current_topic}, have you thought about...",
                f"That reminds me of...",
                f"By the way, about..."
            ]
        }
    
    def _provide_multi_domain_assistance(self, query: str, 
                                       domain_context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide multi-domain assistance"""
        return {
            "query": query,
            "domains_considered": list(domain_context.keys()),
            "primary_domain": domain_context.get("primary", "general"),
            "cross_domain_insights": [],
            "recommended_approach": "integrated_multi_domain"
        }
    
    def _boost_conversation_engagement(self, engagement_level: float, 
                                     conversation_style: str) -> Dict[str, Any]:
        """Boost conversation engagement"""
        techniques = []
        if engagement_level < 0.4:
            techniques = ["Ask open-ended questions", "Share interesting facts", "Use storytelling"]
        elif engagement_level < 0.7:
            techniques = ["Provide deeper insights", "Ask for opinions", "Share relevant examples"]
        else:
            techniques = ["Challenge with thought-provoking questions", "Provide comprehensive analysis", "Connect to broader concepts"]
        
        return {
            "current_engagement": engagement_level,
            "conversation_style": conversation_style,
            "engagement_techniques": techniques,
            "estimated_impact": min(engagement_level + 0.2, 1.0)
        }
    
    def _manage_user_profile(self, user_id: str, action: str) -> Dict[str, Any]:
        """Manage user profiles"""
        profile = self.user_profiles.get(user_id, {})
        
        if action == "get":
            return {
                "user_id": user_id,
                "profile": profile,
                "action": "retrieved"
            }
        elif action == "update":
            return {
                "user_id": user_id,
                "profile": profile,
                "action": "updated",
                "timestamp": datetime.now().isoformat()
            }
        elif action == "reset":
            self.user_profiles[user_id] = {
                "first_interaction": datetime.now().isoformat(),
                "interaction_count": 0
            }
            return {
                "user_id": user_id,
                "action": "reset",
                "timestamp": datetime.now().isoformat()
            }
        
        return {"error": f"Unknown action: {action}"}
    # ============ END OF NEW FUNCTIONS ============

    async def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Implement abstract process method from EnhancedBaseVNI."""
        return await self.process_query(query, context)

    def _initialize_general_knowledge(self) -> Dict[str, Any]:
        """Initialize general knowledge base."""
        return {
            "categories": {
                "science": "Study of the natural world through observation and experiment.",
                "history": "Study of past events, particularly in human affairs.",
                "technology": "Application of scientific knowledge for practical purposes.",
                "arts": "Various branches of creative activity.",
                "culture": "Customs, arts, social institutions of particular groups.",
                "health": "State of complete physical, mental and social well-being.",
                "education": "Process of facilitating learning or acquisition of knowledge.",
                "sports": "Physical activities involving skill and competition.",
                "travel": "Movement between geographical locations.",
                "food": "Substances consumed for nutritional support."
            },
            "conversation_topics": [
                "current events", "technology trends", "science discoveries",
                "book recommendations", "movie reviews", "travel destinations",
                "health tips", "learning resources", "hobbies", "personal development"
            ],
            "response_patterns": {
                "factual": "Provides clear, accurate information",
                "exploratory": "Encourages further discussion",
                "assistive": "Offers help or guidance",
                "conversational": "Maintains engaging dialogue"
            }
        }
    
    async def process_query(self, 
                           query: str, 
                           context: Dict[str, Any] = None,
                           conversation_context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Process general query with enhanced capabilities and biological systems.
        """
        # Extract conversation context if provided
        if context and "conversation_history" in context:
            conversation_context = context["conversation_history"]
        
        # Classify query for better understanding
        classification = self._classify_query(query)
        
        # BIOLOGICAL SYSTEMS INTEGRATION ADDED
        biological_result = None
        if self.enable_biological_systems:
            biological_result = self.process_with_biological_systems(query, context)
            if biological_result.get("biological_processing"):
                # Add biological insights to context
                if context is None:
                    context = {}
                context["biological_insights"] = biological_result.get("biological_insights", {})
                context["activation_result"] = biological_result.get("activation_result", {})
        
        # Build comprehensive context
        enhanced_context = await self._build_enhanced_context(query, classification, conversation_context)
        
        # Retrieve relevant memory
        if hasattr(self, 'memory'):
            memory_context = await self.memory.retrieve_relevant_memory(
                query=query,
                conversation_context=conversation_context
            )
            if memory_context:
                enhanced_context["memory_context"] = memory_context
                
        # Determine generation style with biological influence
        generation_style = self._determine_generation_style(query, classification)
        
        # Apply biological style adjustments if available
        if biological_result and biological_result.get("biological_processing"):
            conversation_style = biological_result.get("biological_insights", {}).get("conversation_style", "")
            if conversation_style == "detailed_analytical":
                generation_style = GenerationStyle.DETAILED.value
            elif conversation_style == "friendly_casual":
                generation_style = GenerationStyle.CASUAL.value
        
        # Generate response with appropriate style
        if hasattr(self, 'generation_module'):
            response_text = self.generation_module.generate(
                query=query,
                context=enhanced_context,
                style=generation_style,
                #temperature=0.7,
                #max_length=300
            )
        else:
            # Fallback to simple response
            response_text = self._generate_fallback_response(query, classification)
        
        # Enhance with conversation flow if applicable
        if conversation_context:
            response_text = self._add_conversation_flow(response_text, conversation_context, query)
        
        # Compile comprehensive result
        result = {
            "response": response_text,
            "confidence": classification.get("confidence", 0.3),
            "vni_instance": self.instance_id,
            "domain": classification.get("primary_domain", "general"),
            "secondary_domains": classification.get("secondary_domains", []),
            "response_type": classification.get("response_type", "general_assistance"),
            "generation_style": generation_style,
            "context_used": {
                "knowledge": bool(enhanced_context.get("knowledge", {})),
                "web_results": len(enhanced_context.get("web_results", [])),
                "conversation_history": bool(conversation_context)
            },
            "suggested_follow_ups": self._generate_follow_up_suggestions(query, response_text, classification)
        }
        
        # Enhance with biological systems if enabled
        if self.enable_biological_systems and biological_result:
            result = self._enhance_general_response(result, query, context, biological_result)
        
        # Record interaction for learning
        self._record_interaction(query, response_text, result, conversation_context)
        
        # Store interaction in memory
        if hasattr(self, 'memory'):
            memory_id = await self.memory.store_interaction(
                query=query,
                response=response_text,
                context=enhanced_context,
                metadata={
                    "conversation_length": len(conversation_context) if conversation_context else 0,
                    "domain": classification.get("primary_domain", "general"),
                    "response_type": classification.get("response_type", "general_assistance"),
                    "biological_processing": biological_result.get("biological_processing", False) if biological_result else False
                }
            )
            result["memory_id"] = memory_id
                
        return result
    
    def _classify_query(self, query: str) -> Dict[str, Any]:
        """Classify query type and intent."""
        query_lower = query.lower()
        
        # Determine primary domain
        domains = []
        for category in self.general_knowledge["categories"]:
            if category in query_lower:
                domains.append(category)
        
        # Check for specific intent patterns
        response_type = "general_assistance"
        confidence = 0.3
        
        if any(word in query_lower for word in ["how to", "tutorial", "guide", "steps", "way to"]):
            response_type = "instructional"
            confidence = 0.6
        elif any(word in query_lower for word in ["what is", "definition", "meaning of", "explain"]):
            response_type = "explanatory"
            confidence = 0.7
        elif any(word in query_lower for word in ["why", "reason", "cause", "because"]):
            response_type = "analytical"
            confidence = 0.5
        elif any(word in query_lower for word in ["compare", "difference", "vs", "versus"]):
            response_type = "comparative"
            confidence = 0.6
        elif any(word in query_lower for word in ["best", "top", "recommend", "suggest"]):
            response_type = "recommendation"
            confidence = 0.4
        elif any(word in query_lower for word in ["hi", "hello", "hey", "how are you", "greeting"]):
            response_type = "conversational"
            confidence = 0.8
        
        # Extract keywords
        keywords = []
        common_words = set(query_lower.split())
        for word in ["what", "how", "why", "when", "where", "who", "which"]:
            if word in common_words:
                keywords.append(word)
        
        return {
            "primary_domain": domains[0] if domains else "general",
            "secondary_domains": domains[1:] if len(domains) > 1 else [],
            "response_type": response_type,
            "confidence": confidence,
            "keywords": keywords,
            "query_complexity": "simple" if len(query.split()) < 10 else "complex"
        }
    
    async def _build_enhanced_context(self, 
                                     query: str, 
                                     classification: Dict[str, Any],
                                     conversation_context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Build comprehensive context for query processing with memory integration."""
        context = {
            "knowledge": {"content": "", "confidence": 0.0},
            "web_results": [],
            "conversation_history": [],
            "query_analysis": classification,
            "memory_context": {}
        }
        
        # Retrieve relevant memory if available
        if hasattr(self, 'memory'):
            try:
                memory_context = await self.memory.retrieve_relevant_memory(
                    query=query,
                    conversation_context=conversation_context
                )
                if memory_context:
                    context["memory_context"] = memory_context
            except Exception as e:
                logger.debug(f"Memory retrieval failed: {e}")
        
        # Add general knowledge from built-in base
        general_info = self._get_general_information(query)
        if general_info:
            context["knowledge"] = {
                "content": general_info,
                "confidence": 0.6,
                "source": "general_knowledge_base"
            }
        
        # Query external knowledge base
        try:
            kb_results = self.query_knowledge(query)
            if kb_results and kb_results[0]["confidence"] > 0.4:
                context["knowledge_base"] = {
                    "content": kb_results[0]["content"],
                    "confidence": kb_results[0]["confidence"],
                    "source": "enhanced_knowledge_base"
                }
        except Exception as e:
            logger.debug(f"Knowledge base query failed: {e}")
        
        # Perform web search for current information if enabled
        if hasattr(self, 'web_search') and classification.get("confidence", 0) < 0.6:
            try:
                search_domain = self._determine_search_domain(query)
                web_results = await self.web_search.search(query, domain=search_domain, num_results=3)
                context["web_results"] = web_results.get("results", [])
            except Exception as e:
                logger.warning(f"Web search failed: {e}")
        
        # Add conversation context
        if conversation_context:
            context["conversation_history"] = conversation_context[-5:]  # Last 5 turns
            context["conversation_summary"] = self._summarize_conversation(conversation_context)
        
        return context
    
    def _record_interaction(self, 
                          query: str, 
                          response: str, 
                          result: Dict[str, Any],
                          conversation_context: Optional[List[Dict]] = None):
        """Record interaction for learning and improvement with memory integration."""
        try:
            metadata = {
                "domain": result["domain"],
                "response_type": result["response_type"],
                "confidence": result["confidence"],
                "generation_style": result["generation_style"],
                "conversation_length": len(conversation_context) if conversation_context else 0,
                "timestamp": hashlib.md5(query.encode()).hexdigest()[:16]
            }
            
            if hasattr(self, 'learning'):
                self.learning.record_interaction(
                    interaction_id=metadata["timestamp"],
                    prompt=query,
                    response=response,
                    domain=result["domain"],
                    metadata=metadata
                )
            
            logger.debug(f"Recorded general interaction: {metadata['timestamp']}")
            
            # Also store in VNIMemory
            if hasattr(self, 'memory'):
                try:
                    import asyncio
                    asyncio.create_task(
                        self.memory.store_interaction(
                            query=query,
                            response=response,
                            context={
                                "conversation_length": metadata["conversation_length"],
                                "domain": metadata["domain"],
                                "confidence": metadata["confidence"]
                            },
                            metadata={
                                "vni_id": self.instance_id,
                                "domain": "general",
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to store interaction in memory: {e}")
            
        except Exception as e:
            logger.error(f"Failed to record interaction: {e}")

    async def engage_in_conversation(self, 
                                    messages: List[Dict[str, str]],
                                    max_turns: int = 10) -> Dict[str, Any]:
        """Engage in a multi-turn conversation with memory integration."""
        
        if not messages:
            # Retrieve conversation starter from memory if available
            starter = "Hello! I'm your general assistant. How can I help you today?"
            if hasattr(self, 'memory'):
                try:
                    past_conversations = await self.memory.retrieve_conversation_patterns()
                    if past_conversations:
                        # Use a personalized starter based on past interactions
                        starter = f"Hello again! I remember we've chatted before. How can I assist you today?"
                except Exception as e:
                    logger.debug(f"Could not retrieve conversation patterns: {e}")
            
            return {
                "response": starter,
                "conversation_state": "initiated",
                "suggested_topics": self.general_knowledge["conversation_topics"][:5]
            }
        
        # Process the last message
        last_message = messages[-1]
        query = last_message.get("content", "")
        
        # Build conversation history
        conversation_history = [
            {"query": msg.get("content", ""), "response": ""}
            for msg in messages[:-1]  # All but last
        ]
        
        # Retrieve relevant past conversations from memory
        memory_context = {}
        if hasattr(self, 'memory'):
            try:
                memory_context = await self.memory.retrieve_relevant_memory(
                    query=query,
                    conversation_context=conversation_history
                )
            except Exception as e:
                logger.debug(f"Memory retrieval failed: {e}")
        
        # Process with context including memory
        context_with_memory = {"conversation_history": conversation_history}
        if memory_context:
            context_with_memory["memory_context"] = memory_context
            
        result = await self.process_query(query, None, conversation_history)
        
        # Store the entire conversation in memory
        if hasattr(self, 'memory') and len(messages) >= 3:  # Store meaningful conversations
            try:
                await self.memory.store_conversation(
                    messages=messages,
                    metadata={
                        "turns": len(messages),
                        "domains_covered": result.get("domain", "general"),
                        "conversation_state": self._determine_conversation_state(messages, result)
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to store conversation in memory: {e}")
        
        # Determine conversation state
        conversation_state = self._determine_conversation_state(messages, result)
        
        return {
            "response": result["response"],
            "conversation_state": conversation_state,
            "follow_up_questions": result.get("suggested_follow_ups", []),
            "current_domain": result["domain"],
            "conversation_length": len(messages) + 1,
            "context_awareness": result["context_used"]["conversation_history"],
            "memory_used": bool(memory_context)
        }
    
    def get_general_stats(self) -> Dict[str, Any]:
        """Get general VNI statistics including memory stats."""
        stats = {
            "vni_id": self.instance_id,
            "categories": len(self.general_knowledge.get("categories", {})),
            "conversation_topics": len(self.general_knowledge.get("conversation_topics", [])),
            "response_patterns": len(self.general_knowledge.get("response_patterns", {}))
        }
        
        # Add memory stats if available
        if hasattr(self, 'memory'):
            try:
                memory_stats = self.memory.get_stats()
                stats["memory"] = {
                    "total_interactions": memory_stats.get("total_interactions", 0),
                    "memory_type": memory_stats.get("memory_type", "unknown"),
                    "conversation_count": memory_stats.get("conversation_count", 0),
                    "retrieval_success_rate": memory_stats.get("retrieval_success_rate", 0.0)
                }
            except Exception as e:
                logger.warning(f"Could not get memory stats: {e}")
                stats["memory"] = {"error": "unavailable"}
        
        return stats
    
    def _get_general_information(self, query: str) -> Optional[str]:
        """Get general information from built-in knowledge base."""
        query_lower = query.lower()
        
        # SAFELY get categories
        categories = self.general_knowledge.get("categories", {})
        
        # Check categories - handle ANY structure
        if isinstance(categories, dict):
            for category, info in categories.items():
                if isinstance(category, str) and category in query_lower:
                    return f"Information about {category}: {info}"
        elif hasattr(categories, '__iter__'):
            # Handle lists or other iterables
            for item in categories:
                topic = self._extract_topic_from_any_type(item)
                if topic and topic in query_lower:
                    return f"Information about {topic}: Found in knowledge base"
        
        # Check conversation topics
        conversation_topics = self.general_knowledge.get("conversation_topics", [])
        if isinstance(conversation_topics, (list, tuple, set)):
            for topic in conversation_topics:
                topic_str = self._extract_topic_from_any_type(topic)
                if topic_str and topic_str in query_lower:
                    return f"That's an interesting topic! {topic_str.replace('_', ' ').title()} often leads to engaging discussions."
        
        return None
    
    def _determine_search_domain(self, query: str) -> str:
        """Determine appropriate search domain for query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["news", "current", "recent", "update", "today"]):
            return "news"
        elif any(word in query_lower for word in ["how to", "tutorial", "guide", "steps", "method"]):
            return "technical"
        elif any(word in query_lower for word in ["definition", "meaning", "what is", "explain"]):
            return "academic"
        elif any(word in query_lower for word in ["best", "top", "review", "rating"]):
            return "review"
        elif any(word in query_lower for word in ["recipe", "cook", "food", "meal"]):
            return "food"
        elif any(word in query_lower for word in ["travel", "destination", "visit", "tourist"]):
            return "travel"
        else:
            return "general"
    
    def _determine_generation_style(self, query: str, classification: Dict[str, Any]) -> str:
        """Determine appropriate generation style for query."""
        query_lower = query.lower()
        response_type = classification.get("response_type", "general_assistance")
        
        style_mapping = {
            "instructional": GenerationStyle.TECHNICAL.value,
            "explanatory": GenerationStyle.DETAILED.value,
            "analytical": GenerationStyle.FORMAL.value,
            "comparative": GenerationStyle.DETAILED.value,
            "recommendation": GenerationStyle.CASUAL.value,
            "conversational": GenerationStyle.CASUAL.value,
            "general_assistance": GenerationStyle.DETAILED.value
        }
        
        # Override based on keywords
        if any(word in query_lower for word in ["creative", "idea", "imagine", "story", "poem"]):
            return GenerationStyle.CREATIVE.value
        elif any(word in query_lower for word in ["business", "professional", "formal", "official"]):
            return GenerationStyle.FORMAL.value
        
        return style_mapping.get(response_type, GenerationStyle.DETAILED.value)
    
    def _generate_fallback_response(self, query: str, classification: Dict[str, Any]) -> str:
        """Generate fallback response when generation module is unavailable."""
        response_type = classification.get("response_type", "general_assistance")
        
        fallback_responses = {
            "instructional": f"I understand you're asking about how to do something. While I can't provide specific instructions right now, I suggest searching for '{query}' online for step-by-step guides.",
            "explanatory": f"You're asking about '{query}'. This is an interesting topic that would benefit from a detailed explanation. For comprehensive information, I recommend consulting reliable sources on this subject.",
            "analytical": f"Regarding your question about '{query}', analysis would require considering multiple factors. Each situation can be different, so context is important for proper analysis.",
            "comparative": f"You're asking to compare aspects of '{query}'. Comparison involves looking at similarities and differences, which can vary based on specific criteria.",
            "recommendation": f"You're looking for recommendations about '{query}'. Preferences can be subjective, so what works best often depends on individual needs and circumstances.",
            "conversational": f"Hello! Thanks for your message about '{query}'. I'm here to help with information and conversation on a wide range of topics.",
            "general_assistance": f"I'm a general-purpose VNI. Regarding '{query}', I can help you think through this topic and suggest where to find more information."
        }
        
        return fallback_responses.get(response_type, 
            f"I'm here to help with your query about '{query}'. This appears to be a {classification.get('primary_domain', 'general')} topic that I'm still learning about.")
    
    def _add_conversation_flow(self, 
                              response: str, 
                              conversation_context: List[Dict], 
                              current_query: str) -> str:
        """Add natural conversation flow to response."""
        if not conversation_context or len(conversation_context) < 2:
            return response
        
        # Get recent context
        last_turn = conversation_context[-1]
        last_response = last_turn.get("response", "").lower()
        
        # Check for continuation patterns
        continuation_keywords = ["also", "further", "additionally", "moreover", "besides"]
        if any(keyword in last_response for keyword in continuation_keywords):
            # This is part of an ongoing discussion
            if not response.lower().startswith(("also", "and", "further", "additionally")):
                response = f"Additionally, {response.lower()}"
        
        # Check for topic switches
        topic_keywords = ["changing topic", "new subject", "different matter", "by the way"]
        if any(keyword in last_response for keyword in topic_keywords):
            response = f"Regarding your new question, {response.lower()}"
        return response
    
    def _extract_topic_from_any_type(self, item) -> str:
        """Dynamically extract a meaningful string topic from ANY data type.
        
        Handles: strings, dicts, lists, tuples, numbers, None, bool, objects
        Returns a clean, human-readable string."""
        # Type 1: String (ideal case)
        if isinstance(item, str):
            # Clean up string
            return item.strip()
        
        # Type 2: Dictionary 
        elif isinstance(item, dict):
            # Strategy: Look for most meaningful string in dict
            candidates = []
            
            # Priority 1: Common key names that might contain topic names
            topic_keys = ['topic', 'name', 'title', 'label', 'category', 'subject', 'key']
            for key in topic_keys:
                if key in item and isinstance(item[key], str) and item[key].strip():
                    candidates.append(item[key].strip())
            
            # Priority 2: All string values
            for value in item.values():
                if isinstance(value, str) and value.strip():
                    candidates.append(value.strip())
            
            # Priority 3: All string keys (fallback)
            for key in item.keys():
                if isinstance(key, str) and key.strip() and key not in topic_keys:
                    candidates.append(key.strip())
            
            # Priority 4: First non-empty element (any type)
            for value in item.values():
                if value and str(value).strip():
                    return str(value).strip()
            
            # Last resort: String representation
            return str(item)[:50]  # Truncate to avoid huge outputs
        
        # Type 3: List or Tuple
        elif isinstance(item, (list, tuple, set)):
            # Strategy: Find first meaningful string element
            for element in item:
                if isinstance(element, str) and element.strip():
                    return element.strip()
                elif isinstance(element, (dict, list, tuple, set)):
                    # Recursively search nested structures
                    nested_result = self._extract_topic_from_any_type(element)
                    if nested_result and nested_result != str(element)[:50]:
                        return nested_result
            
            # Fallback: Join all elements if they're simple types
            try:
                # Try to create a meaningful string from list contents
                elements = []
                for element in item[:5]:  # Limit to first 5
                    if element is None:
                        continue
                    elem_str = str(element).strip()
                    if elem_str and len(elem_str) < 30:  # Avoid huge elements
                        elements.append(elem_str)
                
                if elements:
                    return ', '.join(elements)
            except:
                pass
            
            # Last resort
            return str(item)[:50]
        
        # Type 4: Numbers
        elif isinstance(item, (int, float, complex)):
            return str(item)
        
        # Type 5: None
        elif item is None:
            return "unknown"
        
        # Type 6: Boolean
        elif isinstance(item, bool):
            return "true" if item else "false"
        
        # Type 7: Any other object
        else:
            # Try to get string representation
            try:
                # Check for common attributes
                for attr in ['name', 'title', 'label', 'topic', '__str__']:
                    if hasattr(item, attr):
                        value = getattr(item, attr)
                        if callable(value):
                            value = value()
                        if value and str(value).strip():
                            return str(value).strip()
            except:
                pass
            
            # Last resort: string representation
            return str(item)[:50]
    def _summarize_conversation(self, conversation_context: Optional[List[Dict]]) -> str:
        """Generate a brief summary of the conversation - ULTRA ROBUST VERSION."""
        if not conversation_context:
            return "New conversation"
        
        try:
            topics = []
            seen_topics = set()  # Avoid duplicates
            
            # Get categories - handle ANY possible structure
            categories_container = self.general_knowledge.get("categories", {})
            
            # DYNAMIC: Handle ANY type of categories container
            items_to_process = []
            
            if isinstance(categories_container, dict):
                # Normal case: dict with keys as topics
                items_to_process = list(categories_container.keys())
                # Also add string values as potential topics
                for value in categories_container.values():
                    if isinstance(value, str) and value.strip():
                        items_to_process.append(value)
                    elif isinstance(value, (dict, list, tuple, set)):
                        # Extract topics from nested structures
                        extracted = self._extract_topic_from_any_type(value)
                        if extracted and extracted != str(value)[:50]:
                            items_to_process.append(extracted)
            
            elif isinstance(categories_container, (list, tuple, set)):
                # categories is a list/tuple/set
                items_to_process = list(categories_container)
            
            elif hasattr(categories_container, '__iter__'):
                # Any other iterable
                try:
                    items_to_process = list(categories_container)
                except:
                    items_to_process = [categories_container]
            
            else:
                # Single item or non-iterable
                items_to_process = [categories_container]
            
            # Process each turn in conversation
            for turn in conversation_context[-3:]:
                # Get query - handle ANY type
                query_raw = turn.get("query", "")
                query = str(query_raw).lower().strip() if query_raw else ""
                
                if not query:
                    continue
                
                # Check each potential topic against the query
                for item in items_to_process:
                    # Extract clean topic string from ANY type
                    topic_candidate = self._extract_topic_from_any_type(item)
                    
                    # Skip empty or too long topics
                    if not topic_candidate or len(topic_candidate) > 100:
                        continue
                    
                    # Normalize for comparison
                    topic_lower = topic_candidate.lower().strip()
                    
                    # Check if topic appears in query (flexible matching)
                    if (topic_lower and 
                        topic_lower in query and 
                        topic_lower not in seen_topics and
                        len(topic_lower) > 2):  # Avoid single letters
                        
                        # Add the CLEAN topic string (not the raw item)
                        topics.append(topic_candidate)
                        seen_topics.add(topic_lower)
        
            # Generate summary
            if topics:
                # Clean up topics list
                clean_topics = []
                for topic in topics[:5]:  # Limit to 5 topics
                    if isinstance(topic, str):
                        clean_topics.append(topic)
                    else:
                        # Convert anything else to string
                        clean_topics.append(str(topic)[:30])  # Truncate
                
                return f"Discussion about: {', '.join(clean_topics[:3])}"
            else:
                return f"General conversation ({len(conversation_context)} turns)"
        
        except Exception as e:
            logger.error(f"Error in _summarize_conversation: {e}")
            # Fallback that always works
            return f"Conversation with {len(conversation_context)} messages"
           
    def _generate_follow_up_suggestions(self, 
                                       query: str, 
                                       response: str, 
                                       classification: Dict[str, Any]) -> List[str]:
        """Generate follow-up question suggestions."""
        response_type = classification.get("response_type", "general_assistance")
        domain = classification.get("primary_domain", "general")
        
        suggestions = {
            "instructional": [
                "Would you like more detailed steps?",
                "Should I explain the tools or materials needed?",
                "Do you want troubleshooting tips?",
                "Would you like video tutorial recommendations?"
            ],
            "explanatory": [
                "Would you like more examples?",
                "Should I explain the historical background?",
                "Do you want to know about current applications?",
                "Would you like related concepts explained?"
            ],
            "analytical": [
                "Would you like to explore different perspectives?",
                "Should I provide statistical data?",
                "Do you want case studies?",
                "Would you like expert opinions?"
            ],
            "comparative": [
                "Would you like a feature-by-feature comparison?",
                "Should I discuss pros and cons?",
                "Do you want user experience insights?",
                "Would you like price/performance analysis?"
            ],
            "general_assistance": [
                "What specific aspect interests you most?",
                "Would you like more details on any point?",
                "Should I relate this to other topics?",
                "Do you have any specific questions about this?"
            ]
        }
        
        # Domain-specific additions
        if domain in self.general_knowledge["categories"]:
            suggestions["general_assistance"].extend([
                f"Would you like recent developments in {domain}?",
                f"Should I recommend resources for learning {domain}?",
                f"Do you want practical applications of {domain}?"
            ])
        
        return suggestions.get(response_type, suggestions["general_assistance"])[:3]
    
    def _record_interaction(self, 
                          query: str, 
                          response: str, 
                          result: Dict[str, Any],
                          conversation_context: Optional[List[Dict]] = None):
        """Record interaction for learning and improvement."""
        try:
            metadata = {
                "domain": result["domain"],
                "response_type": result["response_type"],
                "confidence": result["confidence"],
                "generation_style": result["generation_style"],
                "conversation_length": len(conversation_context) if conversation_context else 0,
                "timestamp": hashlib.md5(query.encode()).hexdigest()[:16]
            }
            
            if hasattr(self, 'learning'):
                self.learning.record_interaction(
                    interaction_id=metadata["timestamp"],
                    prompt=query,
                    response=response,
                    domain=result["domain"],
                    metadata=metadata
                )
            
            logger.debug(f"Recorded general interaction: {metadata['timestamp']}")
            
        except Exception as e:
            logger.error(f"Failed to record interaction: {e}")
    
    async def engage_in_conversation(self, 
                                    messages: List[Dict[str, str]],
                                    max_turns: int = 10) -> Dict[str, Any]:
        """Engage in a multi-turn conversation."""
        
        if not messages:
            return {
                "response": "Hello! I'm your general assistant. How can I help you today?",
                "conversation_state": "initiated",
                "suggested_topics": self.general_knowledge["conversation_topics"][:5]
            }
        
        # Process the last message
        last_message = messages[-1]
        query = last_message.get("content", "")
        
        # Build conversation history
        conversation_history = [
            {"query": msg.get("content", ""), "response": ""}
            for msg in messages[:-1]  # All but last
        ]
        
        # Process with context
        result = await self.process_query(query, None, conversation_history)
        
        # Determine conversation state
        conversation_state = self._determine_conversation_state(messages, result)
        
        return {
            "response": result["response"],
            "conversation_state": conversation_state,
            "follow_up_questions": result.get("suggested_follow_ups", []),
            "current_domain": result["domain"],
            "conversation_length": len(messages) + 1,
            "context_awareness": result["context_used"]["conversation_history"]
        }
    
    def _determine_conversation_state(self, 
                                     messages: List[Dict[str, str]], 
                                     current_result: Dict[str, Any]) -> str:
        """Determine the current state of the conversation."""
        
        if len(messages) == 1:
            return "initial"
        
        # Check for topic consistency in recent messages
        recent_topics = []
        for i, msg in enumerate(messages[-3:]):  # Last 3 messages
            content = msg.get("content", "").lower()
            # Simple topic detection
            for category in self.general_knowledge["categories"]:
                if category in content:
                    recent_topics.append(category)
                    break
        
        if len(set(recent_topics)) == 1 and recent_topics:
            return "focused"
        elif len(set(recent_topics)) > 1:
            return "exploratory"
        else:
            return "general"
    
    def setup_generation(self, model_name: str = "microsoft/DialoGPT-medium") -> bool:
        """Setup general generation"""
        # Initialize generation module
        self._init_generation()
        
        success = self.generation_module.setup()
        if success:
            self.generation_enabled = True
            if self.generation_enabled:
                self.configure_generation(
                    #temperature=0.7,  # Medium for general conversation
                    top_p=0.9,
                    #max_length=300
                )        
            self.generation_module.model_name = model_name
            logger.info(f"General generation enabled for {self.instance_id}")
            logger.info(f"Enhanced GeneralVNI capabilities initialized")
        return success
