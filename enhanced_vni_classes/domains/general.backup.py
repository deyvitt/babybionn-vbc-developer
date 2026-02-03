"""General domain VNI implementation"""
from .base_knowledge_loader import BaseKnowledgeLoader
from typing import Dict, Any, Optional, List
from ..core.base_vni import EnhancedBaseVNI
from ..core.capabilities import VNICapabilities, VNIType
from ..modules.generation import EnhancedGenerationModule, GenerationStyle # TextGenerator, GenerationModule
from ..modules.classifier import DomainClassifier
from ..modules.web_search import WebSearch
from ..modules.attention import AttentionMechanism
from ..modules.learning_system import LearningSystem
from ..utils.logger import get_logger
import hashlib

logger = get_logger(__name__)

class GeneralVNI(EnhancedBaseVNI, BaseKnowledgeLoader):
    """General domain VNI"""
    def __init__(self, instance_id: str = "general_001", vni_config: Dict[str, Any] = None, auto_load_knowledge: bool = True):
        # Set up capabilities
        capabilities = VNICapabilities(
            domains=["general", "conversation", "information", "assistance", "multi_domain"],
            can_search=True,
            can_learn=True,
            can_collaborate=True,
            max_context_length=3000,
            special_abilities=["multi_domain", "conversational", "task_assistance", "knowledge_synthesis"],
            vni_type='specialized'
        )
        
        # Initialize base VNI - FIXED VERSION
        super().__init__(
            instance_id=instance_id,
            domain="general", 
            capabilities=capabilities,
            vni_config=vni_config or {}
        )
        
        # General-specific setup
        self.vni_type = "general" 
        self.generation_module = EnhancedGenerationModule(domain="general")
        self.general_knowledge = self._initialize_general_knowledge()
        
        # Initialize enhanced modules - no need already initialized in base
        # self.classifier = DomainClassifier()
        self.web_search = WebSearch(vni_id=self.instance_id)
        # self.attention = AttentionMechanism()
        # self.learning = LearningSystem()

        # Load knowledge from files if enabled
        if auto_load_knowledge:
            self.load_domain_knowledge("general")        
        
        logger.info(f"Initialized GeneralVNI: {instance_id}")

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
        Process general query with enhanced capabilities
        """
        # Extract conversation context if provided
        if context and "conversation_history" in context:
            conversation_context = context["conversation_history"]
        
        # Classify query for better understanding
        classification = self._classify_query(query)
        
        # Build comprehensive context
        enhanced_context = await self._build_enhanced_context(query, classification, conversation_context)
        
        # Determine generation style
        generation_style = self._determine_generation_style(query, classification)
        
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
        
        # Record interaction for learning
        self._record_interaction(query, response_text, result, conversation_context)
        
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
        """Build comprehensive context for query processing."""
        context = {
            "knowledge": {"content": "", "confidence": 0.0},
            "web_results": [],
            "conversation_history": [],
            "query_analysis": classification
        }
        
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
