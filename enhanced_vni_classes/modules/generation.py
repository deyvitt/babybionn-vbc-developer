# enhanced_vni_classes/modules/generation.py
"""
Enhanced Text Generation Module for VNIs
Combines LLM-based generation with style control and fallback mechanisms
"""
import torch
import random
import re
import hashlib
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.logger import get_logger

logger = get_logger(__name__)

class GenerationStyle(Enum):
    """Generation style options"""
    CONCISE = "concise"
    DETAILED = "detailed"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    FORMAL = "formal"
    CASUAL = "casual"
    DOMAIN_SPECIFIC = "domain_specific"

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_length: int = 512
    repetition_penalty: float = 1.2
    do_sample: bool = True
    num_return_sequences: int = 1
    style: str = GenerationStyle.DETAILED.value
    
    @classmethod
    def get_domain_defaults(cls, domain: str) -> 'GenerationConfig':
        """Get default configuration for a domain"""
        defaults = {
            "medical": cls(temperature=0.6, top_p=0.85, max_length=512),
            "legal": cls(temperature=0.5, top_p=0.8, max_length=512),
            "technical": cls(temperature=0.6, top_p=0.85, max_length=1024),
            "general": cls(temperature=0.7, top_p=0.9, max_length=512)
        }
        return defaults.get(domain, cls())

@dataclass
class ResponsePattern:
    """Learned response pattern"""
    prompt_pattern: str
    response_pattern: str
    domain: str
    usage_count: int = 0
    success_rate: float = 0.0
    confidence: float = 0.5
    last_used: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def update(self, success_metric: float):
        """Update pattern metrics"""
        self.usage_count += 1
        self.success_rate = (self.success_rate * (self.usage_count - 1) + success_metric) / self.usage_count
        self.confidence = min(0.95, self.confidence + 0.1 * success_metric)
        self.last_used = datetime.now().isoformat()

@dataclass
class StyleTemplate:
    """Template for generation styles"""
    opening: str
    closing: str
    connector: str
    max_length: int
    description: str

class EnhancedGenerationModule:
    """
    Enhanced text generation with LLM integration and style control
    Supports both LLM-based generation and rule-based enhancement
    """
    
    def __init__(self, 
                 domain: str = "general",
                 model_name: str = "microsoft/DialoGPT-medium",
                 bridge_dim: int = 512,
                 enable_llm: bool = True,
                 vni_id: Optional[str] = None):
        """
        Args:
            domain: Domain for generation
            model_name: LLM model to use
            bridge_dim: Dimension for bridge layer
            enable_llm: Whether to enable LLM generation
            vni_id: Optional VNI identifier for patterns
        """
        self.domain = domain
        self.model_name = model_name
        self.bridge_dim = bridge_dim
        self.enable_llm = enable_llm
        self.vni_id = vni_id or f"gen_{domain}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:6]}"
        
        # LLM components
        self.model = None
        self.tokenizer = None
        self.bridge_layer = None
        
        # Style and pattern management
        self.response_patterns: Dict[str, ResponsePattern] = {}
        self.style_templates = self._init_style_templates()
        self.quality_threshold = 0.7
        
        # Configuration
        self.config = GenerationConfig.get_domain_defaults(domain)
        self.generation_config = {}
        
        # Statistics
        self.generation_count = 0
        self.llm_success_count = 0
        self.fallback_count = 0
        
        logger.info(f"Initialized EnhancedGenerationModule for {domain} (LLM: {enable_llm})")
    
    def _init_style_templates(self) -> Dict[str, StyleTemplate]:
        """Initialize style templates"""
        return {
            GenerationStyle.CONCISE.value: StyleTemplate(
                opening="",
                closing="",
                connector=". ",
                max_length=100,
                description="Brief and to-the-point responses"
            ),
            GenerationStyle.DETAILED.value: StyleTemplate(
                opening="Let me provide a detailed explanation: ",
                closing="",
                connector=". Furthermore, ",
                max_length=500,
                description="Comprehensive and explanatory responses"
            ),
            GenerationStyle.TECHNICAL.value: StyleTemplate(
                opening="From a technical perspective: ",
                closing="",
                connector=". Additionally, ",
                max_length=300,
                description="Technical and precise responses"
            ),
            GenerationStyle.CREATIVE.value: StyleTemplate(
                opening="Here's a creative approach: ",
                closing="What do you think?",
                connector=". Imagine that ",
                max_length=400,
                description="Imaginative and innovative responses"
            ),
            GenerationStyle.FORMAL.value: StyleTemplate(
                opening="I would like to address your inquiry: ",
                closing="Thank you for your question.",
                connector=". Moreover, ",
                max_length=250,
                description="Professional and formal responses"
            ),
            GenerationStyle.CASUAL.value: StyleTemplate(
                opening="Hey there! ",
                closing="Hope that helps!",
                connector=". Also, ",
                max_length=150,
                description="Friendly and informal responses"
            ),
            GenerationStyle.DOMAIN_SPECIFIC.value: StyleTemplate(
                opening=f"As a {self.domain} assistant: ",
                closing=f"This is based on {self.domain} knowledge.",
                connector=". In this context, ",
                max_length=400,
                description=f"Domain-specific responses for {self.domain}"
            )
        }
    
    def setup(self) -> bool:
        """Setup generation model"""
        if not self.enable_llm:
            logger.info("LLM generation disabled, using enhanced rule-based generation")
            return True
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Setup bridge layer if needed
            if self.bridge_dim > 0:
                model_hidden_size = self.model.config.hidden_size
                self.bridge_layer = torch.nn.Linear(self.bridge_dim, model_hidden_size)
                logger.info(f"Bridge layer: {self.bridge_dim} -> {model_hidden_size}")
            
            # Set generation config
            self.generation_config = {
                "max_new_tokens": self.config.max_length,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "do_sample": self.config.do_sample,
                "num_return_sequences": self.config.num_return_sequences,
                "repetition_penalty": self.config.repetition_penalty
            }
            
            logger.info(f"LLM setup complete for {self.domain} using {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"LLM setup failed: {e}")
            logger.info("Falling back to enhanced rule-based generation")
            self.enable_llm = False
            return True
    
    def generate(self,
                query: str,
                context: Dict[str, Any] = None,
                style: Optional[str] = None,
                use_llm: Optional[bool] = None) -> Dict[str, Any]:
        """
        Generate response using best available method
        
        Args:
            query: User query
            context: Additional context for generation
            style: Generation style (overrides default)
            use_llm: Force LLM usage (None = auto-select)
        """
        self.generation_count += 1
        
        # Determine generation method
        should_use_llm = self._should_use_llm(query, context, use_llm)
        
        # Generate response
        if should_use_llm and self.enable_llm and self.model is not None:
            result = self._generate_with_llm(query, context, style)
            if result.get("confidence", 0) > 0.3:
                self.llm_success_count += 1
                return result
        
        # Fallback to enhanced rule-based generation
        self.fallback_count += 1
        result = self._generate_enhanced(query, context, style)
        
        return result
    
    def _should_use_llm(self, 
                       query: str, 
                       context: Dict[str, Any], 
                       use_llm: Optional[bool]) -> bool:
        """Determine whether to use LLM generation"""
        if use_llm is not None:
            return use_llm
        
        # Auto-selection logic
        query_lower = query.lower()
        
        # Don't use LLM for very short queries
        if len(query.split()) < 3:
            return False
        
        # Use LLM for complex queries
        complex_indicators = ['explain', 'describe', 'analyze', 'compare', 'elaborate']
        if any(indicator in query_lower for indicator in complex_indicators):
            return True
        
        # Use LLM if context has sufficient information
        if context and context.get("knowledge") and len(context["knowledge"]) > 100:
            return True
        
        # Default to rule-based for efficiency
        return False
    
    def _generate_with_llm(self,
                          query: str,
                          context: Dict[str, Any],
                          style: Optional[str] = None) -> Dict[str, Any]:
        """Generate response using LLM"""
        try:
            # Create enhanced prompt
            prompt = self._create_enhanced_prompt(query, context, style)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    **self.generation_config,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and extract response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_text = generated_text.replace(prompt, "").strip()
            
            # Apply style enhancement if needed
            if style and style != GenerationStyle.DOMAIN_SPECIFIC.value:
                response_text = self._apply_style_to_text(response_text, style)
            
            # Validate and format response
            return self._format_llm_response(response_text, query, context, style)
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return self._create_fallback_response(query, "llm_error")
    
    def _generate_enhanced(self,
                          query: str,
                          context: Dict[str, Any],
                          style: Optional[str] = None) -> Dict[str, Any]:
        """Generate response using enhanced rule-based methods"""
        # Get base content
        base_content = self._extract_base_content(context)
        
        # Apply response patterns if available
        pattern_response = self._apply_response_patterns(query, base_content)
        if pattern_response:
            response_text = pattern_response
        else:
            # Generate using template-based approach
            response_text = self._generate_from_template(query, base_content, style)
        
        # Assess quality
        quality_score = self._assess_response_quality(response_text, query, context)
        
        # Improve if needed
        if quality_score < self.quality_threshold:
            response_text = self._improve_response(response_text, query, context)
            quality_score = self._assess_response_quality(response_text, query, context)
        
        return self._format_enhanced_response(response_text, query, quality_score, style)
    
    def _create_enhanced_prompt(self, 
                               query: str, 
                               context: Dict[str, Any],
                               style: Optional[str] = None) -> str:
        """Create enhanced prompt for LLM generation"""
        style_info = f" Respond in a {style} style." if style else ""
        
        # Base domain prompts
        domain_prompts = {
            "medical": f"""You are BabyBIONN-Medical, an AI assistant specializing in medical knowledge.
You have expertise in symptoms, treatments, medications, procedures, and health advice.
Always provide accurate, professional medical information while emphasizing the importance of consulting healthcare professionals.

Query: {query}
Context: {context.get('knowledge', '')[:500] if context else ''}
{style_info}

Please provide a professional, accurate medical response based on your knowledge:""",
            
            "legal": f"""You are BabyBIONN-Legal, an AI assistant specializing in legal knowledge.
You have expertise in contracts, rights, liabilities, regulations, and legal procedures.
Always provide accurate legal information while emphasizing the importance of consulting qualified attorneys.

Query: {query}
Context: {context.get('knowledge', '')[:500] if context else ''}
{style_info}

Please provide a professional, accurate legal response based on your knowledge:""",
            
            "general": f"""You are BabyBIONN, an AI assistant specializing in general knowledge.
You have expertise in technical, business, analytical, and creative domains.

Query: {query}
Context: {context.get('knowledge', '')[:500] if context else ''}
{style_info}

Please provide a helpful, informative response based on your knowledge:"""
        }
        
        prompt = domain_prompts.get(self.domain, domain_prompts["general"])
        
        # Add style-specific instructions
        if style in self.style_templates:
            template = self.style_templates[style]
            prompt += f"\nStyle requirements: {template.description}"
        
        return prompt
    
    def _extract_base_content(self, context: Dict[str, Any]) -> str:
        """Extract base content from context"""
        if not context:
            return ""
        
        content_parts = []
        
        # Add knowledge base content
        if context.get("knowledge"):
            content_parts.append(context["knowledge"])
        
        # Add web results
        if context.get("web_results"):
            for result in context["web_results"][:2]:
                snippet = result.get("snippet", "")
                if snippet:
                    content_parts.append(snippet)
        
        # Add collaboration results
        if context.get("collaboration_results"):
            for collab in context["collaboration_results"][:2]:
                response = collab.get("response", "")
                if response:
                    content_parts.append(response)
        
        # Add domain-specific concepts
        if context.get("domain_concepts"):
            concepts = list(context["domain_concepts"].keys())[:5]
            content_parts.append(f"Relevant concepts: {', '.join(concepts)}")
        
        return " ".join(content_parts)
    
    def _apply_response_patterns(self, query: str, base_content: str) -> Optional[str]:
        """Apply learned response patterns"""
        pattern_key = self._create_pattern_key(query)
        
        if pattern_key in self.response_patterns:
            pattern = self.response_patterns[pattern_key]
            
            # Apply pattern if confident
            if pattern.confidence > 0.8 and pattern.usage_count > 3:
                try:
                    # Use pattern to enhance base content
                    response = pattern.response_pattern
                    if "{content}" in response and base_content:
                        response = response.replace("{content}", base_content[:200])
                    elif "{query}" in response:
                        response = response.replace("{query}", query)
                    
                    pattern.update(0.8)  # Success metric
                    return response
                except Exception as e:
                    logger.warning(f"Pattern application failed: {e}")
        
        return None
    
    def _create_pattern_key(self, query: str) -> str:
        """Create key for response pattern"""
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()[:8]
        return f"{self.domain}_{query_hash}"
    
    def _generate_from_template(self, 
                               query: str, 
                               base_content: str,
                               style: Optional[str] = None) -> str:
        """Generate response using templates"""
        # Determine style
        use_style = style or self.config.style
        
        # Get appropriate template
        if use_style in self.style_templates:
            template = self.style_templates[use_style]
        else:
            template = self.style_templates[GenerationStyle.DOMAIN_SPECIFIC.value]
        
        # Create response
        if not base_content:
            response = f"{template.opening}I don't have specific information about '{query}'.{template.closing}"
        else:
            # Process base content
            sentences = self._split_into_sentences(base_content)
            if not sentences:
                response = f"{template.opening}I need more information about '{query}'.{template.closing}"
            else:
                # Build response using template
                response = template.opening
                for i, sentence in enumerate(sentences[:5]):  # Limit to 5 sentences
                    response += sentence
                    if i < min(4, len(sentences) - 1):
                        response += template.connector
                response += template.closing
        
        # Ensure length constraints
        if len(response) > template.max_length:
            response = response[:template.max_length].rsplit('.', 1)[0] + '.'
            if template.closing:
                response += " " + template.closing
        
        return response
    
    def _apply_style_to_text(self, text: str, style: str) -> str:
        """Apply style transformation to existing text"""
        if style not in self.style_templates:
            return text
        
        template = self.style_templates[style]
        
        # Add opening if not present
        if not text.startswith(template.opening) and template.opening:
            text = template.opening + text
        
        # Add closing if not present
        if not text.endswith(template.closing) and template.closing:
            text = text + template.closing
        
        return text
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def _assess_response_quality(self, 
                                response: str, 
                                query: str, 
                                context: Dict[str, Any]) -> float:
        """Assess response quality"""
        quality_score = 0.5  # Base score
        
        # Length appropriateness
        word_count = len(response.split())
        if 30 <= word_count <= 200:
            quality_score += 0.1
        elif word_count > 200:
            quality_score -= 0.1
        
        # Relevance (keyword overlap)
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        response_words = set(re.findall(r'\b\w+\b', response.lower()))
        common_words = query_words.intersection(response_words)
        
        if common_words:
            relevance = len(common_words) / max(len(query_words), 1)
            quality_score += min(0.3, relevance)
        
        # Coherence check
        sentences = self._split_into_sentences(response)
        if len(sentences) >= 2:
            quality_score += 0.1
        
        # Specificity (avoid vague phrases)
        vague_phrases = ['i think', 'maybe', 'perhaps', 'possibly', 'not sure']
        if not any(phrase in response.lower() for phrase in vague_phrases):
            quality_score += 0.05
        
        return min(1.0, max(0.0, quality_score))
    
    def _improve_response(self, 
                         response: str, 
                         query: str, 
                         context: Dict[str, Any]) -> str:
        """Improve low-quality response"""
        improvements = [
            "Based on available information, ",
            "To address your specific question, ",
            "Considering the context provided, ",
            "From what I understand, "
        ]
        
        if response.startswith(('I don\'t', 'I need', 'I can\'t')):
            return f"{random.choice(improvements)}I'll do my best to help with '{query}'. Could you provide more details?"
        
        # Add structure if response is too short
        if len(response.split()) < 20:
            return f"{random.choice(improvements)}{response} I can provide more details if needed."
        
        return response
    
    def _format_llm_response(self, 
                            response_text: str, 
                            query: str,
                            context: Dict[str, Any],
                            style: Optional[str]) -> Dict[str, Any]:
        """Format LLM response"""
        quality_score = self._assess_response_quality(response_text, query, context)
        
        return {
            "response": response_text,
            "confidence": min(0.3 + (len(response_text.split()) / 100), 0.9),
            "quality_score": quality_score,
            "generated": True,
            "generation_method": "llm",
            "generation_model": self.model_name,
            "response_type": f"{self.domain}_llm_generated",
            "style_applied": style or self.config.style,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "metadata": {
                "tokens_estimated": len(response_text.split()),
                "bridge_used": self.bridge_layer is not None,
                "temperature": self.config.temperature
            }
        }
    
    def _format_enhanced_response(self, 
                                 response_text: str, 
                                 query: str,
                                 quality_score: float,
                                 style: Optional[str]) -> Dict[str, Any]:
        """Format enhanced rule-based response"""
        return {
            "response": response_text,
            "confidence": min(0.2 + quality_score * 0.5, 0.8),
            "quality_score": quality_score,
            "generated": True,
            "generation_method": "enhanced_rule",
            "response_type": f"{self.domain}_enhanced",
            "style_applied": style or self.config.style,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "metadata": {
                "pattern_used": self._create_pattern_key(query) in self.response_patterns,
                "base_content_available": bool(response_text and len(response_text.split()) > 10)
            }
        }
    
    def _create_fallback_response(self, query: str, error_type: str) -> Dict[str, Any]:
        """Create fallback response"""
        fallback_responses = {
            "llm_error": f"I need to learn more about '{query}' in the {self.domain} domain.",
            "general": f"As a {self.domain} assistant, I'm still learning about '{query}'.",
            "medical": f"Medical information about '{query}' requires professional consultation.",
            "legal": f"Legal aspects of '{query}' should be reviewed by a qualified attorney."
        }
        
        response = fallback_responses.get(
            error_type, 
            fallback_responses.get(self.domain, fallback_responses["general"])
        )
        
        return {
            "response": response,
            "confidence": 0.1,
            "generated": False,
            "generation_method": "fallback",
            "response_type": "fallback",
            "quality_score": 0.3,
            "timestamp": datetime.now().isoformat(),
            "query": query
        }
    
    def learn_response_pattern(self, 
                              query: str, 
                              response: str, 
                              success_metric: float = 0.8):
        """Learn a successful response pattern"""
        pattern_key = self._create_pattern_key(query)
        
        if pattern_key not in self.response_patterns:
            self.response_patterns[pattern_key] = ResponsePattern(
                prompt_pattern=query,
                response_pattern=response,
                domain=self.domain,
                usage_count=1,
                success_rate=success_metric,
                confidence=0.5 + success_metric * 0.5
            )
        else:
            self.response_patterns[pattern_key].update(success_metric)
        
        logger.info(f"Learned response pattern for query: {query[:50]}...")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            "total_generations": self.generation_count,
            "llm_success_count": self.llm_success_count,
            "fallback_count": self.fallback_count,
            "llm_success_rate": self.llm_success_count / max(self.generation_count, 1),
            "response_patterns_count": len(self.response_patterns),
            "domain": self.domain,
            "llm_enabled": self.enable_llm,
            "model_loaded": self.model is not None
        }
    
    def get_available_styles(self) -> List[Dict[str, Any]]:
        """Get available generation styles"""
        return [
            {
                "name": style_name,
                "description": template.description,
                "max_length": template.max_length,
                "opening_example": template.opening[:50] + "..." if template.opening else "",
                "suitable_for": ["general", self.domain]
            }
            for style_name, template in self.style_templates.items()
        ]
    
    def update_config(self, **kwargs):
        """Update generation configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Update LLM generation config if needed
        if self.enable_llm and self.model:
            self.generation_config.update({
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_new_tokens": self.config.max_length
            })
        
        logger.info(f"Updated generation config: {kwargs}")

# Factory function for easy creation
def create_generation_module(domain: str = "general", 
                           enable_llm: bool = True,
                           model_name: Optional[str] = None,
                           vni_id: Optional[str] = None) -> EnhancedGenerationModule:
    """Create a generation module with sensible defaults"""
    if model_name is None:
        # Select model based on domain
        model_mapping = {
            "medical": "microsoft/DialoGPT-medium",
            "legal": "microsoft/DialoGPT-medium",
            "technical": "gpt2-medium",
            "general": "microsoft/DialoGPT-medium"
        }
        model_name = model_mapping.get(domain, "microsoft/DialoGPT-medium")
    
    module = EnhancedGenerationModule(
        domain=domain,
        model_name=model_name,
        enable_llm=enable_llm,
        vni_id=vni_id
    )
    
    # Setup the module
    success = module.setup()
    if not success and enable_llm:
        logger.warning(f"LLM setup failed for {domain}, falling back to rule-based generation")
    
    return module
