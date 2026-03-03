# File: template_engine.py
"""
Enhanced Hybrid-Adaptive Template Engine
Converts BabyBIONN reasoning output → LLM instructional prompts

Improvements:
- Input validation with Pydantic
- Error handling and fallbacks
- Template caching and performance optimization
- Analytics and metrics tracking
- Extensibility via plugins
- Configuration management
- Better type safety
"""
import re
import json
import hashlib
import logging
from enum import Enum
from typing import Optional, Dict, List, Any, Tuple, DefaultDict, Callable, Set
from datetime import datetime
from pydantic import BaseModel, Field, validator, ValidationError
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger("template_engine")


# ============================================================================
# CONFIGURATION
# ============================================================================

class EngineConfig(BaseModel):
    """Configuration for the template engine"""
    enable_caching: bool = True
    max_cache_size: int = 128
    enable_metrics: bool = True
    enable_component_learning: bool = True
    strict_validation: bool = True
    default_confidence_threshold: float = 0.4
    
    class Config:
        frozen = True  # Immutable config


# ============================================================================
# ENHANCED DATA MODELS
# ============================================================================

class ReasoningAnalysis(BaseModel):
    """Analysis of reasoning output with validation"""
    reasoning_type: 'ReasoningType'
    domains: List[str] = Field(min_items=0, max_items=10)
    has_conflicts: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    complexity: float = Field(ge=0.0, le=10.0)
    metadata: Dict[str, Any] = Field(default_factory=Dict)
    
    @validator('confidence', 'complexity')
    def validate_ranges(cls, v):
        """Ensure values are in valid ranges"""
        return max(0.0, min(v, 1.0 if 'confidence' in str(cls) else 10.0))
    
    class Config:
        use_enum_values = False  # Keep enum objects, not values


class AggregatorOutput(BaseModel):
    """Validated aggregator output structure"""
    domains_used: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=Dict)
    has_conflicts: bool = False
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    query_complexity: float = Field(default=0.5, ge=0.0, le=1.0)
    synthesis_method: str = "balanced_integration"
    reasoning_steps: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        extra = "allow"  # Allow additional fields


class ReasoningType(Enum):
    MEDICAL_EMERGENCY = "medical_emergency"
    LEGAL_ADVISORY = "legal_advisory"
    TECHNICAL_EXPLANATION = "technical_explanation"
    FINANCIAL_ANALYSIS = "financial_analysis"
    CROSS_DOMAIN = "cross_domain"
    UNKNOWN = "unknown"


class TemplateSafetyLevel(Enum):
    MAX_SAFETY = "max_safety"  # Critical: medical, legal, safety
    HIGH_SAFETY = "high_safety"  # Important: financial, technical
    MEDIUM_SAFETY = "medium_safety"  # General: educational, creative
    EXPLORATORY = "exploratory"  # Experimental: novel combinations
    
    def __lt__(self, other):
        """Enable safety level comparison"""
        if not isinstance(other, TemplateSafetyLevel):
            return NotImplemented
        order = [self.EXPLORATORY, self.MEDIUM_SAFETY, self.HIGH_SAFETY, self.MAX_SAFETY]
        return order.index(self) < order.index(other)


@dataclass
class TemplateComponent:
    """Modular template component with learning capabilities"""
    id: str
    content: str
    category: str  # "structure", "tone", "constraint", "enhancement"
    safety_level: TemplateSafetyLevel
    usage_count: int = 0
    success_rate: float = 0.0
    last_used: datetime = field(default_factory=datetime.now)
    tags: Set[str] = field(default_factory=set)
    
    def update_metrics(self, success: bool):
        """Update component metrics after use"""
        self.usage_count += 1
        self.last_used = datetime.now()
        # Exponential moving average for success rate
        alpha = 0.2
        self.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate


@dataclass
class PromptMetrics:
    """Metrics for generated prompts"""
    reasoning_type: str
    safety_level: str
    confidence: float
    complexity: float
    template_used: str
    components_used: List[str]
    generation_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# TEMPLATE ENGINE
# ============================================================================

class HybridAdaptiveTemplateEngine:
    """
    Enhanced hybrid template engine with:
    - Static core templates for safety
    - Dynamic adaptation for flexibility
    - Learning from successful patterns
    - Performance optimization via caching
    - Comprehensive error handling
    - Metrics tracking
    """
    
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        
        # Core static templates (validated, safe)
        self.core_templates = self._load_core_templates()
        
        # Dynamic template components
        self.components = self._load_components()
        
        # Learned patterns
        self.pattern_library: Dict[str, str] = {}
        
        # Analytics
        self.usage_stats: DefaultDict[str, int] = defaultdict(int)
        self.metrics_history: List[PromptMetrics] = []
        
        # Performance tracking
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Plugin system
        self.preprocessors: List[Callable] = []
        self.postprocessors: List[Callable] = []
        
        logger.info(
            f"Template Engine initialized with hybrid architecture "
            f"(caching={self.config.enable_caching}, metrics={self.config.enable_metrics})"
        )
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def create_prompt(
        self,
        aggregator_output: Dict[str, Any],
        user_query: str,
        user_context: Optional[Dict] = None
    ) -> str:
        """
        Create LLM prompt from BabyBIONN reasoning output
        
        Args:
            aggregator_output: Dictionary containing reasoning analysis
            user_query: Original user query string
            user_context: Optional user context (risk level, preferences, etc.)
            
        Returns:
            Formatted LLM prompt string
            
        Raises:
            ValidationError: If input data is invalid
            RuntimeError: If template generation fails
        """
        start_time = datetime.now()
        
        try:
            # Validate and normalize input
            validated_output = self._validate_input(aggregator_output)
            
            # Apply preprocessors
            for preprocessor in self.preprocessors:
                validated_output = preprocessor(validated_output)
            
            # Step 1: Analyze reasoning
            analysis = self._analyze_reasoning(validated_output.Dict())
            
            # Step 2: Determine safety level
            safety_level = self._determine_safety_level(analysis, user_context)
            
            # Step 3: Select or create template (with caching)
            template = self._get_template(analysis, validated_output.Dict())
            
            # Step 4: Enhance with dynamic components
            enhanced_template = self._enhance_template(template, analysis, safety_level)
            
            # Step 5: Format final prompt
            final_prompt = self._format_prompt(
                enhanced_template, user_query, validated_output.Dict()
            )
            
            # Apply postprocessors
            for postprocessor in self.postprocessors:
                final_prompt = postprocessor(final_prompt)
            
            # Update metrics
            self._record_metrics(
                analysis, safety_level, template, 
                (datetime.now() - start_time).total_seconds() * 1000
            )
            
            # Update usage stats
            self.usage_stats[analysis.reasoning_type.value] += 1
            
            logger.info(
                f"Created prompt for {analysis.reasoning_type.value} "
                f"with {safety_level.value} safety "
                f"(confidence: {analysis.confidence:.2%})"
            )
            
            return final_prompt
            
        except ValidationError as e:
            logger.error(f"Input validation failed: {e}")
            if self.config.strict_validation:
                raise
            # Fallback to permissive mode
            return self._create_fallback_prompt(user_query, aggregator_output)
            
        except Exception as e:
            logger.exception(f"Error creating prompt: {e}")
            raise RuntimeError(f"Template generation failed: {e}") from e
    
    def register_preprocessor(self, func: Callable) -> None:
        """Register a preprocessing function"""
        self.preprocessors.append(func)
        logger.info(f"Registered preprocessor: {func.__name__}")
    
    def register_postprocessor(self, func: Callable) -> None:
        """Register a postprocessing function"""
        self.postprocessors.append(func)
        logger.info(f"Registered postprocessor: {func.__name__}")
    
    def add_component(self, component: TemplateComponent) -> None:
        """Add a new template component"""
        self.components[component.id] = component
        logger.info(f"Added component: {component.id} (category: {component.category})")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get analytics summary"""
        if not self.metrics_history:
            return {"message": "No metrics collected yet"}
        
        return {
            "total_prompts": len(self.metrics_history),
            "cache_hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses),
            "reasoning_type_distribution": Dict(self.usage_stats),
            "avg_confidence": sum(m.confidence for m in self.metrics_history) / len(self.metrics_history),
            "avg_generation_time_ms": sum(m.generation_time_ms for m in self.metrics_history) / len(self.metrics_history),
            "component_usage": self._get_component_usage_stats(),
        }
    
    def save_state(self, filepath: Path) -> None:
        """Save engine state for persistence"""
        state = {
            "usage_stats": Dict(self.usage_stats),
            "pattern_library": self.pattern_library,
            "component_metrics": {
                cid: {
                    "usage_count": c.usage_count,
                    "success_rate": c.success_rate,
                }
                for cid, c in self.components.items()
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Engine state saved to {filepath}")
    
    def load_state(self, filepath: Path) -> None:
        """Load engine state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.usage_stats.update(state.get("usage_stats", {}))
            self.pattern_library.update(state.get("pattern_library", {}))
            
            # Update component metrics
            for cid, metrics in state.get("component_metrics", {}).items():
                if cid in self.components:
                    self.components[cid].usage_count = metrics["usage_count"]
                    self.components[cid].success_rate = metrics["success_rate"]
            
            logger.info(f"Engine state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    # ========================================================================
    # INTERNAL METHODS
    # ========================================================================
    
    def _validate_input(self, aggregator_output: Dict[str, Any]) -> AggregatorOutput:
        """Validate and normalize aggregator output"""
        try:
            return AggregatorOutput(**aggregator_output)
        except ValidationError as e:
            logger.warning(f"Validation errors: {e}")
            # Try to fix common issues
            fixed_output = aggregator_output.copy()
            fixed_output.setdefault("domains_used", [])
            fixed_output.setdefault("confidence", 0.5)
            return AggregatorOutput(**fixed_output)
    
    def _get_template(
        self, 
        analysis: ReasoningAnalysis, 
        aggregator_output: Dict
    ) -> str:
        """Get template with optional caching"""
        if self.config.enable_caching:
            # Create cache key
            cache_key = self._create_cache_key(analysis)
            cached = self._get_from_cache(cache_key)
            if cached:
                self._cache_hits += 1
                return cached
            self._cache_misses += 1
        
        # Generate template
        reasoning_type_value = analysis.reasoning_type.value
        
        # Map ReasoningType values to template keys if needed
        template_key_map = {
            ReasoningType.CROSS_DOMAIN.value: "cross_domain_synthesis",
            ReasoningType.UNKNOWN.value: "general_unknown"
        }
        
        template_key = template_key_map.get(reasoning_type_value, reasoning_type_value)
        
        if template_key in self.core_templates:
            template = self.core_templates[template_key]
        else:
            template = self._create_adaptive_template(analysis)
        
        # Cache if enabled
        if self.config.enable_caching:
            self._add_to_cache(cache_key, template)
        
        return template
    
    @lru_cache(maxsize=128)
    def _create_cache_key(self, analysis: ReasoningAnalysis) -> str:
        """Create cache key for template lookup"""
        # Use reasoning type and sorted domains as key
        key_data = {
            "type": analysis.reasoning_type.value,
            "domains": sorted(analysis.domains),
            "has_conflicts": analysis.has_conflicts,
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[str]:
        """Get template from cache"""
        return self.pattern_library.get(key)
    
    def _add_to_cache(self, key: str, template: str) -> None:
        """Add template to cache"""
        self.pattern_library[key] = template
    
    def _record_metrics(
        self,
        analysis: ReasoningAnalysis,
        safety_level: TemplateSafetyLevel,
        template: str,
        generation_time_ms: float
    ) -> None:
        """Record metrics for this prompt generation"""
        if not self.config.enable_metrics:
            return
        
        metrics = PromptMetrics(
            reasoning_type=analysis.reasoning_type.value,
            safety_level=safety_level.value,
            confidence=analysis.confidence,
            complexity=analysis.complexity,
            template_used=template[:50] + "...",
            components_used=[],  # Could track which components were used
            generation_time_ms=generation_time_ms,
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics (last 1000)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def _get_component_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics for components"""
        return {
            cid: {
                "usage_count": comp.usage_count,
                "success_rate": comp.success_rate,
                "category": comp.category,
            }
            for cid, comp in self.components.items()
            if comp.usage_count > 0
        }
    
    def _create_fallback_prompt(
        self, 
        user_query: str, 
        aggregator_output: Dict
    ) -> str:
        """Create a basic fallback prompt when validation fails"""
        logger.warning("Using fallback prompt generation")
        return f"""
# BabyBIONN Generation Instructions (Fallback Mode)

## USER QUERY:
"{user_query}"

## NOTE:
Input validation failed. Using fallback template.

## GENERATION REQUIREMENTS:
1. Provide a helpful response to the user's query
2. Use appropriate caution and disclaimers
3. Recommend consulting professionals if appropriate
4. Be clear about any limitations or uncertainties

Please generate the final response for the user.
"""
    
    def _load_core_templates(self) -> Dict[str, str]:
        """Load validated core templates"""
        return {
            "medical_emergency": self._create_medical_emergency_template(),
            "legal_advisory": self._create_legal_advisory_template(),
            "technical_explanation": self._create_technical_template(),
            "financial_analysis": self._create_financial_template(),
            "cross_domain_synthesis": self._create_cross_domain_template(),
            "general_unknown": self._create_general_template()
        }
    
    def _load_components(self) -> Dict[str, TemplateComponent]:
        """Load modular template components"""
        components = {}
        
        # Structure components
        components["hierarchical_structure"] = TemplateComponent(
            id="hierarchical_structure",
            content="Present information in hierarchical order: most critical first, then important, then background.",
            category="structure",
            safety_level=TemplateSafetyLevel.HIGH_SAFETY,
            tags={"structure", "organization"}
        )
        
        components["step_by_step"] = TemplateComponent(
            id="step_by_step",
            content="Present information in clear, numbered steps for easy follow-along.",
            category="structure",
            safety_level=TemplateSafetyLevel.MEDIUM_SAFETY,
            tags={"structure", "clarity"}
        )
        
        # Tone components
        components["urgent_tone"] = TemplateComponent(
            id="urgent_tone",
            content="Use urgent, directive language. Be clear and forceful about immediate actions.",
            category="tone",
            safety_level=TemplateSafetyLevel.MAX_SAFETY,
            tags={"tone", "urgency"}
        )
        
        components["neutral_tone"] = TemplateComponent(
            id="neutral_tone",
            content="Use neutral, balanced tone. Present multiple perspectives fairly.",
            category="tone",
            safety_level=TemplateSafetyLevel.MEDIUM_SAFETY,
            tags={"tone", "balance"}
        )
        
        components["empathetic_tone"] = TemplateComponent(
            id="empathetic_tone",
            content="Use empathetic, supportive language while maintaining professionalism.",
            category="tone",
            safety_level=TemplateSafetyLevel.MEDIUM_SAFETY,
            tags={"tone", "empathy"}
        )
        
        # Constraint components
        components["disclaimer_required"] = TemplateComponent(
            id="disclaimer_required",
            content="Include appropriate disclaimers: 'This is not medical/legal/financial advice. Consult professionals.'",
            category="constraint",
            safety_level=TemplateSafetyLevel.MAX_SAFETY,
            tags={"safety", "disclaimer"}
        )
        
        components["source_citation"] = TemplateComponent(
            id="source_citation",
            content="Cite sources and reasoning basis. Make claims traceable and verifiable.",
            category="constraint",
            safety_level=TemplateSafetyLevel.HIGH_SAFETY,
            tags={"accuracy", "transparency"}
        )
        
        # Enhancement components
        components["examples_encouraged"] = TemplateComponent(
            id="examples_encouraged",
            content="Include concrete examples to illustrate key points and aid understanding.",
            category="enhancement",
            safety_level=TemplateSafetyLevel.MEDIUM_SAFETY,
            tags={"clarity", "examples"}
        )
        
        components["visual_aids"] = TemplateComponent(
            id="visual_aids",
            content="Suggest visual representations (diagrams, tables) where appropriate.",
            category="enhancement",
            safety_level=TemplateSafetyLevel.MEDIUM_SAFETY,
            tags={"visualization", "clarity"}
        )
        
        return components
    
    def _analyze_reasoning(self, aggregator_output: Dict) -> ReasoningAnalysis:
        """Analyze aggregator output to determine reasoning type"""
        # Extract domains
        domains = aggregator_output.get("domains_used", [])
        context = aggregator_output.get("context", {})
        
        # Normalize domain names
        domains = [d.lower().strip() for d in domains]
        
        # Determine reasoning type based on domains and context
        if "medical" in domains:
            # Check for emergency indicators
            emergency_keywords = ["emergency", "urgent", "critical", "immediate", "life-threatening"]
            is_emergency = any(kw in str(context).lower() for kw in emergency_keywords)
            reasoning_type = ReasoningType.MEDICAL_EMERGENCY if is_emergency else ReasoningType.UNKNOWN
        elif "legal" in domains or "law" in domains:
            reasoning_type = ReasoningType.LEGAL_ADVISORY
        elif "technical" in domains and len(domains) == 1:
            reasoning_type = ReasoningType.TECHNICAL_EXPLANATION
        elif "financial" in domains or "finance" in domains or "investment" in domains:
            reasoning_type = ReasoningType.FINANCIAL_ANALYSIS
        elif len(domains) >= 2:
            reasoning_type = ReasoningType.CROSS_DOMAIN
        else:
            reasoning_type = ReasoningType.UNKNOWN
        
        # Calculate complexity with better heuristics
        domain_complexity = min(len(domains) * 0.3, 1.0)
        query_complexity = aggregator_output.get("query_complexity", 0.5)
        conflict_complexity = 0.2 if aggregator_output.get("has_conflicts") else 0.0
        
        total_complexity = min(domain_complexity + query_complexity + conflict_complexity, 10.0)
        
        return ReasoningAnalysis(
            reasoning_type=reasoning_type,
            domains=domains,
            has_conflicts=aggregator_output.get("has_conflicts", False),
            confidence=aggregator_output.get("confidence", 0.5),
            complexity=total_complexity,
            metadata={
                "reasoning_steps": aggregator_output.get("reasoning_steps"),
                "synthesis_method": aggregator_output.get("synthesis_method"),
            }
        )
    
    def _determine_safety_level(
        self,
        analysis: ReasoningAnalysis,
        user_context: Optional[Dict]
    ) -> TemplateSafetyLevel:
        """Determine required safety level"""
        # Medical emergencies always max safety
        if analysis.reasoning_type == ReasoningType.MEDICAL_EMERGENCY:
            return TemplateSafetyLevel.MAX_SAFETY
        
        # Legal/financial high safety
        if analysis.reasoning_type in [ReasoningType.LEGAL_ADVISORY, ReasoningType.FINANCIAL_ANALYSIS]:
            return TemplateSafetyLevel.HIGH_SAFETY
        
        # Check user context for safety requirements
        if user_context:
            if user_context.get("high_risk_user", False):
                return TemplateSafetyLevel.MAX_SAFETY
            if user_context.get("professional_context", False):
                return TemplateSafetyLevel.HIGH_SAFETY
            if user_context.get("experimental_mode", False):
                return TemplateSafetyLevel.EXPLORATORY
        
        # Adjust based on confidence
        if analysis.confidence < self.config.default_confidence_threshold:
            return TemplateSafetyLevel.HIGH_SAFETY
        
        # Adjust based on complexity
        if analysis.complexity > 5.0:
            return TemplateSafetyLevel.HIGH_SAFETY
        
        return TemplateSafetyLevel.MEDIUM_SAFETY
    
    def _create_adaptive_template(self, analysis: ReasoningAnalysis) -> str:
        """Create template for unknown reasoning patterns"""
        template_parts = []
        
        # Base structure
        template_parts.append("COMPLEX ANALYSIS DETECTED:")
        template_parts.append(f"Domains involved: {', '.join(analysis.domains)}")
        template_parts.append(f"Complexity level: {analysis.complexity:.1f}/10")
        
        # Add structure based on analysis
        if analysis.has_conflicts:
            template_parts.append("\nCONFLICT RESOLUTION REQUIRED:")
            template_parts.append("- Acknowledge different perspectives")
            template_parts.append("- Explain the basis for resolution")
            template_parts.append("- Present balanced conclusion")
            template_parts.append("- Note areas of uncertainty or disagreement")
        
        if len(analysis.domains) >= 2:
            template_parts.append("\nMULTI-DOMAIN SYNTHESIS:")
            template_parts.append("- Integrate perspectives from each domain")
            template_parts.append("- Highlight connections between domains")
            template_parts.append("- Provide comprehensive analysis")
            template_parts.append("- Address interdependencies")
        
        # Confidence handling with graduated approach
        if analysis.confidence < 0.3:
            template_parts.append("\nLOW CONFIDENCE ALERT:")
            template_parts.append("- Clearly state limitations of analysis")
            template_parts.append("- Recommend expert consultation")
            template_parts.append("- Provide sources for further research")
        elif analysis.confidence < 0.6:
            template_parts.append("\nMODERATE CONFIDENCE NOTE:")
            template_parts.append("- Acknowledge uncertainty where appropriate")
            template_parts.append("- Qualify statements based on confidence level")
            template_parts.append("- Suggest additional verification if needed")
        
        # Add reasoning transparency
        if analysis.metadata.get("reasoning_steps"):
            template_parts.append("\nREASONING TRANSPARENCY:")
            template_parts.append("- Explain the reasoning process used")
            template_parts.append("- Show how conclusions were reached")
        
        return "\n".join(template_parts)
    
    def _enhance_template(
        self,
        template: str,
        analysis: ReasoningAnalysis,
        safety_level: TemplateSafetyLevel
    ) -> str:
        """Enhance template with dynamic components"""
        enhanced = template
        components_used = []
        
        # Add components based on safety level
        if safety_level in [TemplateSafetyLevel.MAX_SAFETY, TemplateSafetyLevel.HIGH_SAFETY]:
            enhanced += "\n\nSAFETY REQUIREMENTS:"
            enhanced += "\n" + self.components["disclaimer_required"].content
            components_used.append("disclaimer_required")
            
            # Add source citation for high safety
            enhanced += "\n" + self.components["source_citation"].content
            components_used.append("source_citation")
        
        # Add structure based on domains
        if len(analysis.domains) >= 2:
            enhanced += "\n\nSTRUCTURE:"
            enhanced += "\n" + self.components["hierarchical_structure"].content
            components_used.append("hierarchical_structure")
        
        # Add tone based on confidence and reasoning type
        if analysis.reasoning_type == ReasoningType.MEDICAL_EMERGENCY:
            enhanced += "\n\nTONE:"
            enhanced += "\n" + self.components["urgent_tone"].content
            components_used.append("urgent_tone")
        elif analysis.confidence > 0.8:
            enhanced += "\n\nTONE: Confident, definitive"
        elif analysis.confidence < 0.4:
            enhanced += "\n\nTONE: Cautious, conditional"
            enhanced += "\n" + self.components["empathetic_tone"].content
            components_used.append("empathetic_tone")
        else:
            enhanced += "\n\nTONE:"
            enhanced += "\n" + self.components["neutral_tone"].content
            components_used.append("neutral_tone")
        
        # Add enhancements for complex topics
        if analysis.complexity > 5.0:
            enhanced += "\n\nENHANCEMENTS:"
            enhanced += "\n" + self.components["examples_encouraged"].content
            components_used.append("examples_encouraged")
            enhanced += "\n" + self.components["step_by_step"].content
            components_used.append("step_by_step")
        
        # Update component usage metrics
        if self.config.enable_component_learning:
            for comp_id in components_used:
                if comp_id in self.components:
                    self.components[comp_id].usage_count += 1
                    self.components[comp_id].last_used = datetime.now()
        
        return enhanced
    
    def _format_prompt(
        self,
        template: str,
        user_query: str,
        aggregator_output: Dict
    ) -> str:
        """Format final LLM prompt"""
        confidence = aggregator_output.get("confidence", 0.5)
        domains = aggregator_output.get("domains_used", [])
        synthesis_method = aggregator_output.get("synthesis_method", "balanced_integration")
        
        # Add reasoning steps if available
        reasoning_context = ""
        if aggregator_output.get("reasoning_steps"):
            reasoning_context = "\n## REASONING STEPS:\n"
            for i, step in enumerate(aggregator_output["reasoning_steps"][:5], 1):
                reasoning_context += f"{i}. {step.get('description', 'N/A')}\n"
        
        prompt = f"""
# BabyBIONN Generation Instructions

## USER QUERY:
"{user_query}"

## SYSTEM REASONING ANALYSIS:
{template}

## ADDITIONAL CONTEXT:
- Analysis confidence: {confidence:.0%}
- Domains considered: {', '.join(domains) if domains else 'general'}
- Synthesis method: {synthesis_method}
- Query complexity: {aggregator_output.get('query_complexity', 'N/A')}
{reasoning_context}

## GENERATION REQUIREMENTS:
1. Generate response based on the above analysis
2. Follow the structure and tone guidelines precisely
3. Incorporate all safety requirements
4. Ensure response is helpful, accurate, and actionable
5. Maintain appropriate epistemic humility based on confidence level

Please generate the final response for the user.
"""
        
        return prompt.strip()
    
    # ========================================================================
    # TEMPLATE CREATION METHODS
    # ========================================================================
    
    def _create_medical_emergency_template(self) -> str:
        return """🚨 MEDICAL EMERGENCY CONTEXT DETECTED

RESPONSE STRUCTURE:
1. IMMEDIATE ACTION: State urgent required actions first
2. EMERGENCY CONTACTS: Provide relevant emergency numbers/contacts
3. SAFETY INSTRUCTIONS: Clear step-by-step safety guidance
4. MEDICAL CONTEXT: Brief relevant medical information
5. FOLLOW-UP: Instructions for next steps

CRITICAL REQUIREMENTS:
- Use urgent, directive language
- Prioritize life-saving actions
- Include specific emergency contacts
- Avoid unnecessary details
- Be clear and unambiguous
- Default to recommending professional medical help"""

    def _create_legal_advisory_template(self) -> str:
        return """⚖️ LEGAL ANALYSIS CONTEXT

RESPONSE STRUCTURE:
1. JURISDictION: State applicable jurisDiction
2. LEGAL PRINCIPLES: Explain relevant legal principles
3. PRACTICAL IMPLICATIONS: How principles apply to situation
4. RECOMMENDATIONS: Actionable legal recommendations
5. DISCLAIMERS: Required legal disclaimers

IMPORTANT:
- Use precise legal terminology
- Include jurisDictional qualifiers
- Always include disclaimers
- Recommend professional consultation
- Avoid definitive guarantees
- Note that laws vary by jurisDiction"""

    def _create_technical_template(self) -> str:
        return """🔧 TECHNICAL ANALYSIS CONTEXT

RESPONSE STRUCTURE:
1. OVERVIEW: High-level understanding
2. TECHNICAL DETAILS: In-depth technical explanation
3. IMPLEMENTATION: Practical implementation steps
4. LIMITATIONS: Technical constraints and caveats
5. BEST PRACTICES: Recommended approaches

GUIDELINES:
- Use clear technical terminology
- Include practical examples
- Explain complex concepts accessibly
- Acknowledge technical trade-offs
- Provide actionable technical guidance
- Reference relevant standards or documentation"""

    def _create_financial_template(self) -> str:
        return """💰 FINANCIAL ANALYSIS CONTEXT

RESPONSE STRUCTURE:
1. FINANCIAL OVERVIEW: Key financial considerations
2. RISK ASSESSMENT: Financial risks and implications
3. RECOMMENDATIONS: Actionable financial guidance
4. REGULATORY CONTEXT: Relevant regulations and compliance
5. DISCLAIMERS: Required financial disclaimers

IMPORTANT:
- Use precise financial terminology
- Include risk disclosures
- Always include financial disclaimers
- Recommend professional consultation
- Provide conservative estimates
- Note market volatility and uncertainties"""

    def _create_cross_domain_template(self) -> str:
        return """🌐 CROSS-DOMAIN SYNTHESIS REQUIRED

RESPONSE STRUCTURE:
1. MULTI-DOMAIN OVERVIEW: Integrated perspective
2. DOMAIN-SPECIFIC ANALYSES: Each domain's contribution
3. SYNTHESIS: How domains interact and relate
4. INTEGRATED RECOMMENDATIONS: Cross-domain solutions
5. TRADE-OFFS: Balancing different domain priorities

SYNTHESIS APPROACH:
- Show connections between domains
- Highlight complementary insights
- Address domain conflicts explicitly
- Provide balanced, integrated perspective
- Acknowledge complexity of multi-domain issues
- Note where domain expertise differs"""

    def _create_general_template(self) -> str:
        return """📝 GENERAL ANALYSIS CONTEXT

RESPONSE STRUCTURE:
1. DIRECT ANSWER: Address core query directly
2. SUPPORTING ANALYSIS: Relevant considerations
3. CONTEXTUAL FACTORS: Important context
4. PRACTICAL GUIDANCE: Actionable information
5. ADDITIONAL RESOURCES: Where to learn more

GENERATION GUIDELINES:
- Be helpful and informative
- Consider multiple perspectives
- Acknowledge uncertainties
- Provide balanced information
- Maintain appropriate tone
- Cite sources where applicable"""


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_engine(
    enable_caching: bool = True,
    enable_metrics: bool = True,
    strict_validation: bool = False
) -> HybridAdaptiveTemplateEngine:
    """Factory function to create a configured engine"""
    config = EngineConfig(
        enable_caching=enable_caching,
        enable_metrics=enable_metrics,
        strict_validation=strict_validation
    )
    return HybridAdaptiveTemplateEngine(config)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create engine
    engine = create_engine()
    
    # Example aggregator output
    test_output = {
        "domains_used": ["medical", "legal"],
        "context": {"emergency": True},
        "confidence": 0.75,
        "has_conflicts": False,
        "query_complexity": 0.6,
        "synthesis_method": "expert_override"
    }
    
    # Generate prompt
    prompt = engine.create_prompt(
        aggregator_output=test_output,
        user_query="What should I do if I'm injured at work?",
        user_context={"high_risk_user": False}
    )
    
    print(prompt)
    print("\n" + "="*80 + "\n")
    print("Metrics Summary:")
    print(json.dumps(engine.get_metrics_summary(), indent=2)) 
