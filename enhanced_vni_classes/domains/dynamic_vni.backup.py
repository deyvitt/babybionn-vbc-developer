# enhanced_vni_classes/domains/dynamic_vni.py
import inspect
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib
import logging
from pathlib import Path
import yaml

from .base_knowledge_loader import BaseKnowledgeLoader 
from ..core.base_vni import EnhancedBaseVNI
from ..core.capabilities import VNICapabilities, VNIType
from ..modules.knowledge_base import KnowledgeBase
from ..modules.classifier import DynamicDomainClassifier
from ..core.pipeline_steps import PipelineStep

logger = logging.getLogger(__name__)

@dataclass
class DomainEvolution:
    """Track domain evolution and learning"""
    learned_keywords: Set[str] = field(default_factory=set)
    query_patterns: Dict[str, int] = field(default_factory=dict)
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    confidence_trend: List[float] = field(default_factory=list)
    
    def add_interaction(self, query: str, confidence: float, response_success: bool):
        """Record an interaction for learning"""
        self.confidence_trend.append(confidence)
        
        # Extract potential new keywords (simple version)
        words = {w.lower() for w in query.split() if len(w) > 3}
        self.learned_keywords.update(words)
        
        self.adaptation_history.append({
            'timestamp': datetime.now().isoformat(),
            'query_preview': query[:100],
            'confidence': confidence,
            'success': response_success
        })
        
        # Keep history manageable
        if len(self.adaptation_history) > 1000:
            self.adaptation_history = self.adaptation_history[-1000:]
        if len(self.confidence_trend) > 100:
            self.confidence_trend = self.confidence_trend[-100:]

@dataclass
class DomainConfig:
    """Configuration for a dynamic domain with enhanced features"""
    name: str
    description: str
    keywords: List[str]
    priority_keywords: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.3
    generation_temperature: float = 0.7
    response_templates: Dict[str, List[str]] = field(default_factory=dict)
    default_concepts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    default_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    max_keywords: int = 100
    learning_rate: float = 0.1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling dataclass fields"""
        data = asdict(self)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DomainConfig':
        """Create from dictionary"""
        # Remove any metadata fields that aren't in the dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
    
    def add_keyword(self, keyword: str, is_priority: bool = False):
        """Add a keyword to the configuration"""
        keyword_lower = keyword.lower()
        
        if keyword_lower not in self.keywords and len(self.keywords) < self.max_keywords:
            self.keywords.append(keyword_lower)
            
            if is_priority and keyword_lower not in self.priority_keywords:
                self.priority_keywords.append(keyword_lower)
                
            logger.info(f"Added keyword '{keyword}' to domain '{self.name}'")
            return True
        return False
    
    def remove_keyword(self, keyword: str):
        """Remove a keyword from the configuration"""
        keyword_lower = keyword.lower()
        
        if keyword_lower in self.keywords:
            self.keywords.remove(keyword_lower)
            
        if keyword_lower in self.priority_keywords:
            self.priority_keywords.remove(keyword_lower)
            
        return True
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the configuration"""
        errors = []
        
        if not self.name or not self.name.strip():
            errors.append("Domain name is required")
        
        if len(self.keywords) < 5:
            errors.append(f"Need at least 5 keywords, got {len(self.keywords)}")
        
        if not (0.0 <= self.confidence_threshold <= 1.0):
            errors.append(f"Confidence threshold must be between 0 and 1, got {self.confidence_threshold}")
        
        if not (0.0 <= self.generation_temperature <= 2.0):
            errors.append(f"Generation temperature should be between 0 and 2, got {self.generation_temperature}")
        
        # Check for duplicates
        if len(self.keywords) != len(set(self.keywords)):
            errors.append("Duplicate keywords found")
        
        return len(errors) == 0, errors

class DomainConfigValidator:
    """Validator for domain configurations"""
    
    @staticmethod
    def validate_advanced(vni_config: DomainConfig) -> Dict[str, Any]:
        """Advanced validation with suggestions"""
        is_valid, errors = vni_config.validate()
        
        suggestions = {}
        
        # Keyword coverage suggestions
        if len(vni_config.keywords) < 20:
            suggestions['keyword_coverage'] = {
                'current': len(vni_config.keywords),
                'recommended': '20-50 keywords for better coverage',
                'severity': 'warning'
            }
        
        # Priority keyword suggestions
        if len(vni_config.priority_keywords) < 3:
            suggestions['priority_keywords'] = {
                'current': len(vni_config.priority_keywords),
                'recommended': '3-10 priority keywords',
                'severity': 'warning'
            }
        
        # Check for overly common words
        common_words = {'the', 'and', 'or', 'but', 'not', 'is', 'are', 'was', 'were', 'this', 'that'}
        common_found = [kw for kw in vni_config.keywords if kw in common_words]
        if common_found:
            suggestions['common_words'] = {
                'words': common_found,
                'recommended': 'Avoid common words that cause false positives',
                'severity': 'warning'
            }
        
        return {
            'valid': is_valid,
            'errors': errors,
            'suggestions': suggestions
        }

class DomainSimilarityAnalyzer:
    """Analyze similarity between domains"""
    
    @staticmethod
    def calculate_similarity(config1: DomainConfig, config2: DomainConfig) -> float:
        """Calculate Jaccard similarity between domain keyword sets"""
        keywords1 = set(config1.keywords)
        keywords2 = set(config2.keywords)
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def get_related_domains(vni_config: DomainConfig, all_configs: Dict[str, DomainConfig], 
                           threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Get domains related to this one"""
        related = []
        
        for name, other_config in all_configs.items():
            if name == vni_config.name:
                continue
            
            similarity = DomainSimilarityAnalyzer.calculate_similarity(vni_config, other_config)
            if similarity > threshold:
                related.append((name, similarity))
        
        # Sort by similarity (highest first)
        related.sort(key=lambda x: x[1], reverse=True)
        return related

from enum import Enum  # ADD THIS IMPORT

# Add GenerationStyle enum definition before the DynamicVNI class
class GenerationStyle(Enum):
    """Enumeration of generation styles"""
    CONCISE = "concise"      # Short, direct responses
    DETAILED = "detailed"    # In-depth explanations
    CREATIVE = "creative"    # Creative/narrative responses
    FORMAL = "formal"       # Formal/legal tone
    GENERAL = "general"     # General/default style
    TECHNICAL = "technical" # Technical/domain-specific
    EMERGENCY = "emergency" # Urgent/emergency responses

class DynamicVNI(EnhancedBaseVNI, BaseKnowledgeLoader):
    """Enhanced Dynamic VNI with learning capabilities. Can adapt to any domain without hardcoded classes"""
    def __init__(self, instance_id: str, domain_config: DomainConfig, auto_load_knowledge: bool = True, enable_learning: bool = True, **kwargs):
        """Args:
            domain_config: Configuration defining the domain
            instance_id: Optional instance identifier
            auto_load_knowledge: Whether to automatically load knowledge from files
            enable_learning: Whether to enable continuous learning"""
        self.domain_config = domain_config
        
        # Generate instance_id if not provided
        if instance_id is None:
            domain_hash = hashlib.md5(domain_config.name.encode()).hexdigest()[:8]
            instance_id = f"dynamic_{domain_config.name}_{domain_hash}"
        
        # Initialize with dynamic domain type
        super().__init__(domain_config.name, instance_id)

        self.vni_type = "dynamic" 
         
        # ENABLE GENERATION
        self.generation_enabled = True

        # Enable learning
        self.enable_learning = enable_learning
        self.evolution = DomainEvolution()
        self.interaction_count = 0
        
        # Initialize capabilities
        self._init_capabilities()
        
        # Initialize domain-specific classifier
        self._init_dynamic_classifier()

        # Load knowledge from files
        if auto_load_knowledge:
            self.load_domain_knowledge(domain_config.name)

        # Initialize generation
        self._init_generation()
        
        # Configure generation for this domain
        if self.generation_enabled:
            self.configure_generation(
                temperature=domain_config.generation_temperature,
                top_p=0.9
            )
        
        logger.info(f"✅ Dynamic VNI for '{domain_config.name}' initialized with learning={enable_learning}")

    def get_default_pipeline(self) -> List[str]:
        """Dynamic pipeline based on domain_config"""
        base_pipeline = ["classify", "knowledge_lookup"]
        
        # Add domain-specific steps
        if self.domain_config.name == "medical":
            base_pipeline.insert(1, "medical_safety")  # Add safety check early
        elif self.domain_config.name == "legal":
            base_pipeline.insert(1, "legal_disclaimer")
        
        base_pipeline.append("generate_response")
        return base_pipeline
    
    def get_available_steps(self) -> Dict[str, PipelineStep]:
        """Get all available steps including dynamic ones"""
        from ..core.pipeline_steps import MedicalSafetyStep, LegalDisclaimerStep, EmergencyCheckStep  # ADD THIS
        
        steps = super().get_available_steps()
        
        # Add domain-specific steps
        if self.domain_config.name == "medical":
            steps["medical_safety"] = MedicalSafetyStep()
        elif self.domain_config.name == "legal":
            steps["legal_disclaimer"] = LegalDisclaimerStep()
        
        # Add dynamic steps based on configuration
        if "emergency" in self.domain_config.priority_keywords:
            steps["emergency_check"] = EmergencyCheckStep()
        
        return steps
    
    def process(self, query: str, pipeline: Optional[List[str]] = None, 
                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process query using pipeline - OVERRIDE the existing process method"""
        # Use provided pipeline or default
        processing_pipeline = pipeline or self.get_default_pipeline()
        
        logger.info(f"Processing query with pipeline: {processing_pipeline}")
        
        # Execute pipeline
        result = self._execute_pipeline(query, processing_pipeline, context)
        
        # Add domain metadata
        result['domain_metadata'] = {
            'domain_name': self.domain_config.name,
            'pipeline_used': processing_pipeline,
            'executed_steps': result.get('executed_steps', []),
            'dynamic_vni': True,
            'learning_enabled': self.enable_learning,
            'interaction_count': self.interaction_count
        }
        
        # Learn from interaction (keep existing learning logic)
        if self.enable_learning:
            self._learn_from_interaction(query, result)
        
        return result
    
    def _init_generation(self):
        """Initialize generation module for dynamic domain"""
        try:
            # Import EnhancedGenerationModule
            from ..modules.generation import EnhancedGenerationModule
            
            # Create generation module
            self.generator = EnhancedGenerationModule(
                domain=self.domain_config.name,
                enable_llm=True,
                model_name="microsoft/DialoGPT-medium"  # Default model
            )
            
            # Setup the generator
            success = self.generator.setup()
            if not success:
                logger.warning(f"Generation setup failed for {self.domain_config.name}")
                self.generation_enabled = False
            
        except Exception as e:
            logger.error(f"Generation initialization failed: {e}")
            self.generation_enabled = False

    def configure_generation(self, **kwargs):
        """Configure generation parameters"""
        if hasattr(self, 'generator') and self.generator:
            self.generator.update_config(**kwargs)
        else:
            logger.warning("Cannot configure generation - generator not initialized")

    # Add this method to the DynamicVNI class (around line 350-400 area):
    
    def generate_response(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response using the domain-specific generation module"""
        if not self.generation_enabled or not hasattr(self, 'generator'):
            return {
                'success': False,
                'response': 'Generation not enabled or generator not initialized',
                'confidence': 0.0
            }
        
        try:
            # Determine style based on domain configuration or query content
            style = self._determine_generation_style(query)
            
            # Generate response
            response = self.generator.generate_response(
                query=query,
                context=context,
                style=style,
                temperature=self.domain_config.generation_temperature,
                max_length=self.domain_config.get('max_response_length', 500)        
            )
            
            # Add domain-specific metadata
            response['domain_metadata'] = {
                'domain_name': self.domain_config.name,
                'generation_style': style,
                'temperature_used': self.domain_config.generation_temperature,
                'dynamic_vni': True
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                'success': False,
                'response': f'Generation failed: {str(e)}',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _determine_generation_style(self, query: str) -> str:
        """Determine the appropriate generation style for the query"""
        query_lower = query.lower()
        
        # Check for emergency/urgent content
        emergency_keywords = ['emergency', 'urgent', 'immediate', 'critical', 'life-threatening']
        if any(keyword in query_lower for keyword in emergency_keywords):
            return GenerationStyle.CONCISE.value
        
        # Check for detailed explanation requests
        detail_keywords = ['explain', 'describe', 'details', 'how does', 'what is', 'why']
        if any(keyword in query_lower for keyword in detail_keywords):
            return GenerationStyle.DETAILED.value
        
        # Check for creative/narrative requests
        creative_keywords = ['story', 'narrative', 'creative', 'imagine', 'write a']
        if any(keyword in query_lower for keyword in creative_keywords):
            return GenerationStyle.CREATIVE.value
    
        # Check for formal/legal tone requests
        formal_keywords = ['legal', 'contract', 'formal', 'official', 'document']
        if any(keyword in query_lower for keyword in formal_keywords):
            return GenerationStyle.FORMAL.value
    
        # Check if domain has style preferences in templates
        if hasattr(self.domain_config, 'response_templates'):
            styles = list(self.domain_config.response_templates.keys())
            if styles:
                # Return first available style
                return styles[0]
        
        # Default based on domain
        if self.domain_config.name in ['legal', 'medical', 'academic']:
            return GenerationStyle.FORMAL.value
        elif self.domain_config.name in ['creative', 'writing', 'story']:
            return GenerationStyle.CREATIVE.value
        else:
            return GenerationStyle.DETAILED.value

    def _init_capabilities(self):
        """Initialize enhanced capabilities"""
        self.available_capabilities = VNICapabilities(
            domains=[self.domain_config.name],
            can_search=True,
            can_learn=self.enable_learning,
            can_collaborate=True,
            max_context_length=3000,
            special_abilities=list({
                self.domain_config.name,
                *self.domain_config.keywords[:3],
                "dynamic_adaptation",
                "continuous_learning" if self.enable_learning else "static"
            }),
            vni_type='specialized',
            # Optional additional fields
            abstraction_levels=["semantic", "structural", "adaptive"],
            processing_speed=1.0,
            collaboration_score=0.7,
            learning_enabled=self.enable_learning
        )
    
    def _init_dynamic_classifier(self):
        """Initialize enhanced classifier with learning support"""
        try:
            all_keywords = list(set(self.domain_config.keywords) | self.evolution.learned_keywords)
            
            self.classifier = DynamicDomainClassifier(
                domain_name=self.domain_config.name,
                keywords=all_keywords,
                priority_keywords=self.domain_config.priority_keywords,
                confidence_threshold=self.domain_config.confidence_threshold,
                adaptive_threshold=self.enable_learning  # Allow threshold adjustment
            )
            logger.info(f"✅ Dynamic classifier for '{self.domain_config.name}' initialized with "
                       f"{len(all_keywords)} keywords ({len(self.evolution.learned_keywords)} learned)")
        except Exception as e:
            logger.error(f"❌ Dynamic classifier initialization failed: {e}")
            self.classifier = None
    
    def _fallback_domain_check(self, query: str) -> bool:
        """Enhanced fallback domain check with learned keywords"""
        query_lower = query.lower()
        
        # Check priority keywords first (higher confidence)
        for keyword in self.domain_config.priority_keywords:
            if keyword in query_lower:
                return True
        
        # Check regular keywords
        all_keywords = set(self.domain_config.keywords) | self.evolution.learned_keywords
        keyword_matches = sum(1 for keyword in all_keywords if keyword in query_lower)
        
        # Dynamic threshold based on learning
        required_matches = 2 if self.interaction_count < 100 else 1
        return keyword_matches >= required_matches
    
    def _learn_from_interaction(self, query: str, result: Dict[str, Any]):
        """Learn from the interaction"""
        self.interaction_count += 1
        
        confidence = result.get('confidence', 0)
        success = result.get('success', True)
        
        # Record interaction
        self.evolution.add_interaction(query, confidence, success)
        
        # Learn new keywords periodically
        if self.interaction_count % 10 == 0:  # Every 10 interactions
            self._update_keywords_from_learning()
    
    def _update_keywords_from_learning(self):
        """Update keywords based on learned patterns"""
        if not self.evolution.learned_keywords:
            return
        
        # Add most frequently learned keywords to vni_config
        learned_list = list(self.evolution.learned_keywords)
        
        # Simple heuristic: add keywords that appear in successful interactions
        for keyword in learned_list[:5]:  # Limit to top 5
            if len(self.domain_config.keywords) < self.domain_config.max_keywords:
                self.domain_config.add_keyword(keyword)
        
        # Reinitialize classifier with updated keywords
        self._init_dynamic_classifier()
    
    def get_domain_insights(self) -> Dict[str, Any]:
        """Get insights about domain performance and learning"""
        avg_confidence = (sum(self.evolution.confidence_trend) / 
                         len(self.evolution.confidence_trend) if self.evolution.confidence_trend else 0)
        
        return {
            'domain_name': self.domain_config.name,
            'interaction_count': self.interaction_count,
            'learning_enabled': self.enable_learning,
            'keywords_total': len(self.domain_config.keywords),
            'keywords_learned': len(self.evolution.learned_keywords),
            'average_confidence': round(avg_confidence, 3),
            'confidence_trend': self.evolution.confidence_trend[-10:],  # Last 10
            'recent_patterns': dict(list(self.evolution.query_patterns.items())[-5:])
        }
    
    def get_default_concepts(self) -> Dict[str, Any]:
        """Enhanced default concepts with learning integration"""
        concepts = self.domain_config.default_concepts.copy() if self.domain_config.default_concepts else {}
        
        # Add learned keywords as concepts
        for keyword in list(self.evolution.learned_keywords)[:20]:  # Limit to 20
            if keyword not in concepts:
                concepts[keyword] = {
                    'strength': 0.6,  # Lower initial strength for learned concepts
                    'usage_count': 0,
                    'source': 'learned',
                    'learned_at': datetime.now().isoformat()
                }
        
        # Add vni_config keywords
        for keyword in self.domain_config.keywords[:30]:  # Limit to 30
            if keyword not in concepts:
                concepts[keyword] = {
                    'strength': 0.8,
                    'usage_count': 0,
                    'source': 'vni_config'
                }
        
        return concepts
    
    def get_default_patterns(self) -> Dict[str, Any]:
        """Enhanced patterns with learned responses"""
        patterns = self.domain_config.default_patterns.copy() if self.domain_config.default_patterns else {}
        
        # Create patterns from priority keywords
        for i, keyword in enumerate(self.domain_config.priority_keywords[:5]):
            pattern_id = f"priority_{keyword}"
            patterns[pattern_id] = {
                "triggers": [keyword],
                "responses": [
                    f"This appears to be related to {keyword}, which is a priority area in {self.domain_config.name}.",
                    f"Regarding {keyword}, this requires specific attention in the {self.domain_config.name} domain."
                ],
                "strength": 0.9,
                "usage_count": 0,
                "priority": True
            }
        
        return patterns
    
    def export_config(self, include_learned: bool = True) -> Dict[str, Any]:
        """Export current configuration including learned data"""
        config_dict = self.domain_config.to_dict()
        
        if include_learned:
            config_dict['_learning_data'] = {
                'learned_keywords': list(self.evolution.learned_keywords),
                'interaction_count': self.interaction_count,
                'confidence_trend': self.evolution.confidence_trend,
                'exported_at': datetime.now().isoformat()
            }
        
        return config_dict
    
    @classmethod
    def from_yaml(cls, yaml_path: str, instance_id: str = None, enable_learning: bool = True) -> 'DynamicVNI':
        """Create DynamicVNI from YAML configuration file"""
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        domain_config = DomainConfig.from_dict(config_data)
        return cls(domain_config, instance_id, enable_learning)
    
    @classmethod
    def from_json(cls, json_path: str, instance_id: str = None, enable_learning: bool = True) -> 'DynamicVNI':
        """Create DynamicVNI from JSON configuration file"""
        with open(json_path, 'r') as f:
            config_data = json.load(f)
        
        domain_config = DomainConfig.from_dict(config_data)
        return cls(domain_config, instance_id, enable_learning)

# Enhanced Domain Factory
class EnhancedDomainFactory:
    """Factory for creating and managing enhanced domain configurations"""
    
    # Predefined domain configurations
    PREDEFINED_DOMAINS = {
        'medical': DomainConfig(
            name='medical',
            description='Medical and healthcare domain with adaptive learning',
            keywords=[
                'medical', 'health', 'doctor', 'hospital', 'pain', 'fever',
                'headache', 'medicine', 'treatment', 'sick', 'illness',
                'disease', 'symptom', 'patient', 'clinic', 'advise',
                'emergency', 'medication', 'diagnosis', 'therapy', 'pharmacy',
                'prescription', 'vaccine', 'covid', 'infection', 'virus',
                'allergy', 'asthma', 'cancer', 'diabetes', 'heart', 'lung'
            ],
            priority_keywords=['emergency', 'urgent', 'pain', 'fever', 'symptom'],
            confidence_threshold=0.3,
            generation_temperature=0.6,
            response_templates={
                "general": [
                    "From a medical perspective, {concept} should be evaluated by a healthcare professional.",
                    "Regarding {concept}, consider individual health factors and consult a doctor."
                ],
                "emergency": [
                    "If this is a medical emergency, please seek immediate professional help.",
                    "For urgent medical concerns, contact emergency services immediately."
                ]
            },
            learning_rate=0.15,
            max_keywords=150
        ),
        'legal': DomainConfig(
            name='legal',
            description='Legal domain with comprehensive coverage',
            keywords=[
                'legal', 'law', 'contract', 'rights', 'agreement', 'court',
                'lawyer', 'attorney', 'case', 'evidence', 'justice', 'liability',
                'dispute', 'compliance', 'regulation', 'statute', 'clause',
                'sue', 'lawsuit', 'trial', 'judge', 'jury', 'verdict',
                'copyright', 'patent', 'trademark', 'intellectual', 'property'
            ],
            priority_keywords=['contract', 'rights', 'lawsuit', 'court', 'legal'],
            confidence_threshold=0.4,
            generation_temperature=0.5,
            response_templates={
                "general": [
                    "From a legal standpoint, {concept} requires proper documentation and professional advice.",
                    "Legal matters involving {concept} should be reviewed by a qualified attorney."
                ],
                "dispute": [
                    "Legal disputes regarding {concept} typically require evidence collection and formal proceedings."
                ]
            },
            max_keywords=120
        ),
        'technical': DomainConfig(
            name='technical',
            description='Technical and programming domain',
            keywords=[
                'code', 'programming', 'software', 'debug', 'error', 'bug',
                'algorithm', 'system', 'database', 'technical', 'python',
                'javascript', 'java', 'c++', 'api', 'framework', 'library',
                'function', 'class', 'object', 'variable', 'loop', 'condition',
                'server', 'client', 'network', 'security', 'encryption'
            ],
            priority_keywords=['debug', 'error', 'bug', 'code', 'python'],
            confidence_threshold=0.5,
            generation_temperature=0.7,
            max_keywords=200
        ),
        # Add more domains as needed
    }
    
    @classmethod
    def get_domain(cls, domain_name: str) -> Optional[DomainConfig]:
        """Get predefined domain configuration"""
        return cls.PREDEFINED_DOMAINS.get(domain_name)
    
    @classmethod
    def create_domain(cls, **kwargs) -> DomainConfig:
        """Create and validate a custom domain configuration"""
        vni_config = DomainConfig(**kwargs)
        
        # Validate
        is_valid, errors = vni_config.validate()
        if not is_valid:
            raise ValueError(f"Invalid domain configuration: {errors}")
        
        return vni_config
    
    @classmethod
    def create_validated_domain(cls, **kwargs) -> Tuple[DomainConfig, Dict[str, Any]]:
        """Create domain with validation report"""
        vni_config = cls.create_domain(**kwargs)
        validation = DomainConfigValidator.validate_advanced(vni_config)
        
        return vni_config, validation
    
    @classmethod
    def detect_domain_from_query(cls, query: str, threshold: int = 2) -> List[Tuple[str, int]]:
        """Detect which domains are relevant to a query with match counts"""
        relevant_domains = []
        query_lower = query.lower()
        
        for domain_name, vni_config in cls.PREDEFINED_DOMAINS.items():
            # Count keyword matches
            matches = sum(1 for keyword in vni_config.keywords if keyword in query_lower)
            
            # Add domain if enough matches
            if matches >= threshold:
                relevant_domains.append((domain_name, matches))
        
        # Sort by match count (highest first)
        relevant_domains.sort(key=lambda x: x[1], reverse=True)
        return relevant_domains
    
    @classmethod
    def suggest_best_domain(cls, query: str) -> Optional[str]:
        """Suggest the best domain for a query"""
        domains = cls.detect_domain_from_query(query, threshold=1)
        
        if domains:
            # Return domain with highest match count
            return domains[0][0]
        
        return None
    
    @classmethod
    def create_dynamic_vni(cls, domain_name: str, instance_id: str = None, 
                          enable_learning: bool = True) -> DynamicVNI:
        """Create a DynamicVNI from a predefined domain name"""
        vni_config = cls.get_domain(domain_name)
        if vni_config is None:
            raise ValueError(f"Unknown domain: {domain_name}")
        
        return DynamicVNI(vni_config, instance_id, enable_learning)
    
    @classmethod
    def get_all_domains(cls) -> List[str]:
        """Get all available domain names"""
        return list(cls.PREDEFINED_DOMAINS.keys())

class DomainPersistenceManager:
    """Manage saving and loading domain configurations with versioning"""
    
    def __init__(self, storage_dir: str = "domains"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def save_domain(self, vni: DynamicVNI, filename: str = None) -> Path:
        """Save a DynamicVNI's configuration to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{vni.domain_config.name}_{timestamp}.json"
        
        filepath = self.storage_dir / filename
        
        # Get configuration with learning data
        config_data = vni.export_config(include_learned=True)
        
        # Add persistence metadata
        config_data['_persistence_metadata'] = {
            'saved_at': datetime.now().isoformat(),
            'instance_id': vni.instance_id,
            'interaction_count': vni.interaction_count,
            'file_format': 'json_v1'
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        logger.info(f"✅ Domain '{vni.domain_config.name}' saved to {filepath}")
        return filepath
    
    def save_config_only(self, vni_config: DomainConfig, filename: str = None) -> Path:
        """Save only the domain configuration"""
        if filename is None:
            filename = f"{vni_config.name}_config.json"
        
        filepath = self.storage_dir / filename
        
        config_data = vni_config.to_dict()
        config_data['_metadata'] = {
            'saved_at': datetime.now().isoformat(),
            'type': 'config_only'
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        return filepath
    
    def load_domain_config(self, filename: str) -> DomainConfig:
        """Load a domain configuration from file"""
        filepath = self.storage_dir / filename
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Remove metadata before creating vni_config
        data.pop('_metadata', None)
        data.pop('_persistence_metadata', None)
        data.pop('_learning_data', None)
        
        return DomainConfig.from_dict(data)
    
    def list_saved_domains(self) -> List[Dict[str, Any]]:
        """List all saved domain configurations"""
        domains = []
        
        for filepath in self.storage_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                domains.append({
                    'filename': filepath.name,
                    'domain_name': data.get('name', 'unknown'),
                    'description': data.get('description', ''),
                    'keywords_count': len(data.get('keywords', [])),
                    'file_size': filepath.stat().st_size,
                    'modified': datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
                })
            except Exception as e:
                logger.error(f"Error reading {filepath}: {e}")
        
        return domains

# Quick access functions
def create_domain_vni(domain_name: str, enable_learning: bool = True) -> DynamicVNI:
    """Quick function to create a domain VNI"""
    return EnhancedDomainFactory.create_dynamic_vni(domain_name, enable_learning=enable_learning)

def analyze_query_domains(query: str) -> Dict[str, Any]:
    """Analyze which domains a query belongs to"""
    domains = EnhancedDomainFactory.detect_domain_from_query(query)
    best_domain = EnhancedDomainFactory.suggest_best_domain(query)
    
    return {
        'query': query[:100] + ('...' if len(query) > 100 else ''),
        'relevant_domains': domains,
        'best_domain': best_domain,
        'domain_count': len(domains)
    }

# Usage example
if __name__ == "__main__":
    # Example: Create a medical domain VNI with learning
    medical_vni = create_domain_vni("medical", enable_learning=True)
    
    # Process a query
    result = medical_vni.process("I have a headache and fever, what should I do?")
    print(f"Confidence: {result.get('confidence')}")
    
    # Get insights
    insights = medical_vni.get_domain_insights()
    print(f"Interactions: {insights['interaction_count']}")
    
    # Save the configuration
    persistence = DomainPersistenceManager()
    persistence.save_domain(medical_vni)
    
    # Analyze a query
    analysis = analyze_query_domains("I need help with a legal contract and medical symptoms")
    print(f"Best domain: {analysis['best_domain']}")
