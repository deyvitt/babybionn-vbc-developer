# File: llm_gateway.py
"""
Unified LLM Gateway - Replaces generation.py
Connects to multiple LLM providers with failover, cost optimization, and unified interface
"""
import os
import json
import time
# import openai
import logging
import hashlib
import requests
# import anthropic
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv
from collections import defaultdict
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()
#client = anthropic.Anthropic(    
#    api_key=os.getenv("ANTHROPIC_API_KEY")
#)
logger = logging.getLogger("deepseek_gateway")

class LLMProvider(Enum):
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    MOCK = "mock"
    # ANTHROPIC = "anthropic"
    # QWEN = "qwen"
    OLLAMA = "ollama"  # For local/self-hosted
    TOGETHER = "together"  # For open-source models

@dataclass
class LLMResponse:
    """Unified response format from any LLM"""
    content: str
    provider: LLMProvider
    model: str
    usage: dict #Dict[str, int]  # tokens, cost
    latency: float  # seconds
    # metadata: Dict[str, Any]
    # raw_response: Optional[Dict] = None

@dataclass
class LLMConfig:
    """Configuration for LLM calls"""
    provider: LLMProvider
    model: str
    temperature: float = 0.3
    max_tokens: int = 4000
    top_p: float = 0.95
#    frequency_penalty: float = 0.0
#    presence_penalty: float = 0.0
    system_prompt: Optional[str] = None
    timeout: int = 30

class BaseLLMClient(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        pass
    
    # REMOVE @abstractmethod - make it a regular method with default implementation
    def estimate_cost(self, prompt: str, config: LLMConfig) -> float:
        """Estimate cost - default implementation returns 0"""
        return 0.0  # Default value

class DeepSeekClient(BaseLLMClient):
    """DeepSeek API Client - PRIMARY PROVIDER"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            logger.warning("⚠️ DEEPSEEK_API_KEY not found in environment")
        
        # Deepseek models
        self.models = {
            "general": "deepseek-chat",
            "code": "deepseek-coder",
            "reasoning": "deepseek-chat",  # Same as general for now
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """Generate response from Deepseek API"""
        start_time = time.time()
        
        # Use appropriate model based on config or default
        model = config.model if config.model else self.models["general"]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=config.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            latency = time.time() - start_time
            
            # Extract usage info
            usage = result.get("usage", {})
            usage["estimated_cost"] = self.estimate_cost(
                usage.get("prompt_tokens", 0), 
                usage.get("completion_tokens", 0)
            )
            
            return LLMResponse(
                content=result["choices"][0]["message"]["content"],
                provider=LLMProvider.DEEPSEEK,
                model=model,
                usage=usage,
                latency=latency
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"DeepSeek API error: {str(e)}")
            raise Exception(f"DeepSeek API failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in DeepSeek client: {str(e)}")
            raise
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD (Deepseek is very affordable)"""
        # Deepseek pricing (as of latest): $0.14 per million input, $0.28 per million output
        input_cost = (input_tokens / 1_000_000) * 0.14
        output_cost = (output_tokens / 1_000_000) * 0.28
        return round(input_cost + output_cost, 6)

class OpenAIClient(BaseLLMClient):
    """OpenAI API Client - FALLBACK PROVIDER"""
    
    def __init__(self, api_key: Optional[str] = None, organization: Optional[str] = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("⚠️ OPENAI_API_KEY not found - OpenAI will not be available")
            self.client = None
        else:
            try:
                import openai  # ← IMPORT HERE, not at top of file
                self.client = openai.OpenAI(api_key=api_key, organization=organization)
            except ImportError:
                logger.warning("⚠️ OpenAI package not installed - OpenAI client unavailable")
                self.client = None
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """Generate response from OpenAI API"""
        if not self.client:
            raise Exception("OpenAI client not initialized - check API key")
        
        start_time = time.time()
        
        messages = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p
            )
            
            latency = time.time() - start_time
            
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "estimated_cost": self.estimate_cost(response.usage)
            }
            
            return LLMResponse(
                content=response.choices[0].message.content,
                provider=LLMProvider.OPENAI,
                model=config.model,
                usage=usage,
                latency=latency
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    def estimate_cost(self, usage) -> float:
        """Estimate OpenAI cost"""
        # Simplified cost estimation
        # GPT-4 Turbo: ~$10 per million tokens
        total_tokens = usage.total_tokens
        return (total_tokens / 1_000_000) * 10

class MockClient(BaseLLMClient):
    """Mock LLM Client for development/testing - no external dependencies"""
    def __init__(self, responses: Optional[Dict] = None):  # ← ADD THIS PARAMETER
        self.responses = responses or self._get_default_responses()
        logger.info(f"✅ Mock LLM client initialized with {len(self.responses)} custom responses")
    
    def _get_default_responses(self):
        """Default mock responses if none provided"""
        return {
            "general": f"[Mock LLM] I understand your query.\n\nAs a mock AI assistant, I'm simulating a response. In production, this would come from a real LLM.",
            "technical": f"[Mock Technical VNI] For technical query.\n\nMock code analysis: Simulated technical response.",
            "medical": f"[Mock Medical VNI] For medical query.\n\n⚠️ MOCK MEDICAL DISCLAIMER: This is simulated medical advice. Always consult real healthcare professionals.",
            "legal": f"[Mock Legal VNI] For legal query.\n\n⚠️ MOCK LEGAL DISCLAIMER: This is simulated legal information, not actual legal advice."
        }
    def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """Generate mock response"""
        import time
        
        start_time = time.time()
        
        # Simple mock responses based on VNI context
        mock_responses = self.responses
        #{
        #    "general": f"[Mock LLM] I understand you said: '{prompt[:100]}...'.\n\nAs a mock AI assistant, I'm simulating a response. In production, this would come from a real LLM.",
        #    "technical": f"[Mock Technical VNI] For technical query: '{prompt[:100]}...'\n\nMock code analysis: This appears to be a {len(prompt.split())}-word technical query about programming/technology.",
        #    "medical": f"[Mock Medical VNI] For medical query: '{prompt[:100]}...'\n\n⚠️ MOCK MEDICAL DISCLAIMER: This is simulated medical advice. Always consult real healthcare professionals.",
        #    "legal": f"[Mock Legal VNI] For legal query: '{prompt[:100]}...'\n\n⚠️ MOCK LEGAL DISCLAIMER: This is simulated legal information, not actual legal advice."
        #}
        
        # Simulate API latency
        time.sleep(0.1)  # 100ms delay
        latency = time.time() - start_time
        
        # Determine context from config or prompt
        context = "general"
        if config.system_prompt:
            if "medical" in config.system_prompt.lower():
                context = "medical"
            elif "legal" in config.system_prompt.lower():
                context = "legal"
            elif "technical" in config.system_prompt.lower():
                context = "technical"
        
        content = mock_responses.get(context, mock_responses["general"])
        
        return LLMResponse(
            content=content,
            provider=LLMProvider.MOCK,  # ← CORRECT: Should be MOCK!
            model="mock-model",
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(content.split()),
                "total_tokens": len(prompt.split()) + len(content.split()),
                "estimated_cost": 0.0
            },
            latency=latency
        )
    
class LLMGateway:
    """
    Unified gateway optimized for Deepseek priority with fallback
    """
    def __init__(self, configs: Optional[Dict[LLMProvider, Dict]] = None):
        # Default configuration if none provided
        if configs is None:
            configs = self._get_default_configs()
        
        self.clients = {}
        self.setup_clients(configs)
        self.cache = {}  # Simple in-memory cache
        self.analytics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_cost": 0.0,
            "provider_stats": defaultdict(lambda: {"requests": 0, "cost": 0.0, "failures": 0})
        }
        
        # Priority order: Deepseek first, then fallbacks
        self.priority_order = [
            LLMProvider.MOCK,    
            LLMProvider.DEEPSEEK,
            LLMProvider.OPENAI,
            # Add other providers as needed
        ]

    def _get_default_configs(self) -> Dict[LLMProvider, Dict]:
        """Get default configurations from environment variables"""
        configs = {}
    
        # Check for mock mode FIRST
        use_mock = os.getenv("LLM_PROVIDER", "").lower() == "mock"
        if use_mock:
            configs[LLMProvider.MOCK] = {}  # Mock doesn't need config
            logger.info("🔧 Using MOCK LLM provider for development")
            return configs
            
        # Deepseek configuration (PRIMARY)
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_key:
            configs[LLMProvider.DEEPSEEK] = {"api_key": deepseek_key}
        
        # OpenAI configuration (FALLBACK - optional)
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            configs[LLMProvider.OPENAI] = {
                "api_key": openai_key,
                "organization": os.getenv("OPENAI_ORG")
            }
    
        # If no providers configured, use mock as fallback
        if not configs:
            configs[LLMProvider.MOCK] = {}
            logger.info("🔧 No API keys found, defaulting to MOCK LLM provider")
            
        return configs
    
    def setup_clients(self, configs: Dict[LLMProvider, Dict]):
        """Initialize clients based on configuration"""
        for provider, config in configs.items():
            try:
                if provider == LLMProvider.DEEPSEEK:
                    # Extract API key from config (handle both formats)
                    api_key = self._extract_api_key(config.get("api_key"))
                    base_url = config.get("base_url", "https://api.deepseek.com")
                    self.clients[provider] = DeepSeekClient(api_key=api_key, base_url=base_url)
                    
                elif provider == LLMProvider.OPENAI:
                    # Extract API key from config
                    api_key = self._extract_api_key(config.get("api_key"))
                    organization = config.get("organization")
                    self.clients[provider] = OpenAIClient(api_key=api_key, organization=organization)
                    
                elif provider == LLMProvider.MOCK:
                    # Get responses from config
                    responses = config.get("responses", {})
                    self.clients[provider] = MockClient(responses=responses)
                        
                # Add more providers as needed
                logger.info(f"✅ Initialized {provider.value} client")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize {provider}: {str(e)}")
    
    def _extract_api_key(self, api_key_config):
        """Extract API key from various formats"""
        if api_key_config is None:
            return None
        elif hasattr(api_key_config, 'value'):
            # Object with .value attribute
            return api_key_config.value
        elif isinstance(api_key_config, dict) and 'value' in api_key_config:
            # Dictionary with 'value' key (what you're now sending)
            return api_key_config['value']
        elif isinstance(api_key_config, str):
            # Plain string (old format)
            return api_key_config
        else:
            return str(api_key_config)
    
    def get_cache_key(self, prompt: str, config: LLMConfig) -> str:
        """Generate cache key for prompt/config combination"""
        key_data = f"{prompt[:100]}_{config.provider.value}_{config.model}_{config.temperature}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def generate(
        self,
        prompt: str,
        vni_context: Optional[str] = None,
        preferred_provider: Optional[LLMProvider] = None,
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """
        Generate response using Deepseek priority strategy
        
        Args:
            prompt: User's query
            vni_context: Which VNI is active (medical, technical, legal, etc.)
            preferred_provider: Override provider preference
            config: Custom LLM configuration
        """
        self.analytics["total_requests"] += 1
        
        # Default config optimized for VNI context
        if config is None:
            model = self._get_model_for_vni(vni_context)
            system_prompt = self._get_system_prompt_for_vni(vni_context)
            
            config = LLMConfig(
                provider=LLMProvider.DEEPSEEK,  # Default to Deepseek
                model=model,
                temperature=self._get_temperature_for_vni(vni_context),
                max_tokens=4000,
                system_prompt=system_prompt
            )
        
        # Check cache
        cache_key = self.get_cache_key(prompt, config)
        if cache_key in self.cache:
            logger.info(f"Cache hit for prompt: {prompt[:50]}...")
            return self.cache[cache_key]
        
        # Determine providers to try
        providers_to_try = []
        if preferred_provider and preferred_provider in self.clients:
            providers_to_try.append(preferred_provider)
        
        # Add priority providers
        for provider in self.priority_order:
            if provider not in providers_to_try and provider in self.clients:
                providers_to_try.append(provider)
        
        # Try providers in order
        last_error = None
        for provider in providers_to_try:
            try:
                config.provider = provider
                response = self.clients[provider].generate(prompt, config)
                
                # Update analytics
                self.analytics["successful_requests"] += 1
                self.analytics["provider_stats"][provider]["requests"] += 1
                
                cost = response.usage.get("estimated_cost", 0)
                self.analytics["total_cost"] += cost
                self.analytics["provider_stats"][provider]["cost"] += cost
                
                # Cache successful response
                self.cache[cache_key] = response
                
                # Limit cache size
                if len(self.cache) > 1000:
                    # Remove oldest entry
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                
                logger.info(f"✅ Generated response using {provider.value} ({response.latency:.2f}s)")
                return response
                
            except Exception as e:
                last_error = e
                self.analytics["failed_requests"] += 1
                self.analytics["provider_stats"][provider]["failures"] += 1
                logger.warning(f"Provider {provider} failed: {str(e)}")
                continue
        
        # All providers failed
        error_msg = f"All LLM providers failed. Last error: {str(last_error)}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    def _get_model_for_vni(self, vni_context: Optional[str]) -> str:
        """Get appropriate model based on VNI context"""
        vni_model_map = {
            "technical": "deepseek-coder",
            "code": "deepseek-coder",
            "medical": "deepseek-chat",
            "legal": "deepseek-chat",
            "general": "deepseek-chat",
        }
        
        if vni_context and vni_context.lower() in vni_model_map:
            return vni_model_map[vni_context.lower()]
        
        return "deepseek-chat"  # Default
    
    def _get_system_prompt_for_vni(self, vni_context: Optional[str]) -> str:
        """Get system prompt based on VNI context"""
        base_prompt = "You are BabyBIONN, an AI assistant with specialized Virtual Neural Instances (VNIs)."
        
        vni_prompts = {
            "medical": f"{base_prompt} MEDICAL VNI ACTIVATED: You are now a medical AI assistant. Provide accurate healthcare information, symptom analysis, and medical guidance. Always recommend consulting healthcare professionals for serious issues.",
            "technical": f"{base_prompt} TECHNICAL VNI ACTIVATED: You are now a technical AI assistant. Provide code examples, architecture advice, debugging help, and technical explanations. Be precise and practical.",
            "legal": f"{base_prompt} LEGAL VNI ACTIVATED: You are now a legal AI assistant. Provide legal information, document analysis, and compliance guidance. Always state this is not legal advice.",
            "general": f"{base_prompt} GENERAL VNI ACTIVATED: You are now a general knowledge assistant. Provide helpful, accurate information across all topics."
        }
        
        if vni_context and vni_context.lower() in vni_prompts:
            return vni_prompts[vni_context.lower()]
        
        return base_prompt
    
    def _get_temperature_for_vni(self, vni_context: Optional[str]) -> float:
        """Get appropriate temperature based on VNI context"""
        # Lower temperature for factual domains, higher for creative
        temperature_map = {
            "medical": 0.1,    # More factual
            "legal": 0.1,      # More factual
            "technical": 0.2,  # Balanced
            "general": 0.3,    # Creative but reasonable
        }
        
        if vni_context and vni_context.lower() in temperature_map:
            return temperature_map[vni_context.lower()]
        
        return 0.3  # Default
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get gateway analytics"""
        return self.analytics.copy()
    
    def clear_cache(self):
        """Clear response cache"""
        self.cache.clear()
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return [p.value for p in self.clients.keys()]

# Singleton instance for easy import
def create_gateway() -> LLMGateway:
    """Create and return a configured LLM Gateway instance"""
    return LLMGateway()

# ========== SINGLETON ACCESSORS - ADD THIS SECTION ==========
_GATEWAY_INSTANCE = None

def get_gateway(configs: Optional[Dict[LLMProvider, Dict]] = None) -> LLMGateway:
    """
    SINGLETON accessor - returns the SAME instance every time.
    Use this instead of create_gateway() for all production code.
    """
    global _GATEWAY_INSTANCE
    if _GATEWAY_INSTANCE is None:
        logger.info("🚀 Creating singleton LLM Gateway instance")
        _GATEWAY_INSTANCE = LLMGateway(configs)
    return _GATEWAY_INSTANCE

def create_gateway() -> LLMGateway:
    """
    Legacy function - maintained for backward compatibility.
    Now returns the singleton instance instead of creating new ones.
    """
    return get_gateway()
# ============================================================

# Test function
def test_deepseek_gateway():
    """Test the Deepseek-prioritized gateway"""
    print("🧪 Testing Deepseek-prioritized LLM Gateway...")
    
    try:
        gateway = create_gateway()
        
        # Test with different VNIs
        test_cases = [
            ("Hello, who are you?", "general", "Basic greeting"),
            ("Write a Python function to sort a list", "technical", "Code generation"),
            ("What are common cold symptoms?", "medical", "Medical query"),
        ]
        
        for prompt, vni, description in test_cases:
            print(f"\n🔧 Testing: {description}")
            print(f"VNI: {vni}")
            print(f"Prompt: {prompt[:50]}...")
            
            try:
                response = gateway.generate(
                    prompt=prompt,
                    vni_context=vni
                )
                
                print(f"✅ Success!")
                print(f"   Provider: {response.provider.value}")
                print(f"   Model: {response.model}")
                print(f"   Latency: {response.latency:.2f}s")
                print(f"   Response length: {len(response.content)} chars")
                print(f"   Estimated cost: ${response.usage.get('estimated_cost', 0):.6f}")
                
            except Exception as e:
                print(f"❌ Failed: {str(e)}")
        
        print(f"\n📊 Analytics:")
        analytics = gateway.get_analytics()
        print(f"   Total requests: {analytics['total_requests']}")
        print(f"   Successful: {analytics['successful_requests']}")
        print(f"   Failed: {analytics['failed_requests']}")
        print(f"   Total cost: ${analytics['total_cost']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Gateway test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Run test if executed directly
    test_deepseek_gateway()
