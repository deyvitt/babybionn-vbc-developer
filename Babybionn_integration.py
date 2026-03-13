# File: babybionn_integration.py
"""
Main integration layer connecting all components
"""
import hashlib
import logging
from datetime import datetime
from llm_Gateway import get_gateway, LLMConfig, LLMProvider
from template_engine import HybridAdaptiveTemplateEngine
from bionn_aggregator import UnifiedAggregator
import asyncio
from typing import Dict, Optional, Any


logger = logging.getLogger("babybionn_integration")

class BabyBIONNSystem:
    """
    Complete BabyBIONN system integrating:
    1. Reasoning (your aggregator)
    2. Template Engine (reasoning → instructions)
    3. LLM Gateway (instructions → response)
    """
    
    def __init__(self, llm_configs: Optional[Dict] = None, aggregator_config: Optional[Dict] = None):
        # LLM Gateway is handled by the singleton - we don't pass it through aggregator_config
        # The gateway will be created once via get_gateway() inside aggregator
        
        # Create proper config object for aggregator - NO llm_configs here!
        from bionn_synaptic import SynapticConfig
        
        # Start with default config
        config = SynapticConfig()
        
        # If aggregator_config is provided, update only valid attributes
        if aggregator_config:
            for key, value in aggregator_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"Ignoring unknown aggregator config parameter: {key}")
        
        # Initialize aggregator with config object (NO llm_configs parameter)
        self.aggregator = UnifiedAggregator(config)
        
        # Template engine (no LLM here)
        self.template_engine = HybridAdaptiveTemplateEngine()
        
        # User session management
        self.user_sessions = {}
        logger.info("BabyBIONN System initialized - Aggregator owns LLM Gateway")
    
    async def process_query(
        self,
        user_query: str,
        user_id: Optional[str] = None,
        context: Optional[Dict] = None,
        preferred_llm: Optional[LLMProvider] = None
    ) -> Dict[str, Any]:
        """
        Process user query through complete BabyBIONN pipeline
        
        Returns:
            Complete response with metadata
        """
        start_time = datetime.now()
        
        # Step 1: Get user context
        user_context = self._get_user_context(user_id, context)
        
        # Step 2: Process with aggregator (reasoning)
        logger.info(f"Processing query with aggregator: {user_query[:50]}...")
        aggregator_output = await self.aggregator.process_query_enhanced(
            query=user_query,
            context=user_context
        )
        
        # Step 3: Convert reasoning to LLM instructions
        logger.info("Converting reasoning to LLM instructions")
        llm_prompt = self.template_engine.create_prompt(
            aggregator_output=aggregator_output,
            user_query=user_query,
            user_context=user_context
        )
        
        # Step 4: Generate response via LLM
        logger.info("Generating response via LLM gateway")
        llm_config = LLMConfig(
            provider=preferred_llm or LLMProvider.OPENAI,
            model="gpt-4-turbo-preview",
            temperature=0.3,
            max_tokens=4000,
            system_prompt="You are BabyBIONN, a biological intelligence system."
        )
        
        llm_response = await self.aggregator.llm_gateway.generate(
            prompt=llm_prompt,
            preferred_provider=preferred_llm,
            config=llm_config
        )

        # Step 5: Assemble final response
        processing_time = (datetime.now() - start_time).total_seconds()
        
        final_response = {
            "response": llm_response.content,
            "metadata": {
                "processing_time": processing_time,
                "reasoning_metadata": aggregator_output.get("metadata", {}),
                "llm_metadata": {
                    "provider": llm_response.provider.value,
                    "model": llm_response.model,
                    "latency": llm_response.latency,
                    "tokens_used": llm_response.usage
                },
                "template_engine": {
                    "reasoning_type": "extracted_from_aggregator",
                    "safety_level": "determined_by_analysis"
                }
            },
            "confidence": aggregator_output.get("confidence", 0.5),
            "sources": aggregator_output.get("sources", []),
            "query_id": self._generate_query_id(user_query)
        }
        
        # Step 6: Update user session
        if user_id:
            self._update_user_session(user_id, user_query, final_response)
        
        logger.info(f"Query processed in {processing_time:.2f}s")
        return final_response
    
    def _get_user_context(self, user_id: Optional[str], context: Optional[Dict]) -> Dict:
        """Get or create user context"""
        user_context = context or {}
        
        if user_id:
            # Load from session
            if user_id in self.user_sessions:
                user_context.update(self.user_sessions[user_id].get("context", {}))
            
            # Add user ID to context
            user_context["user_id"] = user_id
        
        return user_context
    
    def _update_user_session(self, user_id: str, query: str, response: Dict):
        """Update user session with query/response"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                "context": {},
                "query_history": [],
                "response_history": []
            }
        
        self.user_sessions[user_id]["query_history"].append({
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "response_confidence": response.get("confidence", 0.5)
        })
        
        # Keep only last 100 queries
        if len(self.user_sessions[user_id]["query_history"]) > 100:
            self.user_sessions[user_id]["query_history"] = \
                self.user_sessions[user_id]["query_history"][-100:]
    
    def _generate_query_id(self, query: str) -> str:
        """Generate unique query ID"""
        import uuid
        return f"query_{hashlib.md5(query.encode()).hexdigest()[:8]}_{uuid.uuid4().hex[:8]}"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and analytics"""
        return {
            "system": "BabyBIONN",
            "version": "2.0.0",
            "status": "operational",
            "components": {
                "aggregator": "active",
                "template_engine": "active",
                "llm_gateway": "active"
            },
            "analytics": {
                "template_engine": self.template_engine.usage_stats,
                "llm_gateway": self.llm_gateway.get_analytics(),
                "active_sessions": len(self.user_sessions)
            }
        } 
