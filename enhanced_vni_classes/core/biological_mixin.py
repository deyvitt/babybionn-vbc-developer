# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

"""Universal biological systems integration mixin"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class BiologicalSystemsMixin:
    """Mixin to provide consistent biological systems across all VNIs"""
    
    def __init__(self, *args, **kwargs):
        # Store biological enable flag BEFORE calling super()
        self._enable_biological_from_init = kwargs.get('enable_biological_systems', True)
        
        # Remove from kwargs to avoid duplicate handling in parent
        if 'enable_biological_systems' in kwargs:
            del kwargs['enable_biological_systems']
        
        # Store domain if provided
        if 'domain' in kwargs:
            self.domain = kwargs.pop('domain')
        
        # Call parent constructor with try-except
        try:
            super().__init__(*args, **kwargs)
        except TypeError as e:
            # If super() fails, try calling without args
            try:
                super().__init__()
            except:
                # If that also fails, just initialize the object
                pass
        
        # Ensure biological systems are initialized
        self._ensure_biological_systems()
    
    def _ensure_biological_systems(self):
        """Ensure biological systems are properly initialized"""
        # Set enable flag
        self.enable_biological_systems = getattr(self, '_enable_biological_from_init', True)
        
        # Initialize if not already done
        if not hasattr(self, 'attention_system') or self.attention_system is None:
            self._initialize_attention_system()
        
        if not hasattr(self, 'activation_router') or self.activation_router is None:
            self._initialize_activation_router()
        
        if not hasattr(self, 'memory_system') or self.memory_system is None:
            self._initialize_memory_system()
        
        logger.debug(f"✅ Biological systems ensured for {getattr(self, 'instance_id', 'unknown')}")
    
    def _initialize_attention_system(self):
        """Initialize attention system with default config"""
        try:
            from neuron.demoHybridAttention import DemoHybridAttention
        
            # Get config from instance or use default
            attention_config = getattr(self, 'attention_config', {})
            if not attention_config:
                attention_config = {
                    'dim': 256,
                    'num_heads': 8,
                    'window_size': 256,
                    'use_sliding': True,
                    'use_global': True,
                    'use_hierarchical': True,
                    'global_token_ratio': 0.05,
                    'memory_tokens': 16,
                    'multi_modal': False
                }
            
            # FIX: Remove invalid parameters that DemoHybridAttention doesn't accept
            valid_params = {
                'dim', 'num_heads', 'window_size', 'use_sliding', 
                'use_global', 'use_hierarchical', 'global_token_ratio', 
                'memory_tokens', 'multi_modal'
            }
            
            # Filter out invalid parameters
            filtered_config = {
                k: v for k, v in attention_config.items() 
                if k in valid_params
            }
            
            # Log removed parameters for debugging
            removed = set(attention_config.keys()) - set(filtered_config.keys())
            if removed:
                logger.warning(f"⚠️ Removing invalid attention config parameters: {removed}")
            
            self.attention_system = DemoHybridAttention(**filtered_config)
            logger.debug(f"🧠 Attention system initialized for {getattr(self, 'instance_id', 'unknown')}")
            
        except ImportError as e:
            logger.error(f"❌ Cannot import DemoHybridAttention: {e}")
            self.attention_system = None
            self.enable_biological_systems = False
        except Exception as e:
            logger.error(f"❌ Failed to initialize attention system: {e}")
            self.attention_system = None
        
    def _initialize_activation_router(self):
        """Initialize activation router"""
        try:
            from neuron.smart_activation_router import SmartActivationRouter
            
            vni_id = getattr(self, 'instance_id', 'unknown')
            domain = getattr(self, 'domain', 'general')
            
            self.activation_router = SmartActivationRouter()
            logger.debug(f"⚡ Activation router initialized for {vni_id}")
            
        except ImportError as e:
            logger.error(f"❌ Cannot import SmartActivationRouter: {e}")
            self.activation_router = None
            self.enable_biological_systems = False
        except Exception as e:
            logger.error(f"❌ Failed to initialize activation router: {e}")
            self.activation_router = None
    
    def _initialize_memory_system(self):
        """Initialize memory system"""
        try:
            from neuron.vni_memory import VniMemory
            
            vni_id = getattr(self, 'instance_id', 'unknown')
            memory_config = getattr(self, 'memory_config', {})
            
            if not memory_config:
                memory_config = {
                    'short_term_capacity': 100,
                    'long_term_capacity': 1000,
                    'consolidation_threshold': 0.7,
                    'retention_period': 86400
                }
            
            self.memory_system = VniMemory(
                vni_id=self.instance_id,
                storage_manager=None
            )
            logger.debug(f"💾 Memory system initialized for {vni_id}")
            
        except ImportError as e:
            logger.error(f"❌ Cannot import VniMemory: {e}")
            self.memory_system = None
            self.enable_biological_systems = False
        except Exception as e:
            logger.error(f"❌ Failed to initialize memory system: {e}")
            self.memory_system = None
    
    def process_with_universal_biological(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """Universal biological processing for all VNIs"""
        if not self.enable_biological_systems:
            return {
                'biological_processing': False,
                'error': 'Biological systems not enabled'
            }
        
        try:
            # 1. Attention mechanism
            attention_result = {}
            if self.attention_system:
                attention_result = self.attention_system.focus(
                    input_data=query,
                    context=context
                )
            
            # 2. Activation routing
            activation_result = {}
            if self.activation_router:
                activation_result = self.activation_router.route(
                    input_text=query,
                    attention_weights=attention_result.get('attention_weights', {}),
                    domain=getattr(self, 'domain', 'general')
                )
            
            # 3. Memory operations
            memory_result = {}
            if self.memory_system:
                # Store current query
                store_result = self.memory_system.store(
                    query=query,
                    context=context,
                    activation_level=activation_result.get('activation_level', 0.5),
                    domain=getattr(self, 'domain', 'general')
                )
                
                # Retrieve relevant memories
                relevant_memories = self.memory_system.retrieve(
                    query=query,
                    context=context,
                    top_k=3
                )
                
                memory_result = {
                    'stored': store_result,
                    'retrieved': relevant_memories
                }
            
            return {
                'attention': attention_result,
                'activation': activation_result,
                'memory': memory_result,
                'biological_processing': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Universal biological processing failed: {e}")
            return {
                'biological_processing': False,
                'error': str(e)
            }
    
    def get_biological_status(self) -> Dict[str, Any]:
        """Get biological systems status"""
        return {
            'enabled': self.enable_biological_systems,
            'attention_system': self.attention_system is not None,
            'activation_router': self.activation_router is not None,
            'memory_system': self.memory_system is not None,
            'health': 'healthy' if all([
                self.attention_system is not None,
                self.activation_router is not None,
                self.memory_system is not None
            ]) else 'degraded'
        } 
