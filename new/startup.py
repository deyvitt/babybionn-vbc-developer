# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

# core/startup.py
import asyncio
import logging
from typing import Optional
from .orchestrator import EnhancedBabyBIONNOrchestrator

logger = logging.getLogger("BabyBIONN-Startup")

class StartupManager:
    """Manages non-blocking startup of the BabyBIONN system"""
    
    def __init__(self):
        self.orchestrator = None
        self.initialized = False
        
    async def initialize(self, background: bool = True):
        """Initialize the system, optionally in background"""
        if background:
            asyncio.create_task(self._initialize_background())
        else:
            await self._initialize_foreground()
    
    async def _initialize_background(self):
        """Initialize in background without blocking"""
        try:
            logger.info("🚀 Starting background initialization...")
            
            # Create orchestrator
            self.orchestrator = EnhancedBabyBIONNOrchestrator()
            
            # Initialize orchestrator (it will internally use core/ modules)
            await self.orchestrator.initialize()
            
            self.initialized = True
            logger.info("✅ Background initialization complete!")
            logger.info(f"📊 Stats: {len(self.orchestrator.vni_instances)} VNIs, "
                       f"{len(self.orchestrator.synaptic_connections)} connections")
            
        except ImportError as e:
            logger.error(f"❌ Import error - Check core/ modules: {e}")
            logger.info("📁 Required core modules:")
            logger.info("   - orchestrator.py (main orchestrator)")
            logger.info("   - attention.py (EnhancedSynapticAttentionBridge)")
            logger.info("   - autonomy.py (AutonomyEngine)")
            logger.info("   - routing.py (SmartActivationRouter)")
            
        except Exception as e:
            logger.error(f"❌ Background initialization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def _initialize_foreground(self):
        """Initialize in foreground (blocking)"""
        await self._initialize_background()
    
    def get_orchestrator(self) -> Optional[EnhancedBabyBIONNOrchestrator]:
        """Get the orchestrator instance"""
        return self.orchestrator 
