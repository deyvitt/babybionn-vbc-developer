# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

"""
Integration with neuron/ directory modules
"""
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def import_neuron_modules():
    """Import modules from the neuron directory"""
    try:
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(PROJECT_ROOT))
        
        # Import demoHybridAttention
        from bionn_attention import DemoHybridAttention
        logger.info("✅ Imported DemoHybridAttention")
        
        # Import aggregator
        from bionn_aggregator import ResponseAggregator, AggregatorConfig
        logger.info("✅ Imported ResponseAggregator")
        
        # Import RL modules if available
        try:
            from neuron.reinforcement_learning.reinforce_learn import RLConfig, VNIReinforcementEngine
            from neuron.reinforcement_learning.vni_rl_integration import VNILearningOrchestrator, VNIStimulus
            logger.info("✅ Imported RL modules")
        except ImportError as e:
            logger.warning(f"RL modules not available: {e}")
            RLConfig = VNIReinforcementEngine = VNILearningOrchestrator = VNIStimulus = None
        
        return {
            'DemoHybridAttention': DemoHybridAttention,
            'ResponseAggregator': ResponseAggregator,
            'AggregatorConfig': AggregatorConfig,
            'RLConfig': RLConfig,
            'VNIReinforcementEngine': VNIReinforcementEngine,
            'VNILearningOrchestrator': VNILearningOrchestrator,
            'VNIStimulus': VNIStimulus
        }
        
    except ImportError as e:
        logger.error(f"❌ Failed to import neuron modules: {e}")
        raise 
