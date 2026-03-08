# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributorsnan

# neuron/reinforcement_learning/training/usage_example.py
# usage_example.py
"""
Example using our EXISTING knowledge_bases folder
"""

import logging
from training_pipeline import create_training_pipeline

logging.basicConfig(level=logging.INFO)

class MockVNIManager:
    def __init__(self):
        self.vnis = {}
    def register_vni(self, vni_id, vni_type):
        self.vnis[vni_id] = {'type': vni_type, 'pretrained_patterns': {}}

class MockRLSystem:
    def __init__(self):
        self.rl_engine = MockRLEngine()

class MockRLEngine:
    def update_synaptic_strength(self, vni_id, pattern_id, strength):
        print(f"Updated synaptic strength for {vni_id}.{pattern_id}: {strength}")

def main():
    print("🚀 Starting BabyBIONN Training with EXISTING knowledge_bases")
    
    vni_manager = MockVNIManager()
    rl_system = MockRLSystem()
    
    # Create training pipeline that uses your existing knowledge_bases
    training_pipeline = create_training_pipeline(vni_manager, rl_system)
    
    # Check what domains are available
    available_domains = training_pipeline.knowledge_loader.get_all_domains()
    print(f"📚 Available domains: {available_domains}")
    
    # Run training on available domains
    if available_domains:
        results = training_pipeline.run_complete_training(available_domains)
        print("✅ Training completed!")
    else:
        print("❌ No domains found in knowledge_bases folder")

if __name__ == "__main__":
    main()
