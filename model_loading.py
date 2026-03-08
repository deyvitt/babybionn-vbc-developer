# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

# model_loading.py
import torch
import logging
from ultralytics import YOLO
import spacy
import os

logger = logging.getLogger("BabyBIONN-ModelManager")

class ModelManager:
    def __init__(self):
        self.device = self.setup_device()
        self.models = {}
        self.models_loaded = False
        
    def setup_device(self):
        """Setup GPU device with optimization"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("🖥️ Using CPU")
        return device
    
    def load_vision_models(self):
        """Load computer vision models"""
        try:
            logger.info("📷 Loading vision models...")
            # YOLO model for object detection
            self.models['yolo'] = YOLO('yolov8n.pt').to(self.device)
            logger.info("✅ Vision models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load vision models: {e}")
            return False
        
    def load_nlp_models(self):
        """Load natural language processing models - simplified without transformers"""
        try:
            logger.info("📝 Loading NLP models...")
            # Only spaCy for basic NLP - no transformers needed
            self.models['spacy'] = spacy.load('en_core_web_sm')
            logger.info("✅ NLP models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load NLP models: {e}")
            return False
        
    def load_rl_models(self):
        """Load reinforcement learning models"""
        try:
            logger.info("🎮 Loading RL models...")
            # Load RL policy models if they exist
            if os.path.exists('models/policy_net.pth'):
                self.models['policy_net'] = torch.load('models/policy_net.pth', map_location=self.device)
                logger.info("✅ RL models loaded successfully")
                return True
            else:
                logger.warning("⚠️ RL models not found, skipping")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to load RL models: {e}")
            return False
    
    def load_all_models(self):
        """Load all models and return success status"""
        logger.info("🚀 Starting model loading sequence...")
        
        success = True
        success &= self.load_vision_models()
        success &= self.load_nlp_models() 
        success &= self.load_rl_models()
        
        self.models_loaded = success
        if success:
            logger.info("🎉 All models loaded successfully!")
        else:
            logger.warning("⚠️ Some models failed to load")
            
        return success
        
    def cleanup(self):
        """Cleanup GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 GPU memory cleaned up")
    
    def get_model_status(self):
        """Return status of all models"""
        return {
            'vision_loaded': 'yolo' in self.models,
            'nlp_spacy_loaded': 'spacy' in self.models,
            'rl_loaded': 'policy_net' in self.models,
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
            'all_models_loaded': self.models_loaded
        }

# Global instance
model_manager = ModelManager()
