# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

"""
Configuration Settings
"""
import os
from typing import Dict, Any

class Config:
    """Application configuration"""
    
    # Server settings
    HOST = os.getenv("BABYBIONN_HOST", "0.0.0.0")
    PORT = 8002
    RELOAD = os.getenv("BABYBIONN_RELOAD", "True").lower() == "true"
    
    # API settings
    API_PREFIX = "/api"
    CORS_ORIGINS = ["*"]
    
    # Model settings
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")
    
    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # VNI settings
    DEFAULT_VNI_COUNTS = {
        "medical": 1,
        "legal": 1,
        "general": 1
    }
    
    # Generation settings
    GENERATION_ENABLED = os.getenv("GENERATION_ENABLED", "True").lower() == "true"
    GENERATION_MAX_LENGTH = int(os.getenv("GENERATION_MAX_LENGTH", 200))
    
    # Autonomy settings
    AUTONOMY_ENABLED = os.getenv("AUTONOMY_ENABLED", "True").lower() == "true"
    AUTONOMY_LEVEL = float(os.getenv("AUTONOMY_LEVEL", "0.6"))
    
    # Learning settings
    LEARNING_ENABLED = os.getenv("LEARNING_ENABLED", "True").lower() == "true"
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.01"))
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            key: value for key, value in cls.__dict__.items() 
            if not key.startswith('__') and not callable(value)
        } 
