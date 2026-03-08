# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors
# config.py
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration management for BabyBIONN"""
    
    # Application
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    PORT = int(os.getenv("PORT", "8001"))
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-in-production")
    
    # Paths
    KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_bases")
    MESH_PATTERNS_DIR = os.getenv("MESH_PATTERNS_DIR", "./mesh_patterns")
    CHECKPOINTS_DIR = os.getenv("CHECKPOINTS_DIR", "./checkpoints")
    
    # Neural Mesh
    NEURAL_MESH_ENABLED = os.getenv("NEURAL_MESH_ENABLED", "true").lower() == "true"
    MESH_LEARNING_RATE = float(os.getenv("MESH_LEARNING_RATE", "0.1"))
    
    @classmethod
    def validate(cls) -> None:
        """Validate required environment variables"""
        required = ["SECRET_KEY"]
        for var in required:
            if not getattr(cls, var):
                raise ValueError(f"Missing required environment variable: {var}")

# Validate on import
Config.validate()

# Export config
config = Config 
