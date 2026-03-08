#!/bin/bash
# setup.sh - Initial BabyBIONN setup script

echo "🚀 Setting up BabyBIONN project..."
echo "================================="

# 1. Create virtual environment (optional)
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# 2. Activate virtual environment (optional)
echo "🔧 Activating virtual environment..."
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install Python dependencies
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Create .env file from template if it doesn't exist
if [ ! -f ".env" ]; then
    echo "🔐 Creating .env file from template..."
    
    # Generate a random secret key
    SECRET_KEY=$(openssl rand -hex 32)
    
    # Create .env file with the generated key
    cat > .env <<EOF
# 🚨 DO NOT COMMIT THIS FILE TO GIT! 🚨
# Copy to .env.example and commit that instead

# ============ APPLICATION ============
ENVIRONMENT=development
LOG_LEVEL=DEBUG
PORT=8001
WORKERS=2
RELOAD=true

# ============ SECURITY & API KEYS ============
SECRET_KEY=$SECRET_KEY
API_KEY=your-api-key-for-external-services

# ============ MODEL PATHS ============
KNOWLEDGE_BASE_DIR=./knowledge_bases
MESH_PATTERNS_DIR=./mesh_patterns
CHECKPOINTS_DIR=./checkpoints

# ============ NEURAL MESH ============
NEURAL_MESH_ENABLED=true
MESH_LEARNING_RATE=0.1
MESH_ACTIVATION_THRESHOLD=0.3
MAX_CONCURRENT_VNIS=10
MESH_CACHE_SIZE=1000
EOF
    
    echo "✅ Generated random SECRET_KEY: $SECRET_KEY"
    echo "✅ .env file created"
    
    # Set secure permissions
    chmod 600 .env
    echo "✅ Set secure permissions on .env file"
else
    echo "✅ .env file already exists"
fi

# 5. Create necessary directories
echo "📁 Creating project directories..."
mkdir -p knowledge_bases mesh_patterns checkpoints logs cache models

# 6. Create .env.example (safe to commit)
if [ ! -f ".env.example" ]; then
    echo "📄 Creating .env.example template..."
    cat > .env.example <<EOF
# Copy this file to .env and fill in your values
ENVIRONMENT=development
PORT=8001
SECRET_KEY=change-this-to-a-random-value
API_KEY=your-api-key-here
KNOWLEDGE_BASE_DIR=./knowledge_bases
MESH_PATTERNS_DIR=./mesh_patterns
EOF
fi

# 7. Test Docker build (optional)
echo "🐳 Testing Docker build..."
if command -v docker &> /dev/null; then
    echo "Building test image..."
    docker build -f Dockerfile.optimized -t babybionn-test .
else
    echo "Docker not installed, skipping Docker test"
fi

echo ""
echo "🎉 Setup complete!"
echo "Next steps:"
echo "1. Review the .env file and update any values"
echo "2. Run locally: python main.py"
echo "3. Or build Docker: docker-compose build"
echo "4. Or run with Docker: docker-compose up"

# Make it executable:
# chmod +x setup.sh

# Run it
# ./setup.sh
#_______________________________________
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors
