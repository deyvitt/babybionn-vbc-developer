#!/bin/bash
# build.sh - Optimized build for BabyBIONN

set -e  # Exit on error

echo "🚀 Building BabyBIONN with optimized Docker setup..."
echo "=================================================="

# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No build environment specified"
    echo "Usage: $0 [dev|prod|test]"
    exit 1
fi

# Option 1: Quick development build
if [ "$1" = "dev" ]; then
    echo "📦 Building DEVELOPMENT image..."
    docker-compose -f docker-compose.dev.yml build --no-cache
    echo "✅ Development build complete!"
    echo "Run with: docker-compose -f docker-compose.dev.yml up"

# Option 2: Production build
elif [ "$1" = "prod" ]; then
    echo "🏗️  Building PRODUCTION image..."
    docker-compose build --no-cache
    echo "✅ Production build complete!"
    echo "Run with: docker-compose up -d"

# Option 3: Test only
elif [ "$1" = "test" ]; then
    echo "🧪 Testing dependencies..."
    docker build -f Dockerfile -t babybionn-test .
    docker run --rm babybionn-test python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Default: Show help
else
    echo "Usage:"
    echo "  ./build.sh dev     - Build development image"
    echo "  ./build.sh prod    - Build production image"
    echo "  ./build.sh test    - Test if build works and CUDA is available"
    exit 1
fi
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors
