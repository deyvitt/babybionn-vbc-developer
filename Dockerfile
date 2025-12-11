# Dockerfile.optimized
# Uses pre-built PyTorch image with CUDA 11.8 for maximum speed
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# ============ ARGUMENTS (can be passed at build time) ============
ARG BUILD_ENV=production
ARG PYTHON_VERSION=3.10

# ============ METADATA ============
LABEL maintainer="BabyBIONN Team"
LABEL description="BabyBIONN: Neural Mesh AI System with FastAPI & PyTorch"
LABEL version="1.0.0"

# ============ ENVIRONMENT VARIABLES ============
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV BUILD_ENV=${BUILD_ENV}
ENV TZ=UTC
ENV LANG=C.UTF-8
ENV PYTHONPATH=/app

# ============ SYSTEM DEPENDENCIES ============
# Install packages with retry logic for network issues
RUN echo "Acquire::Retries \"5\";" > /etc/apt/apt.conf.d/80-retries && \
    echo "Acquire::http::Timeout \"120\";" >> /etc/apt/apt.conf.d/80-retries && \
    echo "Acquire::https::Timeout \"120\";" >> /etc/apt/apt.conf.d/80-retries && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libsndfile1 \
        ffmpeg \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# ============ SET MIRROR ENVIRONMENT VARIABLES ============
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_TRUSTED_HOST=mirrors.aliyun.com

# ============ UPGRADE PIP ============
RUN python -m pip install --upgrade pip setuptools wheel

# ============ COPY AND FILTER REQUIREMENTS ============
# Copy requirements first (better caching)
COPY requirements.txt /tmp/requirements.txt

# Create a filtered requirements file without PyTorch (already in base image)
RUN grep -v -E "^(torch|torchvision|torchaudio|--index-url)" /tmp/requirements.txt > /tmp/filtered_requirements.txt

# ============ INSTALL PACKAGES ============
RUN pip install --no-cache-dir -r /tmp/filtered_requirements.txt

# Clean up
RUN rm /tmp/requirements.txt /tmp/filtered_requirements.txt

# ============ NLP MODELS & DATA ============
# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download NLTK data
RUN python -c "import nltk; \
    nltk.download('punkt'); \
    nltk.download('stopwords'); \
    nltk.download('wordnet'); \
    nltk.download('averaged_perceptron_tagger'); \
    nltk.download('omw-eng')"

# ============ CREATE APP STRUCTURE ============
RUN mkdir -p \
    /app/knowledge_bases \
    /app/models \
    /app/pretraining_data \
    /app/checkpoints \
    /app/logs \
    /app/cache \
    /app/mesh_patterns \
    /app/static \
    /app/temp

# Set permissions
RUN chmod -R 755 /app

# ============ COPY APPLICATION ============
# Copy only necessary files (filtered by .dockerignore)
COPY . /app/

# ============ ENVIRONMENT CONFIGURATION ============
# Model caching directories
ENV TRANSFORMERS_CACHE=/app/models
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
ENV HF_HOME=/app/models
ENV HF_DATASETS_CACHE=/app/cache
ENV XDG_CACHE_HOME=/app/cache
ENV TORCH_HOME=/app/models
ENV TORCH_CACHE=/app/models

# Application configuration
ENV PORT=8001
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO
ENV KNOWLEDGE_BASE_DIR=/app/knowledge_bases
ENV MESH_PATTERNS_DIR=/app/mesh_patterns
ENV CHECKPOINTS_DIR=/app/checkpoints
ENV STATIC_DIR=/app/static

# Neural Mesh specific
ENV NEURAL_MESH_ENABLED=true
ENV MESH_LEARNING_RATE=0.1
ENV MESH_ACTIVATION_THRESHOLD=0.3
ENV MAX_CONCURRENT_VNIS=10

# Set default environment based on build arg
RUN if [ "$BUILD_ENV" = "development" ]; then \
    echo "Building in development mode"; \
    else \
    echo "Building in production mode"; \
    fi

# ============ VALIDATION & TEST ============
# Test critical imports
RUN python <<'EOF'
import sys
print('Python:', sys.version)
print()

import torch
print(f'✅ PyTorch {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'✅ CUDA version: {torch.version.cuda}')
print()

import transformers
print(f'✅ Transformers: {transformers.__version__}')
import sentence_transformers
print(f'✅ Sentence Transformers: {sentence_transformers.__version__}')
print()

import fastapi
print(f'✅ FastAPI: {fastapi.__version__}')
import uvicorn
print(f'✅ Uvicorn: {uvicorn.__version__}')
print()

print('✅ All critical dependencies loaded successfully!')
EOF

# ============ HEALTH CHECK ============
# Install curl if not present
RUN which curl || apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# ============ PORTS ============
# FastAPI
EXPOSE 8001
# Jupyter/optional
EXPOSE 8888

# ============ ENTRYPOINT ============
# Use shell form for environment variable expansion
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8001} --workers ${WORKERS:-2}"]
