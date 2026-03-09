# SPDX-License-Identifier: MPL-2.0
# This file is part of BabyBIONN. See https://mozilla.org/MPL/2.0/ for license terms.
FROM python:3.10-slim

WORKDIR /app

# 1. Install system dependencies (including protobuf compiler)
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    build-essential \
    libgmp-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy requirements.txt
COPY requirements.txt .

# 3. First, install a known-good protobuf version
RUN pip install --no-cache-dir protobuf==3.20.3

# 4. Install libp2p from source (compiled against the protobuf we just installed)
RUN pip install --no-cache-dir --no-binary libp2p libp2p

# 5. Install CPU-only PyTorch
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# 6. Install remaining packages from requirements.txt
# (protobuf and libp2p lines should be removed from requirements.txt to avoid conflict)
RUN pip install --no-cache-dir -r requirements.txt

# 7. Verify protobuf version
RUN python -c "import google.protobuf; print('✅ Installed protobuf version:', google.protobuf.__version__)"

# 8. Download spaCy model
RUN python -c "import spacy; spacy.cli.download('en_core_web_sm')"

# 9. Install NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 10. Copy app code
COPY . .

# 11. Create directories for BabyBIONN
RUN mkdir -p /app/knowledge_bases /app/models /app/logs /app/static /app/templates /app/cache

# 12. Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=-1
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL="INFO"

# 13. Create non-root user for security
RUN useradd -m -u 1000 bionnuser && chown -R bionnuser:bionnuser /app
USER bionnuser

# 14. Test PyTorch is CPU-only
RUN python -c "\
import torch;\
print(f'✅ PyTorch {torch.__version__}');\
print(f'✅ CUDA available: {torch.cuda.is_available()} (False = Good)')\
"

# 15. Test core imports
RUN python <<EOF
import torch
import transformers
import fastapi
import spacy
import libp2p
print("✅ Core dependencies imported successfully")
print(f"PyTorch {torch.__version__}, Transformers {transformers.__version__}")
print(f"libp2p {libp2p.__version__ if hasattr(libp2p, '__version__') else 'installed'}")
EOF

EXPOSE 8002

# 16. Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8002/health || exit 1

CMD ["python", "main.py"]
