FROM python:3.10-slim

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    build-essential \
    libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy requirements with CPU PyTorch
COPY requirements.txt .

# 3. First install CPU-only PyTorch (BEFORE everything else)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# 4. Install OpenSSL for secure API connections (for LLM Gateway)
RUN apt-get update && apt-get install -y \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 5. Then install all other packages
RUN pip install --no-cache-dir -r requirements.txt

# 6. Download spaCy model
RUN python -c "import spacy; spacy.cli.download('en_core_web_sm')"

# 7. Install NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 8. Copy app code
COPY . .

# 9. Create directories for BabyBIONN
RUN mkdir -p /app/knowledge_bases /app/models /app/logs /app/static /app/templates /app/cache

# 10. Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=-1
ENV PYTHONUNBUFFERED=1
#ENV DEEPSEEK_API_KEY=""
#ENV OPENAI_API_KEY=""
ENV LOG_LEVEL="INFO"

# 11. Create non-root user for security
RUN useradd -m -u 1000 bionnuser && chown -R bionnuser:bionnuser /app
USER bionnuser

# 12. Test PyTorch is CPU-only
RUN python -c "\
import torch;\
print(f'✅ PyTorch {torch.__version__}');\
print(f'✅ CUDA available: {torch.cuda.is_available()} (False = Good)')\
"

# 13. Test LLM Gateway imports
RUN python <<EOF
import torch
import transformers
import fastapi
import spacy
print("✅ Core dependencies imported successfully")
print(f"PyTorch {torch.__version__}, Transformers {transformers.__version__}")
EOF

EXPOSE 8002

# 14. Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8002/health || exit 1

CMD ["python", "main.py"] 
