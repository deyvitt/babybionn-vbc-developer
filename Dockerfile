FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
     ULTRALYTICS_CACHE_DIR=/app/ultralytics_cache   

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \   
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Create directories (NO YOLO DOWNLOAD HERE)
RUN mkdir -p /app/knowledge_bases /app/static/uploads /app/models /app/ultralytics_cache

# Copy application code
COPY . .

# Fix knowledge base path (update this if the file structure changed)
RUN if [ -f "enhanced_vni_classes.py" ]; then sed -i 's|filename = f"knowledge_|filename = f"/app/knowledge_bases/knowledge_|' enhanced_vni_classes.py; fi

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
