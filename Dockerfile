FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-only PyTorch first (much smaller than the default CUDA build)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so the first start is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .
RUN mkdir -p data logs

# Most hosts pass $PORT (Render, Railway...); Hugging Face Spaces uses 7860
EXPOSE 7860
CMD ["sh", "-c", "streamlit run main.py --server.port=${PORT:-7860} --server.address=0.0.0.0 --server.headless=true"]
