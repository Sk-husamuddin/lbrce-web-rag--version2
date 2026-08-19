FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface \
    TRANSFORMERS_CACHE=/opt/huggingface \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /app

# Runtime libraries needed by PyMuPDF/Pillow and common scientific wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libglib2.0-0 \
       libgomp1 \
       libjpeg62-turbo \
       libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY backend ./backend

# The API is intentionally backend-only; secrets must be supplied through .env
# or the deployment platform, never copied into the image.
EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
