FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    SUPERCRONIC_URL=https://github.com/aptible/supercronic/releases/download/v0.2.29/supercronic-linux-amd64 && \
    curl -fsSL "$SUPERCRONIC_URL" -o /usr/local/bin/supercronic && \
    chmod +x /usr/local/bin/supercronic && \
    rm -rf /var/lib/apt/lists/*

COPY predictor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --quiet

# Bake the ONNX model into the image so it never downloads at runtime
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding('sentence-transformers/all-MiniLM-L6-v2').embed(['warmup']))"

COPY shared/ ./shared/
COPY predictor/ ./predictor/

CMD ["supercronic", "/app/predictor/crontab"]
