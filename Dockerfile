# SalesIntelligence – Render Dockerfile
# Leichtgewichtig: kein Aurora/PyTorch, nur Flask + RunPod-Client

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python datasets/generate_defaults.py

RUN useradd -m -u 1000 aurora && chown -R aurora:aurora /app
USER aurora

ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Render setzt PORT automatisch
EXPOSE 8000

# 1 Worker reicht (kein lokales Modell, nur API-Calls)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 4 --timeout 120 --worker-class gthread --access-logfile - --error-logfile - app:app"]
