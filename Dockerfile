# ══════════════════════════════════════════════════════════
#  Aurora Web App – Production Dockerfile
#  Basis: Python 3.11 slim  |  CPU-only (kein CUDA)
# ══════════════════════════════════════════════════════════

FROM python:3.11-slim

# System-Pakete (minimiert)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

# Arbeitsverzeichnis
WORKDIR /app

# Abhängigkeiten zuerst (Docker-Cache-Optimierung)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# App-Code kopieren (inkl. aurora_checkpoint.pt ~805 MB)
COPY . .

# Default-Datensätze generieren
RUN python datasets/generate_defaults.py

# Nicht-Root-Benutzer (Azure-Security-Best-Practice)
RUN useradd -m -u 1000 aurora && chown -R aurora:aurora /app
USER aurora

# HuggingFace-Cache in Workdir (wichtig für Azure – kein Schreib-Zugriff auf /home)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Checkpoint-Pfad: aurora_checkpoint.pt liegt neben app.py im Container
# Überschreiben mit: -e AURORA_MODEL_PATH=/mnt/models/aurora_checkpoint.pt
ENV AURORA_MODEL_PATH=aurora_checkpoint.pt

# Port (Azure setzt PORT-Env automatisch)
EXPOSE 8000

# Startup: Gunicorn mit 2 Workern (CPU-bound, mehr bringt nichts)
# --timeout 600: Aurora-Inference kann mehrere Minuten dauern
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--threads", "4", \
     "--timeout", "600", \
     "--worker-class", "sync", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
