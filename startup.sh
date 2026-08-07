#!/bin/bash
# ═══════════════════════════════════════════════════════
#  Azure App Service Startup Script  (ohne Docker)
#  Wird in Azure als "Startup Command" eingetragen:
#  bash startup.sh
# ═══════════════════════════════════════════════════════
set -e

echo "=== Aurora Web App Startup ==="
echo "Python: $(python --version)"
echo "Workdir: $(pwd)"

# Abhängigkeiten sicherstellen
pip install -r requirements.txt -q

# Default-Datensätze erzeugen (wenn noch nicht vorhanden)
if [ ! -f "datasets/gastro_bier.csv" ]; then
    echo "Generiere Default-Datensätze ..."
    python datasets/generate_defaults.py
fi

# HuggingFace Cache konfigurieren (wichtig: Azure verwendet /tmp)
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HOME}/.cache/transformers"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

# Aurora Checkpoint-Pfad (Standard: neben app.py)
# In Azure App-Einstellungen überschreiben mit AURORA_MODEL_PATH=/pfad/zum/checkpoint.pt
export AURORA_MODEL_PATH="${AURORA_MODEL_PATH:-aurora_checkpoint.pt}"
if [ -f "${AURORA_MODEL_PATH}" ]; then
    echo "Checkpoint gefunden: ${AURORA_MODEL_PATH}  ($(du -sh "${AURORA_MODEL_PATH}" | cut -f1))"
else
    echo "WARNUNG: Kein Checkpoint unter '${AURORA_MODEL_PATH}' – wird von HuggingFace geladen."
fi

echo "Starte Gunicorn ..."
exec gunicorn \
    --bind "0.0.0.0:${PORT:-8000}" \
    --workers 2 \
    --threads 4 \
    --timeout 600 \
    --worker-class sync \
    --access-logfile - \
    --error-logfile - \
    app:app
