#!/usr/bin/env bash
# Baut und pusht das GraphReady-Worker-Image für RunPod.
#   ./runpod_graphml/build.sh ghcr.io/salesintelligence-dev [tag]   # wie das Aurora-Image (GitHub-Registry, privat)
#   ./runpod_graphml/build.sh <dockerhub-user> [tag]                # alternativ Docker Hub
# Danach in RunPod: Serverless → New Endpoint → Container Image <user>/graphready-worker:<tag>,
# GPU z. B. RTX A4000/A5000 (16–24 GB), Min Workers 0, Max Workers 1.
# Die Endpoint-ID als RUNPOD_GRAPHML_ENDPOINT_ID im Render-Dashboard eintragen.
set -euo pipefail
USER_NAME="${1:?Registry angeben, z. B. ghcr.io/salesintelligence-dev}"
TAG="${2:-1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CTX="$HERE/.build"

rm -rf "$CTX" && mkdir -p "$CTX"
cp "$HERE/Dockerfile" "$HERE/requirements.txt" "$HERE/handler.py" "$CTX/"
# Paket ohne Rohdownloads und Caches (Demo-Daten als npz sind dabei)
rsync -a --exclude 'data/raw' --exclude '__pycache__' --exclude '.gitignore' "$ROOT/graph_ml/" "$CTX/graph_ml/"

docker build --platform linux/amd64 -t "$USER_NAME/graphready-worker:$TAG" "$CTX"
docker push "$USER_NAME/graphready-worker:$TAG"
echo "Fertig: $USER_NAME/graphready-worker:$TAG"
