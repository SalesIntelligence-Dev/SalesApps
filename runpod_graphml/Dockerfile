# RunPod Worker – GraphReady GPU-Engines (GNN, LightGCN, TabPFN)
# Gleiche Basis wie der bestehende Aurora/CV-Worker (runpod/Dockerfile.runpod),
# aber eigenes Image und eigener Endpoint.
# Bauen über runpod_graphml/build.sh (kopiert graph_ml/ in den Build-Kontext).

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
# blinker liegt im Basis-Image als distutils-Paket vor und lässt sich nicht deinstallieren (Flask braucht ≥ 1.9)
# torch bleibt auf der Version des Basis-Images (CUDA 12.4) – sonst zieht pip ein neueres CUDA nach, das viele
# RunPod-Treiber nicht können (der Worker liefe dann still auf CPU)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --ignore-installed blinker && \
    pip freeze | grep -iE "^(torch|torchvision|torchaudio)==" > /tmp/constraints.txt && cat /tmp/constraints.txt && \
    pip install --no-cache-dir -r requirements.txt -c /tmp/constraints.txt

# TabPFN-v2-Gewichte schon beim Build laden (kein Download beim Kaltstart)
RUN python -c "from tabpfn import TabPFNClassifier; import numpy as np; \
    from tabpfn.constants import ModelVersion; \
    c = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device='cpu'); \
    c.fit(np.random.rand(20, 3), np.r_[np.zeros(10), np.ones(10)]); print('TabPFN v2 bereit')" || \
    echo "TabPFN-Gewichte nicht vorgeladen – Download beim ersten Job"

COPY graph_ml/ /app/graph_ml/
COPY handler.py /app/handler.py

ENV PYTHONUNBUFFERED=1 GRAPHML_WORKER=1

CMD ["python", "-u", "/app/handler.py"]
