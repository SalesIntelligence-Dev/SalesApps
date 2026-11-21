# To-dos: Live-GPU für /graph_ml auf Render (RunPod-Worker)

Stand: 21.11.2026 · GraphReady läuft bereits live auf https://salesapps.onrender.com/graph_ml.
Ohne diese Schritte funktioniert alles, die GPU-Engines (GNN, TabPFN, LightGCN) zeigen auf Render aber nur die
Vorberechnung, „Live neu berechnen“ antwortet mit 409.

## Schon erledigt

- [x] Worker-Image lokal gebaut: `ghcr.io/salesintelligence-dev/graphready-worker:1` (≈ 20 GB)
- [x] Lokal getestet: GPU-Selbsttest 7,6 TFLOP/s (RTX 3060), GNN-Churn-Job ok
- [x] Dockerfile korrigiert: torch bleibt auf 2.4.1 + CUDA 12.4 (wie der Aurora-Worker), `blinker` mit `--ignore-installed`
- [x] Commit `2d49652` (lokal, noch nicht gepusht – nur Worker-Dateien und Doku, ändert nichts an der Web-App)

## Offen

### 1. Registry-Zugang zum Hochladen (ein Weg genügt)

Der erste Upload nach `ghcr.io` scheiterte mit `denied`: Die gespeicherte Anmeldung hat kein Schreibrecht.

**Weg A – GitHub-Registry wie das Aurora-Image `aurora-runpod` (empfohlen)**

- [ ] GitHub → Settings → Developer settings → Personal access tokens (classic) → Generate new token,
      Häkchen bei `write:packages`, Ablauf z. B. 30 Tage
- [ ] Im Terminal anmelden (Passwort = Token):
  ```bash
  docker login ghcr.io -u DEIN_GITHUB_NAME
  ```
- [ ] Hochladen:
  ```bash
  docker push ghcr.io/salesintelligence-dev/graphready-worker:1
  ```

**Weg B – Docker Hub**

- [ ] `docker login` starten und **sofort** https://login.docker.com/activate öffnen, den neu angezeigten Code
      eingeben (läuft nach wenigen Minuten ab). Im Terminal muss danach „Login Succeeded“ stehen.
- [ ] Umbenennen und hochladen (ohne Neubau):
  ```bash
  docker tag ghcr.io/salesintelligence-dev/graphready-worker:1 DEINNAME/graphready-worker:1
  ```
  ```bash
  docker push DEINNAME/graphready-worker:1
  ```
- [ ] Auf hub.docker.com das Repository `graphready-worker` auf **Private** stellen (enthält euren Code)

Ist das lokale Image weg, neu bauen: `./runpod_graphml/build.sh ghcr.io/salesintelligence-dev 1`
(bzw. mit `DEINNAME` für Docker Hub). Der Build dauert ≈ 10 Minuten.

### 2. RunPod: neuen Endpoint anlegen (runpod.io → Serverless → New Endpoint)

**Den bestehenden Aurora-Endpoint `b1ygc0ydqzpogz` nicht ändern** – ihn nutzen `/forecasting` und
`/computer_vision_quality`.

- [ ] Registry-Zugang klären:
  - Weg A: dieselben Credentials wie beim Aurora-Endpoint (dort unter *Edit Endpoint → Container Registry Credentials*
    nachsehen, nichts speichern).
  - Weg B: unter *Settings → Container Registry Auth* einen Eintrag mit Docker-Hub-Name und Access Token anlegen.
- [ ] Endpoint anlegen mit:

  | Feld | Wert |
  |---|---|
  | Container Image | `ghcr.io/salesintelligence-dev/graphready-worker:1` (Weg B: `DEINNAME/graphready-worker:1`) |
  | Container Registry Credentials | siehe oben |
  | Endpoint Name | `graphready-worker` |
  | GPU | 16 oder 24 GB (z. B. RTX A4000, A5000, 4090) |
  | Active / Min Workers | 0 (keine Kosten im Leerlauf) |
  | Max Workers | 1 |
  | Idle Timeout | 5 s |
  | Execution Timeout | 600 s |
  | Container Disk | mindestens 40 GB |
  | Allowed CUDA Versions | 12.4 und neuer |
  | FlashBoot | an |
  | Umgebungsvariablen | keine |

- [ ] Deploy → **Endpoint-ID** kopieren (kurze Zeichenkette oben auf der Endpoint-Seite bzw. in der URL)

### 3. Render: Variable setzen (dashboard.render.com → Service `salesapps` → Environment)

- [ ] Add Environment Variable:

  | Key | Value |
  |---|---|
  | `RUNPOD_GRAPHML_ENDPOINT_ID` | die ID aus Schritt 2 |
  | `KUMO_API_KEY` | optional, nur wenn ein Kumo-Zugang besteht |

- [ ] **Save, rebuild, and deploy**
- `RUNPOD_API_KEY` und `RUNPOD_ENDPOINT_ID` (Aurora) sind schon gesetzt und bleiben unverändert.
- In `render.yaml` nichts eintragen – Schlüssel gehören nur ins Dashboard.

### 4. Prüfen (nach dem Neustart von Render)

- [ ] https://salesapps.onrender.com/graph_ml/api/system → `runpod.graphml.endpoint_set: true`, Health `ok`
- [ ] Seite öffnen → System-Panel → **Selbsttest** → GPU-Name und TFLOP/s erscheinen
      (erster Aufruf 1–3 Minuten Kaltstart, weil RunPod das 20-GB-Image lädt)
- [ ] Schritt 4 → **Live neu berechnen** läuft über RunPod (statt 409)
- [ ] Optional: in Schritt 5 eine nicht vorberechnete Playbook-Kombination anwenden → Job über RunPod
- [ ] `/forecasting` und `/computer_vision_quality` funktionieren weiter (Aurora-Endpoint unberührt)

### 5. Danach

- [ ] Commit `2d49652` pushen (`git push`) – löst ein Render-Deploy aus, ändert aber nur Worker-Dateien und Doku
- [ ] GitHub-Token nach Ablauf löschen bzw. nicht verlängern

## Bei Problemen

| Symptom | Ursache / Lösung |
|---|---|
| `denied` beim Push | Token ohne `write:packages` oder falsche Organisation |
| RunPod: „image pull failed“ | Registry-Credentials im Endpoint fehlen oder Repository privat ohne Zugang |
| Selbsttest meldet `gpu: CPU` | GPU-Typ mit zu altem Treiber gewählt → in RunPod CUDA ≥ 12.4 erzwingen |
| Seite zeigt „RunPod-GPU-Worker nicht erreichbar“ | Endpoint-ID falsch oder Endpoint gelöscht – Fallback auf Vorberechnung greift automatisch |
| Job bricht nach 10 Minuten ab | Execution Timeout im Endpoint erhöhen |
