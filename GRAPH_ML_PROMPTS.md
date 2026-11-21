# GraphReady – Aufbau der Seite `/graph_ml` in 5 Prompts

Stand: 21.11.2026 · Grundlage: `Graph_ML_Readiness_Plattform.pdf` · Ziel-URL: https://salesapps.onrender.com/graph_ml
Backup des Ausgangszustands: `../webapp_backup_21_11_26` (vollständige Kopie von `WebApp`, inkl. `.git`)

## So verwendest du dieses Dokument

1. Gib die Prompts **der Reihe nach** in Claude Code (in diesem Ordner `WebApp`) ein. Jeder Prompt baut auf dem vorherigen auf.
2. Du kannst den Prompt-Block komplett kopieren **oder** kurz schreiben: `Führe Prompt 2 aus GRAPH_ML_PROMPTS.md aus.`
3. Jeder Prompt liest zuerst **Abschnitt 0 (Gesamtkontext)**. Deshalb funktioniert jeder Prompt auch in einer neuen Session.
4. Nach jedem Prompt: Ergebnis unter http://localhost:5200/graph_ml anschauen, dann erst den nächsten Prompt starten.
5. Prompt 3 ist der größte. Wird er zu lang, teile ihn in **3a** (CPU-Engines: Regel, LightGBM, Graph-Features) und **3b** (GPU-Engines: TabPFN, GNN, KumoRFM, Aurora).

| Prompt | Inhalt | PDF-Module | Stufe |
|---|---|---|---|
| 1 | Fundament: Blueprint, Daten-Konnektor, Graph-Mapping, UI-Gerüst, System- und GPU-Status | 1 Konnektoren, 2 Graph-Mapping | – |
| 2 | Readiness-Checks und Use-Case-Bibliothek | 3 Readiness-Checks | 1 Strukturcheck, 2 Profilcheck |
| 3 | Rückblick und Baseline-Vergleich mehrerer Modellklassen, GPU über RunPod | 4 Baseline-Vergleich | 3 Rückblick |
| 4 | Fix-Playbooks, Ampel aus drei Dimensionen, Was-wäre-wenn mit Euro-Potenzial | 5 Fix-Playbooks | 2 + 3 |
| 5 | Aktivierung, Monitoring, Report, Feinschliff und Deployment | 6 Aktivierung | 3 |

---

## 0 Gesamtkontext (wird von jedem Prompt gelesen)

### 0.1 Ziel

Eine neue, eigenständige Seite `/graph_ml` in der bestehenden Flask-App „SalesApps“. Sie setzt die im PDF beschriebene **modellunabhängige Graph-ML-Readiness-Plattform** (Arbeitstitel *GraphReady*) als funktionierende Demo um:

- die **6 Module** der Produktarchitektur (PDF S. 3) als geführter Ablauf,
- die **3 Stufen** Strukturcheck, Profilcheck und Rückblick,
- eine **Use-Case-Readiness-Ampel** aus drei Dimensionen (Datenreife, Vorhersagekraft, Euro-Potenzial),
- eine **Was-wäre-wenn-Analyse** mit zusätzlichem Umsatz unter klar benannten Annahmen,
- einen Vergleich mehrerer Modellklassen (Regel, LightGBM, Tabular Foundation Model, Graph-ML/GNN, relationales Foundation Model, Aurora),
- echte Daten: **Online Retail II**.

### 0.2 Leitplanken: bestehende Inhalte bleiben unverändert

- Alles Neue liegt in `graph_ml/`, `templates/graph_ml.html`, `runpod_graphml/` und `tests/`.
- In bestehenden Dateien sind **nur** diese Änderungen erlaubt:
  - `app.py`: 2 Zeilen (Import und `register_blueprint`), direkt unter dem CV-Blueprint.
  - `requirements.txt`: Pakete nur ergänzen, nichts entfernen und keine Versionen ändern.
  - `.claude/launch.json` ist Entwicklungs-Konfiguration und bleibt unverändert.
- **Nicht anfassen:** `templates/home.html` (keine neue Kachel), `static/`, alle anderen Templates und Engines, `datasets/`, `runpod/` (der Aurora/CV-Worker bleibt unverändert), `render.yaml`, `Dockerfile`, `aurora_inference.py`, `cv_quality/`.
- Das Backup `../webapp_backup_21_11_26` wird nie verändert.
- Keine Secrets in Code oder Git. Schlüssel kommen nur über Umgebungsvariablen (Render-Dashboard).
- Kein `git push` ohne Rückfrage, denn ein Push löst das Render-Deployment aus.
- Nach jedem Prompt: Smoke-Test aller bestehenden Routen (Status 200) und `git diff --stat`, das nur erlaubte Dateien zeigen darf.
- UI-Sprache Deutsch, deutsche Zahlenformate (1.067.371 · 300.000 €).

### 0.3 Datensatz-Entscheidung

**Hauptdatensatz: Online Retail II** (UCI Machine Learning Repository, CC BY 4.0, kommerziell nutzbar mit Quellenangabe im Footer).
Download: https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip (Erreichbarkeit am 21.11.2026 geprüft)

Warum dieser Datensatz:
- **Echte Datenlücken** machen den Readiness-Check sichtbar: 243.007 von 1.067.371 Positionen (22,8 %) haben keine Kundennummer. Dazu kommen Stornos (Rechnungen mit „C“-Präfix) ohne Bezug zur Originalrechnung, Preis 0, negative Mengen, Nicht-Artikel-Codes (POST, M, D, BANK CHARGES, AMAZONFEE …), Artikelvarianten und Dubletten. Artikel haben nur eine Freitext-Beschreibung und keine Artikelgruppe.
- **Zwei Jahre Historie** (12/2009–12/2011) ermöglichen zeitliche Splits und 90-Tage-Labels für den Rückblick.
- **Deckt Sales und After Sales ab:** Kunden, Rechnungen, Positionen, Artikel und Länder bilden einen relationalen Graphen. Über Stornos und Retouren lässt sich ein After-Sales-Use-Case bauen.
- **Die Größe zeigt Engine-Grenzen:** rund 1 Mio. Zeilen gegenüber TabPFN (Kontextgröße) und Kumo (30 MB je Anfrage).
- Spalten: `Invoice, StockCode, Description, Quantity, InvoiceDate, Price (GBP), Customer ID, Country`.

**Ergänzung: AdventureWorks, nur als Schema** (Tabellen, Spalten, PK/FK aus dem öffentlichen Microsoft-DDL, keine Daten). Damit zeigt Stufe 1, wie das ERP-typische Graph-Mapping über viele Tabellen funktioniert.

Nicht gewählt: SAP SALT (nicht-kommerziell), Olist (Lizenz offen), WideWorldImporters (redundant zu AdventureWorks). RelBench ist optional später für die Kalibrierung.

### 0.4 Use-Case-Bibliothek: Abbildung der bisherigen Apps auf Online Retail II

★ = Leuchtturm-Use-Case (wie im PDF: Retention und Root Cause). Alle Labels folgen einem **90-Tage-Fenster nach dem Stichtag**, sofern nicht anders angegeben.

| # | Use Case | Bisherige App | Bereich | Label (Vorhersageziel) | Regel-Baseline („was der Vertrieb heute tut“) | Max. Stufe mit Online Retail II | Hebel |
|---|---|---|---|---|---|---|---|
| 1 | ★ Churn & Retention | `/churn` | Sales | Aktiver Kunde (Kauf in den letzten 365 Tagen) kauft in den nächsten 90 Tagen nicht | „seit 60 Tagen keine Bestellung“, Ranking nach Recency | 3 | Umsatz halten |
| 2 | Graph Cross-Selling | `/cross-selling` | Sales | Kauf aus einer bisher nicht gekauften Artikelgruppe | meistverkaufte noch nicht gekaufte Gruppe | 3 (nach Playbook „Artikelgruppen aus Freitext“) | Umsatz |
| 3 | Produkt-Empfehlung | `/recommender` | Sales | Rangliste der gekauften Artikel (MAP@12, Recall@12) | Topseller der letzten 30 Tage plus Wiederkauf | 3 | Warenkorb |
| 4 | Neukunden- / Lead-Scoring | `/lead-scoring` | Sales | Erstkäufer kauft innerhalb von 90 Tagen erneut | Wert des ersten Warenkorbs | 3 (Cold-Start-Check) | Umsatz, Vertriebsfokus |
| 5 | Next Best Action | `/next-action` | Sales | keine Reaktionshistorie vorhanden; Regelwerk über die Scores 1–4 | – | 2 (Ampel höchstens Gelb, Blocker: keine Kampagnen- und Reaktionsdaten) | Umsatz |
| 6 | Absatzprognose | `/forecasting`, `/forecasting_model` | Sales / SCM | Wochenmenge je Artikelgruppe und Land | saisonal-naiv (Vorjahreswoche) | 3 | Bestand, Kosten |
| 7 | Margenoptimierung | `/margenoptimierung` | Sales | nicht berechenbar, weil der Einstandspreis fehlt; Profil: Preisstreuung je Artikel und Kunde | – | 2 | Marge |
| 8 | ★ Retouren- & Storno-Root-Cause | neu (After Sales) | After Sales | Position wird innerhalb von 30 Tagen storniert oder retourniert; Kohorten mit erhöhtem Risiko | Retourenquote je Artikel aus der Vorperiode | 3 | Kosten, Umsatzerhalt |
| 9 | Warranty Root Cause | neu (After Sales, PDF-Vorlage) | After Sales | Garantiefall in X Monaten je Maschine | – | 1 (Rot: Seriennummer, Charge und Garantiefälle fehlen). AdventureWorks-Schema zeigt Teilabdeckung | Kosten |

Die Use Cases 5, 7 und 9 sind **bewusst ehrlich Gelb oder Rot**. Sie zeigen, dass die Plattform auch sagt, wann ein Use Case *nicht* trägt und welche Daten fehlen.

### 0.5 Modell-Engines (modellunabhängiger Vergleich)

| Engine | Modellklasse | Läuft auf | Use Cases | Hinweis |
|---|---|---|---|---|
| Regel-Baseline | Heuristik | Render-CPU | alle | Pflichtvergleich für „Vorhersagekraft“ |
| LightGBM | Baum-Modell | Render-CPU | 1, 2, 4, 6, 8 | Pflicht-Baseline laut PDF, bereits in `requirements.txt` |
| LightGBM + Graph-Features | Graph-ML light | Render-CPU | 1–4, 8 | Personalized PageRank, Co-Purchase-Embeddings (TruncatedSVD), Nachbarschafts-Churnquote. Graph-Signal ohne GPU |
| TabPFN v2 | Tabular Foundation Model | RunPod-GPU (CPU-Fallback mit ≤ 1.000 Trainingszeilen) | 1, 4, 8 | Lizenz und Gewichte-Zugang vor kommerzieller Nutzung prüfen. Offene Alternative: TabICL |
| RDL-GNN (heterogener GraphSAGE, PyTorch Geometric) | Relational Deep Learning | RunPod-GPU, lokal RTX 3060 für die Vorberechnung | 1, 2, 4, 8 | RelBench-Stil mit zeitlichem Neighbor-Sampling |
| LightGCN | Graph-Empfehlung | RunPod-GPU, CPU für kleine Graphen | 3 | Graph aus Kunden und Artikeln |
| KumoRFM | Relationales Foundation Model (In-Context, Few-Shot, ohne Training) | Kumo-API (`KUMO_API_KEY`), oder offene Gewichte auf RunPod, falls verfügbar | 1–4 | Ohne Key ausgegraut mit Erklärung. Nutzungsbedingungen vorher klären (im PDF eine offene Frage) |
| Aurora | Zeitreihen-Foundation-Model | RunPod-GPU (bestehender Endpoint, `job_type=aurora_forecast`) | 6 | `aurora_inference.py` nur importieren, nicht ändern |
| SAP RPT-1 OSS (optional) | Tabular Foundation Model | RunPod-GPU | 1, 4, 8 | Nur, wenn die Gewichte frei verfügbar sind, sonst weglassen |

Modellwahl nach „Beyond IID“: Bäume liegen bei zeitlichen, gruppierten und großen Daten vorn, TFMs bei kleinen bis mittleren, gut gemischten Daten, relationale Modelle dann, wenn ein Signal über mehrere Schritte im Graphen messbar hilft. **Eine Baum-Baseline läuft immer mit.**

### 0.6 Die 6 Module und die 3 Stufen

| Modul (PDF S. 3) | Umsetzung auf `/graph_ml` | Prompt |
|---|---|---|
| 1 Konnektoren | Online Retail II, AdventureWorks-Schema, eigener Upload, Profil-JSON | 1 (Profil in 2) |
| 2 Graph-Mapping | Schlüssel, Zeitspalten, Verknüpfungsquoten, Schema-Graph, Beispiel-Subgraph | 1 |
| 3 Readiness-Checks | 9 Prüfdimensionen für 9 Use Cases, Stufen 1 und 2 | 2 |
| 4 Baseline-Vergleich | Regel vs. Bäume vs. TFM vs. GNN vs. RFM mit zeitlichem Split | 3 |
| 5 Fix-Playbooks | Bereinigen, Lücken schließen, neu prüfen; Ampel und Was-wäre-wenn | 4 |
| 6 Aktivierung | Listen fürs CRM, Kontrollgruppe, Monitoring, Report | 5 |

| Stufe | Eingabe | Aussage | Risiko |
|---|---|---|---|
| 1 Strukturcheck | nur Tabellen- und Feldliste bzw. Graph-Mapping | „Use Case ist grundsätzlich abbildbar“ | Falsches Grün: Felder existieren, sind aber leer |
| 2 Profilcheck | nur Kennzahlen (Zeilen, Füllgrad, Verknüpfungsquote, Zeitraum, Ereignisse), keine Rohdaten | „Menge, Historie und Verknüpfung reichen“ | keine Aussage zur Vorhersagekraft |
| 3 Rückblick | historische Rohdaten | „So gut wäre die Vorhersage gewesen, so viel Potenzial unter Annahmen“ | Annahme zur Wirkung |

Trichter im Geschäftsmodell: Light-Scan (Stufe 1, kostenlos bis 2.500 €) → Readiness-Sprint (Stufe 2/3, 8.000–20.000 €) → Pilot (25.000–60.000 €) → Lizenz (1.500–4.000 € je Use Case und Monat). Alle Preise sind Annahmen aus dem PDF.

### 0.7 Ampel-Logik und Was-wäre-wenn

**Dimension 1, Datenreife** (Score 0–100 aus den 9 Prüfungen; die ersten fünf Kern-Prüfungen zählen zusammen 70 %):
Grün ab 75 Punkten ohne rote Kern-Prüfung · Gelb bei 50–74 Punkten oder einer roten Nicht-Kern-Prüfung · Rot unter 50 Punkten oder bei roter Prüfung „Struktur“ oder „Label“.

**Dimension 2, Vorhersagekraft** (nur Stufe 3): Lift der besten Engine gegenüber der Regel-Baseline auf der Hauptmetrik (z. B. Recall@Top-10 %), mit Bootstrap-Konfidenzintervall.
Grün ab +20 % relativ und untere KI-Grenze über 0 · Gelb bei +5 bis +20 % · Rot unter +5 % oder nicht signifikant · Grau, wenn nicht messbar (Next Best Action).

**Dimension 3, Euro-Potenzial** (12 Monate, realistisches Szenario, netto nach Kosten):
Grün ab 100.000 € · Gelb ab 25.000 € · Rot darunter. Die Schwellen sind in der UI einstellbar.

**Gesamtampel** = die schwächste belastbare Dimension, mit Begründung. Stufe 1 ist „vorläufig“ (schraffiert), in Stufe 2 ist die Datenreife belastbar, in Stufe 3 alle drei Dimensionen.

**Was-wäre-wenn (Churn):**
`Potenzial = erkannte Abwanderer in Top-X % × Rettungsquote × Ø Jahresumsatz × (optional Marge) − Listengröße × Kosten je Kontakt`
`Mehrwert Graph ML = Potenzial(beste Engine) − Potenzial(Regel-Baseline)`
Kontrollrechnung aus dem PDF: 2.000 Kunden, 150 Abwanderer, 20.000 € Umsatz je Kunde, 60 davon erkannt, 25 % gerettet → **300.000 €**.

Pflicht-Hinweis in der UI: *„Gemessen: wie gut die Vorhersage gewesen wäre. Nicht gemessen: ob der Vertrieb reagiert hätte. Den echten Effekt belegt erst der Pilot mit Kontrollgruppe.“* Der Rückblick ist streng nach Datum getrennt (kein Leakage).
Währung: Online Retail II ist in GBP. Umrechnung in EUR mit einem konfigurierbaren Kurs (Standard 1,15), sichtbar ausgewiesen.

### 0.8 Technik und Infrastruktur (System- und GPU-Zugriff)

- **Render:** Plan *starter*, 1 Worker, 4 Threads, 120 s Timeout, CPU-only torch 2.1, etwa 512 MB RAM.
  Daraus folgt: keine Langläufer im Request, alles Rechenintensive als Hintergrund-Job mit Polling. Daten im kompakten NumPy-Format (`.npz`, Kategorien als Codes), lazy geladen; Schritt 1 und 2 kommen aus Artefakten. Ziel: unter 400 MB RSS für die ganze App (Stand nach Prompt 1: 195 MB Basis, Spitze 329 MB).
- **GPU wie bei den bestehenden Seiten:** `/forecasting` (Aurora) und `/computer_vision_quality` nutzen RunPod Serverless über `RUNPOD_API_KEY` und `RUNPOD_ENDPOINT_ID`.
  `/forecasting_model` selbst rechnet LightGBM auf der CPU und holt Wetterdaten von Open-Meteo. Die GPU-Anbindung stammt also aus den beiden anderen Seiten.
- **Für Graph ML gibt es einen eigenen RunPod-Worker** in `runpod_graphml/` (Dockerfile, Handler, Requirements). Er läuft als eigener Endpoint (`RUNPOD_GRAPHML_ENDPOINT_ID`, gleicher `RUNPOD_API_KEY`), damit der bestehende Aurora/CV-Worker unberührt bleibt. Ist die Variable nicht gesetzt, laufen nur die CPU-Engines und die GPU-Engines werden ausgegraut.
  Payload höchstens 8 MB (RunPod-Limit für `/run` ist 10 MB). Alternativ lädt der Worker Online Retail II selbst von UCI und baut den Graphen deterministisch nach.
- **Aurora** für die Absatzprognose nutzt den bestehenden Endpoint über `aurora_inference.py`, ohne Änderung.
- **Lokal:** RTX 3060 (12 GB) und GTX 1650 sind vorhanden. Das System-Python hat kein torch. Für die Vorberechnung wird eine venv in `.venv/` angelegt (steht schon in `.gitignore`), mit torch+CUDA und torch_geometric.
- **Vorberechnete Ergebnisse** liegen in `graph_ml/artifacts/*.json` (klein, werden committet). Render zeigt sie sofort an, „Live neu berechnen“ startet echte Jobs.
- **Dev-Server:** `.claude/launch.json` → Konfiguration „WebApp“, Port 5200.

### 0.9 UI-Leitlinien

- **Optisch hochwertiges dunkles Dashboard**, passend zu den bestehenden SalesApps-Seiten (Schrift Inter, CSS-Variablen, Logo verlinkt auf `/`).
- **Stack** wie in der CV-App: Tailwind (CDN), Alpine.js, Chart.js 4.4, dazu Cytoscape.js (jsdelivr) für die Graphen.
- **Layout:**
  - Header mit Titel „GraphReady · Graph-ML-Readiness“ und Status-Chips (CPU · RunPod-GPU · Kumo · TabPFN).
  - Stufen-Umschalter als Trichter (1 Strukturcheck · 2 Profilcheck · 3 Rückblick).
  - Horizontaler Stepper mit 6 Schritten (Konnektoren → Aktivierung), Status je Schritt.
  - Darunter das Panel des aktiven Schritts.
  - Readiness-Ampel-Matrix als dauerhaft erreichbare Übersicht.
- **Ampeln barrierearm:** Farbe plus Icon plus Text. Jede Bewertung hat ein „Warum?“ (Tooltip oder Drawer). Annahmen sind immer sichtbar.
- Lade-Skeletons, Job-Fortschritt mit Log-Zeilen (wie in der CV-App), leere Zustände und Fehlerzustände mit klarer Handlungsanweisung.
- **Responsiv** von 375 bis 1920 px, keine horizontale Scrollbar der Seite (Tabellen scrollen intern).
- Footer mit Quellenangabe Online Retail II (UCI, CC BY 4.0) und dem PDF als Grundlage.

### 0.10 Dateistruktur (Ziel)

```
graph_ml/
  __init__.py           # graph_bp
  routes.py             # /graph_ml und /graph_ml/api/*
  prepare_data.py       # Download + Aufbereitung Online Retail II (python -m graph_ml.prepare_data)
  build_adventureworks_schema.py  # AdventureWorks-DDL → Schema-JSON
  data_loader.py        # Konnektoren: Online Retail II, Upload, Schema-Import, Profil-JSON
  graph_mapping.py      # Schlüssel-, Zeit- und FK-Erkennung, Schema-Graph
  usecases.py           # Use-Case-Bibliothek (Vorlagen)
  checks.py             # 9 Prüfdimensionen, Stufe 1/2
  profiler_cli.py       # Stufe-2-Profil lokal beim Kunden erzeugen (nur Kennzahlen)
  backtest.py           # zeitliche Splits, Labels, Metriken
  engines/              # rule.py, lgbm.py, graph_features.py, tabpfn_engine.py, gnn.py, kumo.py, aurora_adapter.py
  playbooks.py          # Fix-Playbooks
  business_case.py      # Ampel aus 3 Dimensionen, Was-wäre-wenn
  activation.py         # Listen, Kontrollgruppe, Export, Monitoring
  jobs.py               # Hintergrund-Jobs, RunPod-Client, Device-Erkennung
  precompute.py         # Vorberechnung der Artefakte
  data/                 # online_retail_ii.npz (+ .meta.json), adventureworks_schema.json; raw/ wird nicht committet
  artifacts/            # vorberechnete Ergebnisse (JSON)
templates/graph_ml.html
runpod_graphml/         # Dockerfile, handler.py, requirements.txt (eigener GPU-Endpoint)
tests/test_graph_ml_*.py
```

---

## Prompt 1 – Fundament: Blueprint, Daten-Konnektor, Graph-Mapping, UI-Gerüst, System- und GPU-Status

```text
Lies zuerst GRAPH_ML_PROMPTS.md (Abschnitt 0 komplett) und Graph_ML_Readiness_Plattform.pdf (vor allem „Produktarchitektur“, S. 3). Halte dich strikt an die Leitplanken in 0.2.

Aufgabe: Lege die neue, isolierte Seite /graph_ml an (Flask-Blueprint nach dem Muster von cv_quality). Baue Modul 1 „Konnektoren“ und Modul 2 „Graph-Mapping“ vollständig sowie das UI-Gerüst für alle 6 Module.

1. Blueprint und Routen
   - Paket graph_ml/ laut 0.10; graph_bp mit /graph_ml und /graph_ml/ sowie API unter /graph_ml/api/…
   - In app.py nur 2 Zeilen ergänzen (Import + register_blueprint), direkt unter dem CV-Blueprint. Keine Kachel auf home.html.

2. Daten-Konnektor (Modul 1)
   - graph_ml/prepare_data.py lädt Online Retail II (URL in 0.3) und liest beide Excel-Blätter.
     Spalten vereinheitlichen: invoice, stock_code, description, quantity, invoice_date, price_gbp, customer_id, country.
     Speichern als graph_ml/data/online_retail_ii.parquet (Kategorien, float32/Int32, zstd).
     [Umgesetzt als online_retail_ii.npz – siehe „Umsetzungsnotizen nach Prompt 1“ am Ende.]
     Die Rohdaten bleiben vollständig, also alle 1.067.371 Zeilen inklusive Lücken, Stornos und Dubletten. Die Bereinigung übernehmen später die Fix-Playbooks.
     Prüfe die Dateigröße. Ist sie über 30 MB, schlag mir eine Alternative vor, bevor du sie committest.
   - Daraus ein relationales Schema ableiten (customers, invoices, invoice_lines, products, countries) mit PK/FK. Das ist der „ERP-Export“, auf dem der Graph entsteht.
   - graph_ml/data/adventureworks_schema.json: Tabellen, Spalten, Typen und PK/FK der Schemata Sales, Production, Person und Purchasing aus dem öffentlichen Microsoft-DDL. Nur das Schema, keine Daten.
   - Upload-Konnektor: CSV/XLSX/Parquet (eine oder mehrere Tabellen) sowie Schema-Import (JSON oder CSV „tabelle,spalte,typ“) für Stufe 1. Limit 64 MB wie in der App.
   - Speicher schonen: lazy laden, Singleton-Cache mit Lock.

3. Graph-Mapping (Modul 2): graph_ml/graph_mapping.py
   - Automatisch erkennen: Primärschlüssel-Kandidaten (eindeutig, nicht leer), Fremdschlüssel per Werte-Inklusion zwischen Tabellen, Zeitspalten, Kategorien mit sehr vielen Ausprägungen, Freitextspalten.
   - Kennzahlen je Beziehung: Verknüpfungsquote, verwaiste Datensätze, Dubletten.
     Beispiele: Positionen ohne customer_id (22,8 %), Stornos ohne Bezug zur Originalrechnung.
   - Ausgabe als JSON: Knotentypen, Kantentypen, Kennzahlen, Zeitspalten je Tabelle.
   - Visualisierung mit Cytoscape.js:
     (a) Schema-Graph: Tabellen als Knoten, FKs als Kanten, Kantenfarbe nach Verknüpfungsquote.
     (b) Beispiel-Subgraph für einen auswählbaren Kunden: Kunde → Rechnungen → Artikel → andere Kunden, höchstens etwa 150 Knoten.

4. System- und GPU-Zugriff (wie bei /forecasting und /computer_vision_quality, siehe 0.8)
   - graph_ml/jobs.py
     - Hintergrund-Jobs: Thread plus In-Memory-Store mit Status, Fortschritt und Log, dazu eine Polling-API.
     - RunPod-Client: /run senden, /status abfragen, Timeout über RUNPOD_TIMEOUT. Er nutzt RUNPOD_API_KEY und RUNPOD_GRAPHML_ENDPOINT_ID.
     - Device-Erkennung in dieser Reihenfolge: lokale CUDA-GPU (falls torch vorhanden), dann RunPod, dann CPU.
   - GET /graph_ml/api/system liefert:
     - Python- und Paketversionen (lightgbm, torch, torch_geometric, tabpfn)
     - lokale GPU (Name, Speicher)
     - RunPod Graph-ML-Endpoint: konfiguriert und Health (GET /v2/{id}/health, mit Timeout, 60 s Cache)
     - Aurora-Endpoint konfiguriert (RUNPOD_ENDPOINT_ID)
     - KUMO_API_KEY gesetzt (nur ja/nein, nie den Wert)
     - RAM des Prozesses
   - Im Header als Status-Chips anzeigen; ein Klick öffnet ein Detail-Panel.

5. UI-Gerüst templates/graph_ml.html (gemäß 0.9, optisch hochwertig)
   - Header, Stufen-Trichter, Stepper mit 6 Schritten, je Schritt ein Panel.
   - Schritte 3–6 vorerst mit ansprechendem Platzhalter („folgt“). Die Ampel-Matrix als Platzhalter mit den 9 Use Cases aus 0.4.
   - Schritt 1: Karten für die Datenquellen.
     - Online Retail II ist vorausgewählt, mit Kennzahlen: Zeilen, Zeitraum, Kunden, Artikel, Länder, Stornos, Anteil ohne Kundennummer.
     - Dazu AdventureWorks-Schema und eigener Upload, plus eine Vorschautabelle der Daten.
   - Schritt 2: Schema-Graph, Kennzahlen-Tabelle und Beispiel-Subgraph, dazu die Box „Was bedeutet das für Graph ML?“.

6. Prüfen und abschließen
   - preview_start „WebApp“ und /graph_ml im Browser testen (Desktop und 375 px). Die Konsole muss fehlerfrei sein. Zeig mir Screenshots.
   - Smoke-Test: Diese bestehenden Routen liefern weiterhin 200:
     /, /forecasting, /forecasting_model, /computer_vision_quality, /margenoptimierung, /cross-selling, /konfiguration, /retrieval, /lead-scoring, /recommender, /opportunity, /next-action, /churn, /api/health
   - git diff --stat darf nur erlaubte Dateien zeigen. Lokal committen, nicht pushen.
```

---

## Prompt 2 – Readiness-Checks und Use-Case-Bibliothek (Stufe 1 und 2)

```text
Lies GRAPH_ML_PROMPTS.md Abschnitt 0 und den Code in graph_ml/ aus Prompt 1. Die Leitplanken in 0.2 gelten.

Aufgabe: Modul 3 „Readiness-Checks“ mit der Use-Case-Bibliothek und den Stufen 1 (Strukturcheck) und 2 (Profilcheck).

1. Use-Case-Bibliothek graph_ml/usecases.py
   Die 9 Use Cases aus 0.4 als Vorlagen (Dataclass oder JSON). Jede Vorlage enthält:
   - benötigte Tabellen und Felder (Pflicht oder optional), mit Synonymen für ERP-Feldnamen, z. B. customer_id ~ KUNNR ~ CustomerID
   - Label-Definition (Fenster, Mindestfallzahl) und Regel-Baseline
   - typische Blocker und Hebel
   - Link zur bisherigen App-Route

2. Die 9 Prüfdimensionen in graph_ml/checks.py (laut PDF S. 4–5), je Use Case
   Struktur & Schlüssel · Zeit & Leakage · Label & Historie · Datenlage & Modellwahl · Abdeckung ähnlicher Fälle (Cold Start) · Spaltenqualität · Freitext · Größe & Limits · Governance & Rückspielweg

   Jede Prüfung liefert:
   - Status (grün, gelb, rot oder „nicht prüfbar“) und einen Score von 0–100
   - Messwerte und eine Begründung in einem Satz
   - konkrete Maßnahmen mit Verweis auf eine Fix-Playbook-ID (PB-01 … PB-10 aus Prompt 4)
   Gewichtung wie in 0.7.

   Konkrete Messungen für Online Retail II, unter anderem:
   - Verknüpfungsquote Position → Kunde; Stornos ohne Bezug zur Originalrechnung
   - Zeitstempel vorhanden; Positivfälle je 90-Tage-Fenster und Use Case; Historie in Monaten
   - Cold Start: Anteil Kunden mit weniger als 3 Rechnungen; neue Artikel im Testzeitraum
   - Spalten-Ampel: Leerwerte, Preis 0, negative Mengen, stock_code-Varianten, uneinheitliche Beschreibungen, Nicht-Artikel-Codes
   - Freitext: Die Beschreibung ist die einzige Artikelinfo, es gibt keine Artikelgruppen
   - Größe gegenüber den Engine-Limits: TabPFN-Kontextgröße, Kumo 30 MB je Anfrage
   - Governance: customer_id ist pseudonym, aber personenbezogen; Rückspielweg nicht vorhanden → Checkliste
   - Leakage-Kandidaten markieren, also Felder, die erst nach dem Vorhersagezeitpunkt entstehen (z. B. Stornos zur selben Rechnung)

   „Datenlage & Modellwahl“ nach „Beyond IID“:
   - Klassifiziere Größe, Breite, zeitliche Verschiebung (PSI zwischen Zeiträumen), Gruppierung, Kardinalität und Text.
   - Daraus folgt eine empfohlene Engine-Klasse (Baum, TFM oder relational) mit Begründung. Die Baum-Baseline ist immer dabei.

3. Stufen
   - Stufe 1, Strukturcheck: arbeitet NUR auf der Tabellen- und Feldliste (aus dem Graph-Mapping oder einem Schema-Import, z. B. AdventureWorks).
     - Aussage: „grundsätzlich abbildbar“, plus fehlende Verknüpfungen (z. B. Warranty: Seriennummer → Charge fehlt).
     - Warnhinweis: „Falsches Grün möglich: Felder können leer sein.“
   - Stufe 2, Profilcheck: arbeitet NUR auf Kennzahlen, ohne Rohdaten.
     - graph_ml/profiler_cli.py führt ein Kunde lokal aus (python -m graph_ml.profiler_cli <export-ordner> --out profil.json). Es exportiert nur aggregierte Kennzahlen.
     - Upload dieser JSON in der UI. Für Online Retail II wird das Profil intern erzeugt.
     - Gut sichtbarer Hinweis: „Es verlassen keine personenbezogenen Daten das Unternehmen.“
   - Stufe 3 bleibt als „verfügbar nach dem Rückblick“ markiert. Die Ampel-Datenstruktur enthält die Dimensionen Vorhersagekraft und Euro-Potenzial aber schon.

4. API: POST /graph_ml/api/checks {dataset, stage, use_cases[]} liefert das Ergebnis-JSON, gecacht je Datensatz-Version.

5. UI Schritt 3
   - Ampel-Matrix: Zeilen = Use Cases, Spalten = 9 Prüfdimensionen plus Datenreife gesamt. Filter Sales / After Sales, Leuchttürme markiert.
   - Detail-Drawer je Use Case:
     - Label-Definition
     - benötigte gegenüber vorhandenen Tabellen als Mini-Schema-Graph, fehlende Knoten rot gestrichelt
     - Prüfergebnisse mit Begründung und Maßnahmen
     - empfohlene Engine-Klasse
   - Spalten-Ampel je Tabelle.
   - Der Stufen-Umschalter wirkt live: Stufe 1 zeigt nur die Struktur, schraffiert als „vorläufig“; Stufe 2 zeigt die volle Datenreife.
   - Ampel immer mit Icon und Text, nicht nur mit Farbe.

6. Tests in tests/test_graph_ml_checks.py
   - Verknüpfungsquote customer_id ≈ 77,2 %
   - Stufe 1 liest keine einzige Datenzeile
   - Stufe 2 verwendet nur die Profil-JSON
   Danach Browser-Prüfung mit Screenshot, Smoke-Test der bestehenden Routen, git diff --stat, lokal committen.
```

---

## Prompt 3 – Rückblick und Baseline-Vergleich mehrerer Modellklassen, GPU über RunPod (Stufe 3)

```text
Lies GRAPH_ML_PROMPTS.md Abschnitt 0 (vor allem 0.5, 0.7 und 0.8) und den Code in graph_ml/. Die Leitplanken in 0.2 gelten.

Aufgabe: Stufe 3 „Rückblick“. Ein streng zeitlich getrennter Backtest je Use Case und ein modellunabhängiger Vergleich mehrerer Engines (Modul 4 „Baseline-Vergleich“), inklusive GPU über RunPod.

1. Backtest-Rahmen graph_ml/backtest.py (RelBench-Stil)
   - Trainings-Snapshots monatlich ab 2010-03-01, Label-Fenster 90 Tage.
   - Test-Snapshot T_test = 2011-09-01; sein Label-Fenster endet am 2011-11-30.
   - Ins Training kommen nur Snapshots, deren Label-Fenster vor T_test endet. Validierung ist der letzte Trainings-Snapshot.
   - Features je Snapshot ausschließlich aus Ereignissen vor dem Snapshot-Zeitpunkt (RFM, Trends, Artikelgruppen-Mix, Land, Stornoquote, Saisonalität …).
   - Leakage-Guard als harte Assertion plus Test.
   - Labels laut 0.4 für Churn, Cross-Selling, Empfehlung, Neukunden-Scoring, Retouren-Root-Cause und Absatzprognose.
   - Next Best Action hat keine Label-Historie: nur ein Regelwerk, Vorhersagekraft „nicht messbar“.
   - Metriken:
     - Klassifikation: AUROC, AUPRC, Precision/Recall@Top-10 %, Anteil erfasster Umsatz, Kalibrierung (Brier, Reliability-Kurve)
     - Empfehlung: MAP@12, Recall@12
     - Prognose: WAPE, MASE
     - Bootstrap-Konfidenzintervalle
   - Bis Prompt 4 die Artikelgruppen aus dem Freitext liefert, nutzt Cross-Selling eine vorläufige Gruppierung per Keyword-Regeln (klar gekennzeichnet).

2. Engines in graph_ml/engines/ laut 0.5
   Einheitliches Interface: fit, predict, explain, meta (Name, Klasse, Device, Laufzeit, Lizenzhinweis).
   - Regel-Baseline je Use Case (z. B. „seit 60 Tagen keine Bestellung“)
   - LightGBM auf der CPU, mit Feature-Importance und SHAP über pred_contrib
   - LightGBM + Graph-Features: Personalized PageRank, Grad, Co-Purchase-Embeddings (TruncatedSVD der Kunde×Artikel-Matrix), Nachbarschafts-Churnquote
   - TabPFN v2 (Paket tabpfn) auf der RunPod-GPU
     - CPU-Fallback mit stratifiziertem Subsample von höchstens 1.000 Zeilen, klar gekennzeichnet
     - Sind Lizenz oder Gewichte nicht verfügbar: TabICL als offene Alternative, sonst ausgegraut
   - RDL-GNN: heterogener GraphSAGE (PyTorch Geometric) über customers–invoices–invoice_lines–products–countries
     - zeitliches Neighbor-Sampling, nur Kanten vor dem Snapshot
     - für Churn, Cross-Selling, Neukunden und Retouren; LightGCN für die Empfehlung
   - KumoRFM-Adapter (kumoai-SDK oder REST, Variable KUMO_API_KEY): In-Context-Vorhersage per Predictive Query.
     Beispiel: PREDICT COUNT(invoices.*, 0, 90, days) = 0 FOR EACH customers.customer_id
     Ohne Key ausgegraut mit Hinweis, kein Fehler.
   - Aurora für die Absatzprognose über den bestehenden RunPod-Job aurora_forecast. aurora_inference.py nur importieren, nicht ändern.

3. GPU über einen eigenen RunPod-Worker (siehe 0.8)
   - Neuer Ordner runpod_graphml/ mit Dockerfile (Basis wie runpod/Dockerfile.runpod), handler.py und requirements.txt.
     - job_types: graphml_selftest (Benchmark, wird vom Selbsttest aus Prompt 1 schon aufgerufen; Rückgabe {"gpu": Name, "gflops": Zahl}), graphml_gnn_train, graphml_lightgcn, graphml_tabpfn
     - TabPFN-Gewichte schon beim Image-Build laden
   - Der Ordner runpod/ bleibt unverändert.
   - Payload höchstens 8 MB (komprimiertes npz/parquet, base64). Alternativ lädt der Worker Online Retail II selbst und baut den Graphen deterministisch nach.
   - Im UI je Engine anzeigen, wo sie lief (Render-CPU, lokale GPU oder RunPod-GPU mit GPU-Name) und wie lange sie gebraucht hat.
   - Gib mir am Ende die exakten Befehle zum Bauen und Pushen des Images und zum Anlegen des Endpoints. Ich führe sie selbst aus und setze RUNPOD_GRAPHML_ENDPOINT_ID in Render.

4. Vorberechnung graph_ml/precompute.py
   - Berechnet alle Backtests für Online Retail II.
   - Speichert kompakte Ergebnisse in graph_ml/artifacts/*.json: Metriken, Kurven, Top-Listen, Feature-Wichtigkeit, Laufzeit, Device.
   - Render zeigt diese sofort an. Der Button „Live neu berechnen“ startet Jobs mit Polling, Fortschritt und Log.
   - Wenn möglich jetzt lokal auf der RTX 3060 ausführen (venv .venv mit torch-CUDA und torch_geometric).
     Klappt das nicht, die CPU-Engines rechnen und die GPU-Engines als „ausstehend“ markieren.

5. Empfehlung zur Modellwahl
   - Kombiniere die Datenlage-Klassifikation aus Prompt 2 mit dem Backtest-Ergebnis.
   - Ergebnis: „Empfohlene Engine je Use Case“ mit Begründung nach „Beyond IID“.

6. UI Schritt 4
   - Use-Case-Auswahl; die Zeitachse des Splits (Trainings-Snapshots, Lücke, Test-Fenster) als Grafik.
   - Leaderboard: Engine, Klasse, Device, Metriken mit KI, Laufzeit, Lift gegenüber der Regel.
   - Charts: ROC, Precision@k, Kalibrierungskurve, Top-Features, Beispiel-Erklärung für einen einzelnen Kunden.
   - Box mit dem Hinweis: „So gut wäre die Vorhersage gewesen. Die Umsatzwirkung ist eine Annahme (siehe Was-wäre-wenn).“
   - Die Ampel-Dimension „Vorhersagekraft“ wird aus dem Lift gegenüber der Regel-Baseline befüllt (Schwellen aus 0.7).

7. Tests und Abschluss
   - Leakage-Test (kein Feature-Zeitstempel ≥ Snapshot), feste Seeds, Engine-Interface.
   - Browser-Prüfung mit Screenshots, Smoke-Test der bestehenden Routen.
   - Speicherbedarf des Flask-Prozesses messen (Ziel unter 400 MB RSS).
   - git diff --stat, lokal committen.
```

---

## Prompt 4 – Fix-Playbooks, Ampel aus drei Dimensionen, Was-wäre-wenn mit Euro-Potenzial

```text
Lies GRAPH_ML_PROMPTS.md Abschnitt 0 (vor allem 0.6 und 0.7) und den Code in graph_ml/. Die Leitplanken in 0.2 gelten.

Aufgabe: Modul 5 „Fix-Playbooks“ mit Rückkopplung zu den Checks, dazu die Use-Case-Readiness-Ampel aus drei Dimensionen und die Was-wäre-wenn-Analyse mit Euro-Potenzial.

1. Fix-Playbooks in graph_ml/playbooks.py
   Jedes Playbook hat:
   - ID, Problem, betroffene Checks und Use Cases
   - Optionen mit Vor- und Nachteilen
   - eine Vorschau der Wirkung: betroffene Zeilen, Kunden, Umsatz
   - einen reproduzierbaren Bereinigungsschritt. Pipeline-Versionen v0, v1 …; die Rohdaten werden nie überschrieben.
   Mindestens diese Playbooks:
   PB-01 Fehlende customer_id: (A) ausschließen oder (B) Gast-Pseudo-Kunde je Rechnung, nur für Warenkorb und Empfehlung zulässig
   PB-02 Stornos ohne Originalbezug: heuristisches Matching (Kunde, Artikel, Menge, höchstens 90 Tage), Matching-Quote anzeigen
   PB-03 Nicht-Artikel-Codes ausfiltern (POST, D, M, BANK CHARGES, AMAZONFEE, DOT, C2, ADJUST, TEST …)
   PB-04 Artikelstamm konsolidieren: stock_code-Varianten, Groß- und Kleinschreibung, abweichende Beschreibungen
   PB-05 Freitext → Artikelgruppen: TF-IDF und Clustering, etwa 20–30 Gruppen mit sprechenden Namen; optional per Sprachmodell. Schaltet Cross-Selling voll frei und ersetzt die vorläufige Gruppierung aus Prompt 3.
   PB-06 Exakte Dubletten entfernen
   PB-07 Preis 0, negative Mengen ohne Storno und Ausreißer-Mengen markieren
   PB-08 Währung GBP → EUR (Kurs konfigurierbar, Standard 1,15)
   PB-09 Governance-Checkliste (Pseudonymisierung, Speicherort, Rückspielweg) zum Abhaken
   PB-10 Fehlende Tabellen für Warranty und Marge: Datenanforderung generieren, also welche Tabellen, Felder und Verknüpfungen aus dem ERP nötig sind, inklusive Mapping-Vorlage

2. Rückkopplung
   - „Playbooks anwenden“ führt die Checks neu aus (Stufe 2). Auf Wunsch läuft auch der Backtest neu (Stufe 3, als Job).
   - Danach Vorher/Nachher-Vergleich der Ampel mit Delta-Pfeilen und Score-Änderung.
   - Das ist der Kreislauf aus dem PDF: Die Playbooks „führen zurück zu den Checks, bis ein Use Case trägt“.

3. Ampel aus drei Dimensionen in graph_ml/business_case.py (Regeln aus 0.7)
   - Datenreife, Vorhersagekraft, Euro-Potenzial → Gesamtampel. Konservativ: Es zählt die schwächste Dimension, mit Begründung.
   - Je Stufe zeigen, welche Dimensionen belastbar sind.

4. Was-wäre-wenn: Rückblick-Rechnung mit echten Zahlen aus dem Backtest, nichts erfunden
   - Churn:
     - Daten: aktive Kunden am Stichtag, tatsächlich abgewanderte und deren Vorjahresumsatz; wie viele davon in der Top-X-%-Liste der Regel und der besten Engine stehen.
     - Schieberegler: Listengröße (Top 5–30 %), Rettungsquote (Standard 25 %), Kosten je Kontakt, Bruttomarge.
     - Ergebnis: Potenzial in € für die Regel und für Graph ML; Mehrwert = Differenz.
   - Cross-Selling und Empfehlung: Treffer × Ø Warenkorb der Gruppe × Konversionsannahme.
   - Neukunden-Scoring: zusätzliche Zweitkäufer durch gezieltes Onboarding.
   - Retouren-Root-Cause: vermeidbare Retouren in Risikokohorten × Kosten je Retoure (Standard 15 €) × Vermeidungsquote.
   - Absatzprognose: Fehlerreduktion (WAPE) × Bestandswert × Kapitalkostensatz.
   - Für alle: Hochrechnung auf 12 Monate, Sensitivität als Tornado-Chart, Szenarien konservativ / realistisch / optimistisch.
   - Der Pflicht-Hinweis aus 0.7 ist immer sichtbar, ebenso der Hinweis auf die zeitliche Trennung.

5. Trichter-Ansicht zum Geschäftsmodell (PDF S. 6–7, Preise aus 0.6, als Annahmen gekennzeichnet)
   - Am Ende der Analyse eine Empfehlung für den nächsten Schritt, passend zum Ergebnis.
   - Dazu das berechnete Potenzial im Verhältnis zu den Kosten der Stufe (ROI-Indikator).

6. UI
   - Schritt 5: Playbook-Karten mit Vorschau, „Anwenden“ und Vorher/Nachher.
   - Ampel-Dashboard: eine Karte je Use Case mit drei Mini-Ampeln, Gesamtampel und €-Potenzial.
   - Was-wäre-wenn-Panel: Schieberegler mit Live-Neuberechnung im Browser, ohne Server-Anfragen bei Slider-Bewegung.
   - Wasserfall-Chart: verlorener Umsatz → in der Liste erkannt → gerettet (Annahme) → netto nach Kosten.

7. Tests und Abschluss
   - Die Potenzialrechnung reproduziert das Rechenbeispiel aus dem PDF (300.000 €, siehe 0.7).
   - Playbooks sind idempotent.
   - Browser-Prüfung inklusive 375 px, Smoke-Test, git diff --stat, lokal committen.
```

---

## Prompt 5 – Aktivierung, Monitoring, Report, Feinschliff und Deployment

```text
Lies GRAPH_ML_PROMPTS.md Abschnitt 0 und den Code in graph_ml/. Die Leitplanken in 0.2 gelten.

Aufgabe: Modul 6 „Aktivierung“, Monitoring, Report-Export und der Feinschliff der ganzen Seite, inklusive Deployment-Vorbereitung.

1. Aktivierung in graph_ml/activation.py
   - Je Use Case eine priorisierte Aktionsliste aus der empfohlenen Engine. Jeder Eintrag enthält:
     - Kunde bzw. Artikel, Score und kalibrierte Wahrscheinlichkeit
     - die drei wichtigsten Gründe in Klartext und den erwarteten Wert
     - die empfohlene Aktion. Next Best Action ist ein Regelwerk über die Scores für Churn, Cross-Selling, Empfehlung und Neukunden, mit Konfliktauflösung.
   - Kalibrierte Schwellen und ein Mensch in der Schleife:
     - Status je Eintrag: offen, übernommen oder verworfen mit Grund
     - unsichere Fälle markieren. Das ist die Lehre aus dem Kumo-Praxistest im PDF: zu viele falsche „sichere“ Vorhersagen.
   - Kontrollgruppe:
     - zufällige, stratifizierte Holdout-Gruppe (Standard 20 %) mit festem Seed
     - Fallzahlschätzung für den Pilot: Welcher Effekt ist mit n Kunden nachweisbar?
   - Export als CSV, XLSX (openpyxl) und JSON im Format eines generischen CRM-Imports, mit Mapping-Vorlage für Salesforce, HubSpot und MS Dynamics.
     Nur eine Vorschau des Rückspielwegs, KEIN echtes Senden an externe Systeme.

2. Monitoring
   - Datenqualität im Monatsverlauf (Füllgrad, Verknüpfungsquote).
   - Drift: PSI je Feature und für den Score zwischen Trainings- und aktuellem Zeitraum.
   - Score-Verteilung, Alarmschwellen und ein Button „Erneut prüfen“.

3. Report
   - „Readiness-Report“ als druckoptimierte HTML-Seite (Route /graph_ml/report, druckbar als PDF) und als XLSX.
   - Inhalt: Zusammenfassung, Ampel je Use Case mit drei Dimensionen, Prüfergebnisse, Modellvergleich, Reparaturplan mit Aufwandsschätzung, Business Case mit Annahmen, nächster Schritt.

4. Feinschliff der UI (alle 6 Schritte)
   - Geführte Demo-Tour „Pitch in 3 Akten“, jederzeit überspringbar:
     Akt 1: Online Retail II, Lücken werden sichtbar.
     Akt 2: Graph-Mapping und AdventureWorks.
     Akt 3: Rückblick und €-Potenzial.
   - Kompakte Infobox „Markt & Zeitleiste“ (AutoML 2012/13 → GNN 2017 → TabPFN 2022 → RDL/RelBench 2024 → KumoRFM 2025 → 2026) mit den Quellen aus dem PDF.
   - Einheitliche Animationen, Lade-Skeletons, leere Zustände.
   - Fehlerzustände, z. B. RunPod nicht erreichbar → klarer Hinweis auf den Fallback.
   - Tastaturbedienung, ausreichender Kontrast, 375–1920 px, Dark-Theme.
   - Performance: erster Seitenaufbau unter 2 s auf Render, API-Antworten unter 1 s dank Artefakten, lange Läufe nur als Jobs.

5. Qualität und Deployment
   - Komplette Testsuite ausführen. Im Browser end-to-end durch alle 6 Schritte klicken, mit Screenshot je Schritt.
   - Render-Tauglichkeit prüfen: Build-Zeit, RSS unter 400 MB, requirements.txt nur ergänzt. render.yaml nur ändern, wenn zwingend nötig, und vorher fragen.
   - Abgleich mit dem Backup ../webapp_backup_21_11_26:
     - Smoke-Test aller bestehenden Routen
     - git diff --stat und diff -rq der bestehenden Templates und Engines → nur erlaubte Dateien verändert
   - Committen. FRAG MICH, bevor du pushst; der Push löst das Render-Deployment aus.
     Nach dem Deploy https://salesapps.onrender.com/graph_ml, /graph_ml/api/system und /api/health prüfen.
   - RunPod: Befehle zum Bauen und Pushen des Graph-ML-Images auflisten und die nötigen Umgebungsvariablen in Render nennen (RUNPOD_GRAPHML_ENDPOINT_ID, optional KUMO_API_KEY).
```

---

## Bewusste Entscheidungen und offene Punkte

- **Eigener RunPod-Endpoint für Graph ML:** Der bestehende Worker in `runpod/` (Aurora + CV) bleibt unverändert, so kann ein Image-Neubau `/forecasting` und `/computer_vision_quality` nicht beschädigen. Das kostet einen zweiten Endpoint; bei Serverless zahlt man nur die Laufzeit.
- **Systemzugriff wie bei `/forecasting_model`:** Diese Seite rechnet selbst auf der CPU (LightGBM + Open-Meteo). Die GPU-Anbindung über RunPod stammt aus `/forecasting` und `/computer_vision_quality` und wird für `/graph_ml` nach demselben Muster gebaut.
- **KumoRFM:** Ob und wie es frei nutzbar ist, muss vor einer kommerziellen Demo geklärt werden (im PDF als offene Frage genannt). Der Adapter ist optional und fällt sauber weg, wenn kein Zugang besteht.
- **TabPFN:** Lizenz und Zugang zu den Gewichten vor kommerzieller Nutzung prüfen. TabICL ist die offene Alternative.
- **Warranty Root Cause** lässt sich mit Online Retail II nicht trainieren, weil Seriennummern, Chargen und Garantiefälle fehlen. Die Seite zeigt das ehrlich als Rot im Strukturcheck. Der After-Sales-Leuchtturm ist deshalb **Retouren- & Storno-Root-Cause**.
- **Render Starter (≈ 512 MB RAM):** Wird der Speicher trotz kompaktem Datenformat und Artefakten knapp, ist ein Upgrade auf *Standard* die einfachste Lösung. Das wäre eine Änderung am Render-Plan; Claude fragt vorher.
- **Kein Link auf der Startseite:** Wie `/forecasting_model` und `/computer_vision_quality` ist `/graph_ml` nur direkt über die URL erreichbar, weil `home.html` unverändert bleibt. Eine Kachel lässt sich später mit einem Einzeiler ergänzen.

---

## Umsetzungsnotizen nach Prompt 1 (für die folgenden Prompts)

- **Datenformat:** Online Retail II liegt als `graph_ml/data/online_retail_ii.npz` (5,6 MB) vor, nicht als Parquet. Grund: Es lädt ohne pyarrow und braucht nur etwa halb so viel Arbeitsspeicher (49 statt 120 MB). Laden mit `graph_ml.data_loader.load_online_retail()`. Neu erzeugen: `python -m graph_ml.prepare_data` (mit `--skip-download`, wenn die Zip-Datei in `data/raw/` liegt).
- **Daten-API:** `get_registry()` liefert `raw_online_retail()` (flacher Export), `tables(id)` (customers, invoices, invoice_lines, products, countries), `mapping(id)`, `overview(id)`, `customer_stats()`, `customer_subgraph(id)`. Kategorie-Operationen immer über Codes ausführen (`cat_startswith`, `_mode_codes`), nicht über 1 Mio. Strings.
- **Artefakte:** `graph_ml/artifacts/*.json` tragen `source_version` und `artifact_version`. Ändert sich die Logik in `graph_mapping.py`, `ARTIFACT_VERSION` erhöhen und `prepare_data` neu laufen lassen.
- **Befunde** aus dem Mapping haben schon die Felder `check` (Prüfdimension) und `playbook` (PB-01 … PB-10). Prompt 2 und 4 bauen darauf auf.
- **Jobs/GPU:** `graph_ml.jobs.store.submit(kind, title, fn, **kw)`; `fn(ctx, …)` meldet über `ctx.progress()`, `ctx.log()` und `ctx.set_device()`. `RunPodClient()` nutzt `RUNPOD_GRAPHML_ENDPOINT_ID`. torch wird nur importiert, wenn ein NVIDIA-Treiber vorhanden ist; auf Render spart das rund 200 MB.
- **Frontend:** Alpine ruft `init()` automatisch auf, deshalb kein zusätzliches `x-init` setzen. Zum Zeichnen nach DOM-Updates `afterRender(fn)` verwenden, nicht `$nextTick` (das lief im Vorschaufenster nach dem Start nicht mehr) und nicht `requestAnimationFrame` (pausiert in Hintergrund-Tabs). Gezeichnet wird über `$watch` auf `mapping`, `overview` und `subgraph`. Cytoscape-Container bekommen `guardWheel(el)`, damit das Mausrad die Seite scrollt; gezoomt wird mit Strg/⌘ oder den Buttons. Farben als `hexA()`/rgba angeben; Cytoscape kennt kein 8-stelliges Hex.
- **Tests:** `python -m unittest tests.test_graph_ml_mapping -v` (pytest ist lokal nicht installiert). Der Test `test_existing_routes_unchanged` prüft alle bestehenden Routen.

---

## Umsetzungsnotizen nach Prompt 2 (für die folgenden Prompts)

- **Bausteine:**
  - `usecases.py` enthält den Feldkatalog `FIELDS` mit ERP-Synonymen und die Vorlagen `USE_CASES`. Jede Vorlage hat `requires` als Tupel (Feld, pflicht/optional/messung, Ersatzfeld, Playbook) und `links` für Felder, die über Schlüssel verbunden sein müssen.
  - `semantics.py` stellt `schema_view()` (Struktur ohne Zahlen) und `bind_fields()` bereit. `bind_fields()` liefert die Feldzuordnung, z. B. customer_key → `invoices.customer_id`, `Sales.SalesOrderHeader.CustomerID` oder `KNA1.KUNNR`.
  - `profiler.py` enthält `build_profile()`, `events_from_online_retail()` und `assemble_events()`. Letzteres baut eine flache Ereignistabelle über FK-Pfade.
  - `checks.py` enthält `run_checks()` mit den 9 Dimensionen, der Readiness-Aggregation, `max_stage` und der Spalten-Ampel.
- **Stufen-Isolation:** `Registry.checks(ds, stage)` ruft Stufe 1 nur mit `schema_view(mapping)` auf, Stufe ≥ 2 nur mit `Registry.profile(ds)`. Die Tests sperren dafür jeden Datenzugriff (`tests/test_graph_ml_checks.py`). Diese Trennung in Prompt 3 beibehalten: Der Backtest liest Rohdaten, die Checks nicht.
- **Profil:**
  - Für Online Retail II liegt das Profil als Artefakt `artifacts/profile_online_retail_ii.json` vor. Download über `/graph_ml/api/datasets/<id>/profile`.
  - Ein hochgeladenes Profil-JSON (Schlüssel `profile_version`) wird zu einem Datensatz der Art `profile`: höchstens Stufe 2, Schritt 2 zeigt den Graphen aus dem Profil, es gibt keinen Kunden-Subgraphen.
  - Kunden-CLI: `python -m graph_ml.profiler_cli <ordner> --out profil.json`. `assert_privacy()` verhindert Beispiel- und Schlüsselwerte im Export.
- **Kennzahlen für Prompt 3** in `profile["events"]["usecase_stats"]`:
  - Zeitfenster je Use Case (`windows`: cutoff, eligible, positives)
  - `cutoffs`: monatlich, mit ≥ 12 Monaten Historie und 90 Tagen Label
  - Für Online Retail II: Test-Stichtag 2011-09-01; Churn 4.335 aktiv / 2.189 abgewandert; Neukunden 307 / 160; Stornos 80,6 % heuristisch zuordenbar; PSI max. 0,027
- **Ampel-Struktur** je Use Case:
  - `ampel.datenreife` ist berechnet.
  - `ampel.vorhersagekraft` und `ampel.euro` stehen auf `status: "pending"` und werden in Prompt 3 bzw. 4 befüllt.
  - Die Gesamtampel ist bis dahin „vorläufig“ und gleich der Datenreife.
- **Engine-Empfehlung** liegt je Use Case in `engine` (`primary`, `challengers`, `baseline`, `reason`). Prompt 3 soll sie mit dem Backtest bestätigen oder korrigieren.
- **Frontend:** Der Drawer-Graph wird über `$watch('ucOpen')` gezeichnet und wiederholt den Versuch, bis der Container sichtbar ist. Die Checks laden bei Änderung von `stage` oder `dsId` neu; `checksSeq` verwirft veraltete Antworten.

