# GraphReady v2 – Ausbau: mehr Datensätze, 20 Use Cases, schlankere Seite

Stand 21.11.2026. Ausgangspunkt: `/graph_ml` nach Prompt 5 (Backup: `../graph_ml_backup_v1_21_11_26`, Git-Tag `graph_ml-v1`).
Arbeitsweise: **ein Prompt nach dem anderen**, jeweils mit Tests, Browser-Prüfung und lokalem Commit. Rechnen lokal
auf der RTX 3060, Ergebnisse als Artefakte committen – **keine RunPod-Anbindung nötig** (siehe `GRAPH_ML_GPU_TODO.md`).

---

## 0 Gemeinsamer Kontext (gilt für jeden Prompt)

### 0.1 Leitplanken (unverändert aus v1)
- Nur `graph_ml/`, `templates/graph_ml.html`, `runpod_graphml/`, `tests/` und diese Doku ändern.
  `app.py`, `requirements.txt` (nur ergänzen), `render.yaml`, andere Seiten, `runpod/`, `datasets/` bleiben unberührt.
- Backups `../webapp_backup_21_11_26` und `../graph_ml_backup_v1_21_11_26` werden nie verändert.
- Kein `git push` ohne Rückfrage. Render bleibt ohne torch; GPU-Ergebnisse kommen als Artefakte (Ziel < 400 MB RSS).
- Datensätze mit Lizenz **nur nicht-kommerziell** (SAP SALT, Olist) werden nicht auf Render ausgeliefert, sondern nur
  lokal bzw. als Schema (Stufe 1) genutzt – Entscheidung siehe 0.5.
- Jeder Use Case sagt ehrlich, wo **Graph ML / relationales Lernen** messbar hilft – und wo LightGBM genügt.

### 0.2 Seitenaufbau v2 (Punkte 3–5 der Anfrage)
- **Entfällt:** Hero-Panel „Readiness-Trichter“, Buttons „Demo-Tour“ und „Markt & Zeitleiste“.
  Die Stufenwahl (1 Struktur · 2 Profil · 3 Rückblick) bleibt als kompakter Umschalter im Schritt „Readiness-Checks“.
- **Neue Reihenfolge der Pipeline (vereinfacht, nur die Kernpunkte):**

  | # | Schritt | Kernaussage auf der Kachel |
  |---|---|---|
  | 1 | Daten | Datensatz wählen, Tabellen und Lücken auf einen Blick |
  | 2 | Graph-Mapping | Tabellen als vernetzter Schema-Graph (Darstellung wie AdventureWorks, für alle Datensätze) |
  | 3 | Fix-Playbooks | Lücken schließen, Pipeline v0 → v1 (früher Schritt 5) |
  | 4 | Readiness | Ampel je Use Case aus Datenreife (Stufe 1–2) |
  | 5 | Rückblick & € | Modellvergleich und Euro-Potenzial (Stufe 3, inkl. Was-wäre-wenn) |
  | 6 | Aktivierung | Listen, Kontrollgruppe, Monitoring, Report |

- **Jeder Use Case bekommt eine eigene Detailseite** als Slide-over (wie der heutige Use-Case-Drawer, aber breiter),
  aufrufbar aus Ampel-Matrix, Rückblick und Aktivierung. Feste Gliederung:
  1. *Worum geht es* – Hebel und Wirkung laut Literatur (mit Quelle/Einordnung)
  2. *Input-Daten* – Tabellen und Felder, Abdeckung im gewählten Datensatz, fehlende Felder → Playbook/Datenanforderung
  3. *Warum relational / Graph?* – welches Signal über mehrere Tabellen läuft (Mini-Graph des Pfads),
     und **gemessen**: Lift von Graph-Engine vs. LightGBM vs. TabPFN vs. Regel
  4. *Vorhersage* – Label, Zeitfenster, Einheit (Kunde, Position, Maschine …)
  5. *Nutzen in Euro* – Formel, Annahmen als Schieberegler, Ergebnis realistisch/konservativ
  6. *Readiness-Ampel* – drei Dimensionen mit Begründung, nächster Schritt
- Graph-Mapping: Der Cytoscape-Schema-Graph (heute nur AdventureWorks) wird Standard für **alle** Datensätze, inkl.
  Verknüpfungsquote an den Kanten und Hervorhebung der Pfade, die ein gewählter Use Case braucht.

### 0.3 Datensätze v2

| Datensatz | Inhalt | Lizenz | Einsatz in GraphReady | Stufe |
|---|---|---|---|---|
| Online Retail II | echte Transaktionen, 1,07 Mio. Zeilen | CC BY 4.0 | Sales-Kern (bleibt) | 3 |
| AdventureWorks **mit Daten** | Fahrradhersteller: Aufträge mit Rabatten, Sonderangebote, Gebiete, Vertriebler mit Quoten, Stücklisten, Fertigungsaufträge mit Ausschuss, Lieferanten | MIT (künstlich) | Preis/Rabatt, Leakage, Gebiete, Lieferanten, Konfiguration | 3 |
| WideWorldImporters | Großhändler: Bestellung → Rechnung, Einkaufsgemeinschaften, Sonderkonditionen, Lieferantentransaktionen, Lager | MIT (künstlich) | Angebot → Auftrag, Account-Scoring über Einkaufsgemeinschaften, Ersatzteil-/Lagerlogik | 3 |
| Microsoft Azure Predictive Maintenance | 100 Maschinen, Telemetrie, Fehler, Wartungen, Ausfälle, Komponenten (5 Tabellen) | Lizenz in Prompt 2 prüfen | Predictive Maintenance, Ersatzteilbedarf, First-Time-Fix | 3 |
| NASA C-MAPSS | Turbinen-Sensordaten bis zum Ausfall | Public Domain (NASA) | Restlebensdauer – Benchmark für PdM | 3 |
| Olist | brasilianischer E-Commerce, 9 Tabellen, Bewertungen, Lieferungen, Verkäufer | CC BY-NC-SA 4.0 | nur lokal: Lieferqualität, Bewertungen, Verkäufer-Graph | 1–3 lokal |
| Odoo-Demodaten | Mittelstands-ERP | LGPL (Odoo) | Schema-Import (Stufe 1); Daten nur mit lokaler Odoo-Instanz | 1 |
| SAP SALT | echte SAP-Kundenaufträge, 4 Tabellen | CC BY-NC-SA 4.0, Registrierung | nur Schema (Stufe 1) bzw. lokal mit eigenem Zugang | 1 |
| **Synthetischer After-Sales-Datensatz** *(Vorschlag)* | Installationen mit Seriennummer, Serviceverträge, Serviceaufträge, Garantiefälle, Ersatzteile – auf AdventureWorks-Produkten und -Kunden | eigen, klar als synthetisch gekennzeichnet | Garantie, Betrug, Regress, installierte Basis, Serviceverträge, Ersatzteilpreise | 3 (nur Methodik) |

Für After-Sales gibt es keinen frei nutzbaren, relationalen Echtdatensatz. Ein synthetischer Datensatz mit dokumentierten,
eingebauten Mustern zeigt, dass die Pipeline funktioniert – **nicht**, wie gut ein Modell beim Kunden wäre. Die UI kennzeichnet das sichtbar.

### 0.4 Use-Case-Bibliothek v2 (20 + bestehende)

Bestehende Use Cases bleiben erhalten und werden zugeordnet (→). „Graph-Signal“ = was relationales Lernen zusätzlich sieht.

| # | Use Case | Hebel · Wirkung laut Literatur | Input-Daten | Vorhersage | Graph-Signal (warum relational) | Erwartung Graph vs. LightGBM | Datensatz |
|---|---|---|---|---|---|---|---|
| 1 | Preis- und Rabattempfehlung → *Margenoptimierung* | Marge · +2–7 Pp. Umsatzrendite | Auftragspositionen, Rabatte, Kunde, Produkt, Gebiet | Rabatt, bei dem der Auftrag gerade noch gewonnen wird / Mengenreaktion | Rabatte ähnlicher Kunden für ähnliche Produkte (Kunde–Produkt–Kunde) | mittel | AdventureWorks, WWI |
| 2 | Preis-Leakage aufdecken | Marge · > 6 % des Umsatzes versickern | Listenpreis, Netto, Rabatt, Sonderkonditionen | Positionen mit Rabatt über Peer-Niveau | Abweichung vom Peer-Preis im Kunde–Produkt-Graph | gering–mittel | AdventureWorks, Online Retail II |
| 3 | Potenzial der installierten Basis | Umsatz · 15–25 % (ein Fall) | Installationen, Alter, Verträge, Serviceaufträge | Kunde/Anlage mit Upgrade- oder Servicebedarf | ähnliche Anlagen und deren Historie | hoch | synthetisch |
| 4 | Abwanderung früh erkennen → *Churn & Retention* | Umsatz · 5 % mehr Bindung = 25–95 % mehr Gewinn | Belege, Positionen, Stornos | kein Kauf in 90 Tagen | inaktive Nachbarkunden über gemeinsame Artikel | hoch (gemessen: GNN +15 %) | Online Retail II |
| 5 | Cross-Selling und Produktempfehlung → *Cross-Selling*, *Produktempfehlung* | Umsatz · meist 10–15 % | Positionen, Artikelgruppen | nächste Gruppe / Top-12-Artikel | Kunde–Artikel–Kunde: „Kunden wie du kauften …“ | hoch | Online Retail II |
| 6 | Ersatzteilpreise optimieren | Marge · +2 Pp. EBIT (ein Fall) | Ersatzteile, Preise, Verfügbarkeit, Anlage | preisunelastische Teile | Teil–Anlage–Kunde: Bindung an installierte Basis | mittel | synthetisch, AdventureWorks (Komponenten) |
| 7 | Serviceverträge: Abschluss und Verlängerung | Umsatz · Abschlussquote +5–15 % | Verträge, Anlagen, Serviceaufträge | Verlängerung / Abschluss in 90 Tagen | Verträge ähnlicher Kunden und Anlagen | hoch | synthetisch |
| 8 | Garantie: Frühwarnung und Root Cause → *Warranty Root Cause*, *Retouren-Root-Cause* | Kosten · −4 bis −20 % | Seriennummer, Charge, Lieferant, Garantiefälle | Garantiefall in X Monaten; auffällige Chargen | Charge–Lieferant–Bauteil: Fehler pflanzen sich fort | hoch | synthetisch; Retouren: Online Retail II |
| 9 | Predictive Maintenance als Service | Kosten, Umsatz · Stillstand −30 bis −50 % | Telemetrie, Fehler, Wartungen, Komponenten | Ausfall einer Komponente in 24 h / Restlebensdauer | Maschine–Komponente–Fehler-Historie | mittel (oft genügt Zeitreihe + LightGBM) | Azure PdM, C-MAPSS |
| 10 | Next Best Action im Vertrieb → *Next Best Action* | Umsatz +5–8 %, Betreuungskosten −20–30 % | alle Scores + Reaktionen | Aktion mit höchstem Wert je Kunde | kombiniert die Graph-Scores | hoch (Regelwerk ohne Reaktionsdaten) | Online Retail II |
| 11 | Ersatzteilbedarf und Bestände | Kosten · Bestand −20 bis −27 % | Komponenten-Wechsel, Anlagenbestand, Lager | Bedarf je Teil und Monat | Bedarf aus der installierten Basis über Stücklisten | mittel | Azure PdM, WWI |
| 12 | Absatzprognose → *Absatzprognose* | Prognosefehler −20 bis −50 % | Wochenmengen, Artikelgruppen | Menge je Artikel und Woche | hierarchisch über Gruppen | mittel (gemessen: LightGBM gewinnt) | Online Retail II, WWI |
| 13 | Vertriebsabdeckung und Gebiete | Vertriebsproduktivität +5–10 % | Gebiete, Vertriebler, Quoten, Kunden, Umsatz | Potenzial je Gebiet / unterversorgte Kunden | Kunde–Gebiet–Vertriebler | mittel | AdventureWorks |
| 14 | Garantiebetrug und Fehlbuchungen | 3–15 % der Garantiekosten | Garantiefälle, Händler, Techniker, Teile | auffällige Fälle | Betrugsringe: Händler–Techniker–Teil | hoch | synthetisch |
| 15 | Lieferantenregress | Zulieferer tragen ~10 %, fair bis 37 % | Garantiefälle, Bauteile, Chargen, Lieferanten | regressfähige Fälle je Lieferant | Fall–Bauteil–Lieferant | hoch | synthetisch + AdventureWorks (Vendor, Ausschuss) |
| 16 | First-Time-Fix im Außendienst | Fehlbesuch: 1,6 Zusatzfahrten à 200–300 $ | Serviceaufträge, Fehlercodes, Teile, Techniker | benötigte Teile vor dem Einsatz | Fehler–Teil-Muster ähnlicher Anlagen | mittel | Azure PdM, synthetisch |
| 17 | Service-Triage und Diagnose | Produktivität +30–45 % (GenAI) | Fehlertexte, Codes | Kategorie/Lösung | gering – eher Sprachmodell | gering (ehrlich Rot/Gelb) | synthetisch (Texte) |
| 18 | Gewinnwahrscheinlichkeit und Umsatzprognose | nur 45 % vertrauen ihrer Prognose | Angebote/Bestellungen, Phasen, Kunde | Auftrag ja/nein, Pipeline-Umsatz | Erfolg ähnlicher Angebote beim selben Konzern/Käuferverbund | mittel | WWI (Bestellung → Rechnung) |
| 19 | Lead- und Account-Scoring → *Neukunden-Scoring* | Lead-ROI +77 % (ältere Umfrage) | Erstkauf, Branche, Verbund | Wiederkauf / Account-Potenzial | Einkaufsgemeinschaft, ähnliche Erstkäufer | mittel | Online Retail II, WWI |
| 20 | Konfigurations- und Angebotsprüfung | 62 % Margenverlust zwischen Angebot und Lieferung | Angebote, Konfigurationen/Stücklisten, Kosten | Angebote mit Margenrisiko | Konfiguration–Bauteil–Kosten | hoch | AdventureWorks (BOM, Kosten) |

**Zur Produktempfehlung (Frage aus der Anfrage):** Die heutige Siegerin (LightGBM-Ranker) nutzt Kandidaten aus dem
Co-Kauf-Graphen aller Kunden (ItemKNN: „wer A kaufte, kaufte auch B“) – andere Kunden fließen also ein, aber nur als
Artikel-Ähnlichkeit („Graph-ML light“), nicht als gelernter Kunden-Graph. Das echte Graph-Modell (LightGCN) verlor
(0,138 vs. 0,257 MAP@12). Prompt 3 macht das sichtbar (2-Sprung-Pfad Kunde → Artikel → Nachbarkunde → Artikel) und
testet zusätzlich ein heterogenes GNN für Link-Prediction.

### 0.5 Offene Entscheidungen (vor Prompt 2 klären)
1. Synthetischer After-Sales-Datensatz ja/nein (sonst bleiben 3, 6, 7, 8, 14, 15 auf Stufe 1 = nur Struktur).
2. Olist / SAP SALT nur lokal und nicht auf Render – einverstanden?
3. Azure-PdM-Lizenz: falls unklar, nur C-MAPSS (Public Domain) verwenden.

---

## Prompt 1 – Seitenumbau und Use-Case-Detailseiten (Punkte 3, 4, 5)
- Hero-Trichter, Demo-Tour-Button und Markt-Button entfernen (Code der Tour darf bleiben, nur nicht sichtbar).
- Pipeline neu ordnen (0.2): Fix-Playbooks wird Schritt 3, Readiness 4, Rückblick & € 5 (Was-wäre-wenn wandert aus dem
  alten Schritt 5 hierher), Aktivierung 6. Stepper-Kacheln auf Kernpunkte kürzen; Hash-Links `#s1`–`#s6` neu zuordnen.
- Stufen-Umschalter kompakt in Schritt 4.
- Use-Case-Detailseite als Slide-over mit den 6 Abschnitten aus 0.2 – zunächst für die 9 bestehenden Use Cases befüllt.
- Graph-Mapping: Schema-Graph-Darstellung wie AdventureWorks auch für Online Retail II und Uploads; Pfad-Hervorhebung je Use Case.
- Tests, 375 px, Smoke-Test, Commit.

## Prompt 2 – Datensätze einlesen und mappen
- Konnektoren: AdventureWorks mit Daten (CSV aus `sql-server-samples`), WideWorldImporters (Restore der `.bak` in einem
  lokalen SQL-Server-Container → kompakte `.npz`), Azure PdM bzw. C-MAPSS, Olist (lokal), Odoo- und SALT-Schema.
- Optional: Generator für den synthetischen After-Sales-Datensatz (Seed, dokumentierte Muster, Kennzeichnung).
- Für jeden Datensatz: Graph-Mapping, Profil (Stufe 2), Lizenzhinweis, Größe auf Render prüfen.

## Prompt 3 – Sales-Kern: Abwanderung · Cross-Selling & Produktempfehlung · Lead- & Account-Scoring
- Detailseiten vollständig; Produktempfehlung mit sichtbarem Nachbarschafts-Graph und zusätzlichem hetero-GNN.
- Account-Scoring über WWI-Einkaufsgemeinschaften (relationales Signal).

## Prompt 4 – Next Best Action · Absatzprognose · Vertriebsabdeckung & Gebiete
- Gebiete aus AdventureWorks (SalesTerritory, SalesPerson, Quoten): Potenzial je Gebiet, unterversorgte Kunden.

## Prompt 5 – Pricing: Preis- & Rabattempfehlung · Preis-Leakage · Konfigurations- & Angebotsprüfung
- AdventureWorks/WWI-Rabatte und Sonderkonditionen; Stücklisten-Kosten für Angebotsprüfung.

## Prompt 6 – Installierte Basis · Serviceverträge · Ersatzteilpreise
## Prompt 7 – Garantie-Frühwarnung & Root Cause · Garantiebetrug · Lieferantenregress
## Prompt 8 – Predictive Maintenance · Ersatzteilbedarf · First-Time-Fix
## Prompt 9 – Gewinnwahrscheinlichkeit & Umsatzprognose · Service-Triage · Abgleich der Altfälle (Marge, Retouren)
Je Prompt: Labels mit Leakage-Guard, Regel-Baseline, LightGBM, LightGBM + Graph-Features, TabPFN, GNN (lokale GPU),
Euro-Formel mit Annahmen, Aktivierungsliste falls sinnvoll, Detailseite, Tests, Artefakte, Commit.

## Prompt 10 – Gesamtabgleich und Deployment
- Ampel-Matrix über alle Use Cases × Datensätze, Report um neue Use Cases ergänzen, Render-Speicher prüfen,
  Artefakte committen, nach Freigabe pushen.

---

## Umsetzungsnotizen nach Prompt 1

- **Seite:** Hero ohne Trichter, ohne Demo- und Markt-Button (Tour-Code bleibt, ist aber nicht mehr erreichbar). Stepper:
  1 Daten · 2 Graph-Mapping · 3 Fix-Playbooks · 4 Readiness · 5 Rückblick & € · 6 Aktivierung.
  Das Euro-Potenzial (Ampel-Dashboard, Was-wäre-wenn, Trichter) steht jetzt unter dem Rückblick in Schritt 5.
- **Stufen:** Umschalter „1 · Struktur / 2 · Profil / 3 · Rückblick“ im Banner von Schritt 4, auch auf dem Handy sichtbar.
  Schritt 5 und 6 schalten automatisch auf Stufe 3 (auch beim direkten Aufruf über `#s5`/`#s6`).
- **Graph-Mapping:** kompakte Darstellung wie AdventureWorks für alle Datenbanken (kleine Knoten in Gruppenfarbe, dünne Kanten, nur Verknüpfungen unter 95 % farbig mit Prozentwert). Tabellen ohne DB-Schema bekommen eine fachliche
  Gruppe (`domainOf`: Vertrieb, Kunden, Artikel, Stammdaten, Daten). Verknüpfungsquoten bleiben an den Kanten;
  Graph und Beziehungen immer in voller Breite.
- **Use-Case-Detailseite** (Slide-over, 900 px): 1 Worum geht es · 2 Input-Daten · 3 Warum relational · 4 Vorhersage ·
  5 Nutzen in Euro · 6 Readiness-Ampel, darunter aufklappbar die 9 Prüfungen. Texte in `usecases.STORY`.
  Abschnitt 3 vergleicht die Modellfamilien aus dem Rückblick getrennt:
  Regel · Baum · Tabular FM · Graph-Merkmale (LightGBM mit Graph-Features, ItemKNN, Ranker) · Graph-Modell (GNN, LightGCN).
  Urteile: Churn GNN +15,1 % ggü. LightGBM; Empfehlung Graph-Merkmale +15,5 %, LightGCN −37,9 % ggü. Regel;
  Cross-Selling noch ohne GNN.
- **Aufruf:** Use-Case-Name in der Ampel-Matrix, „ⓘ Use-Case-Seite“ am Leaderboard (Schritt 5) und an den Aktionslisten (Schritt 6).
- **Qualität:** 48 Tests grün (neu `tests/test_graph_ml_v2.py`), 375 px ohne seitliches Scrollen in allen Schritten,
  19 Seiten 200.

## Umsetzungsnotizen nach Prompt 2

- **Antworten aus 0.5:**
  - After-Sales-Benchmark = **SCANIA Component X** (echte Lkw-Flotte, CC BY 4.0, DOI 10.5878/jvb5-d390) – ersetzt den synthetischen Datensatz.
  - Predictive Maintenance mit **NASA C-MAPSS** (Public Domain).
  - Olist und SAP SALT: erst lokal, danach ggf. GitHub – offen (siehe unten).
- **Neu: `graph_ml/connectors.py`** – Aufbereitung einmal lokal, Ablage als `data/<id>.npz` + `<id>.meta.json`
  (Tabellen spaltenweise, Texte als Kategorien, Quelle, Lizenz, Gruppen, deklarierte Schlüssel). Render lädt nur diese Dateien.
  `python -m graph_ml.connectors --only adventureworks_data,cmapss,scania,wwi`

  | Datensatz | Tabellen | Zeilen | Beziehungen | Datei | Hinweis |
  |---|---|---|---|---|---|
  | AdventureWorks (mit Daten) | 61 | 739.124 | 79 deklariert | 5,0 MB | CSV aus `AdventureWorks-oltp-install-script.zip`; Binär-/XML-Tabellen ausgelassen |
  | WideWorldImporters | 31 | 1.057.599 | 69 deklariert | 10,1 MB | `.bak` in einem lokalen SQL-Server-Container wiederhergestellt; JSON-Blobs entfernt |
  | SCANIA Component X | 12 | 68.503 | 11 | 1,4 MB | 8 Konfigurationsmerkmale als eigene Tabellen (Graph: Fahrzeuge teilen Konfigurationen); 1,1 Mio. Auslesungen nur lokal (`data/local/scania_readouts.npz`, 337 MB, nicht in Git) |
  | NASA C-MAPSS | 3 | 266.676 | 2 | 8,9 MB | Flotten → Triebwerke → Zyklen, Training (bis Ausfall) und Test (mit Rest-RUL) |

- **Mapping:** `build_mapping` misst deklarierte Fremdschlüssel (`extra["declared_fks"]`, auch zusammengesetzt; 279.0 = 279) und
  übernimmt Schema-Gruppen (`extra["schemas"]`) → die kompakte AdventureWorks-Darstellung gilt automatisch für alle.
  Echte Befunde: AdventureWorks – nur 50 % der Positionen haben einen Vertriebler (Online-Aufträge), 89,5 % der Kunden < 3 Belege;
  WideWorldImporters – 663 Kunden ohne einen einzigen Abwanderer (künstliche Daten → Churn nicht lernbar).
- **Speicher (Render):** immer nur ein Datensatz v2 im Speicher; Text-Blobs (> 50 % eindeutig, Ø > 120 Zeichen) werden nicht
  gespeichert. Spitze beim Durchklicken aller Datensätze + Report + Export: 329 MB.
- **UI:** Datensatz-Karten mit Symbol, Tabellen/Zeilen/FKs, Chip „künstlich“ bei Beispieldaten.
- **Offen:** Olist (Kaggle-Login nötig), SAP SALT (Freigabe auf Hugging Face mit eigenem Konto nötig), Odoo-Schema.

## Umsetzungsnotizen nach Prompt 3

- **Abwanderung (Churn):** unverändert aus v1 – GNN +15,1 % ggü. LightGBM, Detailseite vollständig.
- **Cross-Selling:** Hetero-GraphSAGE jetzt auch für Paare Kunde × Artikelgruppe (Kunden-Einbettung + Gruppen-Einbettung +
  Paar-Merkmale). Ergebnis Recall@Top-10 %: Regel 0,113 · **LightGBM 0,267** · TabPFN 0,252 · LightGBM + Graph 0,251 ·
  GNN 0,204 → LightGBM genügt, das GNN verliert.
- **Produktempfehlung:** neuer **GNN-Ranker** (`gnn_rank`): Kunden- und Artikel-Einbettungen aus dem Kaufgraphen bewerten
  dieselben Kandidaten wie der LightGBM-Ranker. MAP@12: Regel 0,223 · ItemKNN 0,203 · **LightGBM-Ranker 0,257** ·
  GNN-Ranker 0,247 · LightGCN 0,138. Antwort auf die Frage aus der Anfrage: Andere Kunden fließen ein; das gelernte
  Graph-Modell holt fast auf (+11 % ggü. Regel), die Graph-Merkmale im einfacheren Ranker bleiben vorn.
  Neu auf der Detailseite: Nachbarschafts-Pfad (Kunde → eigene Artikel → 5 Nachbarn mit den meisten gemeinsamen Artikeln →
  empfohlene Artikel, die diese Nachbarn kaufen), `activation.build_reco_paths()` → `artifacts/reco_paths.json`
  (40 Beispielkunden, Ø 89 % der Nachbarn kaufen den empfohlenen Artikel).
- **Lead- und Account-Scoring:** Neukunden-Scoring unverändert (Online Retail II). Neu: **Account-Scoring auf
  WideWorldImporters** – „Quartalsumsatz wächst um ≥ 20 %“, monatliche Stichtage 01/2014–01/2016, Test 01.02.2016.
  WWI wird über `connectors.wwi_as_retail()` ins Positionsformat gebracht; `backtest.dataset_time()` setzt die Zeitachse
  je Datensatz (`DATASET_OF = {"account_scoring": "wwi"}`). Graph-Signal: Wachstum der Peers in Einkaufsgemeinschaft,
  Kundenkategorie und Region (ohne den Account selbst) + Kunde–Artikel-Nachbarschaft.
  AUPRC: Regel (Trend) 0,307 – unter Zufall (42 % Positivrate, Rückkehr zum Mittelwert) · LightGBM 0,798 ·
  LightGBM + Graph 0,784 · GNN 0,823 · **TabPFN 0,826**.
- **Euro:** Account-Scoring 491.144 €/Jahr – auf künstlichen Daten, deshalb als WWI gekennzeichnet und aus „nächster Schritt“
  und Report für Online Retail II ausgenommen (je Datensatz getrennt).
- **Ampel:** Rückblick-Ergebnisse fließen nur in die Matrix des Datensatzes, auf dem sie gerechnet wurden (Stufe 3 für WWI neu).
- **Tests:** 53 grün (u. a. Peer-Signal ohne Selbstbezug, Leakage des Account-Tasks, Zeitachse zurückgesetzt, GNN-Ergebnisse
  vorhanden, keine Datensatz-Vermischung).

## Umsetzungsnotizen nach Prompt 4

- **Next Best Action – jetzt messbar** (`graph_ml/nba_eval.py`): Offline-Policy-Vergleich am Rückblick-Stichtag 01.09.2011.
  Je Kunde wählt das Regelwerk (Kontrollgruppe, Retention vor Verkauf, Neukunden, höchster erwarteter Wert, 1 Aktion je Quartal)
  über die Scores von LightGBM + Graph-Merkmalen; gezählt wird, ob das Zielereignis danach eintrat (Abwanderung, Gruppenkauf,
  Wiederkauf, Artikelkauf) – Wirkungsquoten wie in Schritt 5. Netto-Wert je Quartal: **NBA 49.822 €** · Regel-Policy
  („Inaktive anrufen, sonst Topseller-Newsletter“) 26.862 € · Einheitsaktion 1.254 € → **+85,5 %** (90-%-Intervall +50 … +161 %),
  Euro 199.288 €/Jahr. Ein gelerntes NBA-Modell (Uplift) braucht weiterhin Reaktionsdaten – steht so auf der Detailseite.
  Artefakt `backtest_next_best_action.json` (Art `policy`), `evaluation.POLICY_UCS`, Formel in Python und JS identisch.
- **Absatzprognose:** neue Engine **LightGBM + Gruppen-/Co-Kauf-Graph** (`lgbm_graph`, Forecast): Ø 4/13 Wochen und Trend der
  Geschwister-Artikel derselben Gruppe (ohne die Reihe selbst) und der 5 ähnlichsten Artikel im Co-Kauf-Graphen.
  WAPE: Regel 1,100 · LightGBM 0,748 · **Graph 0,721** (−3,6 % Fehler) – erste Prognose, bei der das Graph-Signal gewinnt.
  Euro bleibt klein (9.261 €/Jahr, Lagerkosten bei 100 Artikeln) → Gesamt Rot.
- **Vertriebsabdeckung & Gebiete (neu, AdventureWorks):** `connectors.aw_as_retail()` (Gebiet, Produktkategorie,
  Vertriebler-Portfolio über Store.SalesPersonID, Händler/Endkunde), Zeitachse 06/2023–03/2025.
  Label: Händler bestellt im nächsten Quartal; Liste nach Wahrscheinlichkeit × Jahresumsatz.
  Graph-Signal: Anteil der Portfolio- und Gebiets-Peers mit frischem Auftrag, Portfolio-Wachstum.
  Umsatz-Recall@Top-10 %: Regel (größte Händler zuerst) 0,384 · **LightGBM 0,457** · GNN 0,449 · TabPFN 0,405 ·
  LightGBM + Graph 0,391.
  Gebietstabelle (`activation.build_territory_table()` → `territory_table.json`): Händler, Vertriebler, Quote,
  Umsatz Vorquartal, erwartet vs. tatsächlich (Abweichung je Gebiet 33 %, Prognose zu hoch – bewusst sichtbar),
  11 überfällige umsatzstarke Händler – das Modell gab ihnen Ø 1 % Bestellwahrscheinlichkeit, keiner bestellte.
- **Fehler aus v1 behoben:** `Registry.checks()` lieferte beim zweiten Aufruf für Stufe 3 die Ampel ohne Rückblick und Euro
  (Cache vor dem Einmischen). Jetzt wird bei jedem Aufruf frisch eingemischt (Regressionstest).
- **Tests:** 57 grün. Render: 19 Seiten 200, 230 MB.

## Umsetzungsnotizen nach Prompt 5

- **Deployment:** v2-Prompts 1–4 gepusht (`1f46457`), Render live; nach dem Durchklicken aller Datensätze 389 MB von 512 MB
  auf Render (lokal 230 MB) – beobachten, bei Engpass Plan „Standard“ (Rückfrage nötig).
- **Neu: `graph_ml/pricing.py`** (AdventureWorks, Händlerpositionen) mit Listenpreis und Standardkosten **zum Auftragsdatum**
  (Preis- und Kostenhistorie), Rabattstaffel aus `SpecialOffer`, Stücklisten-Tiefe aus `BillOfMaterials`.
  Training 8 Quartale ab 01/2023, Test 01.01.–29.06.2025; Peer-Merkmale eines Quartals nur aus den 365 Tagen davor
  (Test: Zukunft verändern → Merkmale gleich). Der gewährte Rabatt ist die Entscheidung und deshalb **kein** Merkmal.
- **Preis- & Rabattempfehlung** (erzielbarer Nettopreis in % vom Listenpreis, MAE):
  Regel (zuletzt erzielter Artikelpreis) 1,31 Pp. · LightGBM 0,98 · LightGBM + Peer-Preise (Graph) 0,98 · **TabPFN 0,85**
  → +35,2 % (signifikant). Graph-Signal (Peers im Gebiet, Händler-Historie, Vertriebler-Portfolio) bringt hier nichts.
- **Preis-Leakage** (Preis > 3 Pp. unter dem erwartbaren): TabPFN markiert 170 Positionen / 89.412 € im Halbjahr
  (Regel: 555 Positionen / 108.797 € – mehr Fehlalarme). Beispiel: „Mountain-500 Silver“ zu 12 % vom Listenpreis,
  erwartbar 60 %; die Abverkaufsaktion (40 %) erklärt höchstens 36 %.
- **Konfigurations- & Angebotsprüfung:** Regelprüfung (Marge mit Kosten zum Datum < 10 %), Vorhersagekraft „nicht anwendbar“:
  492 von 11.699 Aufträgen, 4.175 Verlustpositionen, −445.655 € Deckungsbeitrag im Halbjahr; Verluste nach Sonderaktion und
  Stücklisten-Tiefe auf der Detailseite. Graph ML erst sinnvoll, wenn Kosten im Angebot geschätzt werden müssen.
- **Euro (realistisch, AdventureWorks = künstlich, nur im AW-Kontext):** Preisempfehlung 53.017 € · Leakage 70.270 € ·
  Angebotsprüfung 581.018 € pro Jahr. Formeln (Lücke × Quote − Prüfungen × Kosten, × 2 Halbjahre) in Python und JS identisch;
  „Mehrwert ggü. Regel“ bei Lücken-Use-Cases bewusst leer (mehr gefundene Lücke ≠ besser).
- **Fehler behoben:** neue Felder wurden nie zugeordnet, weil `semantics.FIELD_ORDER` sie nicht enthielt – betraf auch
  `account_group` aus Prompt 3 (Einkaufsgemeinschaft in WWI jetzt erkannt). Neu: Felder `list_price`, `discount`, `sales_rep`;
  Profile neu erzeugt. Route `/backtest/<uc>` lässt jetzt auch die eigenen Rückblicke (NBA, Pricing) zu.
- **Tests:** 60 grün. Render: 19 Seiten 200, 230 MB.

## Umsetzungsnotizen nach Prompt 6

- **Neu: `graph_ml/aftersales.py`** – drei eigene Rückblicke im Artefaktformat (`POLICY_UCS`), Datensätze über
  `DATASET_OF` (`installed_base`, `service_contracts` → `scania`, `spare_parts_pricing` → `aw`). Neue Felder
  `service_event` (Reparatur/Serviceauftrag, Fertigungsaufträge ausgeschlossen), `contract`, `config`; Synonyme für
  Fahrzeug-/Triebwerks-IDs und Auslesungen. Profile neu erzeugt.
- **Installierte Basis (SCANIA, echte Flotte):** 232.670 Schnitte (bis 10 je Fahrzeug, 23.550 Fahrzeuge), Label Reparatur
  Komponente X in 48 Zeiteinheiten (Positivrate 2,8 %). SCANIA hat keine Kalenderzeit → Test auf 20 % der Fahrzeuge.
  Peer-Quoten je Konfigurationswert fold-basiert nur aus Trainingsfahrzeugen (Test: eigenes Label kippen → eigene Merkmale
  gleich; ein Leave-one-out-Encoding hatte vorher das eigene Label gespiegelt). AUPRC: Regel (längste Laufzeit) 0,038 ·
  **LightGBM 0,115** · LightGBM + Konfigurations-Peers 0,105 · GNN Fahrzeug ↔ Konfiguration 0,103 · TabPFN 0,068 → +205 %
  (signifikant). **Ehrlich:** Das Signal steckt in den Auslesungen, die Konfiguration erklärt Ausfälle kaum – Graph bringt
  hier nichts zusätzlich. Euro 69.750 €/Jahr (Testflotte, Auftragswert 2.500 € = Annahme).
- **Serviceverträge:** keine Vertragsdaten → Vorhersagekraft und Euro „nicht messbar“, Datenreife max. Gelb (`cap`), eigener
  Text zum fehlenden Label (`nolabel`). Gezeigt: risikobasierte Prämie je Konfiguration (spec_2) aus dem Reparaturmodell.
- **Ersatzteilbedarf & -preise (AdventureWorks):** 111 Komponenten, 101 in Rädern verbaut (1.150 Stücklisten-Kanten);
  installierte Basis = verkaufte Räder × Stückmenge. Skalenfreies Ziel (Faktor zum Vorquartal, L1). WAPE Folgequartal
  (Prognose 01/2025 und 04/2025): Regel 0,463 · LightGBM 0,288 · **mit installierter Basis 0,274** → +40,7 % (signifikant).
  14 gebundene Teile (Korrelation ≥ 0,5), 123.375 € Umsatz im Test; Euro (Aufschlag 3 %, 30 % Mengenverlust) nur 3.782 €/Jahr
  → Rot. Neue Was-wäre-wenn-Art `uplift` in Python und JS identisch.
- **UI:** Detailseiten mit Kalkulationstabelle (Verträge) und Panel „gebundene Teile“; Schritt 5 zeigt bei Daten ohne
  Kalenderzeit den Einheiten-Split statt „Invalid Date“; der Chip „undefined Zeitschnitte“ (betraf auch Pricing) behoben.
- **Tests:** 63 grün (System-Python; im GPU-`.venv` fehlt `openpyxl` für den Excel-Export). Web-Modus: alle Datensätze
  und Stufen 200, 206 MB RSS, kein torch. Volle SCANIA-Auslesungen nur lokal (`data/local/`), Precompute-Hook überspringt
  sonst.

## Umsetzungsnotizen nach Prompt 7

- **Deployment:** Prompts 5 und 6 gepusht (`eb58afe`), Render live (17 Use Cases, Routen 200).
- **Kein offener Garantiedatensatz** → gemessen wird über die AdventureWorks-Kette Lieferant → Bauteil (Wareneingang mit
  Beanstandungen) → Stückliste → Produkt → Fertigungsauftrag (Ausschuss mit Grund). Neu: `graph_ml/quality.py`, Felder
  `rejected_qty` und `technician`; Leakage-Guard `past_stats` zählt nur Fälle, die vor dem Start abgeschlossen waren
  (Beanstandung ab Fälligkeit, Ausschuss ab Auftragsende; Test: Zukunft verändern → Merkmale davor gleich).
- **Lieferantenregress** (8.845 Bestellpositionen, 86 Lieferanten, Quote 6,4 %; Test ab 01.04.2025): AUPRC Regel (bisherige
  Lieferantenquote) 0,089 · LightGBM 0,136 · LightGBM + Bauteil-/Lieferanten-Peers 0,138 · **TabPFN 0,166** · GNN Position ↔
  Lieferant ↔ Bauteil 0,117 → +87,9 % (signifikant). Graph-Peers bringen kaum etwas. Regress-Volumen im Test 465.413 €,
  Top-10-Lieferanten 64,6 %. Euro 149.336 €/Jahr (Rückholquote 60 % = Annahme) → Gesamt Gelb (Datenreife).
- **Qualitäts-Frühwarnung & Root Cause** (72.591 Fertigungsaufträge, Ausschuss 1,0 %; Test ab 01.01.2025): Regel (größte
  Menge) 0,095 · LightGBM 0,089 · + Graph 0,086 · TabPFN 0,099 · GNN 0,079 → +4 % (nicht signifikant). Ausschuss entsteht in
  AdventureWorks nur bei großen Aufträgen und folgt nicht der Lieferantenqualität (gute Lieferanten 2,2 %, schwache 0,7 %) –
  ehrlich Rot; Euro negativ (−40.076 €). Ursachen nach Ausschussgrund und Produktgruppe auf der Detailseite.
- **Garantiebetrug:** keine offenen Daten (Garantiefälle mit Händler, Techniker, Prüfergebnis) → nur Struktur (`NOT_MEASURABLE`).
  `warranty_root_cause` bleibt Stufe 1 und verweist auf die Qualitäts-Frühwarnung.
- **Tests:** 66 grün (System-Python). Web-Modus 211 MB, kein torch.

## Umsetzungsnotizen nach Prompt 8

- **Deployment:** Prompt 7 gepusht (`92337a3`).
- **Neu: `graph_ml/maintenance.py`**; `quality.evaluate_binary` kann jetzt auch nach Einheiten splitten (`masks`). Neues Feld
  `failure_event`; `stockitemid` als Artikelschlüssel (WWI-Artikel wurden bisher nicht erkannt – betraf auch Preisempfehlung
  und Cross-Selling auf WWI). Profile neu erzeugt.
- **Predictive Maintenance (NASA C-MAPSS, Public Domain):** 14.160 Schnitte (10 je Triebwerk, 1.416 Triebwerke, 4 Flotten),
  Label Ausfall in 30 Zyklen (8,7 %), Test 20 % der Triebwerke je Flotte. Graph: Sensoren normiert gegen Peer-Triebwerke unter
  derselben Betriebsbedingung (nur Trainings-Triebwerke, erste 30 Zyklen; Test: Test-Triebwerke verändern → Peer-Werte der
  anderen gleich) und Ausfallquote der Flotten-Peers im gleichen Alter. AUPRC: Regel (meiste Zyklen) 0,344 · LightGBM 0,905 ·
  + Graph 0,923 · **TabPFN 0,936** · GNN 0,496 → +172 % (signifikant). Euro 5,6 Mio. € (Testflotte über die Lebensdauer,
  50.000 € Mehrkosten je ungeplantem Ausfall = Annahme; Hochrechnung bewusst 1×, die Testschnitte decken die ganze Lebensdauer ab).
- **Ersatzteilbedarf & Bestände (WWI):** 227 Lagerartikel, Bedarf Folgemonat, Test 4 Monate, WAPE wertgewichtet: Regel
  (Ø 3 Monate) 0,226 · **LightGBM 0,206** · + Warengruppe/Co-Kauf 0,213 → +8,8 % (signifikant, < 20 % → Gelb). Euro 52.877 €
  (Sicherheitsbestand-Formel wie Absatzprognose). Bestandsampel zum 31.05.2016: 9 Artikel knapp; 213 von 227 mit > 1 Jahr
  Reichweite (künstliche Endbestände – Kapital-Kennzahl bewusst weggelassen).
- **First-Time-Fix:** keine offenen Serviceaufträge mit Fehlercode, Techniker und Teilen → nur Struktur (`NOT_MEASURABLE`).
- **Tests:** 68 grün. Web-Modus 213 MB, kein torch.

## Umsetzungsnotizen nach Prompt 9

- **Deployment:** Prompt 8 gepusht (`a2ade6f`).
- **Neu: `graph_ml/sales_pipeline.py`** – Gewinnwahrscheinlichkeit & Umsatzprognose auf WWI. Es gibt keine Angebote; fast alle
  nicht fakturierten Aufträge sind Rückstandsaufträge. Label: Rückstand wird nicht binnen 30 Tagen fakturiert (7.330 Aufträge,
  60 %; Test ab 01.01.2016, 796). Graph: Artikel → andere offene Rückstände, Artikel → Lieferant, Kunde → Einkaufsgemeinschaft,
  Artikel → offene Einkaufsbestellungen (alles zum Auftragsdatum bekannt; Test: Zukunft verändern → Quoten davor gleich).
- **Ehrlich:** Der WWI-Generator folgt fast einer Mengenregel (ab ~50 Stück bleibt der Rückstand liegen; Einzelmerkmal Menge
  AUC 0,89, Wert 0,94). Erst mit einer Kundenquote als Regel sah es nach +52 % aus – Baseline deshalb „größte Menge zuerst“:
  AUPRC Regel 0,896 · LightGBM 0,990 · + Graph 0,998 · TabPFN 1,000 · GNN 0,937 → +11,6 % (signifikant, Gelb).
  Umsatzprognose je Woche (WAPE): Regel (Lieferquote je Mengenklasse) 0,38 · LightGBM 0,15 · + Graph 0,11 · **TabPFN 0,06**
  · GNN 0,19. Euro 18.899 €/Jahr (Rettungsquote 25 %, Marge 30 % = Annahmen) → Rot.
- **Service-Triage:** keine offenen Servicetexte (WWI-Kommentare/Gutschriftgründe leer) → nur Struktur, Feld `service_text`;
  ehrlich als Sprachmodell-Thema markiert.
- **Abgleich der Altfälle:** Marge → Verweis auf Preisempfehlung, Leakage und Angebotsprüfung (AdventureWorks); Retouren-Root-Cause
  bleibt auf Online Retail II (WWI ohne Gutschriften, AW ohne Retouren), Garantie-Pendant = Qualitäts-Frühwarnung.
- **Tests:** 70 grün. Web-Modus 213 MB, alle Routen 200, kein torch. 25 Use Cases.

## Umsetzungsnotizen nach Prompt 10 (Abschluss v2)

- **Deployment:** Prompt 9 gepusht (`806e57e`).
- **Neu: `graph_ml/matrix.py`** – Ampel-Matrix 25 Use Cases × 6 eingebaute Datensätze (Stufe 3 bzw. höchste mögliche Stufe).
  Je Use Case genau ein Heimat-Datensatz (Rückblick + Euro, ★), sonst Datenreife; vier Use Cases ohne offene Daten
  (Garantie, Garantiebetrug, First-Time-Fix, Service-Triage) als „nur Struktur“. Artefakt `artifacts/matrix.json` (36 KB,
  Precompute schreibt es zuletzt), Route `/graph_ml/api/matrix`, aufklappbares Panel „Gesamtabgleich“ in Schritt 4 – Zelle
  öffnet die Detailseite im passenden Datensatz (Stufe 3 automatisch).
- **Stand:** 20 von 25 Use Cases mit Rückblick gemessen · Gesamt Grün 0 · Gelb 14 · Rot 7 · nur Struktur 4. Kein Use Case ist
  insgesamt Grün, weil Datenreife oder Euro die schwächste Dimension bleiben.
- **Report (HTML + Excel):** neuer Abschnitt 8 „Gesamtabgleich“ (Matrix + Rückblicke auf weiteren Datensätzen mit Hinweis
  „künstlich, nicht addierbar“), Excel-Blätter „Matrix“ und „Weitere Datensätze“.
- **Ergebnisse v2 je Modellklasse (Sieger, 18 Rückblicke mit Modellvergleich):** TabPFN 7× · LightGBM + Graph-Merkmale 5× ·
  LightGBM 4× · gelerntes GNN 2× (Churn, Neukunden-Scoring). Vorhersagekraft Grün bei 11 Use Cases. Ehrliches Fazit:
  relationale Signale helfen am häufigsten als Merkmale in Baum-/Tabellenmodellen; ein gelerntes GNN gewinnt selten.
- **Tests:** 72 grün. Web-Modus: 34 Routen + 6 Datensätze × 3 Stufen = 200, Spitze 240 MB, kein torch.
- **Offen (außerhalb v2):** Olist und SAP SALT (Zugang durch den Nutzer), Odoo-Schema, RunPod-Worker für Live-GPU
  (`GRAPH_ML_GPU_TODO.md`).

## Nachtrag Schritt 3: Data-Classifier-Analyse

- **Neu: `graph_ml/classifier_scan.py`** – durchsucht jede Datenbank nach manuell gepflegten oder lückenhaften Feldern
  (Kennzeichen, Klassifikationen, Parameter) und prüft je Feld, ob ML sie vorschlagen kann. Muster: Kaskade aus dem Konzept
  „Ersatzteil-Klassifikation in Kfz-Stücklisten“ (Stufe 0 Regeln, Stufe 1 ML gemessen; Stufe 2 LLM-Prüfer, Stufe 3 Mensch als Konzept).
- **Erkennung:** Tabellenrollen (Stamm-, Bewegungsdaten, Wertelisten; Personen-, Geo- und Systemfelder ausgeschlossen),
  Wahrscheinlichkeit „manuell“ aus Rolle, Name, Füllgrad, Kardinalität. Kontext: eigene Felder, verbundene Stammdaten,
  Strukturtabellen (Stückliste, Lieferant je Artikel), Graph-Nachbarschaft (verschachtelt out-of-fold, ohne eigenes Label –
  Test), Text (Benennung + Benennung des Vaters), sprechende Nummern.
- **Folgefelder:** Trennt ein Einzelfeld ≥ 97 %, ist es Regel oder Folge der Entscheidung; ist ein Merkmal einer verbundenen
  Tabelle betroffen, fällt die ganze Tabelle heraus (Ersatzteil: Sonderangebote, Preis- und Kostenhistorie).
- **Leitbeispiel Ersatzteil (AdventureWorks):** 325 Stücklisten-Bauteile, 240 verkaufsfähig. Regel 0,965 · LightGBM 0,985 ·
  Text allein 0,974 · mit Stückliste/Graph 1,000 (künstliche Daten – ehrlich markiert). Auto-ET 73,9 %, recall-sicher
  Auto-Negativ 26,2 %, 0 gemessene Fehlalarme; nachweisbare Fehlergrenze mit 240 ET nur 1,2 % (für 0,1 % ≈ 3.000 ET).
  Euro: 1.300 € Pflegeaufwand/Jahr, Bevorratung 25.664 € je ET und Jahr, Prüfgrenze getrennt ausgewiesen.
- **Gesamt:** AdventureWorks 36 Felder (20 ML vorhersagbar, 4 teilweise, 4 kaum, 8 nicht prüfbar), WWI 33 (24 · 3 · 1 · 5);
  Online Retail II, SCANIA, C-MAPSS: keine manuell gepflegten Felder (Sensor-/Transaktionsdaten) – Lückenliste mit Grund.
- **UI:** Panel „Data-Classifier-Analyse“ am Ende von Schritt 3 mit Kennzahlen, Tabelle, Detail je Feld (Modellvergleich,
  Folgefelder, Merkmale, Graubereich, Lückenvorschläge) und editierbaren Euro-Annahmen (Python und JS identisch).
- **Tests:** 76 grün. Web-Modus 232 MB.
