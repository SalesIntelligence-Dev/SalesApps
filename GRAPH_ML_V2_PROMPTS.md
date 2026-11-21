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
