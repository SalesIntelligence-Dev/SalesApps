"""
Lernintuition je Use Case – was jede Stufe sieht, wie die Graph-Verbindung aussieht und ob der Graph hilft
=========================================================================================================
  Inhalte für die Use-Case-Detailseite (Abschnitte 2 und 3) und die Übersicht in Schritt 5:

  * Daten: was eine Zeile ist, was ins Modell hineingeht und was herauskommt
  * Lernintuition: je Stufe (Regel → LightGBM → TabPFN → LightGBM + Graph-Merkmale → GNN → relationales FM) in einem
    Satz, was das Modell sieht – dazu der gemessene Wert aus dem Rückblick-Artefakt
  * Graph-Verbindung: kleines Beispiel (erfundene Werte), wie das Signal über andere Datensätze läuft
  * Urteil „Graph hilft?“ und „Tabular FM hilft?“: aus den Messwerten berechnet (bestes Graph-Modell bzw. TabPFN gegen
    LightGBM ohne Graph), dazu eine fachliche Erklärung – ausdrücklich als Hypothese gekennzeichnet

  Die Werte sind Punktwerte auf dem Testfenster, auf dem auch das beste Modell gewählt wird; ein Signifikanztest gegen
  LightGBM fehlt. Das steht in der Oberfläche bei jedem Urteil.
"""
from __future__ import annotations

from . import evaluation as ev

FAMILY_OF = {"rule": "rule", "popularity": "rule", "single": "rule", "check": "check", "lgbm": "tree", "tabpfn": "tfm",
             "lgbm_graph": "light", "itemknn": "light", "lgbm_rank": "light", "nba": "policy", "gnn": "gnn", "gnn_rank": "gnn",
             "lightgcn": "gnn", "kumorfm": "rfm", "aurora": "tsfm", "cascade": "cascade"}
FAMILY_NAME = {"rule": "Regel", "tree": "LightGBM", "tfm": "TabPFN (Tabular FM)", "light": "LightGBM + Graph-Merkmale",
               "gnn": "Graph-Modell (GNN)", "rfm": "Relationales FM (KumoRFM)", "tsfm": "Zeitreihen-FM (Aurora)",
               "policy": "Policy mit Modell-Scores", "check": "Regelprüfung / Kalkulation", "cascade": "Kaskade je Feld"}
ORDER = ["rule", "tree", "tfm", "light", "gnn", "rfm", "tsfm", "policy", "cascade", "check"]
GRAPH_FAMS = ("light", "gnn")

METRICS = {
    "auprc": ("AUPRC", "Wie sauber ist die Rangliste? Man geht die Liste von oben durch und misst, welcher Anteil der Einträge "
              "wirklich zutrifft – gemittelt über alle Listenlängen. Zufälliges Sortieren ergibt die Grundquote der Fälle "
              "(bei 3 % Fällen also 0,03), perfektes Sortieren 1,0.", "höher ist besser"),
    "recall_at_10": ("Recall@Top-10 %", "Welcher Anteil aller echten Fälle steht in den oberen 10 % der Liste? 0,40 heißt: Mit einer "
                     "Liste von 10 % der Datensätze erreicht man 40 % aller späteren Fälle. Zufall ergibt 0,10.", "höher ist besser"),
    "value_captured_at_10": ("Umsatz-Recall@Top-10 %", "Wie Recall@Top-10 %, aber nach Umsatz gewichtet: Welcher Anteil des "
                             "betroffenen Umsatzes steckt in den oberen 10 % der Liste? Große Kunden zählen mehr.", "höher ist besser"),
    "map_at_12": ("MAP@12", "Wie viele der 12 empfohlenen Artikel kauft der Kunde danach wirklich, und wie weit oben stehen sie? "
                  "Ein Treffer auf Platz 1 zählt mehr als auf Platz 12. 0 = kein Treffer, 1 = alle Käufe ganz oben.", "höher ist besser"),
    "wape": ("WAPE", "Gewichteter Prognosefehler: Summe der absoluten Abweichungen geteilt durch die tatsächliche Menge. "
             "0,25 heißt: Die Prognose liegt im Schnitt 25 % daneben.", "kleiner ist besser"),
    "mae": ("MAE", "Mittlere absolute Abweichung des vorhergesagten vom tatsächlich erzielten Preis, in Anteilen vom "
            "Listenpreis. 0,01 heißt: im Schnitt 1 Prozentpunkt daneben.", "kleiner ist besser"),
    "net_value": ("Netto-Wert", "Was hätten die gewählten Aktionen im Rückblick-Quartal eingebracht? Ob ein Kunde danach kaufte, ist "
                  "gemessen; die Wirkung der Aktion ist eine Annahme.", "höher ist besser"),
    "auto_98": ("automatisch entscheidbar", "Anteil der Einträge, die bei mindestens 98 % Trefferquote ohne Mensch gesetzt werden "
                "könnten – gemittelt über die geprüften Felder.", "höher ist besser"),
}
CHECK_METRIC = ("keine Prognose-Metrik", "Hier wird nichts vorhergesagt: Es ist eine Regelprüfung bzw. Kalkulation mit bekannten "
                "Größen.", "")

DISCLAIMER = ("Die Erklärung ist eine Hypothese, die zum Messergebnis passt, nicht separat belegt. Die Werte sind Punktwerte "
              "aus einem Testfenster (dort wird auch das beste Modell gewählt); ein Signifikanztest gegen LightGBM fehlt.")


def _g(nodes, edges, caption):
    """Beispiel-Graph: nodes = (id, Beschriftung, Spalte, Rolle); Rolle ∈ focus, entity, risk, ok, peer, target."""
    return {"nodes": [{"id": i, "label": lab, "col": c, "role": r} for i, lab, c, r in nodes],
            "edges": [{"source": a, "target": b, "label": lab} for a, b, lab in edges], "caption": caption}


# Je Use Case: Zeile/Input/Output, was die Stufen sehen (Regel kommt aus dem Artefakt), Beispiel-Graph und Erklärung
INTUITION: dict[str, dict] = {
    "churn": {
        "row": "eine Zeile je aktivem Kunden zum Stichtag (Kauf in den letzten 180 Tagen)",
        "input": "Kundenzeile (Recency, Frequenz, Umsatz, Sortiment, Treue) + Kunde–Artikel-Graph der 365 Tage vor dem Stichtag",
        "output": "Wahrscheinlichkeit, dass der Kunde in den nächsten 90 Tagen nichts kauft → Anrufliste der oberen 10 %",
        "sees": {"tree": "eine Zeile je Kunde mit Aggregaten der eigenen Belege (Recency, Frequenz, Umsatz, Bestellrhythmus, Stornos)",
                 "tfm": "dieselbe Zeile; schließt aus ähnlichen Kunden im Kontext, ohne Training",
                 "light": "dieselbe Zeile plus fest vorgegebene Aggregate über Kunde → Artikel → andere Kunden (Anteil inaktiver "
                          "Nachbarn, PageRank, Sortimentsprofil)",
                 "gnn": "dieselben Rohdaten als Kunde–Artikel-Graph; die Verdichtung über die Nachbarn lernt das Netz selbst "
                        "(2 Sprünge, 64 Größen je Knoten)",
                 "rfm": "die verknüpften Tabellen direkt; vortrainiert, lernt aus Beispielen im Kontext (nicht gerechnet: kein API-Schlüssel)"},
        "graph": _g([("A", "Kunde A\n(Prognose?)", 0, "focus"), ("X", "Weihnachtsdeko X", 1, "entity"), ("Y", "Kerzenhalter Y", 1, "entity"),
                     ("B", "Kunde B\ninaktiv", 2, "risk"), ("C", "Kunde C\naktiv", 2, "ok"), ("D", "Kunde D\ninaktiv", 2, "risk")],
                    [("A", "X", "kauft"), ("A", "Y", "kauft"), ("X", "B", "auch gekauft von"), ("X", "C", ""), ("Y", "C", ""), ("Y", "D", "")],
                    "Anteil inaktiver Nachbarn über gemeinsame Artikel: (1 + 1) / (2 + 2) = 0,50 – C zählt doppelt, weil er zwei "
                    "Artikel mit A teilt. LightGBM ohne Graph sieht nur die Zeile von A."),
        "why": "Die Kunden sind überwiegend Wiederverkäufer. Wer dieselben Nischenartikel bezieht (z. B. Weihnachtsdeko), hängt an "
               "denselben Trends und bricht oft gemeinsam weg. Das steht in keiner einzelnen Kundenzeile."},
    "lead_scoring": {
        "row": "eine Zeile je Neukunde nach dem Erstkauf",
        "input": "Erstkauf (Wert, Positionen, Artikel, Preis, Storno) + Erstkäufer-Artikel-Graph",
        "output": "Wahrscheinlichkeit, dass der Neukunde in 90 Tagen wieder kauft → Liste für Neukunden-Betreuung",
        "sees": {"tree": "eine Zeile je Neukunde mit den Kennzahlen des Erstkaufs",
                 "tfm": "dieselbe Zeile; schließt aus ähnlichen Erstkäufen im Kontext",
                 "light": "dieselbe Zeile plus Wiederkaufquote früherer Erstkäufer derselben Artikel, Wiederkaufquote im Land, "
                          "Beliebtheit der Artikel",
                 "gnn": "der Erstkauf als Knoten im Kunde–Artikel-Graph; lernt, was frühere Käufer derselben Artikel taten"},
        "graph": _g([("N", "Neukunde N\n(Wiederkauf?)", 0, "focus"), ("P", "Artikel P", 1, "entity"), ("Q", "Artikel Q", 1, "entity"),
                     ("E1", "Erstkäufer 1\nkam wieder", 2, "ok"), ("E2", "Erstkäufer 2\nkam wieder", 2, "ok"), ("E3", "Erstkäufer 3\nnicht", 2, "risk")],
                    [("N", "P", "Erstkauf"), ("N", "Q", "Erstkauf"), ("P", "E1", "Erstkauf von"), ("P", "E2", ""), ("Q", "E3", "")],
                    "Wiederkaufquote ähnlicher Erstkäufer: 2 von 3 = 0,67 – der Ersatz für die fehlende eigene Historie."),
        "why": "Kaltstart: Ein Neukunde hat außer dem Erstkauf keine Historie, ähnliche Erstkäufer derselben Artikel sind der naheliegende "
               "Ersatz. Gemessen bringt das wenig: Die festen Graph-Merkmale verschlechtern LightGBM, das GNN liegt knapp vorn. "
               "Vermutlich bestimmt die Art des Erstkaufs (Händler- oder Privatbestellung) den Wiederkauf stärker als die Artikel."},
    "cross_selling": {
        "row": "eine Zeile je Kunde × noch nicht gekaufter Artikelgruppe",
        "input": "Kunde (Umsatz, Breite), Gruppe (Beliebtheit, Wachstum), Co-Kauf-Affinität zwischen Gruppen",
        "output": "Wahrscheinlichkeit, dass der Kunde in 90 Tagen erstmals in dieser Gruppe kauft → Angebotsliste",
        "sees": {"tree": "eine Zeile je Kunde × Gruppe mit Kennzahlen von Kunde und Gruppe",
                 "tfm": "dieselbe Zeile; schließt aus ähnlichen Paaren im Kontext",
                 "light": "dieselbe Zeile plus Co-Kauf-Affinität: Wie oft kaufen Kunden der eigenen Gruppen auch die Zielgruppe?",
                 "gnn": "Kunden-Einbettung aus dem Kaufgraphen + Gruppe + Paar-Merkmale; lernt die Affinität selbst"},
        "graph": _g([("A", "Kunde A", 0, "focus"), ("K", "Gruppe Kerzen\n(gekauft)", 1, "entity"), ("P1", "Kunden\nder Gruppe Kerzen", 2, "peer"),
                     ("L", "Gruppe Laternen\n(Kauf?)", 3, "target")],
                    [("A", "K", "kauft"), ("K", "P1", "auch gekauft von"), ("P1", "L", "60 % kaufen auch")],
                    "Co-Kauf-Affinität: 60 % der Kerzen-Käufer kaufen auch Laternen – A hat noch keine Laternen gekauft."),
        "why": "Die Co-Kauf-Affinität zwischen Gruppen ist ein echtes Graph-Signal. Bei nur wenigen Artikelgruppen ist sie aber fast eine "
               "feste Tabelle, die LightGBM über Beliebtheit und Wachstum der Gruppe schon abbildet; das GNN verliert deutlich. Ob "
               "Graph ML auf Artikelebene hilft (Linkvorhersage statt weniger Gruppen), ist offen."},
    "recommendation": {
        "row": "eine Zeile je Kunde × Kandidaten-Artikel (Kandidaten aus Co-Kauf-Graph, eigenen Artikeln, Topsellern)",
        "input": "eigene Käufe, Aktualität, Co-Kauf-Score, Beliebtheit, Einbettungs-Ähnlichkeit",
        "output": "12 Artikel je Kunde, die er in den nächsten 30 Tagen am wahrscheinlichsten kauft",
        "sees": {"light": "Kandidaten und Ähnlichkeit aus dem Co-Kauf-Graphen aller Kunden („wer X kaufte, kaufte auch Z“), "
                          "bewertet von LightGBM bzw. ItemKNN",
                 "gnn": "Kunden- und Artikel-Einbettungen aus dem Kaufgraphen, gelernt (GNN-Ranker, LightGCN)"},
        "graph": _g([("A", "Kunde A", 0, "focus"), ("X", "Artikel X\n(gekauft)", 1, "entity"), ("B", "Kunde B", 2, "peer"),
                     ("C", "Kunde C", 2, "peer"), ("Z", "Artikel Z\n(Empfehlung)", 3, "target")],
                    [("A", "X", "kauft"), ("X", "B", "auch gekauft von"), ("X", "C", ""), ("B", "Z", "kauft"), ("C", "Z", "kauft")],
                    "2 Sprünge: A → X → Kunden mit X → Z. Je mehr Nachbarn Z kaufen, desto höher Z in A's Liste."),
        "why": "Empfehlung ist der klassische Graph-Fall: „Kunden, die X kauften, kauften auch Z“ läuft immer über andere Kunden. Die "
               "Siegerin nutzt Kandidaten und Ähnlichkeit aus dem Co-Kauf-Graphen; ein Vergleich mit einem Modell ganz ohne Graph "
               "fehlt, deshalb ist die Regel die Messlatte. LightGCN nutzt nur die Kaufkanten ohne Wiederkauf- und Aktualitätssignal "
               "und verliert deshalb vermutlich."},
    "returns_root_cause": {
        "row": "eine Zeile je Verkaufsposition",
        "input": "Menge, Preis gegenüber üblich, Positionen im Beleg, Kundenhistorie + Retourenquoten von Artikel, Nachbarartikeln, Kunde",
        "output": "Wahrscheinlichkeit, dass die Position binnen 30 Tagen storniert wird → Prüfliste",
        "sees": {"tree": "eine Zeile je Position mit Menge, Preis, Belegumfang und Kundenhistorie",
                 "tfm": "dieselbe Zeile inklusive Graph-Merkmale; schließt aus ähnlichen Positionen im Kontext",
                 "light": "dieselbe Zeile plus Retourenquote des Artikels, seiner Nachbarartikel (gleiche Käufer) und des Kunden"},
        "graph": _g([("P", "Position\n(Storno?)", 0, "focus"), ("X", "Artikel X\nQuote 8 %", 1, "entity"), ("K", "Kunde K\nQuote 20 %", 1, "risk"),
                     ("N", "Nachbarartikel\nQuote 15 %", 2, "risk")],
                    [("P", "X", "Artikel"), ("P", "K", "Kunde"), ("X", "N", "gleiche Käufer")],
                    "Drei Quoten, die in der Position selbst nicht stehen: Artikel, Nachbarartikel über gemeinsame Käufer, Kunde."),
        "why": "Retouren hängen an Artikel und Kunde zugleich: Ein Artikel, den dieselben Kunden wie retourenanfällige Artikel kaufen, "
               "wird eher zurückgeschickt, und wer oft storniert, tut es wieder. Diese Quoten stehen in keiner Position selbst; "
               "einfache Graph-Merkmale reichen, ein GNN wurde hier nicht gerechnet."},
    "demand_forecast": {
        "row": "eine Zeile je Artikel × Prognosewoche (13 Wochen voraus)",
        "input": "Vorwochen, Vorjahreswoche, gleitende Mittel, Kalender + Trend der Geschwister-Artikel und Co-Kauf-Nachbarn",
        "output": "Menge je Artikel und Woche → Sicherheitsbestand",
        "sees": {"tree": "die eigene Absatzreihe des Artikels als Verzögerungs-Merkmale",
                 "light": "dieselbe Reihe plus Ø und Trend der Geschwister derselben Gruppe und der 5 ähnlichsten Co-Kauf-Artikel",
                 "tsfm": "nur die eigene Tagesreihe, zero-shot (nicht gerechnet: Endpoint nicht gesetzt)"},
        "graph": _g([("X", "Artikel X\n(Menge?)", 0, "focus"), ("G", "Gruppe Deko", 1, "entity"), ("S1", "Geschwister 1\nTrend ↑", 2, "ok"),
                     ("S2", "Geschwister 2\nTrend ↑", 2, "ok"), ("N", "Co-Kauf-Nachbar\nTrend →", 2, "peer")],
                    [("X", "G", "gehört zu"), ("G", "S1", "enthält"), ("G", "S2", ""), ("X", "N", "oft zusammen gekauft")],
                    "Steigt der Absatz der Geschwister-Artikel, kündigt das auch für X mehr Absatz an."),
        "why": "Artikel derselben Gruppe teilen Saison und Trend, ein steigender Gruppentrend kündigt auch beim einzelnen Artikel mehr "
               "Absatz an. Der Effekt ist klein: In der Prognose tragen die eigenen Vorwochen das meiste."},
    "account_scoring": {
        "row": "eine Zeile je aktivem Account (WideWorldImporters) zum Stichtag",
        "input": "Umsatz, Frequenz, Trend, Kreditlimit, Kategorie + Wachstum der Einkaufsgemeinschaft, Kategorie und Region",
        "output": "Wahrscheinlichkeit, dass der Quartalsumsatz um ≥ 20 % wächst → Betreuungsliste",
        "sees": {"tree": "eine Zeile je Account mit Umsatz, Frequenz, Trend und Stammdaten",
                 "tfm": "dieselbe Zeile; bei wenigen hundert Accounts schließt TabPFN aus ähnlichen Accounts",
                 "light": "dieselbe Zeile plus Wachstum der Peers: Einkaufsgemeinschaft, Kundenkategorie, Region (ohne den Account)",
                 "gnn": "Account im Kunde–Artikel-Graph; lernt aus den Nachbarn"},
        "graph": _g([("A", "Account A\n(Wachstum?)", 0, "focus"), ("EG", "Einkaufsgemeinschaft", 1, "entity"), ("P1", "Peer 1\n+30 %", 2, "ok"),
                     ("P2", "Peer 2\n+25 %", 2, "ok"), ("P3", "Peer 3\n−5 %", 2, "risk")],
                    [("A", "EG", "Mitglied"), ("EG", "P1", ""), ("EG", "P2", ""), ("EG", "P3", "")],
                    "Wachstum der Einkaufsgemeinschaft ohne A selbst: Ø +17 %."),
        "why": "Accounts einer Einkaufsgemeinschaft könnten gemeinsam wachsen (gemeinsame Konditionen und Aktionen). Gemessen trägt das "
               "kaum: Die festen Peer-Merkmale bringen nichts, GNN und TabPFN liegen gleichauf knapp vor LightGBM. Bei wenigen "
               "hundert Accounts ist das eher ein Fall für ein Tabellenmodell; WideWorldImporters ist zudem künstlich."},
    "territory_coverage": {
        "row": "eine Zeile je Händler zum Quartalsstichtag (AdventureWorks)",
        "input": "Recency, Frequenz, Jahresumsatz, Bestellrhythmus + Anteil der Portfolio- und Gebiets-Händler mit frischem Auftrag",
        "output": "Wahrscheinlichkeit eines Auftrags im nächsten Quartal, gewichtet mit dem Umsatz → Besuchsliste",
        "sees": {"tree": "eine Zeile je Händler mit der eigenen Bestellhistorie",
                 "tfm": "dieselbe Zeile; schließt aus ähnlichen Händlern",
                 "light": "dieselbe Zeile plus Anteil der Händler desselben Vertrieblers und Gebiets mit frischem Auftrag",
                 "gnn": "Händler im Kunde–Artikel-Graph; lernt aus den Nachbarn"},
        "graph": _g([("H", "Händler H\n(Auftrag?)", 0, "focus"), ("V", "Vertriebler", 1, "entity"), ("T", "Gebiet", 1, "entity"),
                     ("P1", "Portfolio-Händler\nfrischer Auftrag", 2, "ok"), ("P2", "Portfolio-Händler\nkein Auftrag", 2, "risk"),
                     ("G1", "Gebiets-Händler\nfrischer Auftrag", 2, "ok")],
                    [("H", "V", "betreut von"), ("H", "T", "liegt in"), ("V", "P1", ""), ("V", "P2", ""), ("T", "G1", "")],
                    "Besuchsrunde: Hat der Vertriebler gerade das Portfolio besucht, steigt die Auftragschance auch für H."),
        "why": "Arbeiten Vertriebler in Besuchsrunden, sagt ein frischer Auftrag im Portfolio etwas über den nächsten Händler. In "
               "AdventureWorks bestellen Händler aber in festen Rhythmen: Die eigene Historie trägt, das Portfolio- und Gebiets-"
               "Signal verschlechtert LightGBM sogar."},
    "next_best_action": {
        "row": "eine Zeile je Kunde mit den Scores der Einzel-Use-Cases",
        "input": "Churn-, Cross-Selling-, Empfehlungs- und Neukunden-Scores mit erwartetem Wert und Kosten je Aktion",
        "output": "eine Aktion je Kunde und Quartal (Anruf, Angebot, Empfehlung, Newsletter oder keine)",
        "sees": {"policy": "die Scores der anderen Use Cases (LightGBM + Graph-Merkmale) und eine Regel zur Konfliktauflösung"},
        "graph": _g([("K", "Kunde", 0, "focus"), ("S1", "Churn-Score", 1, "entity"), ("S2", "Cross-Selling-Score", 1, "entity"),
                     ("S3", "Empfehlungs-Score", 1, "entity"), ("A", "eine Aktion\n(höchster Netto-Wert)", 2, "target")],
                    [("K", "S1", ""), ("K", "S2", ""), ("K", "S3", ""), ("S1", "A", ""), ("S2", "A", ""), ("S3", "A", "")],
                    "Die Aktion mit dem höchsten erwarteten Netto-Wert gewinnt; Graph-Signale wirken nur über die Scores."),
        "why": "Next Best Action ist eine Entscheidungsregel über die Scores der anderen Use Cases. Der Graph wirkt nur indirekt über "
               "diese Scores; für ein gelerntes NBA-Modell bräuchte es Reaktionsdaten auf frühere Aktionen."},
    "price_recommendation": {
        "row": "eine Zeile je Händler-Auftragsposition (AdventureWorks)",
        "input": "Artikel, Kategorie, Menge und Mengenstaffel, Listenpreis, Gebiet + Peer-Preise der letzten 365 Tage",
        "output": "erzielbarer Nettopreis in % vom Listenpreis (der gewährte Rabatt ist die Entscheidung und kein Merkmal)",
        "sees": {"tree": "eine Zeile je Position mit Artikel, Menge, Staffel, Listenpreis und Gebiet",
                 "tfm": "dieselbe Zeile; schätzt die Preisstaffel aus ähnlichen Positionen",
                 "light": "dieselbe Zeile plus Preise der Peers: gleicher Artikel im Gebiet, Händlerhistorie, Vertriebler-Portfolio"},
        "graph": _g([("P", "Position\n(Preis?)", 0, "focus"), ("X", "Artikel X", 1, "entity"), ("R", "Händler R", 1, "entity"),
                     ("H1", "Peer-Händler im Gebiet\n92 % vom Liste", 2, "peer"), ("H2", "Peer-Händler im Gebiet\n88 % vom Liste", 2, "peer")],
                    [("P", "X", "Artikel"), ("P", "R", "Händler"), ("X", "H1", "auch gekauft von"), ("X", "H2", "")],
                    "Peer-Preis: Ø 90 % vom Listenpreis für denselben Artikel im selben Gebiet."),
        "why": "Was Peers für denselben Artikel zahlen, ist intuitiv wichtig. Es ist aber ein einfacher Mittelwert, den LightGBM über "
               "Artikel, Menge und Gebiet schon fast sieht; die Peer-Merkmale bringen nichts. TabPFN schätzt die Preisstaffel genauer."},
    "price_leakage": {
        "row": "eine Zeile je Händler-Auftragsposition – mit dem erwartbaren Preis aus der Preisempfehlung",
        "input": "wie Preisempfehlung; Leakage = erzielter Preis mehr als 3 Prozentpunkte unter dem erwartbaren",
        "output": "Positionen mit Leakage und Lücke in Euro → Freigabe-Prüfliste",
        "sees": {"tree": "das Preismodell der Preisempfehlung (LightGBM)", "tfm": "das Preismodell der Preisempfehlung (TabPFN)",
                 "light": "das Preismodell der Preisempfehlung mit Peer-Preisen"},
        "graph": _g([("P", "Position\n12 % vom Liste", 0, "focus"), ("E", "erwartbar 60 %\n(Peers, Staffel)", 1, "entity"),
                     ("L", "Leakage\n48 Pp.", 2, "risk")],
                    [("P", "E", "verglichen mit"), ("E", "L", "Lücke")],
                    "Leakage hängt am Preismodell: erwartbarer Preis minus erzielter Preis."),
        "why": "Leakage ist die Abweichung vom erwartbaren Preis aus der Preisempfehlung. Gemessen ist deshalb nur die Preisgenauigkeit "
               "(MAE) des zugrunde liegenden Modells, nicht wie gut Leakage-Fälle erkannt werden. Der Graph hilft dabei so wenig wie dort."},
    "quote_check": {
        "row": "eine Zeile je Auftrag mit seinen Positionen",
        "input": "Nettopreis, Standardkosten zum Auftragsdatum, Stücklisten-Tiefe, Sonderangebot",
        "output": "Aufträge mit Marge unter 10 % → Prüfliste",
        "sees": {"check": "Nettopreis minus Standardkosten zum Datum – eine Rechnung, kein Modell"},
        "graph": _g([("A", "Angebot", 0, "focus"), ("K", "Konfiguration", 1, "entity"), ("B1", "Bauteil 1\nKosten", 2, "peer"),
                     ("B2", "Bauteil 2\nKosten", 2, "peer")],
                    [("A", "K", ""), ("K", "B1", "Stückliste"), ("K", "B2", "")],
                    "Kosten = Summe der Bauteilkosten entlang der Stückliste."),
        "why": "Mit bekannten Standardkosten ist die Marge eine Rechnung, kein Prognoseproblem. Der Stücklisten-Pfad wird nur gebraucht, "
               "um Kosten zu summieren; Graph ML hilft erst, wenn Kosten im Angebot fehlen und geschätzt werden müssen."},
    "installed_base": {
        "row": "eine Zeile je Fahrzeug und Werkstatt-Auslesung (SCANIA), bis zu 10 je Fahrzeug",
        "input": "Sensor-Zähler und -Histogramme zum Schnitt, Zuwachs seit der letzten Auslesung, Laufzeit + Konfiguration",
        "output": "Wahrscheinlichkeit einer Reparatur der Komponente X in 48 Zeiteinheiten → Liste für Serviceangebote",
        "sees": {"tree": "eine Zeile je Auslesung mit den eigenen Sensorwerten und ihrem Zuwachs",
                 "tfm": "dieselbe Zeile (100 Merkmale mit der größten Streuung); schließt aus ähnlichen Auslesungen",
                 "light": "dieselbe Zeile plus Reparaturquote der Fahrzeuge mit gleicher Konfiguration (nur Trainingsfahrzeuge)",
                 "gnn": "Fahrzeug-Schnitt ↔ Konfigurationswerte als Graph; lernt aus gleich konfigurierten Fahrzeugen"},
        "graph": _g([("V", "Fahrzeug V\n(Reparatur?)", 0, "focus"), ("C", "Konfiguration\nspec_2 = Cat5", 1, "entity"),
                     ("F1", "Fahrzeug\nrepariert", 2, "risk"), ("F2", "Fahrzeug\nnicht repariert", 2, "ok"), ("F3", "Fahrzeug\nnicht repariert", 2, "ok")],
                    [("V", "C", "hat"), ("C", "F1", "auch bei"), ("C", "F2", ""), ("C", "F3", "")],
                    "Reparaturquote gleich konfigurierter Fahrzeuge: 1 von 3."),
        "why": "Gleich konfigurierte Fahrzeuge könnten ähnlich ausfallen. In SCANIA erklärt die Konfiguration die Reparaturen aber kaum; "
               "das Signal steckt in den eigenen Sensor-Auslesungen und ihrem Verlauf. Peer-Quoten und GNN verschlechtern LightGBM."},
    "service_contracts": {
        "row": "eine Zeile je Konfigurationssegment",
        "input": "erwartete Reparaturquote je Konfiguration aus dem Modell der installierten Basis",
        "output": "Risikoprämie je Fahrzeug (Reparaturquote × Auftragswert × 1,2)",
        "sees": {"check": "eine Kalkulation aus dem Reparaturmodell – ohne Vertragsdaten kein eigenes Label"},
        "graph": _g([("V", "Vertrag", 0, "focus"), ("F", "Fahrzeug", 1, "entity"), ("C", "Konfiguration", 2, "entity"),
                     ("R", "Reparaturrisiko", 3, "target")],
                    [("V", "F", ""), ("F", "C", ""), ("C", "R", "")],
                    "Die Prämie folgt dem erwarteten Servicebedarf der Konfiguration."),
        "why": "Ohne Vertragsdaten gibt es nichts zu lernen; die Prämie je Konfiguration kommt aus dem Reparaturmodell der installierten "
               "Basis. Mit Verträgen je Seriennummer wäre die Verlängerung ein Churn-artiger Fall."},
    "spare_parts_pricing": {
        "row": "eine Zeile je Komponente × Quartal (AdventureWorks)",
        "input": "Menge Vorquartal, Ø vier Quartale, Vorjahresquartal + installierte Basis über die Stückliste",
        "output": "Absatz der Komponente im Folgequartal; gebundene Teile tragen den Preisaufschlag",
        "sees": {"tree": "die eigene Absatzreihe des Teils",
                 "light": "dieselbe Reihe plus installierte Basis: verkaufte Räder × Stückmenge laut Stückliste"},
        "graph": _g([("T", "Ersatzteil\n(Bedarf?)", 0, "focus"), ("S", "Stückliste", 1, "entity"), ("R1", "Rad-Modell A\n1.200 verkauft", 2, "ok"),
                     ("R2", "Rad-Modell B\n300 verkauft", 2, "peer")],
                    [("T", "S", "verbaut in"), ("S", "R1", "2 Stück je Rad"), ("S", "R2", "1 Stück je Rad")],
                    "Installierte Basis: 1.200 × 2 + 300 × 1 = 2.700 verbaute Teile – das steht in keiner Zeile der Teile-Historie."),
        "why": "Der Teilebedarf folgt der installierten Basis: Wie viele Räder mit diesem Teil verkauft wurden, steht nur über Stückliste "
               "und Verkaufsaufträge. Fachlich ist das der überzeugendste Graph-Pfad; der Effekt ist aber klein."},
    "quality_root_cause": {
        "row": "eine Zeile je Fertigungsauftrag (AdventureWorks)",
        "input": "Auftragsmenge, Fertigungstage, Kosten, Arbeitsgänge, bisherige Ausschussquote + Stückliste und Bauteil-Lieferanten",
        "output": "Wahrscheinlichkeit von Ausschuss → Prüfung vor dem Start; Ursachen nach Grund und Produktgruppe",
        "sees": {"tree": "eine Zeile je Auftrag mit Menge, Produkt und bisheriger Ausschussquote",
                 "tfm": "dieselbe Zeile inklusive Graph-Merkmale",
                 "light": "dieselbe Zeile plus Ausschuss der Geschwister-Produkte, der Produkte mit gleichen Bauteilen und "
                          "Beanstandungsquote der Bauteil-Lieferanten",
                 "gnn": "Auftrag ↔ Produkt ↔ Unterkategorie als Graph"},
        "graph": _g([("A", "Fertigungsauftrag\n(Ausschuss?)", 0, "focus"), ("P", "Produkt", 1, "entity"), ("B", "Bauteil", 2, "entity"),
                     ("L", "Lieferant\nQuote 12 %", 3, "risk")],
                    [("A", "P", ""), ("P", "B", "Stückliste"), ("B", "L", "geliefert von")],
                    "Idee: Schlechte Bauteil-Lieferanten pflanzen sich als Ausschuss fort."),
        "why": "Die Idee: Schlechte Bauteil-Lieferanten pflanzen sich als Ausschuss in die Fertigung fort. In AdventureWorks entsteht "
               "Ausschuss aber nur bei großen Aufträgen und ist sonst zufällig; kein Modell schlägt die Mengen-Regel belastbar."},
    "supplier_recourse": {
        "row": "eine Zeile je Bestellposition beim Lieferanten (AdventureWorks)",
        "input": "Bestellmenge, Positionswert, Stückpreis, Lieferzeit, Bonität + bisherige Beanstandungsquote des Lieferanten (nur "
                 "Vergangenheit) + Quote anderer Lieferanten mit gleichen Bauteilen",
        "output": "Wahrscheinlichkeit einer Beanstandung beim Wareneingang (zum Bestellzeitpunkt) → Prüfliste",
        "sees": {"tree": "eine Zeile je Position mit Menge, Preis und der eigenen Lieferantenhistorie",
                 "tfm": "dieselbe Zeile; schließt vorsichtig aus wenigen Beanstandungen ähnlicher Positionen im Kontext",
                 "light": "dieselbe Zeile plus Quote desselben Bauteils bei anderen Lieferanten und der Lieferanten-Peers",
                 "gnn": "Position ↔ Lieferant ↔ Bauteil als Graph"},
        "graph": _g([("P", "Bestellposition\n(Beanstandung?)", 0, "focus"), ("L", "Lieferant L\nbisher 16 %", 1, "risk"), ("B", "Bauteil 908", 1, "entity"),
                     ("L2", "anderer Lieferant\nbisher 5 %", 2, "ok"), ("L3", "anderer Lieferant\nbisher 4 %", 2, "ok")],
                    [("P", "L", "bestellt bei"), ("P", "B", "Bauteil"), ("B", "L2", "auch geliefert von"), ("B", "L3", "")],
                    "Geglättete Quote: (Beanstandungen + 10 × Ø) / (Positionen + 10) – bei wenig Historie nah am Durchschnitt."),
        "why": "Das Signal steckt in der eigenen Historie des Lieferanten, nicht bei anderen Lieferanten mit gleichen Bauteilen. TabPFN "
               "nutzt die wenigen Beanstandungen am besten. Bauteil-Merkmale wie Materialgruppe oder Erstmuster fehlen in AdventureWorks. "
               "Das Label ist „Ware beanstandet“, nicht „Regress durchgesetzt“."},
    "predictive_maintenance": {
        "row": "eine Zeile je Triebwerk und Betriebszyklus (C-MAPSS), bis zu 10 je Triebwerk",
        "input": "21 Sensoren, Ø und Änderung über 10 Zyklen, Zyklenzahl + Sensoren relativ zu Peers gleicher Betriebsbedingung",
        "output": "Wahrscheinlichkeit eines Ausfalls in 30 Zyklen → Inspektionsliste",
        "sees": {"tree": "eine Zeile je Zyklus mit den eigenen Sensorwerten und ihrem Verlauf",
                 "tfm": "dieselbe Zeile inklusive Peer-Normierung",
                 "light": "dieselbe Zeile plus Sensoren relativ zum Normalwert der Peers unter derselben Bedingung und Ausfallquote "
                          "der Flotte im gleichen Alter",
                 "gnn": "Schnitt ↔ Triebwerk ↔ Bedingung ↔ Flotte als Graph"},
        "graph": _g([("S", "Triebwerk T\nZyklus 180 (Ausfall?)", 0, "focus"), ("C", "Betriebsbedingung 3", 1, "entity"), ("F", "Flotte FD002", 1, "entity"),
                     ("P1", "Peer-Triebwerke\nNormalwert", 2, "peer"), ("P2", "Flotten-Peers\nLebensdauer", 2, "peer")],
                    [("S", "C", "läuft unter"), ("S", "F", "gehört zu"), ("C", "P1", ""), ("F", "P2", "")],
                    "Sensor relativ zum Normalwert der Peers unter derselben Bedingung – „normal“ hängt von der Bedingung ab."),
        "why": "Was „normal“ ist, hängt von der Betriebsbedingung ab; die Normierung gegen Peer-Triebwerke hilft LightGBM ein wenig. Das "
               "eigentliche Signal ist der Sensorverlauf des Triebwerks selbst; ein GNN über Triebwerke und Bedingungen verliert "
               "deutlich. C-MAPSS ist eine Simulation."},
    "spare_parts_demand": {
        "row": "eine Zeile je Lagerartikel × Monat (WideWorldImporters)",
        "input": "Vormonat, Ø 3 und 12 Monate, Vorjahresmonat, Trend, Preis + Trend der Warengruppe und Co-Kauf-Nachbarn",
        "output": "Bedarf im Folgemonat → Sicherheitsbestand und Bestandsampel",
        "sees": {"tree": "die eigene Bedarfsreihe des Artikels",
                 "light": "dieselbe Reihe plus Trend der Warengruppen-Geschwister und der Artikel derselben Kunden"},
        "graph": _g([("A", "Lagerartikel\n(Bedarf?)", 0, "focus"), ("W", "Warengruppe", 1, "entity"), ("K", "Kunden des Artikels", 1, "entity"),
                     ("S", "Geschwister\nTrend ↑", 2, "peer"), ("N", "übrige Artikel\nder Kunden", 2, "peer")],
                    [("A", "W", ""), ("A", "K", "gekauft von"), ("W", "S", ""), ("K", "N", "kaufen auch")],
                    "Trend der Geschwister und der Co-Kauf-Artikel als Zusatzsignal."),
        "why": "Artikel derselben Warengruppe oder derselben Kunden könnten sich gemeinsam bewegen. Gemessen nicht: Der eigene Verlauf "
               "trägt, die Graph-Signale verschlechtern die Prognose leicht."},
    "win_probability": {
        "row": "eine Zeile je Rückstandsauftrag (WideWorldImporters)",
        "input": "Auftragswert, Menge, Kunde, Kreditlimit + offene Rückstände desselben Artikels, Lieferantenquote, offener Nachschub",
        "output": "Wahrscheinlichkeit, dass der Rückstand nicht binnen 30 Tagen fakturiert wird; Pipeline-Umsatz je Woche",
        "sees": {"tree": "eine Zeile je Auftrag mit Wert, Menge und Kundenhistorie",
                 "tfm": "dieselbe Zeile inklusive Graph-Merkmale",
                 "light": "dieselbe Zeile plus offene Rückstände desselben Artikels, Quote des Lieferanten, offene Einkaufsbestellungen",
                 "gnn": "Auftrag ↔ Artikel ↔ Kunde ↔ Lieferant als Graph"},
        "graph": _g([("O", "Rückstand\n(Umsatz gefährdet?)", 0, "focus"), ("I", "Artikel", 1, "entity"), ("R", "andere offene\nRückstände", 2, "risk"),
                     ("S", "Lieferant", 2, "entity"), ("PO", "offene\nEinkaufsbestellung", 2, "ok")],
                    [("O", "I", ""), ("I", "R", "wartet auch"), ("I", "S", "geliefert von"), ("I", "PO", "Nachschub")],
                    "Druck auf den Bestand: Wie viele warten noch auf denselben Artikel, kommt Nachschub?"),
        "why": "Ob ein Rückstand zu Umsatz wird, hängt am Nachschub des Artikels; das wäre ein Graph-Signal. In WideWorldImporters folgt "
               "der Generator aber fast einer Mengenregel (ab etwa 50 Stück bleibt der Rückstand liegen), alle Modelle liegen nahe der "
               "Obergrenze. Ein Datenleck wurde geprüft: Die Merkmale nutzen nur, was zum Auftragsdatum feststand."},
    "data_prediction": {
        "row": "eine Zeile je Datensatz mit dem manuell gepflegten Feld (z. B. je Stücklisten-Bauteil)",
        "input": "übrige Felder der Zeile, verbundene Stammdaten, Stückliste (Vater, Geschwister), Benennung",
        "output": "vorgeschlagener Wert je Feld, automatisch bei ≥ 98 % Sicherheit, sonst Graubereich zum Menschen",
        "sees": {"rule": "häufigster Wert je bestem Einzelfeld (Stufe 0)",
                 "cascade": "je Feld das beste aus LightGBM, LightGBM + Verknüpfungen, Text-Modell, TabPFN und Ensemble"},
        "graph": _g([("T", "Bauteil\n(Ersatzteil?)", 0, "focus"), ("V", "Vater-Baugruppe", 1, "entity"), ("G1", "Geschwister\nErsatzteil", 2, "ok"),
                     ("G2", "Geschwister\nkein Ersatzteil", 2, "risk"), ("L", "Lieferant", 1, "entity")],
                    [("T", "V", "Stückliste"), ("V", "G1", "enthält auch"), ("V", "G2", ""), ("T", "L", "geliefert von")],
                    "Kontext aus der Stückliste: Was sind die Geschwister in derselben Baugruppe?"),
        "why": "Ein manuell gepflegtes Feld hängt oft am Kontext: Ob ein Bauteil Ersatzteil ist, verraten Vater-Baugruppe, Geschwister und "
               "Benennung. Welche Information trägt, ist je Feld verschieden; die Kaskade entscheidet je Feld."},
    # nicht gemessen – Struktur
    "margin": {"row": "eine Zeile je Position", "input": "Verkaufspreis, Einstandspreis", "output": "Deckungsbeitrag",
               "graph": _g([("P", "Position", 0, "focus"), ("X", "Artikel", 1, "entity"), ("K", "Peer-Kunden", 2, "peer")],
                           [("P", "X", ""), ("X", "K", "Preis der Peers")], "Peer-Preise über Artikel."),
               "why": "Ohne Einstandspreis keine Marge; gemessen wird das in Preisempfehlung, Leakage und Angebotsprüfung."},
    "warranty_root_cause": {"row": "eine Zeile je Maschine", "input": "Seriennummer, Charge, Lieferant, Garantiefälle", "output": "Garantiefall",
                            "graph": _g([("M", "Maschine", 0, "focus"), ("C", "Charge", 1, "entity"), ("M2", "Maschinen\nderselben Charge", 2, "risk")],
                                        [("M", "C", ""), ("C", "M2", "auch in")], "Gleiche Charge, ähnliche Ausfälle."),
                            "why": "Fehler pflanzen sich über Charge und Lieferant fort – ein klassisches Graph-Signal; ohne offene Garantiedaten nicht gemessen."},
    "warranty_fraud": {"row": "eine Zeile je Garantiefall", "input": "Händler, Techniker, Teile, Kosten", "output": "auffällig ja/nein",
                       "graph": _g([("F", "Garantiefall", 0, "focus"), ("H", "Händler", 1, "entity"), ("T", "Techniker", 1, "entity"),
                                    ("F2", "weitere Fälle\nselber Ring", 2, "risk")],
                                   [("F", "H", ""), ("F", "T", ""), ("H", "F2", ""), ("T", "F2", "")], "Betrugsringe wiederholen Händler, Techniker und Teile."),
                       "why": "Betrug tritt in Ringen auf – Muster über mehrere Sprünge, die nur ein Graph sieht; ohne offene Daten nicht gemessen."},
    "first_time_fix": {"row": "eine Zeile je Serviceauftrag", "input": "Fehlercode, Anlage, Techniker", "output": "benötigte Teile",
                       "graph": _g([("S", "Serviceauftrag", 0, "focus"), ("E", "Fehlercode", 1, "entity"), ("P", "frühere Aufträge\n→ Teile", 2, "peer")],
                                   [("S", "E", ""), ("E", "P", "gleicher Code")], "Teile früherer Aufträge mit gleichem Fehlercode."),
                       "why": "Welche Teile ein Einsatz braucht, zeigt das Muster ähnlicher Fälle; ohne offene Serviceaufträge nicht gemessen."},
    "service_triage": {"row": "eine Zeile je Servicemeldung", "input": "Fehlertext, Code", "output": "Kategorie/Lösung",
                       "graph": _g([("M", "Meldung", 0, "focus"), ("T", "Fehlertext", 1, "entity"), ("L", "ähnliche Meldungen\n→ Lösung", 2, "peer")],
                                   [("M", "T", ""), ("T", "L", "ähnlich")], "Ähnliche Texte, frühere Lösungen."),
                       "why": "Der Text trägt das Signal – ein Sprachmodell-Thema; der Graph hilft wenig. Ohne offene Texte nicht gemessen."},
}


def _rel(a, b, lower):
    if a is None or b in (None, 0):
        return None
    return (b - a) / b if lower else (a - b) / abs(b)


def _verdict(delta, basis):
    if delta is None:
        return {"status": "na", "label": "nicht verglichen", "delta": None, "basis": basis}
    if delta >= 0.10:
        s, lab = "ok", "ja, deutlich"
    elif delta >= 0.03:
        s, lab = "warn", "leicht (nicht abgesichert)"
    elif delta > -0.03:
        s, lab = "bad", "nein – gleichauf"
    else:
        s, lab = "bad", "nein – schlechter"
    return {"status": s, "label": lab, "delta": round(delta, 4), "basis": basis}


def _data_prediction_graph():
    from . import classifier_scan as cs
    art = cs.load("adventureworks_data")
    if not art:
        return None
    fields = [r for r in art["results"] if r.get("engines") and r.get("status") != "na" and not r.get("inherited")]
    graph = [r for r in fields if r.get("best") == "lgbm_graph"]
    tfm = [r for r in fields if r.get("best") == "tabpfn"]
    return {"n": len(fields), "graph": len(graph), "tfm": len(tfm), "ensemble": sum(1 for r in fields if r.get("best") == "ensemble"),
            "text": sum(1 for r in fields if r.get("best") == "text"), "tree": sum(1 for r in fields if r.get("best") == "lgbm")}


def for_use_case(uc: str) -> dict:
    base = INTUITION.get(uc, {})
    r = ev.load(uc)
    out = {"uc": uc, "row": base.get("row"), "input": base.get("input"), "output": base.get("output"),
           "graph": base.get("graph"), "why": base.get("why"), "disclaimer": DISCLAIMER, "measured": bool(r)}
    if not r:
        out.update({"stages": [], "graph_verdict": {"status": "na", "label": "nicht gemessen", "delta": None, "basis": ""},
                    "tfm_verdict": {"status": "na", "label": "nicht gemessen", "delta": None, "basis": ""}, "metric": None})
        return out
    m, lower = r["main_metric"], bool(r.get("lower_is_better"))
    met = METRICS.get(m, CHECK_METRIC if r.get("kind") == "check" else (r.get("main_metric_label", m), "", ""))
    out["metric"] = {"key": m, "name": met[0], "explain": met[1], "direction": met[2], "label": r.get("main_metric_label")}
    best = {}
    for e in r["engines"]:
        fam = FAMILY_OF.get(e["id"], "rule")
        v = (e.get("metrics") or {}).get(m) if e.get("status") == "ok" else None
        cur = best.get(fam)
        better = cur is None or (v is not None and (cur["value"] is None or (v < cur["value"] if lower else v > cur["value"])))
        if better:
            best[fam] = {"family": fam, "engine": e["id"], "name": e["meta"]["name"], "value": v, "status": e.get("status")}
    sees = base.get("sees", {})
    rule_v = (best.get("rule") or {}).get("value")
    tree_v = (best.get("tree") or {}).get("value")
    stages = []
    for fam in ORDER:
        if fam not in best and fam not in sees:
            continue
        b = best.get(fam, {"family": fam, "engine": None, "name": "nicht gerechnet", "value": None, "status": "na"})
        txt = sees.get(fam) or (r.get("rule_text") if fam == "rule" else "")
        if fam == "rule" and r.get("rule_text"):
            txt = f"eine feste Regel: {r['rule_text']}"
        stages.append({**b, "family_name": FAMILY_NAME.get(fam, fam), "sees": txt,
                       "vs_rule": _rel(b["value"], rule_v, lower) if fam != "rule" else None,
                       "vs_tree": _rel(b["value"], tree_v, lower) if fam not in ("rule", "tree") else None})
    vals = [s for s in stages if s["value"] is not None]
    win = (min(vals, key=lambda s: s["value"]) if lower else max(vals, key=lambda s: s["value"])) if vals and r.get("kind") != "check" else None
    for s in stages:
        s["best"] = bool(win) and s is win
    out["stages"] = stages
    out["winner"] = {"family": win["family"], "family_name": win["family_name"], "name": win["name"], "value": win["value"]} if win else None
    kind = r.get("kind")
    gvals = [best[f] for f in GRAPH_FAMS if f in best and best[f]["value"] is not None]
    g_best = (min(gvals, key=lambda b: b["value"]) if lower else max(gvals, key=lambda b: b["value"])) if gvals else None
    if kind in ("check", "policy"):
        gv = {"status": "na", "label": "kein Modellvergleich", "delta": None, "basis": "Regelprüfung bzw. Policy"}
        tv = dict(gv)
    elif kind == "classifier":
        dp = _data_prediction_graph() or {}
        share = (dp.get("graph", 0) / dp["n"]) if dp.get("n") else None
        gv = {"status": "warn" if share and share >= 0.25 else "bad",
              "label": f"je Feld: Graph-Kontext allein gewinnt bei {dp.get('graph', 0)}, das Ensemble aus Text und Verknüpfungen bei "
                       f"{dp.get('ensemble', 0)} von {dp.get('n', 0)} geprüften Feldern", "delta": share, "basis": "AdventureWorks, Anteil der Felder"}
        tv = {"status": "warn" if dp.get("tfm") else "bad", "label": f"je Feld: TabPFN gewinnt bei {dp.get('tfm', 0)} von {dp.get('n', 0)} Feldern",
              "delta": (dp.get("tfm", 0) / dp["n"]) if dp.get("n") else None, "basis": "AdventureWorks, Anteil der Felder"}
    elif not lower and tree_v is not None and tree_v >= 0.98:
        gv = {"status": "na", "label": "nicht aussagekräftig – alle Modelle nahe der Obergrenze", "delta": _rel(g_best and g_best["value"], tree_v, lower),
              "basis": "ggü. LightGBM"}
        tv = {**gv, "delta": _rel((best.get("tfm") or {}).get("value"), tree_v, lower)}
    else:
        if tree_v is not None:
            gv = _verdict(_rel(g_best["value"], tree_v, lower) if g_best else None, "bestes Graph-Modell ggü. LightGBM ohne Graph")
        else:
            gv = _verdict(_rel(g_best["value"], rule_v, lower) if g_best else None, "bestes Graph-Modell ggü. Regel (LightGBM ohne Graph fehlt)")
        if g_best:
            gv["model"] = g_best["name"]
        tfm_v = (best.get("tfm") or {}).get("value")
        tv = _verdict(_rel(tfm_v, tree_v, lower) if tfm_v is not None else None, "TabPFN ggü. LightGBM")
        if tfm_v is None:
            tv["label"] = "nicht gerechnet"
    vk = r.get("vorhersagekraft") or {}
    if vk.get("status") == "bad":
        for v in (gv, tv):
            if v.get("status") != "na":
                v["note"] = "kein Modell schlägt die Regel belastbar"
    out["graph_verdict"], out["tfm_verdict"] = gv, tv
    out["dataset"] = r.get("dataset") or "online_retail_ii"
    return out


def overview() -> dict:
    from .usecases import USE_CASES
    rows = {u["id"]: for_use_case(u["id"]) for u in USE_CASES}
    return {"use_cases": rows, "disclaimer": DISCLAIMER, "family_names": FAMILY_NAME}
