"""
Business Case: Ampel aus drei Dimensionen und Was-wäre-wenn
==========================================================
Datenreife (Checks) · Vorhersagekraft (Rückblick) · Euro-Potenzial (hier).

Euro-Potenzial = Rückblick-Rechnung mit echten Zahlen aus dem Backtest:
wie viele Fälle bzw. wie viel Umsatz die Top-Liste im Testfenster erfasst
hätte – multipliziert mit klar benannten Annahmen (Rettungsquote, Kosten …).

  Gemessen:     wie gut die Vorhersage gewesen wäre.
  Nicht gemessen: ob der Vertrieb reagiert hätte. Beleg erst im Pilot mit Kontrollgruppe.

Die Formeln existieren identisch in templates/graph_ml.html (Live-Schieberegler
ohne Server-Anfrage); tests/test_graph_ml_business.py prüft das PDF-Rechenbeispiel.
"""
from __future__ import annotations

from . import evaluation as ev

FX_DEFAULT = 1.15
EURO_GREEN, EURO_YELLOW = 100_000, 25_000
DISCLAIMER = ("Gemessen: wie gut die Vorhersage gewesen wäre. Nicht gemessen: ob der Vertrieb reagiert hätte. "
              "Den echten Effekt belegt erst der Pilot mit Kontrollgruppe.")

# role: rate (Szenario skaliert ×0,5/×1,5), cost (×1,5/×0,75), fixed
PARAMS: dict[str, list[dict]] = {
    "churn": [
        {"id": "list_share", "label": "Listengröße (oberste % der Kunden)", "unit": "%", "default": 10, "min": 5, "max": 30, "step": 1, "role": "fixed"},
        {"id": "rate", "label": "Rettungsquote", "unit": "%", "default": 25, "min": 5, "max": 60, "step": 1, "role": "rate"},
        {"id": "cost", "label": "Kosten je Kontakt", "unit": "€", "default": 50, "min": 0, "max": 300, "step": 5, "role": "cost"},
        {"id": "margin", "label": "Ergebnisanteil (100 % = Umsatz, sonst Deckungsbeitrag)", "unit": "%", "default": 100, "min": 10, "max": 100, "step": 5, "role": "fixed"},
        {"id": "periods", "label": "Kampagnen pro Jahr (je Quartal eine neue Liste)", "unit": "×", "default": 4, "min": 1, "max": 4, "step": 1, "role": "fixed"}],
    "lead_scoring": [
        {"id": "list_share", "label": "Listengröße (oberste % der Erstkäufer)", "unit": "%", "default": 20, "min": 5, "max": 30, "step": 1, "role": "fixed"},
        {"id": "rate", "label": "Mehrumsatz durch gezielte Betreuung", "unit": "%", "default": 20, "min": 5, "max": 60, "step": 1, "role": "rate"},
        {"id": "cost", "label": "Kosten je betreutem Neukunden (Willkommens-Anruf)", "unit": "€", "default": 30, "min": 0, "max": 300, "step": 5, "role": "cost"},
        {"id": "margin", "label": "Ergebnisanteil", "unit": "%", "default": 100, "min": 10, "max": 100, "step": 5, "role": "fixed"},
        {"id": "periods", "label": "Kohorten pro Jahr", "unit": "×", "default": 4, "min": 1, "max": 4, "step": 1, "role": "fixed"}],
    "cross_selling": [
        {"id": "list_share", "label": "Listengröße (oberste % der Kunde×Gruppe-Paare)", "unit": "%", "default": 10, "min": 5, "max": 30, "step": 1, "role": "fixed"},
        {"id": "rate", "label": "Anteil der Käufe, den das Angebot auslöst", "unit": "%", "default": 20, "min": 5, "max": 60, "step": 1, "role": "rate"},
        {"id": "cost", "label": "Kosten je Angebot (E-Mail/Webshop)", "unit": "€", "default": 1, "min": 0, "max": 20, "step": 0.5, "role": "cost"},
        {"id": "margin", "label": "Ergebnisanteil", "unit": "%", "default": 100, "min": 10, "max": 100, "step": 5, "role": "fixed"},
        {"id": "periods", "label": "Aktionen pro Jahr", "unit": "×", "default": 4, "min": 1, "max": 4, "step": 1, "role": "fixed"}],
    "recommendation": [
        {"id": "coverage", "label": "Kunden mit persönlicher Empfehlung", "unit": "%", "default": 100, "min": 10, "max": 100, "step": 5, "role": "fixed"},
        {"id": "rate", "label": "Anteil der Treffer, den die Empfehlung auslöst", "unit": "%", "default": 10, "min": 2, "max": 40, "step": 1, "role": "rate"},
        {"id": "cost", "label": "Kosten je Kunde und Quartal", "unit": "€", "default": 1, "min": 0, "max": 10, "step": 0.5, "role": "cost"},
        {"id": "margin", "label": "Ergebnisanteil", "unit": "%", "default": 100, "min": 10, "max": 100, "step": 5, "role": "fixed"},
        {"id": "periods", "label": "Quartale pro Jahr", "unit": "×", "default": 4, "min": 1, "max": 4, "step": 1, "role": "fixed"}],
    "returns_root_cause": [
        {"id": "list_share", "label": "Geprüfte Positionen (oberste % nach Risiko)", "unit": "%", "default": 10, "min": 5, "max": 30, "step": 1, "role": "fixed"},
        {"id": "rate", "label": "Vermeidungsquote erkannter Retouren", "unit": "%", "default": 30, "min": 5, "max": 60, "step": 1, "role": "rate"},
        {"id": "cost_return", "label": "Prozesskosten je Retoure", "unit": "€", "default": 15, "min": 5, "max": 50, "step": 1, "role": "fixed"},
        {"id": "value_loss", "label": "Wertverlust je Retoure (Anteil Warenwert)", "unit": "%", "default": 10, "min": 0, "max": 50, "step": 1, "role": "fixed"},
        {"id": "cost", "label": "Prüfkosten je markierter Position (automatisch)", "unit": "€", "default": 0.2, "min": 0, "max": 5, "step": 0.1, "role": "cost"},
        {"id": "periods", "label": "Zeiträume pro Jahr (Testfenster = 2 Monate)", "unit": "×", "default": 6, "min": 1, "max": 6, "step": 1, "role": "fixed"}],
    "account_scoring": [
        {"id": "list_share", "label": "Accounts in der Betreuungsliste (oberste %)", "unit": "%", "default": 10, "min": 5, "max": 30, "step": 1, "role": "fixed"},
        {"id": "rate", "label": "zusätzlich gesicherter Anteil des Wachstums durch Key-Account-Betreuung", "unit": "%", "default": 15, "min": 5, "max": 50, "step": 1, "role": "rate"},
        {"id": "cost", "label": "Kosten je betreutem Account und Quartal", "unit": "€", "default": 150, "min": 0, "max": 1000, "step": 10, "role": "cost"},
        {"id": "margin", "label": "Ergebnisanteil", "unit": "%", "default": 100, "min": 10, "max": 100, "step": 5, "role": "fixed"},
        {"id": "periods", "label": "Quartale pro Jahr", "unit": "×", "default": 4, "min": 1, "max": 4, "step": 1, "role": "fixed"}],
    "territory_coverage": [
        {"id": "list_share", "label": "Besuchskapazität (oberste % der Händler)", "unit": "%", "default": 20, "min": 5, "max": 50, "step": 1, "role": "fixed"},
        {"id": "rate", "label": "Mehrumsatz durch gezielten Besuch (Literatur: Produktivität +5–10 %)", "unit": "%", "default": 7.5, "min": 1, "max": 20, "step": 0.5, "role": "rate"},
        {"id": "cost", "label": "Kosten je Besuch", "unit": "€", "default": 120, "min": 0, "max": 500, "step": 10, "role": "cost"},
        {"id": "margin", "label": "Ergebnisanteil", "unit": "%", "default": 100, "min": 10, "max": 100, "step": 5, "role": "fixed"},
        {"id": "periods", "label": "Besuchsrunden pro Jahr", "unit": "×", "default": 4, "min": 1, "max": 4, "step": 1, "role": "fixed"}],
    "price_recommendation": [
        {"id": "rate", "label": "Anteil der Preislücke, der mit Preisempfehlung im Angebot geschlossen wird", "unit": "%", "default": 30, "min": 5, "max": 80, "step": 5, "role": "rate"},
        {"id": "cost", "label": "Kosten je geprüftem Auftrag", "unit": "€", "default": 5, "min": 0, "max": 50, "step": 1, "role": "cost"},
        {"id": "periods", "label": "Testzeiträume pro Jahr (Test = Halbjahr)", "unit": "×", "default": 2, "min": 1, "max": 4, "step": 1, "role": "fixed"}],
    "price_leakage": [
        {"id": "rate", "label": "Anteil des Leakage, der durch Freigabeprozess vermieden wird", "unit": "%", "default": 40, "min": 5, "max": 80, "step": 5, "role": "rate"},
        {"id": "cost", "label": "Kosten je geprüftem Auftrag", "unit": "€", "default": 10, "min": 0, "max": 50, "step": 1, "role": "cost"},
        {"id": "periods", "label": "Testzeiträume pro Jahr (Test = Halbjahr)", "unit": "×", "default": 2, "min": 1, "max": 4, "step": 1, "role": "fixed"}],
    "quote_check": [
        {"id": "rate", "label": "Anteil der Margenlücke, der in der Angebotsprüfung zurückgeholt wird", "unit": "%", "default": 25, "min": 5, "max": 80, "step": 5, "role": "rate"},
        {"id": "cost", "label": "Kosten je geprüftem Angebot", "unit": "€", "default": 15, "min": 0, "max": 100, "step": 5, "role": "cost"},
        {"id": "periods", "label": "Testzeiträume pro Jahr (Test = Halbjahr)", "unit": "×", "default": 2, "min": 1, "max": 4, "step": 1, "role": "fixed"}],
    "installed_base": [
        {"id": "list_share", "label": "Fahrzeuge mit proaktivem Serviceangebot (oberste %)", "unit": "%", "default": 10, "min": 5, "max": 30, "step": 1, "role": "fixed"},
        {"id": "rate", "label": "Anteil erkannter Reparaturen, die als eigener Werkstattauftrag gewonnen werden", "unit": "%", "default": 30, "min": 5, "max": 70, "step": 1, "role": "rate"},
        {"id": "cost", "label": "Kosten je Serviceangebot (Kontakt, Terminierung)", "unit": "€", "default": 25, "min": 0, "max": 200, "step": 5, "role": "cost"},
        {"id": "margin", "label": "Ergebnisanteil des Werkstattauftrags", "unit": "%", "default": 35, "min": 10, "max": 100, "step": 5, "role": "fixed"},
        {"id": "periods", "label": "Angebotsrunden pro Jahr", "unit": "×", "default": 4, "min": 1, "max": 6, "step": 1, "role": "fixed"}],
    "spare_parts_pricing": [
        {"id": "uplift", "label": "Preisaufschlag auf gebundene Teile", "unit": "%", "default": 3, "min": 1, "max": 10, "step": 0.5, "role": "rate"},
        {"id": "loss", "label": "davon durch Mengenverlust wieder verloren (Elastizität)", "unit": "%", "default": 30, "min": 0, "max": 90, "step": 5, "role": "fixed"},
        {"id": "cost", "label": "Pflegekosten je gebundenem Teil und Halbjahr", "unit": "€", "default": 50, "min": 0, "max": 500, "step": 10, "role": "cost"},
        {"id": "periods", "label": "Testzeiträume pro Jahr (Test = Halbjahr)", "unit": "×", "default": 2, "min": 1, "max": 4, "step": 1, "role": "fixed"}],
    "next_best_action": [
        {"id": "periods", "label": "Kampagnen pro Jahr (eine Aktion je Kunde und Quartal)", "unit": "×", "default": 4, "min": 1, "max": 4, "step": 1, "role": "fixed"}],
    "demand_forecast": [
        {"id": "coverage_weeks", "label": "Reichweite des Bestands", "unit": "Wochen", "default": 8, "min": 2, "max": 26, "step": 1, "role": "fixed"},
        {"id": "safety_share", "label": "Anteil Sicherheitsbestand am Bestand", "unit": "%", "default": 30, "min": 5, "max": 60, "step": 1, "role": "rate"},
        {"id": "holding_rate", "label": "Lagerhaltungskostensatz p. a. (Kapital, Lager, Abschriften)", "unit": "%", "default": 25, "min": 5, "max": 40, "step": 1, "role": "rate"}],
}
SCENARIOS = {"konservativ": {"rate": 0.5, "cost": 1.5}, "realistisch": {"rate": 1.0, "cost": 1.0},
             "optimistisch": {"rate": 1.5, "cost": 0.75}}
STEP_LABELS = {"churn": ("Umsatz der Abwanderer", "davon in der Liste", "gerettet (Annahme)"),
               "lead_scoring": ("Umsatz der Wiederkäufer", "davon in der Liste", "Mehrumsatz (Annahme)"),
               "cross_selling": ("Umsatz in neuen Gruppen", "davon in der Liste", "ausgelöst (Annahme)"),
               "recommendation": ("Umsatz mit empfohlenen Artikeln", "erreichte Kunden", "ausgelöst (Annahme)"),
               "returns_root_cause": ("Kosten aller Retouren", "davon erkannt", "vermieden (Annahme)"),
               "account_scoring": ("Mehrumsatz wachsender Accounts", "davon in der Liste", "gesichert (Annahme)"),
               "territory_coverage": ("Händlerumsatz im Quartal", "davon in der Besuchsliste", "Mehrumsatz (Annahme)"),
               "next_best_action": ("Wert aller Kandidaten-Aktionen", "gewählte Aktionen", "realisiert (Annahme)"),
               "price_recommendation": ("Preislücke zum erwartbaren Preis", "geprüfte Aufträge", "geschlossen (Annahme)"),
               "price_leakage": ("Leakage im Testzeitraum", "geprüfte Aufträge", "vermieden (Annahme)"),
               "quote_check": ("Margenlücke unter 10 %", "geprüfte Angebote", "zurückgeholt (Annahme)"),
               "installed_base": ("Wert aller Reparaturen im Test", "davon in der Liste", "gewonnen (Annahme)"),
               "spare_parts_pricing": ("Umsatz gebundener Teile", "gebundene Teile", "Mehrmarge (Annahme)"),
               "demand_forecast": ("Sicherheitsbestand", "Fehlerreduktion", "Kostensenkung")}
FUNNEL = [
    {"id": "scan", "name": "Light-Scan", "stage": 1, "price": "kostenlos – 2.500 €", "cost": 2_500,
     "what": "Automatischer Check eines Exports, Readiness-Ampel je Use Case"},
    {"id": "sprint", "name": "Readiness-Sprint (2–4 Wochen)", "stage": 2, "price": "8.000 – 20.000 €", "cost": 14_000,
     "what": "Vollständiger Check, Baseline-Vergleich der Modellklassen, Reparaturplan, Business Case"},
    {"id": "pilot", "name": "Pilot & Integration (6–12 Wochen)", "stage": 3, "price": "25.000 – 60.000 €", "cost": 42_500,
     "what": "Datenbereinigung, ein Use Case live in CRM/ERP, Messung mit Kontrollgruppe"},
    {"id": "license", "name": "Plattform-Lizenz", "stage": 3, "price": "1.500 – 4.000 € je Use Case/Monat", "cost": 33_000,
     "what": "Laufende Vorhersagen, Monitoring von Datenqualität und Drift, weitere Use Cases"},
]


def _p(params: dict, key: str, default=0.0) -> float:
    return float(params.get(key, default))


def intervention(value_in_list: float, n_list: int, rate: float, margin: float, cost_per_contact: float,
                 periods: float) -> dict:
    """Grundformel (PDF-Rechenbeispiel): erfasster Wert × Quote × Ergebnisanteil − Kontaktkosten, × Zeiträume."""
    gain = value_in_list * rate * margin
    cost = n_list * cost_per_contact
    return {"gain": gain, "cost": cost, "net": gain - cost, "annual": (gain - cost) * periods}


def compute(uc: str, w: dict, params: dict, engine: str, fx: float = FX_DEFAULT, scen: dict | None = None) -> dict | None:
    """Euro-Potenzial einer Engine für einen Use Case (identisch zur JS-Funktion bcCompute)."""
    if not w or engine not in w.get("engines", {}):
        return None
    scen = scen or {"rate": 1.0, "cost": 1.0}
    fx = fx if w.get("currency", "GBP") == "GBP" else 1.0
    if w.get("kind") in ("leakage", "check"):     # Pricing: Lücke × Quote − geprüfte Aufträge × Kosten (pricing.py)
        e = w["engines"][engine]
        lk = e.get("leak_eur", e.get("gap_eur", 0.0))
        gain, cost = lk * _p(params, "rate", 30) / 100 * scen["rate"], e["n"] * _p(params, "cost") * scen["cost"]
        return {"base": lk, "caught": lk, "gain": gain, "cost": cost, "net": gain - cost,
                "annual": (gain - cost) * _p(params, "periods", 2), "n_list": e["n"], "tp": e["n"]}
    if w.get("kind") == "uplift":                # Ersatzteile: Umsatz gebundener Teile × Aufschlag × (1 − Mengenverlust)
        e = w["engines"][engine]
        gain = e["rev"] * _p(params, "uplift", 3) / 100 * scen["rate"] * (1 - _p(params, "loss", 30) / 100)
        cost = e["n"] * _p(params, "cost") * scen["cost"]
        return {"base": e["rev"], "caught": e["rev"], "gain": gain, "cost": cost, "net": gain - cost,
                "annual": (gain - cost) * _p(params, "periods", 2), "n_list": e["n"], "tp": e["n"]}
    if w.get("kind") == "policy":                # realisierter Netto-Wert der gewählten Aktionen (nba_eval)
        net = w["engines"][engine]["net"] * scen["rate"]
        return {"base": abs(net), "caught": abs(net), "gain": max(net, 0.0), "cost": 0.0, "net": net,
                "annual": net * _p(params, "periods", 4), "n_list": 0, "tp": 0}
    e = w["engines"][engine]
    r = lambda k: _p(params, k) / 100 * scen["rate"]          # noqa: E731
    c = lambda k: _p(params, k) * scen["cost"]                # noqa: E731
    if w["kind"] == "list":
        i = max(0, min(len(w["k"]) - 1, int(round(_p(params, "list_share", 10))) - 1))
        n, tp, val = w["n"][i], e["tp"][i], e["value"][i] * fx
        total = w["value_total"] * fx
        if uc == "returns_root_cause":
            per = _p(params, "cost_return") + 0
            avoided = tp * r("rate") * per + val * r("rate") * _p(params, "value_loss") / 100
            all_cost = w["positives"] * per + total * _p(params, "value_loss") / 100
            caught = tp * per + val * _p(params, "value_loss") / 100
            cost = n * c("cost")
            net = avoided - cost
            return {"base": all_cost, "caught": caught, "gain": avoided, "cost": cost, "net": net,
                    "annual": net * _p(params, "periods", 6), "n_list": n, "tp": tp}
        res = intervention(val, n, r("rate"), _p(params, "margin", 100) / 100, c("cost"), _p(params, "periods", 4))
        return {"base": total, "caught": val, **res, "n_list": n, "tp": tp}
    if w["kind"] == "reco":
        cov = _p(params, "coverage", 100) / 100
        val = e["hit_value"] * fx * cov
        res = intervention(val, int(e["customers"] * cov), r("rate"), _p(params, "margin", 100) / 100,
                           c("cost"), _p(params, "periods", 4))
        return {"base": e["hit_value"] * fx, "caught": val, **res, "n_list": int(e["customers"] * cov),
                "tp": e["hits"]}
    if w["kind"] == "forecast":
        rule = w["engines"].get("rule", {}).get("wape")
        if rule is None:
            return None
        annual_demand = (w["demand_value"] or 0) * fx * 4
        safety = annual_demand * _p(params, "coverage_weeks", 8) / 52 * r("safety_share")
        red = max(0.0, (rule - e["wape"]) / rule) if rule else 0.0
        gain = safety * red * r("holding_rate")
        return {"base": safety, "caught": safety * red, "gain": gain, "cost": 0.0, "net": gain, "annual": gain,
                "reduction": red}
    return None


def defaults(uc: str) -> dict:
    return {p["id"]: p["default"] for p in PARAMS.get(uc, [])}


def euro_status(annual: float | None, green: float = EURO_GREEN, yellow: float = EURO_YELLOW) -> dict:
    if annual is None:
        return {"status": "na", "reason": "nicht berechenbar"}
    st = "ok" if annual >= green else "warn" if annual >= yellow else "bad"
    return {"status": st, "annual": round(annual), "reason": f"{_eur(annual)} pro Jahr (realistisch, netto)"}


def _eur(x: float) -> str:
    return f"{x:,.0f} €".replace(",", ".")


def tornado(uc: str, w: dict, params: dict, engine: str, fx: float) -> list[dict]:
    base = (compute(uc, w, params, engine, fx) or {}).get("annual", 0.0)
    out = []
    for p in PARAMS[uc]:
        lo, hi = dict(params), dict(params)
        lo[p["id"]] = max(p["min"], p["default"] * 0.5) if p["id"] in params else p["default"]
        hi[p["id"]] = min(p["max"], p["default"] * 1.5) if p["id"] in params else p["default"]
        a = (compute(uc, w, lo, engine, fx) or {}).get("annual", 0.0)
        b = (compute(uc, w, hi, engine, fx) or {}).get("annual", 0.0)
        out.append({"param": p["id"], "label": p["label"], "low": round(a - base), "high": round(b - base)})
    return sorted(out, key=lambda x: -abs(x["high"] - x["low"]))


def use_case_case(uc: str, variant: str = "v0", fx: float = FX_DEFAULT) -> dict | None:
    res = ev.load(uc, variant)
    if not res or not res.get("whatif"):
        return None
    w = res["whatif"]
    params = defaults(uc)
    model = res.get("best")
    rec = res.get("recommendation", {}).get("engine") or model
    out = {"use_case": uc, "variant": variant, "whatif": w, "params": PARAMS[uc], "defaults": params,
           "model": model, "model_name": next((e["meta"]["name"] for e in res["engines"] if e["id"] == model), model),
           "recommended": rec, "labels": STEP_LABELS[uc], "currency": w.get("currency"), "fx": fx,
           "dataset": res.get("dataset", "online_retail_ii")}
    if out["dataset"] in ("wwi", "adventureworks_data"):   # künstliche Beispieldaten
        name = {"wwi": "WideWorldImporters", "adventureworks_data": "AdventureWorks"}[out["dataset"]]
        out["note"] = (f"Gerechnet auf {name} (künstliche Beispieldaten) – die Methode ist übertragbar, "
                       "die Euro-Größenordnung nicht.")
    elif out["dataset"] == "scania":
        out["note"] = ("Gerechnet auf SCANIA Component X (echte, anonymisierte Lkw-Flotte) – der Auftragswert je Reparatur "
                       f"({_eur(res.get('aftersales', {}).get('repair_eur', 2500))}) ist eine Annahme, SCANIA veröffentlicht keine Preise.")
    out["scenarios"] = {}
    for name, sc in SCENARIOS.items():
        m = compute(uc, w, params, model, fx, sc) if model else None
        r_ = compute(uc, w, params, "rule", fx, sc)
        added = (m["annual"] - r_["annual"]) if m and r_ else None
        if w.get("kind") in ("leakage", "check", "uplift"):
            added = None                  # mehr gefundene Lücke ≠ besser – die Regel markiert mehr Fehlalarme
        out["scenarios"][name] = {"model": m, "rule": r_, "added": added}
    real = out["scenarios"]["realistisch"]
    out["euro"] = euro_status(real["model"]["annual"] if real["model"] else None)
    out["tornado"] = tornado(uc, w, params, model, fx) if model else []
    return out


def overview(variant: str = "v0", fx: float = FX_DEFAULT) -> dict:
    """Euro-Potenzial aller messbaren Use Cases + Trichter-Empfehlung."""
    cases = {uc: use_case_case(uc, variant, fx) for uc in ev.UC_ORDER + ev.POLICY_UCS}
    for uc, why in ev.NOT_MEASURABLE.items():
        cases[uc] = {"use_case": uc, "euro": {"status": "na", "reason": why}}
    for uc in ev.POLICY_UCS:                     # Rückblick ohne Was-wäre-wenn (Serviceverträge): Euro nicht bewertbar
        r = ev.load(uc, variant) if cases.get(uc) is None else None
        if r and r.get("vorhersagekraft", {}).get("status") == "na":
            cases[uc] = {"use_case": uc, "dataset": r.get("dataset"), "euro": {"status": "na", "reason": r["vorhersagekraft"]["reason"]}}
    return {"variant": variant, "fx": fx, "cases": cases, "funnel": FUNNEL, "next_step": next_step(cases),
            "disclaimer": DISCLAIMER, "thresholds": {"green": EURO_GREEN, "yellow": EURO_YELLOW}}


def next_step(cases: dict, vk: dict | None = None) -> dict:
    """Empfehlung für den nächsten Schritt: größter belastbarer Hebel gegen die Kosten von Sprint + Pilot + Lizenz."""
    ranked = []
    for uc, c in cases.items():
        if not c or c.get("euro", {}).get("status") in (None, "na"):
            continue
        if c.get("dataset", "online_retail_ii") != "online_retail_ii":
            continue                     # nächster Schritt je Datensatz – keine Werte aus anderen Daten mischen
        ranked.append((c["euro"].get("annual", 0) or 0, uc))
    ranked.sort(reverse=True)
    if not ranked or ranked[0][0] <= 0:
        return {"stage": "scan", "text": "Light-Scan: zuerst Datenreife herstellen – noch kein positiver Hebel."}
    top_val, top = ranked[0]
    year1 = FUNNEL[1]["cost"] + FUNNEL[2]["cost"] + FUNNEL[3]["cost"]
    roi = top_val / year1
    stage = "pilot" if roi >= 2 else "sprint"
    names = {"churn": "Churn & Retention", "lead_scoring": "Neukunden-Scoring", "cross_selling": "Cross-Selling",
             "recommendation": "Produkt-Empfehlung", "returns_root_cause": "Retouren-Root-Cause",
             "demand_forecast": "Absatzprognose", "account_scoring": "Account-Scoring",
             "territory_coverage": "Vertriebsabdeckung", "next_best_action": "Next Best Action",
             "price_recommendation": "Preis- & Rabattempfehlung", "price_leakage": "Preis-Leakage", "quote_check": "Angebotsprüfung",
             "installed_base": "Installierte Basis", "service_contracts": "Serviceverträge", "spare_parts_pricing": "Ersatzteilpreise"}
    step = "Pilot" if stage == "pilot" else "Readiness-Sprint"
    roi_txt = str(round(roi, 1)).replace(".", ",")
    return {"stage": stage, "use_case": top, "annual": round(top_val), "year1_cost": year1, "roi": round(roi, 1),
            "text": (f"{step} für {names.get(top, top)}: Potenzial {_eur(top_val)} pro Jahr gegenüber "
                     f"{_eur(year1)} Kosten im ersten Jahr (Sprint + Pilot + Lizenz) – ROI-Indikator {roi_txt}×.")}


def merge_ampel(u: dict, case: dict | None) -> None:
    """Gesamtampel = schwächste belastbare Dimension (Datenreife, Vorhersagekraft, Euro-Potenzial)."""
    rank = {"ok": 0, "warn": 1, "bad": 2}
    euro = (case or {}).get("euro") or {"status": "pending"}
    u["ampel"]["euro"] = euro
    dims = [d.get("status") for d in (u["ampel"]["datenreife"], u["ampel"]["vorhersagekraft"], euro)]
    real = [d for d in dims if d in rank]
    worst = max(real, key=lambda x: rank[x]) if real else "na"
    names = {0: "Datenreife", 1: "Vorhersagekraft", 2: "Euro-Potenzial"}
    weakest = [names[i] for i, d in enumerate(dims) if d == worst]
    complete = all(d in rank or d == "na" for d in dims)
    u["ampel"]["gesamt"] = {"status": worst, "provisional": not complete,
                            "note": ("Schwächste Dimension: " + ", ".join(weakest)) if real else "nicht bewertbar"}
