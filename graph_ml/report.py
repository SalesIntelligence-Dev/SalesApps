"""
Readiness-Report (Modul 6)
==========================
Druckoptimierte HTML-Seite (/graph_ml/report, im Browser „als PDF speichern“) und Excel-Mappe
(/graph_ml/report.xlsx) aus den vorberechneten Artefakten – ohne Neuberechnung, Render-tauglich.

Inhalt: Zusammenfassung · Ampel je Use Case (drei Dimensionen) · Prüfergebnisse · Modellvergleich ·
Reparaturplan mit Aufwand · Business Case mit Annahmen · Aktivierung & Monitoring · nächster Schritt.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd

from . import business_case as bc
from . import evaluation as ev
from . import playbooks as pb

TEMPLATE = Path(__file__).parent / "report_template.html"
ORDER = {"bad": 0, "warn": 1, "ok": 2}
STATUS_TEXT = {"ok": "Grün", "warn": "Gelb", "bad": "Rot", "na": "nicht messbar", None: "–"}
SOURCES = [
    ("Gartner: Lack of AI-Ready Data", "https://www.gartner.com/en/newsroom/press-releases/2025-02-26-lack-of-ai-ready-data-puts-ai-projects-at-risk"),
    ("Fey et al.: Relational Deep Learning", "https://arxiv.org/pdf/2312.04615"),
    ("NVIDIA: RelBench", "https://docs.nvidia.com/sdgm/research/relbench.md"),
    ("Beyond IID: How General Are Tabular Foundation Models, Really? (NeurIPS 2026)", "https://neurips.cc/virtual/2026/poster/139490"),
    ("Productize: Test von NVIDIA Kumo", "https://productize.life/blog/nvidia-kumo-ai-test/en"),
    ("Online Retail II, UCI Machine Learning Repository (CC BY 4.0)", "https://archive.ics.uci.edu/dataset/502/online+retail+ii"),
]


def de_int(x) -> str:
    try:
        return f"{float(x):,.0f}".replace(",", ".")
    except Exception:          # auch Jinja-Undefined
        return "–"


def de_eur(x) -> str:
    v = de_int(x)
    return "–" if v == "–" else f"{v} €"


def de_pct(x, d: int = 0) -> str:
    try:
        return f"{float(x) * 100:.{d}f} %".replace(".", ",")
    except Exception:          # auch Jinja-Undefined
        return "–"


def de_num(x, d: int = 3) -> str:
    try:
        return f"{float(x):.{d}f}".replace(".", ",")
    except Exception:          # auch Jinja-Undefined
        return "–"


def _weakest(*dims) -> str:
    st = [d for d in dims if d in ORDER]
    return min(st, key=ORDER.get) if st else "na"


def _recommended_vid() -> str | None:
    p = pb.ARTIFACTS / "pipeline_recommended.json"
    return json.loads(p.read_text())["variant"] if p.exists() else None


def collect(variant: str = "v0") -> dict:
    from .data_loader import get_registry
    reg = get_registry()
    facts = reg.overview("online_retail_ii")["dataset"]["facts"]
    checks = reg.checks("online_retail_ii", 3)
    dims = {d["id"]: d for d in checks["dimensions"]}
    rec = _recommended_vid()
    bc0 = bc.overview("v0")
    bc1 = bc.overview(rec) if rec else None
    pipe = pb.load_variant(rec) if rec else None
    after = {c["id"]: c for c in (pipe or {}).get("comparison", [])}

    ampel, check_rows = [], []
    for u in checks["use_cases"]:
        a = u.get("ampel") or {}
        dr, vk, eu, ge = (a.get("datenreife") or {}), (a.get("vorhersagekraft") or {}), (a.get("euro") or {}), (a.get("gesamt") or {})
        v1 = None
        if pipe:
            r1 = ev.load(u["id"], rec) if u["id"] in ev.UC_ORDER else None
            d1 = (after.get(u["id"]) or {}).get("after", {})
            vk1 = (r1 or {}).get("vorhersagekraft", {}).get("status", "na") if r1 else "na"
            eu1 = (bc1["cases"].get(u["id"]) or {}).get("euro", {}).get("status", "na") if bc1 else "na"
            v1 = {"score": d1.get("score"), "datenreife": d1.get("status"), "vorhersagekraft": vk1, "euro": eu1,
                  "gesamt": _weakest(d1.get("status"), vk1, eu1),
                  "annual": (bc1["cases"].get(u["id"]) or {}).get("euro", {}).get("annual") if bc1 else None}
        ampel.append({"id": u["id"], "name": u["name"], "area": u["area"], "lighthouse": u.get("lighthouse"),
                      "label": u["label"], "max_stage": u.get("max_stage"),
                      "datenreife": dr, "vorhersagekraft": vk, "euro": eu, "gesamt": ge, "v1": v1,
                      "blockers": u.get("blockers") or []})
        for c in u["checks"]:
            check_rows.append({"uc": u["name"], "dim": dims[c["dim"]]["name"], "status": c["status"],
                               "score": c["score"], "reason": c["reason"],
                               "actions": "; ".join(f"{x['text']}" + (f" ({x['playbook']})" if x.get("playbook") else "")
                                                    for x in c.get("actions") or [])})

    models = []
    for uc in ev.UC_ORDER:
        r = ev.load(uc)
        if not r:
            continue
        m = r["main_metric"]
        engines = [{"name": e["meta"]["name"], "klass": e["meta"]["klass"], "status": e["status"],
                    "value": (e.get("metrics") or {}).get(m), "device": (e.get("meta") or {}).get("device"),
                    "best": e["id"] == r["best"]} for e in r["engines"]]
        models.append({"uc": uc, "name": next((u["name"] for u in checks["use_cases"] if u["id"] == uc), uc),
                       "metric": r["main_metric_label"], "lower": r.get("lower_is_better"), "engines": engines,
                       "vk": r["vorhersagekraft"], "recommendation": r["recommendation"]["text"],
                       "timeline": r.get("timeline")})

    cat = {c["id"]: c for c in pb.catalog()}
    plan, lo, hi = [], 0.0, 0.0
    for s in pb.recommended_steps() + [{"id": "PB-09"}, {"id": "PB-10"}]:
        c = cat[s["id"]]
        eff = next((e for e in (pipe or {}).get("effects", []) if e.get("id") == s["id"]), {})
        opt = next((o for o in c.get("options") or [] if o["id"] == s.get("option", c.get("default"))), None)
        plan.append({"id": c["id"], "title": c["title"], "problem": c["problem"], "option": opt["label"] if opt else "",
                     "effort": c["effort"], "effort_note": c.get("effort_note"),
                     "effect": (f"{de_int(eff['rows_removed'])} Zeilen entfernt" if eff.get("rows_removed") else
                                f"Zuordnung {de_pct(eff['match_rate'])}" if eff.get("match_rate") else
                                f"{len(eff.get('groups') or [])} Gruppen" if eff.get("groups") else "")})
        lo += c["effort_days"][0]
        hi += c["effort_days"][1]
    req = pb.data_request(reg.checks("online_retail_ii", 2))

    cases = []
    for uc, c in bc0["cases"].items():
        if not c or c.get("euro", {}).get("status") in (None, "na"):
            continue
        sc = c.get("scenarios") or {}
        cases.append({"uc": uc, "name": next((u["name"] for u in checks["use_cases"] if u["id"] == uc), uc),
                      "model": c.get("model_name"), "euro": c["euro"],
                      "konservativ": ((sc.get("konservativ") or {}).get("model") or {}).get("annual"),
                      "realistisch": ((sc.get("realistisch") or {}).get("model") or {}).get("annual"),
                      "optimistisch": ((sc.get("optimistisch") or {}).get("model") or {}).get("annual"),
                      "added": (sc.get("realistisch") or {}).get("added"),
                      "v1": ((bc1 or {}).get("cases", {}).get(uc) or {}).get("euro", {}).get("annual") if bc1 else None,
                      "assumptions": [f"{p['label']}: {de_num((c.get('defaults') or {}).get(p['id'], p['default']), 0 if float(p['default']).is_integer() else 1)} {p['unit']}"
                                      for p in bc.PARAMS.get(uc, [])]})
    cases.sort(key=lambda x: -(x["euro"].get("annual") or 0))

    from . import activation as act
    a = act.summary()
    m = act.load_monitoring()
    counts = {s: sum(1 for x in ampel if x["gesamt"].get("status") == s) for s in ("ok", "warn", "bad")}
    best = max(cases, key=lambda x: x["euro"].get("annual") or 0) if cases else None
    return {
        "date": pd.Timestamp.now().strftime("%Y-%m-%d"), "date_de": pd.Timestamp.now().strftime("%d.%m.%Y"),
        "variant": variant, "recommended_variant": rec,
        "dataset": {"name": "Online Retail II", "rows": facts["rows"], "customers": facts["customers"],
                    "period": f"{facts['date_min'][:10]} bis {facts['date_max'][:10]}", "months": facts["months"],
                    "without_customer": facts["share_without_customer"], "invoices": facts["invoices"],
                    "countries": facts["countries"]},
        "stage_note": checks.get("stage_note"), "counts": counts, "ampel": ampel, "checks": check_rows,
        "models": models, "plan": plan, "plan_effort": (lo, hi), "data_request": req["sections"],
        "cases": cases, "best_case": best, "next_step": bc0["next_step"], "funnel": bc0["funnel"],
        "disclaimer": bc.DISCLAIMER, "fx": bc0["fx"], "thresholds": bc0["thresholds"],
        "activation": a, "monitoring": m, "sources": SOURCES, "status_text": STATUS_TEXT,
    }


def render_html(data: dict) -> str:
    from flask import render_template_string
    return render_template_string(TEMPLATE.read_text(encoding="utf-8"), d=data, de_int=de_int, de_eur=de_eur,
                                  de_pct=de_pct, de_num=de_num, st=STATUS_TEXT)


def to_xlsx(data: dict) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    wb = Workbook()
    head, fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor="4C3FB0")
    colors = {"ok": "D1FAE5", "warn": "FEF3C7", "bad": "FEE2E2"}

    def table(ws, header, rows, status_cols=()):
        ws.append(header)
        for c in ws[1]:
            c.font, c.fill = head, fill
        for r in rows:
            ws.append(r)
            for ci in status_cols:
                cell = ws.cell(row=ws.max_row, column=ci + 1)
                key = {v: k for k, v in STATUS_TEXT.items()}.get(cell.value)
                if key in colors:
                    cell.fill = PatternFill("solid", fgColor=colors[key])
        ws.freeze_panes = "A2"
        for col in ws.columns:
            w = min(60, max(10, *(len(str(c.value or "")) for c in col[:300])))
            ws.column_dimensions[col[0].column_letter].width = w + 1
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(vertical="top", wrap_text=isinstance(c.value, str) and len(c.value) > 60)

    ws = wb.active
    ws.title = "Zusammenfassung"
    ds = data["dataset"]
    rows = [("Report", "GraphReady · Readiness-Report"), ("Erstellt", data["date_de"]),
            ("Datensatz", f"{ds['name']} · {de_int(ds['rows'])} Positionen · {de_int(ds['customers'])} Kunden · {ds['period']}"),
            ("Ampel gesamt", f"{data['counts']['ok']} Grün · {data['counts']['warn']} Gelb · {data['counts']['bad']} Rot"),
            ("Größter Hebel", f"{data['best_case']['name']}: {de_eur(data['best_case']['euro'].get('annual'))} pro Jahr"
             if data["best_case"] else "–"),
            ("Nächster Schritt", data["next_step"]["text"]),
            ("Reparaturplan", f"{len(data['plan'])} Playbooks · {de_num(data['plan_effort'][0], 1)}–{de_num(data['plan_effort'][1], 1)} PT"),
            ("Hinweis", data["disclaimer"])]
    for k, v in rows:
        ws.append([k, v])
        ws.cell(row=ws.max_row, column=1).font = Font(bold=True)
    ws.column_dimensions["A"].width = 20
    ws.column_dimensions["B"].width = 120

    st = STATUS_TEXT
    table(wb.create_sheet("Ampel"), ["Use Case", "Bereich", "Max. Stufe", "Datenreife", "Score", "Vorhersagekraft", "Lift",
                                     "Euro-Potenzial", "€ / Jahr", "Gesamt", "Begründung", "Gesamt nach Playbooks (v1)"],
          [[a["name"], a["area"], a["max_stage"], st.get(a["datenreife"].get("status")), a["datenreife"].get("score"),
            st.get(a["vorhersagekraft"].get("status")), a["vorhersagekraft"].get("lift"),
            st.get(a["euro"].get("status")), a["euro"].get("annual"), st.get(a["gesamt"].get("status")),
            a["gesamt"].get("note"), st.get((a["v1"] or {}).get("gesamt")) if a["v1"] else "–"] for a in data["ampel"]],
          status_cols=(3, 5, 7, 9, 11))
    table(wb.create_sheet("Prüfungen"), ["Use Case", "Prüfdimension", "Status", "Score", "Befund", "Maßnahmen"],
          [[c["uc"], c["dim"], st.get(c["status"]), c["score"], c["reason"], c["actions"]] for c in data["checks"]],
          status_cols=(2,))
    mrows = []
    for m in data["models"]:
        for e in m["engines"]:
            mrows.append([m["name"], m["metric"], e["name"], e["klass"], e["status"], e["value"], "★" if e["best"] else "",
                          st.get(m["vk"].get("status")), m["vk"].get("reason")])
    table(wb.create_sheet("Modellvergleich"), ["Use Case", "Hauptmetrik", "Engine", "Modellklasse", "Status", "Wert",
                                               "Sieger", "Vorhersagekraft", "Begründung"], mrows, status_cols=(7,))
    table(wb.create_sheet("Reparaturplan"), ["Playbook", "Titel", "Problem", "Option", "Wirkung", "Aufwand", "Hinweis"],
          [[p["id"], p["title"], p["problem"], p["option"], p["effect"], p["effort"], p.get("effort_note") or ""]
           for p in data["plan"]])
    table(wb.create_sheet("Business Case"), ["Use Case", "Engine", "konservativ €", "realistisch €", "optimistisch €",
                                             "Mehrwert ggü. Regel €", "nach Playbooks (v1) €", "Euro-Ampel", "Annahmen"],
          [[c["name"], c["model"], c["konservativ"], c["realistisch"], c["optimistisch"], c["added"], c["v1"],
            st.get(c["euro"].get("status")), " · ".join(c["assumptions"])] for c in data["cases"]], status_cols=(7,))
    a = data.get("activation")
    if a:
        table(wb.create_sheet("Aktivierung"), ["Liste", "Engine", "Stichtag", "Kapazität", "in der Liste", "erwarteter Wert €",
                                               "unsicher", "Kontrollgruppe (Fälle)", "nachweisbarer Effekt (MDE)", "Effekt nachweisbar"],
              [[L["name"], L["engine"]["name"], L["anchor"], L["capacity"], L["summary"]["in_list"], L["summary"]["ev_list"],
                L["summary"]["uncertain_list"], L["pilot"]["n_c"], de_pct(L["pilot"]["mde_rel"]),
                "ja" if L["pilot"]["detectable"] else "nein"] for L in a["lists"].values()])
    m = data.get("monitoring")
    if m:
        table(wb.create_sheet("Monitoring"), ["Monat", "Positionen", "Füllgrad Kunde", "Beschreibung", "Preis 0",
                                              "Nicht-Artikel", "Dubletten", "Stornos mit Original"],
              [[r["month"], r["rows"], r["cust_fill"], r["desc_fill"], r["price0"], r["non_article"], r["dup"],
                r["storno_linked"]] for r in m["months"]])
    table(wb.create_sheet("Quellen"), ["Quelle", "Link"], [list(s) for s in data["sources"]])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
