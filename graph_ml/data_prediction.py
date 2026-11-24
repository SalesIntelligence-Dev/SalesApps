"""
Data Prediction – ein Sammel-Use-Case aus der Data-Classifier-Analyse (Schritt 3)
=================================================================================
  Jedes gefundene, manuell gepflegte Feld, das ML (oder eine Regel) sicher vorschlagen kann, ist ein eigener Use Case.
  In Schritt 4 und 5 erscheint dafür genau eine Kachel mit der Anzahl dieser Feld-Use-Cases je Datensatz.

  Vorhersagekraft: mittlerer automatisch entscheidbarer Anteil (≥ 98 % richtig) der Kaskade gegen die Regel, über die
  prüfbaren Felder des Datensatzes (Bootstrap über Felder). Euro: eingesparte Pflegestunden × Stundensatz.

  python3 -m graph_ml.data_prediction        (liest classifier_<ds>.json, schreibt backtest_data_prediction.json)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import classifier_scan as cs

UC = "data_prediction"
HOME = "adventureworks_data"
DATASETS = ("adventureworks_data", "wwi", "online_retail_ii", "scania", "cmapss")
NAMES = {"adventureworks_data": "AdventureWorks", "wwi": "WideWorldImporters", "online_retail_ii": "Online Retail II",
         "scania": "SCANIA Component X", "cmapss": "NASA C-MAPSS"}


def _hours(r: dict, auto: float) -> float:
    v = r.get("value") or {}
    minutes = r.get("minutes", cs.MINUTES.get(r["kind"], 2.0))
    return 0.0 if r.get("inherited") else (v.get("entries_per_year") or 0) * minutes / 60 * auto


def for_dataset(ds_id: str) -> dict | None:
    """Feld-Use-Cases eines Datensatzes: vorhersagbar (grün) oder teilweise (gelb); Rest bleibt manuell."""
    art = cs.load(ds_id)
    if not art:
        return None
    fields, rule_auto, casc_auto, hrs_c, hrs_r = [], [], [], 0.0, 0.0
    for r in art["results"]:
        if r.get("status") == "na" or not r.get("engines"):
            continue
        rule = r["engines"][0]["metrics"]
        ra = rule.get("auto_98", rule.get("within_10", 0.0)) or 0.0
        ca = r.get("auto_98") or 0.0
        rule_auto.append(ra)
        casc_auto.append(ca)
        hrs_c += _hours(r, ca)
        hrs_r += _hours(r, ra)
        if r.get("status") in ("ok", "warn") and not r.get("inherited"):      # übernommene Werte tippt niemand ein
            fields.append({"id": r["id"], "name": r.get("name") or r["column"], "table": r["table"], "column": r["column"],
                           "kind": r["kind"], "status": r["status"], "verdict": r["verdict"], "auto_98": ca, "rule_auto": ra,
                           "best": next((e["name"] for e in r["engines"] if e["id"] == r.get("best")), None),
                           "special": bool(r.get("special")), "inherited": bool(r.get("inherited")),
                           "net_year": (r.get("euro") or {}).get("net_year")})
    fields.sort(key=lambda f: (not f["special"], f["status"] != "ok", -(f["net_year"] or 0)))
    n_ok = sum(1 for f in fields if f["status"] == "ok")
    out = {"dataset": ds_id, "count": len(fields), "ok": n_ok, "warn": len(fields) - n_ok, "checked": len(casc_auto),
           "likely": art.get("n_likely", 0), "fields": fields, "hours_year": round(hrs_c, 1), "hours_rule": round(hrs_r, 1),
           "mean_auto": round(float(np.mean(casc_auto)), 4) if casc_auto else None,
           "mean_rule": round(float(np.mean(rule_auto)), 4) if rule_auto else None}
    if casc_auto:
        c, r_ = np.array(casc_auto), np.array(rule_auto)
        out["lift"] = float((c.mean() - r_.mean()) / max(r_.mean(), 1e-9))
        rng = np.random.default_rng(7)
        boot = []
        for _ in range(500):
            i = rng.integers(0, len(c), len(c))
            boot.append((c[i].mean() - r_[i].mean()) / max(r_[i].mean(), 1e-9))
        out["lift_ci"] = [round(float(np.percentile(boot, 5)), 4), round(float(np.percentile(boot, 95)), 4)]
    return out


def readiness(dp: dict | None) -> dict:
    """Datenreife der Kachel: Anteil der prüfbaren Felder, die ein Vorschlagsmodell tragen."""
    if not dp or not dp["checked"]:
        return {"status": "bad", "score": 20, "reason": "Keine manuell gepflegten Felder mit genug Beispielen gefunden", "provisional": False}
    share = dp["count"] / max(dp["checked"], 1)
    score = int(round(40 + 55 * share))
    status = "ok" if dp["ok"] >= 3 and score >= 75 else "warn" if dp["count"] else "bad"
    return {"status": status, "score": score, "provisional": False,
            "reason": f"{dp['count']} von {dp['checked']} geprüften Feldern vorschlagbar ({dp['ok']} sicher, {dp['warn']} mit Graubereich)"}


def to_result() -> dict | None:
    from . import evaluation as ev
    per = {ds: for_dataset(ds) for ds in DATASETS}
    per = {k: v for k, v in per.items() if v}
    home = per.get(HOME)
    if not home or not home.get("checked"):
        return None
    vk = ev.vorhersagekraft(home["lift"], home["lift_ci"])
    return {"result_version": ev.RESULT_VERSION, "variant": "v0", "currency": "EUR", "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "host": "lokal", "audit": [], "top_list": [], "use_case": UC, "kind": "classifier", "dataset": HOME,
            "compute_seconds": 0, "main_metric": "auto_98", "main_metric_label": "automatisch entscheidbar (≥ 98 % richtig)",
            "lower_is_better": False,
            "timeline": {"test": "5-fach Kreuzvalidierung", "test_label_end": "", "train": [], "val": None, "horizon_days": 0,
                         "label": "Wert eines manuell gepflegten Felds", "unit": "Datensatz"},
            "rule_text": "Regel: häufigster Wert je bestem Einzelfeld", "sizes": {"fields": home["checked"], "use_cases": home["count"]},
            "engines": [{"id": "rule", "status": "ok", "meta": {"name": "Regel (Stufe 0) je Feld", "klass": "Regel", "device": "CPU"},
                         "metrics": {"auto_98": home["mean_rule"]}},
                        {"id": "cascade", "status": "ok", "meta": {"name": "Kaskade: bestes Modell je Feld (Regel, LightGBM, Graph-Kontext, Text, TabPFN)",
                                                                   "klass": "Kaskade", "device": "GPU"},
                         "metrics": {"auto_98": home["mean_auto"]}}],
            "best": "cascade", "vorhersagekraft": vk,
            "whatif": {"kind": "classifier", "currency": "EUR",
                       "engines": {"cascade": {"hours": home["hours_year"]}, "rule": {"hours": home["hours_rule"]}}},
            "recommendation": {"engine": "cascade", "name": "Kaskade je Feld",
                               "text": f"{home['count']} Feld-Use-Cases auf {NAMES[HOME]} – {vk.get('reason', '')}."},
            "notes": ["Je Feld ein eigener Use Case; Details, Folgefelder und Graubereich in Schritt 3 (Data-Classifier-Analyse)."],
            "data_prediction": {"datasets": per, "names": NAMES}}


def main() -> int:
    from . import evaluation as ev
    r = to_result()
    if r:
        ev.save(r)
        for ds, d in r["data_prediction"]["datasets"].items():
            print(f"{ds:20s} {d['count']:3d} Feld-Use-Cases ({d['ok']} sicher) · {d['hours_year']} h/Jahr")
        print(r["vorhersagekraft"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
