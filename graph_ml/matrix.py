"""
Ampel-Matrix (Prompt 10) – alle Use Cases × alle eingebauten Datensätze
=======================================================================
  Je Zelle: Datenreife (Stufe 3, sonst höchste mögliche Stufe), Vorhersagekraft und Euro nur dort, wo der Rückblick
  gerechnet wurde („Heimat-Datensatz“), Gesamt = schwächste belastbare Dimension. Die Matrix wird beim Precompute als
  Artefakt geschrieben (Render lädt so nicht alle Datensätze nacheinander); fehlt es, wird sie live gerechnet.

  .venv/bin/python -m graph_ml.matrix
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ARTIFACT = Path(__file__).parent / "artifacts" / "matrix.json"
_CACHE: dict | None = None


STRUCTURE_ONLY = ("warranty_root_cause", "warranty_fraud", "first_time_fix", "service_triage")


def home_of(uc: str) -> str | None:
    """Datensatz, auf dem der Rückblick gerechnet wurde – None für Use Cases ohne offene Daten (nur Struktur)."""
    from .backtest import DATASET_NAMES, DATASET_OF
    if uc in STRUCTURE_ONLY:
        return None
    return DATASET_NAMES.get(DATASET_OF.get(uc, "v0"), "online_retail_ii")


def _synthetic(ds_id: str) -> bool:
    from . import connectors as cn
    try:
        return bool(cn.load_meta(ds_id).get("synthetic"))
    except Exception:
        return ds_id.startswith("adventureworks")


def build(reg=None) -> dict:
    from . import evaluation as ev
    from .data_loader import get_registry
    reg = reg or get_registry()
    datasets = [d for d in reg.list() if d.get("builtin")]
    summ = ev.summary()
    cols, cells, ucs = [], {}, None
    for d in datasets:
        stage = reg.max_stage(d["id"])
        res = reg.checks(d["id"], min(stage, 3))
        cols.append({"id": d["id"], "name": d.get("name"), "stage": stage, "synthetic": _synthetic(d["id"])})
        if ucs is None:
            ucs = [{"id": u["id"], "name": u["name"], "area": u["area"], "home": home_of(u["id"])} for u in res["use_cases"]]
        for u in res["use_cases"]:
            a = u.get("ampel") or {}
            dr = a.get("datenreife") or {}
            home = home_of(u["id"]) == d["id"]
            s = summ.get(u["id"]) or {}
            measured = home and s.get("status") == "ok"
            cells[f"{u['id']}|{d['id']}"] = {
                "datenreife": dr.get("status"), "score": dr.get("score"), "home": home, "measured": measured,
                "vorhersagekraft": (a.get("vorhersagekraft") or {}).get("status") if home else None,
                "lift": (a.get("vorhersagekraft") or {}).get("lift") if home else None,
                "euro": (a.get("euro") or {}).get("status") if home else None,
                "annual": (a.get("euro") or {}).get("annual") if home else None,
                "gesamt": (a.get("gesamt") or {}).get("status") if home else dr.get("status"),
                "best": s.get("best_name") if measured else None}
    rows = []
    for u in ucs or []:
        hc = cells.get(f"{u['id']}|{u['home']}") or {}
        why = ev.NOT_MEASURABLE.get(u["id"])
        rows.append({**u, "measured": bool(hc.get("measured")), "not_measurable": why,
                     "ready_in": [c["id"] for c in cols if (cells.get(f"{u['id']}|{c['id']}") or {}).get("datenreife") in ("ok", "warn")]})
    counts = {k: sum(1 for r in rows if r["home"] and (cells.get(f"{r['id']}|{r['home']}") or {}).get("gesamt") == k)
              for k in ("ok", "warn", "bad")}
    counts["na"] = sum(1 for r in rows if not r["home"])
    return {"generated_at": pd.Timestamp.now().isoformat(timespec="seconds"), "datasets": cols, "use_cases": rows, "cells": cells,
            "counts": counts, "measured": sum(1 for r in rows if r["measured"]),
            "note": "Vorhersagekraft und Euro je Use Case nur auf dem Datensatz, auf dem der Rückblick gerechnet wurde (★). "
                    "Euro-Werte verschiedener Datensätze sind nicht addierbar."}


def save(m: dict) -> None:
    ARTIFACT.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")


def load() -> dict:
    global _CACHE
    if _CACHE is None:
        if ARTIFACT.exists():
            _CACHE = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        else:
            _CACHE = build()
    return _CACHE


def main() -> int:
    m = build()
    save(m)
    print(f"Matrix: {len(m['use_cases'])} Use Cases × {len(m['datasets'])} Datensätze · gemessen {m['measured']} · "
          f"Gesamt Grün/Gelb/Rot {m['counts']}")
    for r in m["use_cases"]:
        c = m["cells"].get(f"{r['id']}|{r['home']}") or {}
        print(f"  {r['name'][:38]:38s} {r['home'] or 'nur Struktur':20s} {c.get('gesamt') or '-':5s} reif in: {', '.join(r['ready_in']) or '–'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
