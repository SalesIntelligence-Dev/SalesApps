"""
Engine-Registry für den modellunabhängigen Vergleich (Modul 4).

  Regel-Baseline · LightGBM · LightGBM + Graph-Features · ItemKNN · LightGBM-Ranker
  TabPFN v2 (TFM, GPU) · Hetero-GraphSAGE (RDL-GNN, GPU) · LightGCN (GPU)
  KumoRFM (API, optional) · Aurora (Zeitreihen-FM, bestehender RunPod-Endpoint)
"""
from __future__ import annotations

from .api import AuroraEngine, KumoRFMEngine
from .cpu import (ItemKNNEngine, LGBMEngine, LGBMForecastEngine, LGBMGraphEngine, LGBMRankEngine,
                  NaiveForecastEngine, PopularityEngine, RecoRuleEngine, RuleEngine)
from .gpu import GNNEngine, LightGCNEngine, TabPFNEngine

ENGINE_CLASSES = [RuleEngine, RecoRuleEngine, NaiveForecastEngine, PopularityEngine, LGBMEngine, LGBMGraphEngine,
                  ItemKNNEngine, LGBMRankEngine, LGBMForecastEngine, TabPFNEngine, GNNEngine, LightGCNEngine,
                  KumoRFMEngine, AuroraEngine]
ENGINE_ORDER = ["rule", "popularity", "lgbm", "lgbm_graph", "itemknn", "lgbm_rank", "tabpfn", "gnn", "lightgcn",
                "kumorfm", "aurora"]


def make_engines(task) -> list:
    engines = [cls() for cls in ENGINE_CLASSES if cls().supports(task)]
    return sorted(engines, key=lambda e: ENGINE_ORDER.index(e.id))


def engine_catalog() -> list[dict]:
    """Für die UI: alle Engines mit Klasse, Gerät und Verfügbarkeit."""
    seen, out = set(), []
    for cls in ENGINE_CLASSES:
        e = cls()
        if e.id in seen and e.id in ("rule", "lgbm"):
            continue
        seen.add(e.id)
        ok, why = e.available()
        out.append({**e.meta(), "available": ok, "reason": why, "tasks": list(e.tasks)})
    return out
