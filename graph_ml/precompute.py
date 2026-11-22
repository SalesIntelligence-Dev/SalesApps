"""
Vorberechnung des Rückblicks (Stufe 3) für Online Retail II
===========================================================
Rechnet alle Backtests und speichert kompakte Ergebnisse in
graph_ml/artifacts/backtest_<use_case>.json – Render zeigt sie sofort an.

  .venv/bin/python -m graph_ml.precompute              # mit lokaler GPU (RTX 3060): alle Engines
  python3 -m graph_ml.precompute --engines rule,lgbm  # nur CPU-Engines
  python3 -m graph_ml.precompute --use-cases churn
  .venv/bin/python -m graph_ml.precompute --only-activation   # Modul 6: Aktionslisten, NBA, Monitoring

GPU-Engines, die in der aktuellen Umgebung nicht laufen, behalten ihr
Ergebnis aus einer früheren Vorberechnung (falls vorhanden).
"""
from __future__ import annotations

import argparse
import sys
import time


def main(argv: list[str] | None = None) -> int:
    from . import evaluation as ev
    from .engines.runtime import local_gpu
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--use-cases", default=",".join(ev.UC_ORDER))
    ap.add_argument("--engines", default="", help="Komma-Liste, leer = alle")
    ap.add_argument("--playbooks", action="store_true",
                    help="zusätzlich: Wirkung je Playbook + empfohlenes Paket (v1) inkl. Rückblick")
    ap.add_argument("--only-playbooks", action="store_true", help="nur den Playbook-Teil rechnen")
    ap.add_argument("--activation", action="store_true",
                    help="zusätzlich: Aktionslisten, Next Best Action, Kontrollgruppe und Monitoring (Modul 6)")
    ap.add_argument("--only-activation", action="store_true", help="nur Modul 6 rechnen")
    args = ap.parse_args(argv)
    gpu = local_gpu()
    print(f"[precompute] Gerät: {'Lokale GPU · ' + gpu[1] if gpu[0] else 'nur CPU'}", flush=True)
    engines = [e for e in args.engines.split(",") if e] or None
    t_all = time.time()
    for uc in ([] if args.only_playbooks or args.only_activation else args.use_cases.split(",")):
        t0 = time.time()
        res = ev.run_use_case(uc, engine_ids=engines, previous=ev.load(uc))
        ev.save(res)
        vk = res["vorhersagekraft"]
        print(f"[precompute] {uc:20s} {time.time() - t0:6.1f} s · Sieger {res['best']} · "
              f"Vorhersagekraft {vk.get('status')} ({vk.get('reason')})", flush=True)
        for e in res["engines"]:
            m = e.get("metrics", {}).get(res["main_metric"])
            print(f"    {e['id']:11s} {e['status']:12s} {res['main_metric']}={m if m is None else round(m, 4)}"
                  f"  {e.get('meta', {}).get('device', '')}  {e.get('reason', '')[:70]}", flush=True)
    if (args.playbooks or args.only_playbooks) and not args.only_activation:
        import json
        from . import playbooks as pb
        pb.compute_single_effects(log=lambda m: print(m, flush=True))
        res = pb.run_variant(pb.recommended_steps(), backtest=True, reuse_backtest=args.only_playbooks)
        (pb.ARTIFACTS / "pipeline_recommended.json").write_text(json.dumps({"variant": res["variant"]}),
                                                                 encoding="utf-8")
        print(f"[precompute] Paket {res['variant']}: {res['seconds']} s", flush=True)
        for c in res["comparison"]:
            print(f"    {c['id']:20s} {c['before']['score']} → {c['after']['score']} ({c['delta']:+d})", flush=True)
        for uc, b in res.get("backtest", {}).items():
            print(f"    Rückblick {uc:20s} {b['best']} · {b['vorhersagekraft'].get('reason')}", flush=True)
    if args.activation or args.only_activation:
        from . import activation as act
        art = act.build(log=lambda m: print(m, flush=True))
        act.build_reco_paths(log=lambda m: print(m, flush=True))
        act.build_territory_table(log=lambda m: print(m, flush=True))
        from . import nba_eval, pricing
        ev.save(nba_eval.to_result(nba_eval.evaluate(log=lambda m: print(m, flush=True))))
        for res in pricing.to_results(pricing.evaluate(log=lambda m: print(m, flush=True))).values():
            ev.save(res)
        from . import quality
        quality.main()
        from . import aftersales
        from .connectors import LOCAL_DIR
        if (LOCAL_DIR / "scania_readouts.npz").exists():   # volle SCANIA-Auslesungen liegen nur lokal (337 MB)
            aftersales.main()
        for uc, L in art["lists"].items():
            print(f"    {uc:20s} {L['engine']['name']:28s} Liste {L['summary']['in_list']:5d} · "
                  f"Wert {L['summary']['ev_list']:>12,.0f} € · ECE {L['calibration']['ece_before']} → "
                  f"{L['calibration']['ece_after']}", flush=True)
        print(f"    NBA {art['nba']['counts']} · Kontrollgruppe {art['control']['control']} von "
              f"{art['control']['customers']}", flush=True)
    print(f"[precompute] fertig in {time.time() - t_all:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
