"""
GraphReady Profiler – lokal beim Kunden ausführen (Stufe 2, Profilcheck)
=======================================================================
Liest einen Datenexport (Ordner mit CSV-, XLSX- oder Parquet-Dateien, je
Datei eine Tabelle), erkennt Schlüssel und Verknüpfungen und schreibt ein
Profil mit ausschließlich aggregierten Kennzahlen. Das Profil enthält keine
Datenzeilen, keine Schlüsselwerte und keine Beispielwerte – es kann ohne
personenbezogene Daten an GraphReady übergeben werden.

  python -m graph_ml.profiler_cli <export-ordner> --out profil.json
  python -m graph_ml.profiler_cli --online-retail --out profil.json   # Demo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

EXTS = {".csv", ".txt", ".xlsx", ".xls", ".parquet"}


def read_export(folder: Path) -> dict[str, pd.DataFrame]:
    from .data_loader import _coerce, _read_csv, _safe_name
    tables: dict[str, pd.DataFrame] = {}
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in EXTS:
            continue
        if f.suffix.lower() in (".csv", ".txt"):
            tables[_safe_name(f.name)] = _read_csv(f.read_bytes())
        elif f.suffix.lower() == ".parquet":
            tables[_safe_name(f.name)] = pd.read_parquet(f)
        else:
            for sheet, df in pd.read_excel(f, sheet_name=None).items():
                tables[_safe_name(f.name) if len(tables) == 0 else f"{_safe_name(f.name)}_{_safe_name(sheet)}"] = df
    if not tables:
        raise SystemExit(f"Keine Tabellen in {folder} gefunden (CSV, XLSX, Parquet)")
    return {k: _coerce(v) for k, v in tables.items()}


def profile_folder(folder: Path, name: str | None = None) -> dict:
    from .graph_mapping import build_mapping
    from .profiler import assemble_events, build_profile
    from .semantics import bind_fields, schema_view
    tables = read_export(folder)
    mapping = build_mapping(tables, dataset_id="export", sample_note="Stichprobe")
    events = assemble_events(tables, mapping, bind_fields(schema_view(mapping)))
    return build_profile(dataset_name=name or folder.name, mapping=mapping, tables=tables, events=events,
                         source="GraphReady Profiler (lokal)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Erzeugt ein Kennzahlen-Profil ohne personenbezogene Daten.")
    ap.add_argument("folder", nargs="?", help="Ordner mit dem Datenexport")
    ap.add_argument("--out", default="profil.json", help="Zieldatei (JSON)")
    ap.add_argument("--name", help="Anzeigename des Datensatzes")
    ap.add_argument("--online-retail", action="store_true", help="Demo-Profil für Online Retail II erzeugen")
    args = ap.parse_args(argv)

    from .profiler import assert_privacy
    if args.online_retail:
        from .data_loader import get_registry
        prof = dict(get_registry().profile("online_retail_ii"))
        prof.pop("source_version", None)
        prof.pop("artifact_version", None)
    elif args.folder:
        prof = profile_folder(Path(args.folder), args.name)
    else:
        ap.error("Ordner angeben oder --online-retail")
    assert_privacy(prof)
    Path(args.out).write_text(json.dumps(prof, ensure_ascii=False, indent=1), encoding="utf-8")
    ev = prof.get("events") or {}
    print(f"Profil geschrieben: {args.out} ({Path(args.out).stat().st_size / 1024:.0f} KB)")
    print(f"  Tabellen: {len(prof['tables'])} · Ereignisse: {ev.get('rows', '–')} · "
          f"Zeitraum: {(ev.get('time') or {}).get('months', '–')} Monate")
    print("  Enthält keine Datenzeilen, keine Schlüssel- und keine Beispielwerte.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
