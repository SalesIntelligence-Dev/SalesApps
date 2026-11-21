"""
AdventureWorks-Schema (nur Struktur, keine Daten) aus dem öffentlichen
Microsoft-DDL erzeugen → graph_ml/data/adventureworks_schema.json

  python -m graph_ml.build_adventureworks_schema

Quelle: github.com/microsoft/sql-server-samples (MIT-Lizenz),
samples/databases/adventure-works/oltp-install-script/instawdb.sql
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import requests

DDL_URL = ("https://raw.githubusercontent.com/microsoft/sql-server-samples/master/"
           "samples/databases/adventure-works/oltp-install-script/instawdb.sql")
RAW_PATH = Path(__file__).parent / "data" / "raw" / "instawdb.sql"
OUT_PATH = Path(__file__).parent / "data" / "adventureworks_schema.json"

KEEP_SCHEMAS = {"Sales", "Production", "Person", "Purchasing", "HumanResources"}

_CREATE_RE = re.compile(r"CREATE TABLE \[(\w+)\]\.\[(\w+)\]\((.*?)\n\)\s*ON", re.S)
_COL_RE    = re.compile(r"^\s*\[(\w+)\]\s+(?:\[(\w+)\](\([^)]*\))?|(AS)\s|(uniqueidentifier))(.*)$")
_TYPE_RE   = re.compile(r"CREATE TYPE \[(\w+)\] FROM (\w+(?:\(\d+\))?)")
_PK_RE     = re.compile(r"ALTER TABLE \[(\w+)\]\.\[(\w+)\] WITH CHECK ADD\s+CONSTRAINT \[\w+\] PRIMARY KEY "
                        r"(?:CLUSTERED|NONCLUSTERED)\s*\((.*?)\)", re.S)
# Ein ALTER-TABLE-Block kann mehrere FK-Constraints enthalten
_ALTER_RE  = re.compile(r"ALTER TABLE \[(\w+)\]\.\[(\w+)\] ADD(.*?)\nGO", re.S)
_FK_RE     = re.compile(r"CONSTRAINT \[(\w+)\] FOREIGN KEY\s*\((.*?)\)\s*"
                        r"REFERENCES \[(\w+)\]\.\[(\w+)\]\((.*?)\)", re.S)


def _cols(block: str) -> list[str]:
    return re.findall(r"\[(\w+)\]", block)


def _semantic_type(sql_type: str) -> str:
    t = sql_type.lower()
    if t.startswith(("datetime", "date", "smalldatetime", "time")):
        return "datetime"
    if t.startswith(("int", "smallint", "tinyint", "bigint")):
        return "integer"
    if t.startswith(("money", "smallmoney", "decimal", "numeric", "float", "real")):
        return "number"
    if t.startswith("bit"):
        return "boolean"
    if t.startswith(("uniqueidentifier", "hierarchyid")):
        return "id"
    if t.startswith(("xml",)):
        return "text"
    return "string"


def parse(ddl: str) -> dict:
    user_types = {name: base for name, base in _TYPE_RE.findall(ddl)}
    tables: dict[str, dict] = {}
    for schema, name, body in _CREATE_RE.findall(ddl):
        if schema not in KEEP_SCHEMAS:
            continue
        cols = []
        for line in body.split("\n"):
            m = _COL_RE.match(line)
            if not m:
                continue
            col, typ, length, computed, guid, rest = m.groups()
            if computed:
                sql_type, nullable, is_computed = "computed", True, True
            else:
                sql_type = guid or (typ + (length or ""))
                sql_type = user_types.get(sql_type, sql_type)
                nullable = "NOT NULL" not in rest
                is_computed = False
            cols.append({
                "name": col, "sql_type": sql_type, "type": _semantic_type(sql_type),
                "nullable": nullable, "computed": is_computed,
            })
        tables[f"{schema}.{name}"] = {"name": f"{schema}.{name}", "schema": schema, "columns": cols,
                                      "primary_key": []}
    for schema, name, block in _PK_RE.findall(ddl):
        key = f"{schema}.{name}"
        if key in tables:
            tables[key]["primary_key"] = _cols(block)
    fks = []
    for s1, t1, body in _ALTER_RE.findall(ddl):
        for fk_name, c1, s2, t2, c2 in _FK_RE.findall(body):
            a, b = f"{s1}.{t1}", f"{s2}.{t2}"
            if a in tables and b in tables:
                fks.append({"name": fk_name, "from_table": a, "from_columns": _cols(c1),
                            "to_table": b, "to_columns": _cols(c2)})
    return {
        "id": "adventureworks",
        "name": "AdventureWorks (OLTP-Schema)",
        "kind": "schema",
        "source": "Microsoft sql-server-samples · instawdb.sql",
        "source_url": DDL_URL,
        "license": "MIT (nur Schema, keine Daten)",
        "description": "Fahrradhersteller mit Vertrieb, Produktion, Einkauf, Personal – ERP-typisches Schema.",
        "tables": sorted(tables.values(), key=lambda t: t["name"]),
        "foreign_keys": fks,
    }


def main() -> int:
    if not RAW_PATH.exists():
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(DDL_URL, timeout=60)
        r.raise_for_status()
        RAW_PATH.write_bytes(r.content)
    ddl = RAW_PATH.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    schema = parse(ddl)
    OUT_PATH.write_text(json.dumps(schema, indent=1, ensure_ascii=False), encoding="utf-8")
    n_cols = sum(len(t["columns"]) for t in schema["tables"])
    print(f"{len(schema['tables'])} Tabellen, {n_cols} Spalten, {len(schema['foreign_keys'])} FKs → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
