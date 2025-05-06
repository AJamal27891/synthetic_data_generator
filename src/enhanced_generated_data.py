#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Config-driven Synthetic Data Generator for HeteroData Graphs.

Reads:
  data_generation_config/tables.yml
  data_generation_config/lineage.yml

Writes:
  ../data/*.csv
  ../metadata/schema_metadata.json
  ../metadata/generation_lineage.yaml
  ../metadata/labels.json
"""

import os
import yaml
import json
import random
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

# ──────────────────────────────────────────────────────────────────────────────
# 0) Paths & Load Configs
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(__file__)
ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CFG_DIR    = os.path.join(ROOT_DIR, "data_generation_config")
DATA_DIR   = os.path.join(ROOT_DIR, "data")
META_DIR   = os.path.join(ROOT_DIR, "metadata")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

tbl_path = os.path.join(CFG_DIR, "tables.yml")
lin_path = os.path.join(CFG_DIR, "lineage.yml")
for p in (tbl_path, lin_path):
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing config file: {p!r}")

with open(tbl_path, "r", encoding="utf-8") as f:
    tbl_cfg = yaml.safe_load(f)
with open(lin_path, "r", encoding="utf-8") as f:
    lin_cfg = yaml.safe_load(f)

# ──────────────────────────────────────────────────────────────────────────────
# 1) Initialize RNGs & Faker
# ──────────────────────────────────────────────────────────────────────────────
seed = tbl_cfg.get("seed", 42)
random.seed(seed)
np.random.seed(seed)
Faker.seed(seed)
fake = Faker()

# ──────────────────────────────────────────────────────────────────────────────
# 2) Helpers for derived expressions
# ──────────────────────────────────────────────────────────────────────────────
def lookup_value(df: pd.DataFrame, key_col: str, val_col: str, key):
    """Return the val_col value in df where key_col == key."""
    return df.loc[df[key_col] == key, val_col].iat[0]

def group_sum(df: pd.DataFrame, key_col: str, val_col: str) -> dict:
    """Return a dict mapping each key_col value to the sum of val_col."""
    return df.groupby(key_col)[val_col].sum().to_dict()

def eval_derived(expr: str, row: pd.Series, context: dict, all_tables: dict):
    """
    Evaluate a derived-column expression in a safe namespace:
    - locals(): row fields + context + row index 'i'
    - globals(): random, np, datetime, timedelta, fake,
                 lookup_value, group_sum, plus all_tables
    """
    local_vars = row.to_dict()
    local_vars.update(context)
    local_vars["i"] = row.name

    glob_vars = {
        "random": random,
        "np": np,
        "datetime": datetime,
        "timedelta": timedelta,
        "fake": fake,
        "lookup_value": lookup_value,
        "group_sum": group_sum,
        **all_tables
    }
    return eval(expr, glob_vars, local_vars)

# ──────────────────────────────────────────────────────────────────────────────
# 3) First Pass: Generate non‐derived columns & collect derived specs
# ──────────────────────────────────────────────────────────────────────────────
generated     = {}   # tbl_name -> DataFrame
table_order   = []   # preserve YAML order
derived_specs = {}   # tbl_name -> list of (col_name, col_def)

for tbl_name, tbl_def in tbl_cfg["tables"].items():
    n = tbl_def["row_count"]
    rows = {}
    context = {
        "base_date": datetime.now() - timedelta(days=tbl_def.get("offset_days", 365)),
        "current_year": datetime.now().year
    }

    for col_name, col_def in tbl_def["columns"].items():
        ctype = col_def["type"]

        # collect derived specs for later
        if ctype == "derived":
            derived_specs.setdefault(tbl_name, []).append((col_name, col_def))
            continue

        # non‐derived generation
        if ctype == "int":
            lo, _ = col_def["range"]
            rows[col_name] = list(range(lo, lo + n))

        elif ctype == "float":
            lo, hi = col_def["range"]
            prec   = col_def.get("precision", 2)
            rows[col_name] = [round(random.uniform(lo, hi), prec) for _ in range(n)]

        elif ctype == "faker":
            method = getattr(fake, col_def["method"], None)
            if not method:
                raise AttributeError(f"Faker has no method {col_def['method']!r}")
            args = col_def.get("args", [])
            rows[col_name] = [method(*args) for _ in range(n)]

        elif ctype == "choice":
            if "choices" in col_def:
                choices = col_def["choices"]
                weights = col_def.get("weights", [1]*len(choices))
            else:
                include_null = col_def.get("include_null", False)
                lo, hi        = col_def.get("range", [None, None])
                w_null        = col_def.get("weight_null", 0.0)
                w_rest        = col_def.get("weight_rest", 1.0 - w_null)
                choices, weights = [], []
                if include_null:
                    choices.append(None)
                    weights.append(w_null)
                vals = list(range(lo, hi+1))
                per  = w_rest / len(vals) if vals else 0
                choices += vals
                weights += [per]*len(vals)
            rows[col_name] = random.choices(choices, weights, k=n)

        elif ctype == "lookup":
            if "reference" in col_def:
                ref_tbl, ref_col = col_def["reference"].split(".", 1)
                src_df = generated.get(ref_tbl)
                if src_df is None:
                    raise KeyError(f"lookup reference {ref_tbl!r} not ready")
                pool = src_df[ref_col].tolist()
                rows[col_name] = [random.choice(pool) for _ in range(n)]
            elif "from" in col_def and "map" in col_def:
                parent = col_def["from"]
                mp     = col_def["map"]
                if parent not in rows:
                    raise KeyError(f"{parent!r} must precede lookup for {col_name!r}")
                rows[col_name] = [random.choice(mp[val]) for val in rows[parent]]
            else:
                raise KeyError(f"Invalid lookup spec for {col_name!r}")

        elif ctype == "constant":
            rows[col_name] = [col_def["value"]] * n

        else:
            raise ValueError(f"Unknown column type: {ctype!r}")

    # Build DataFrame without derived columns
    df = pd.DataFrame(rows)
    generated[tbl_name] = df
    table_order.append(tbl_name)
    df.to_csv(os.path.join(DATA_DIR, f"{tbl_name}.csv"), index=False)
    print(f"Generated {tbl_name}.csv ({len(df)} rows)")

# ──────────────────────────────────────────────────────────────────────────────
# 4) Topologically sort derived tables by inter-table dependencies
# ──────────────────────────────────────────────────────────────────────────────
deps = {tbl: set() for tbl in derived_specs}
for tbl, specs in derived_specs.items():
    for _, col_def in specs:
        expr = col_def["expr"]
        for other in derived_specs:
            if other != tbl and other in expr:
                deps[tbl].add(other)

ready = [t for t, ds in deps.items() if not ds]
order = []
while ready:
    t = ready.pop(0)
    order.append(t)
    for u in list(deps):
        if t in deps[u]:
            deps[u].remove(t)
            if not deps[u]:
                ready.append(u)
    deps.pop(t, None)
order += list(deps.keys())

# ──────────────────────────────────────────────────────────────────────────────
# 5) Second Pass: Evaluate & append derived columns
# ──────────────────────────────────────────────────────────────────────────────
for tbl_name in order:
    df = generated[tbl_name]
    specs = derived_specs.get(tbl_name, [])
    if not specs:
        continue
    context = {
        "base_date": datetime.now() - timedelta(days=tbl_cfg["tables"][tbl_name].get("offset_days",365)),
        "current_year": datetime.now().year
    }
    for col_name, col_def in specs:
        expr = col_def["expr"]
        prec = col_def.get("precision", None)
        df[col_name] = df.apply(
            lambda r: eval_derived(expr, r, context, generated),
            axis=1
        )
        if prec is not None:
            df[col_name] = df[col_name].round(prec)
    df.to_csv(os.path.join(DATA_DIR, f"{tbl_name}.csv"), index=False)
    print(f"Added derived cols → {tbl_name}.csv")

# ──────────────────────────────────────────────────────────────────────────────
# 6) Inject Anomalies
# ──────────────────────────────────────────────────────────────────────────────
for an_type, rules in tbl_cfg.get("anomalies", {}).items():
    if an_type == "missing_fk":
        for key, frac in rules.items():
            tbl, col = key.split(".", 1)
            df = generated[tbl]
            idx = df.sample(frac=frac, random_state=seed).index
            df.loc[idx, col] = df[col].max() + 999
            print(f"Injected missing_fk in {tbl}.{col} ({len(idx)} rows)")

# ──────────────────────────────────────────────────────────────────────────────
# 7) Write schema_metadata.json
# ──────────────────────────────────────────────────────────────────────────────
schema = {"tables": {}, "relationships": tbl_cfg.get("relationships", [])}
for tbl in table_order:
    df = generated[tbl]
    schema["tables"][tbl] = {
        "row_count": len(df),
        "domain": tbl_cfg["tables"][tbl]["domain"],
        "columns": {c: str(dt) for c, dt in df.dtypes.to_dict().items()},
        "distinct_counts": {c: int(df[c].nunique()) for c in df.columns},
        "null_rates": {c: float(df[c].isna().mean()) for c in df.columns},
    }
with open(os.path.join(META_DIR, "schema_metadata.json"), "w") as f:
    json.dump(schema, f, indent=2)

# ──────────────────────────────────────────────────────────────────────────────
# 8) Build lineage edges & labels.json
# ──────────────────────────────────────────────────────────────────────────────
reads, writes = [], []
for step in lin_cfg["steps"]:
    for f in step.get("inputs", []):
        tbl = os.path.splitext(os.path.basename(f))[0]
        if tbl in table_order:
            reads.append([step["name"], tbl])
    for f in step.get("outputs", []):
        tbl = os.path.splitext(os.path.basename(f))[0]
        if tbl in table_order:
            writes.append([step["name"], tbl])

with open(os.path.join(META_DIR, "generation_lineage.yaml"), "w") as f:
    yaml.safe_dump({"steps": lin_cfg["steps"]}, f, sort_keys=False)

labels = {
    "step_labels": [
        {"name": s["name"], "label": lin_cfg["labels"].get(s["name"], "normal")}
        for s in lin_cfg["steps"]
    ],
    "anomalies": tbl_cfg.get("anomalies_list", [])
}
with open(os.path.join(META_DIR, "labels.json"), "w") as f:
    json.dump(labels, f, indent=2)

print("✅ All data, metadata, lineage & labels generated.")
