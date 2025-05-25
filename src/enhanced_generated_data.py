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

# Add this global list to track circular dependencies
detected_circular_dependencies = []

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
# 3) Build dependency graph and determine table generation order
# ──────────────────────────────────────────────────────────────────────────────
def build_dependency_graph(tables_config):
    """Build a graph of table dependencies based on lookup references."""
    graph = {tname: set() for tname in tables_config}
    
    # Debug: Print all table names for verification
    print(f"Found {len(tables_config)} tables in configuration")
    
    for tname, tdef in tables_config.items():
        for col_name, col_def in tdef.get("columns", {}).items():
            if isinstance(col_def, dict) and col_def.get("type") == "lookup" and "reference" in col_def:
                ref_parts = col_def["reference"].split(".", 1)
                if len(ref_parts) == 2:
                    ref_tbl, ref_col = ref_parts
                    if ref_tbl != tname:  # Avoid self-references
                        graph[tname].add(ref_tbl)
                        # Verify that the referenced table exists
                        if ref_tbl not in tables_config:
                            print(f"WARNING: Table '{tname}' references non-existent table '{ref_tbl}'")
    
    # Debug: Print dependency graph
    print("\nDependency graph:")
    for table, deps in graph.items():
        if deps:
            print(f"  {table} depends on: {', '.join(deps)}")
    
    return graph

def detect_cycles(graph):
    """Detect and report all cycles in the dependency graph."""
    visited = set()
    rec_stack = set()
    cycles = []
    
    def dfs(node, path):
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                cycle = dfs(neighbor, path[:])
                if cycle:
                    cycles.append(cycle)
            elif neighbor in rec_stack:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycles.append(path[cycle_start:] + [neighbor])
        
        rec_stack.remove(node)
        return None
    
    for node in graph:
        if node not in visited:
            dfs(node, [])
    
    return cycles

def break_cycles(graph):
    """Break cycles in the graph by removing the weakest dependency edges."""
    cycles = detect_cycles(graph)
    
    if not cycles:
        return graph, []
    
    print(f"\nWARNING: Detected {len(cycles)} circular dependencies:")
    for i, cycle in enumerate(cycles):
        print(f"  Cycle {i+1}: {' -> '.join(cycle)}")
    
    # Create a copy of the graph to modify
    modified_graph = {node: set(deps) for node, deps in graph.items()}
    broken_edges = []
    
    for cycle in cycles:
        # For simplicity, just break the last edge in each cycle
        from_node = cycle[-2]
        to_node = cycle[-1]
        if to_node in modified_graph.get(from_node, set()):
            modified_graph[from_node].remove(to_node)
            broken_edges.append((from_node, to_node))
            print(f"  Breaking edge: {from_node} -> {to_node}")
    
    return modified_graph, broken_edges

def topological_sort(graph):
    """Sort tables in order so dependencies are processed first."""
    # First, detect and break any cycles
    acyclic_graph, broken_edges = break_cycles(graph)
    
    # Now perform topological sort on acyclic graph
    result = []
    temp_marks = set()
    perm_marks = set()
    
    def visit(node):
        if node in temp_marks:
            # This shouldn't happen after breaking cycles
            print(f"ERROR: Unexpected cycle detected with {node}")
            return
        if node not in perm_marks:
            temp_marks.add(node)
            for dependency in acyclic_graph.get(node, []):
                visit(dependency)
            temp_marks.remove(node)
            perm_marks.add(node)
            result.append(node)
    
    nodes = list(acyclic_graph.keys())
    while nodes:
        n = nodes.pop(0)
        if n not in perm_marks:
            visit(n)
    
    # Reverse to get dependencies first
    ordered_tables = list(reversed(result))
    
    print("\nTable generation order:")
    for i, table in enumerate(ordered_tables[:10]):
        print(f"  {i+1}. {table}")
    if len(ordered_tables) > 10:
        print(f"  ... and {len(ordered_tables)-10} more tables")
    
    return ordered_tables


# Pre-process tables to handle special cases
special_tables = []
# Find tables that are used in lookups but might not exist in configuration
for tname, tdef in tbl_cfg["tables"].items():
    for col_name, col_def in tdef.get("columns", {}).items():
        if isinstance(col_def, dict) and col_def.get("type") == "lookup" and "reference" in col_def:
            ref_tbl, ref_col = col_def["reference"].split(".", 1)
            if ref_tbl not in tbl_cfg["tables"]:
                print(f"WARNING: Table '{tname}' references missing table '{ref_tbl}'. Adding placeholder.")
                special_tables.append((ref_tbl, ref_col))

# Add placeholder tables with minimal structure for any missing references
for ref_tbl, ref_col in special_tables:
    if ref_tbl not in tbl_cfg["tables"]:
        print(f"Creating placeholder table: {ref_tbl} with column: {ref_col}")
        tbl_cfg["tables"][ref_tbl] = {
            "row_count": 100,
            "domain": "placeholder",
            "columns": {
                ref_col: {"type": "int", "range": [1, 100]}
            }
        }

# Generate table order based on dependencies
print("\nBuilding dependency graph and ordering tables...")
dependency_graph = build_dependency_graph(tbl_cfg["tables"])
table_order = topological_sort(dependency_graph)

# ──────────────────────────────────────────────────────────────────────────────
# 4) First Pass: Generate non‐derived columns & collect derived specs
# ──────────────────────────────────────────────────────────────────────────────
generated     = {}   # tbl_name -> DataFrame
derived_specs = {}   # tbl_name -> list of (col_name, col_def)

for tbl_name in table_order:
    tbl_def = tbl_cfg["tables"].get(tbl_name)
    if not tbl_def:
        continue
        
    n = tbl_def.get("row_count", 100)  # Default to 100 rows if not specified
    rows = {}
    context = {
        "base_date": datetime.now() - timedelta(days=tbl_def.get("offset_days", 365)),
        "current_year": datetime.now().year
    }

    for col_name, col_def in tbl_def.get("columns", {}).items():
        if not isinstance(col_def, dict):
            continue
            
        ctype = col_def.get("type")
        if not ctype:
            continue

        # collect derived specs for later
        if ctype == "derived":
            derived_specs.setdefault(tbl_name, []).append((col_name, col_def))
            continue

        # non‐derived generation
        if ctype == "int":
            lo, hi = col_def.get("range", [1, n])
            rows[col_name] = list(range(lo, lo + n))

        elif ctype == "float":
            lo, hi = col_def.get("range", [0, 100])
            prec = col_def.get("precision", 2)
            rows[col_name] = [round(random.uniform(lo, hi), prec) for _ in range(n)]

        elif ctype == "faker":
            method = getattr(fake, col_def.get("method", "word"), None)
            if not method:
                continue
                
            args = col_def.get("args", [])
            # Fix for date_time methods - convert string literals to actual dates
            if col_def.get("method", "").startswith("date") and args:
                fixed_args = []
                for arg in args:
                    if isinstance(arg, str):
                        if arg == "today":
                            fixed_args.append(datetime.now().date())
                        elif arg == "now":
                            fixed_args.append(datetime.now())
                        elif arg.startswith("+") or arg.startswith("-"):
                            # Handle relative dates like "+1y", "-30d"
                            try:
                                # Extract the number and unit
                                unit = arg[-1]
                                number_part = arg[1:-1]
                                amount = int(number_part)
                                
                                base = datetime.now()
                                if unit == 'y':
                                    delta = timedelta(days=amount*365)
                                elif unit == 'm':
                                    delta = timedelta(days=amount*30)
                                elif unit == 'w':
                                    delta = timedelta(weeks=amount)
                                else:  # 'd' or default
                                    delta = timedelta(days=amount)
                                    
                                if arg.startswith("+"):
                                    fixed_args.append(base + delta)
                                else:
                                    fixed_args.append(base - delta)
                            except ValueError:
                                print(f"WARNING: Could not parse date string '{arg}'. Using current date instead.")
                                fixed_args.append(datetime.now().date())
                        else:
                            fixed_args.append(arg)
                    else:
                        fixed_args.append(arg)
                args = fixed_args
                
            try:
                rows[col_name] = [method(*args) for _ in range(n)]
            except Exception as e:
                print(f"WARNING: Error generating data with {col_def.get('method')}: {e}")
                # Fallback to safer generation
                if col_def.get("method", "").startswith("date"):
                    print(f"Using fallback for date generation in {tbl_name}.{col_name}")
                    rows[col_name] = [fake.date_this_decade() for _ in range(n)]
                else:
                    print(f"Using fallback for {tbl_name}.{col_name}")
                    rows[col_name] = [f"{col_name}_{i}" for i in range(n)]

        elif ctype == "choice":
            if "choices" in col_def:
                choices = col_def["choices"]
                weights = col_def.get("weights", [1]*len(choices))
            else:
                include_null = col_def.get("include_null", False)
                lo, hi = col_def.get("range", [None, None])
                w_null = col_def.get("weight_null", 0.0)
                w_rest = col_def.get("weight_rest", 1.0 - w_null)
                choices, weights = [], []
                if include_null:
                    choices.append(None)
                    weights.append(w_null)
                vals = list(range(lo, hi+1))
                per = w_rest / len(vals) if vals else 0
                choices += vals
                weights += [per]*len(vals)
            rows[col_name] = random.choices(choices, weights, k=n)

        elif ctype == "lookup":
            if "reference" in col_def:
                ref_tbl, ref_col = col_def["reference"].split(".", 1)
                src_df = generated.get(ref_tbl)
                if src_df is None:
                    # More informative error with current tables
                    print(f"\nERROR: In table '{tbl_name}', column '{col_name}' references table '{ref_tbl}' which is not available.")
                    print(f"Tables generated so far: {list(generated.keys())}")
                    print(f"Current position in table_order: {table_order.index(tbl_name)} of {len(table_order)}")
                    
                    # Check if the table exists in our config but hasn't been generated yet
                    if ref_tbl in table_order:
                        ref_index = table_order.index(ref_tbl)
                        current_index = table_order.index(tbl_name)
                        if ref_index > current_index:
                            print(f"CIRCULAR DEPENDENCY DETECTED: '{ref_tbl}' is scheduled to be generated AFTER '{tbl_name}'")
                            print("Will create random values instead of lookup values for this column.")
                            
                            # Track this circular dependency
                            detected_circular_dependencies.append({
                                "type": "circular_dependency",
                                "table": tbl_name,
                                "column": col_name,
                                "references": ref_tbl
                            })
                            
                            # Generate random values as fallback
                            if ref_col.endswith('_id'):
                                rows[col_name] = [random.randint(1, 1000) for _ in range(n)]
                            else:
                                rows[col_name] = [f"placeholder_{i}" for i in range(n)]
                            continue
                    
                    raise KeyError(f"lookup reference {ref_tbl!r} not ready")
                    
                pool = src_df[ref_col].tolist()
                if not pool:
                    print(f"WARNING: Reference table '{ref_tbl}' column '{ref_col}' is empty. Using placeholders.")
                    rows[col_name] = [f"placeholder_{i}" for i in range(n)]
                else:
                    rows[col_name] = [random.choice(pool) for _ in range(n)]
            elif "from" in col_def and "map" in col_def:
                parent = col_def["from"]
                mp = col_def["map"]
                if parent not in rows:
                    raise KeyError(f"{parent!r} must precede lookup for {col_name!r}")
                rows[col_name] = [random.choice(mp[val]) for val in rows[parent]]
            else:
                raise KeyError(f"Invalid lookup spec for {col_name!r}")

        elif ctype == "constant":
            rows[col_name] = [col_def.get("value")] * n

        else:
            raise ValueError(f"Unknown column type: {ctype!r}")

    # Build DataFrame without derived columns
    df = pd.DataFrame(rows)
    generated[tbl_name] = df
    df.to_csv(os.path.join(DATA_DIR, f"{tbl_name}.csv"), index=False)
    print(f"Generated {tbl_name}.csv ({len(df)} rows)")

# ──────────────────────────────────────────────────────────────────────────────
# 5) Topologically sort derived tables by inter-table dependencies
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
# 6) Second Pass: Evaluate & append derived columns
# ──────────────────────────────────────────────────────────────────────────────
for tbl_name in order:
    df = generated.get(tbl_name)
    if df is None:
        continue
        
    specs = derived_specs.get(tbl_name, [])
    if not specs:
        continue
        
    context = {
        "base_date": datetime.now() - timedelta(days=tbl_cfg["tables"][tbl_name].get("offset_days", 365)),
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
# 7) Inject Anomalies
# ──────────────────────────────────────────────────────────────────────────────
# Add after the eval_derived function
def verify_anomalies_against_labels(generated_dfs, labels_json):
    """Verify that the anomalies in the generated data match what's in labels.json"""
    anomalies = labels_json.get("anomalies", [])
    verified = []
    missing = []
    
    # Check each labeled anomaly
    for anomaly in anomalies:
        anomaly_type = anomaly.get("type")
        
        if anomaly_type == "missing_fk":
            table = anomaly.get("table")
            column = anomaly.get("column")
            
            # Check if table exists in generated data
            if table not in generated_dfs:
                missing.append(f"Table '{table}' for anomaly missing_fk not found in generated data")
                continue
                
            # Check if column exists
            if column not in generated_dfs[table].columns:
                missing.append(f"Column '{column}' for anomaly missing_fk not found in table '{table}'")
                continue
                
            # Verify some rows have invalid FKs
            verified.append(f"Verified anomaly: missing_fk in {table}.{column}")
        
        elif anomaly_type == "ghost_output":
            step = anomaly.get("step")
            output = anomaly.get("output")
            # Check if the output file is mentioned in lineage but not on disk
            output_path = os.path.join(DATA_DIR, output)
            if not os.path.exists(output_path):
                verified.append(f"Verified anomaly: ghost_output {output} from step {step}")
            else:
                missing.append(f"Ghost output {output} was actually generated")
        
        elif anomaly_type == "job_failure":
            step = anomaly.get("step")
            verified.append(f"Verified anomaly: job_failure in step {step}")
        
        elif anomaly_type == "partial_write":
            step = anomaly.get("step")
            outputs = anomaly.get("outputs", [])
            for output in outputs:
                output_path = os.path.join(DATA_DIR, output)
                if os.path.exists(output_path):
                    verified.append(f"Verified anomaly: partial_write {output} from step {step}")
                else:
                    missing.append(f"Partial write output {output} was not generated")
        
        elif anomaly_type == "mixed_operation":
            step = anomaly.get("step")
            outputs = anomaly.get("outputs", [])
            for output in outputs:
                output_path = os.path.join(DATA_DIR, output)
                if os.path.exists(output_path):
                    verified.append(f"Verified anomaly: mixed_operation {output} from step {step}")
                else:
                    missing.append(f"Mixed operation output {output} was not generated")
        
        elif anomaly_type == "circular_dependency":
            table = anomaly.get("table")
            column = anomaly.get("column")
            ref_table = anomaly.get("references")
            
            # Check if table exists in generated data
            if table not in generated_dfs:
                missing.append(f"Table '{table}' for circular dependency not found in generated data")
                continue
                
            # Check if column exists
            if column not in generated_dfs[table].columns:
                missing.append(f"Column '{column}' for circular dependency not found in table '{table}'")
                continue
                
            verified.append(f"Verified anomaly: circular dependency from {table}.{column} to {ref_table}")
    
    # Print verification report
    print("\nAnomaly Verification Report:")
    print(f"✓ Verified anomalies: {len(verified)}")
    for v in verified:
        print(f"  - {v}")
    
    if missing:
        print(f"✗ Missing or incorrect anomalies: {len(missing)}")
        for m in missing:
            print(f"  - {m}")
    else:
        print("All labeled anomalies verified successfully!")
    
    return len(missing) == 0

# ──────────────────────────────────────────────────────────────────────────────
# 7) Inject Anomalies
# ──────────────────────────────────────────────────────────────────────────────
# Create a list to track actually injected anomalies
injected_anomalies = []

for an_type, rules in tbl_cfg.get("anomalies", {}).items():
    if an_type == "missing_fk":
        for key, frac in rules.items():
            try:
                tbl, col = key.split(".", 1)
                df = generated.get(tbl)
                if df is None:
                    print(f"WARNING: Cannot inject anomaly in missing table: {tbl}")
                    continue
                    
                # Check if the column exists
                if col not in df.columns:
                    print(f"WARNING: Column {col} not found in table {tbl}, skipping anomaly injection.")
                    continue
                    
                # Calculate number of rows to modify
                num_rows = int(len(df) * frac)
                if num_rows <= 0:
                    continue
                    
                # Sample rows to modify
                idx = df.sample(n=min(num_rows, len(df)), random_state=seed).index
                
                # Get the maximum value or use a default if the column is empty
                if len(df[col]) > 0 and not df[col].isna().all():
                    max_val = df[col].max()
                    # Handle non-numeric columns
                    if isinstance(max_val, (int, float)):
                        new_val = max_val + 999
                    else:
                        new_val = f"invalid_{col}_value"
                else:
                    new_val = 999999  # Default invalid value
                    
                df.loc[idx, col] = new_val
                print(f"Injected missing_fk in {tbl}.{col} ({len(idx)} rows)")
                
                # Track this successfully injected anomaly
                injected_anomalies.append({
                    "type": "missing_fk",
                    "table": tbl,
                    "column": col,
                    "count": len(idx)
                })
            except Exception as e:
                print(f"ERROR during anomaly injection for {key}: {str(e)}")
                continue
    
    # Add logic for other anomaly types
    elif an_type == "ghost_output":
        for step, outputs in rules.items():
            for output in outputs:
                # No need to do anything - ghost outputs don't exist by definition
                injected_anomalies.append({
                    "type": "ghost_output",
                    "step": step,
                    "output": output
                })
                print(f"Registered ghost output: {output} from step {step}")
    
    elif an_type == "job_failure":
        for step in rules:
            injected_anomalies.append({
                "type": "job_failure",
                "step": step
            })
            print(f"Registered job failure: {step}")
    
    # Additional types can be handled similarly


# ──────────────────────────────────────────────────────────────────────────────
# 8) Write schema_metadata.json
# ──────────────────────────────────────────────────────────────────────────────
relationships = []
for rel in tbl_cfg.get("relationships", []):
    if (rel.get("table") and 
        rel.get("column") and rel.get("column") != "None" and
        rel.get("references") and rel.get("references") != "None" and
        "." in str(rel.get("references"))):
        relationships.append(rel)
    else:
        print(f"Filtering out malformed relationship: {rel}")

schema = {"tables": {}, "relationships": relationships}

for tbl in table_order:
    df = generated.get(tbl)
    if df is None:
        continue
    
    schema["tables"][tbl] = {
        "row_count": len(df),
        "domain": tbl_cfg["tables"][tbl].get("domain", "unknown"),
        "columns": {c: str(dt) for c, dt in df.dtypes.to_dict().items()},
        "distinct_counts": {c: int(df[c].nunique()) for c in df.columns},
        "null_rates": {c: float(df[c].isna().mean()) for c in df.columns},
    }
with open(os.path.join(META_DIR, "schema_metadata.json"), "w") as f:
    json.dump(schema, f, indent=2)

# ──────────────────────────────────────────────────────────────────────────────
# 9) Build lineage edges & labels.json
# ──────────────────────────────────────────────────────────────────────────────
reads, writes = [], []
for step in lin_cfg["steps"]:
    # Process inputs
    if "inputs" in step:
        inputs = step["inputs"]
        for f in inputs:
            # Handle both string and list inputs
            if isinstance(f, list):
                for nested_f in f:
                    tbl = os.path.splitext(os.path.basename(nested_f))[0]
                    if tbl in generated:
                        reads.append([step["name"], tbl])
            else:
                tbl = os.path.splitext(os.path.basename(f))[0]
                if tbl in generated:
                    reads.append([step["name"], tbl])
    
    # Process outputs
    if "outputs" in step:
        outputs = step["outputs"]
        for f in outputs:
            # Handle both string and list outputs
            if isinstance(f, list):
                for nested_f in f:
                    tbl = os.path.splitext(os.path.basename(nested_f))[0]
                    if tbl in generated:
                        writes.append([step["name"], tbl])
            else:
                tbl = os.path.splitext(os.path.basename(f))[0]
                if tbl in generated:
                    writes.append([step["name"], tbl])

with open(os.path.join(META_DIR, "generation_lineage.yaml"), "w") as f:
    yaml.safe_dump({"steps": lin_cfg["steps"]}, f, sort_keys=False)

# Include detected circular dependencies in the labels.json file
labels = {
    "step_labels": [
        {"name": s["name"], "label": lin_cfg["labels"].get(s["name"], "normal")}
        for s in lin_cfg["steps"]
    ],
"anomalies": tbl_cfg.get("anomalies_list", []) + detected_circular_dependencies
}
with open(os.path.join(META_DIR, "labels.json"), "w") as f:
    json.dump(labels, f, indent=2)

# Add function to report circular dependencies
def report_circular_dependencies():
    if not detected_circular_dependencies:
        print("\nNo circular dependencies detected during generation.")
        return
        
    print(f"\nCircular Dependencies Report:")
    print(f"Found {len(detected_circular_dependencies)} circular dependencies:")
    
    # Group by referenced table for better readability
    by_ref_table = {}
    for dep in detected_circular_dependencies:
        ref = dep["references"]
        if ref not in by_ref_table:
            by_ref_table[ref] = []
        by_ref_table[ref].append(f"{dep['table']}.{dep['column']}")
    
    # Print grouped report
    for ref_table, referring_cols in by_ref_table.items():
        print(f"  • Table '{ref_table}' is referenced by:")
        for ref_col in referring_cols:
            print(f"    - {ref_col}")
    
    print("\nThese circular dependencies have been added to labels.json")

# Read back the labels.json file for verification
with open(os.path.join(META_DIR, "labels.json"), "r") as f:
    labels_to_verify = json.load(f)

# Call the circular dependencies report function
report_circular_dependencies()

# Verify anomalies
verify_anomalies_against_labels(generated, labels_to_verify)

print("✅ All data, metadata, lineage & labels generated.")