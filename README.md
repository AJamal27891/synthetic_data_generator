# Synthetic Data Generation

A fully-configurable synthetic TPC-DS–like dataset generator, complete with schema metadata and ETL lineage documentation. This repository produces CSV files and accompanying JSON/YAML metadata for downstream analytics or data-warehouse experiments.

---

## Table of Contents

- [Overview](#overview)  
- [Folder Structure](#folder-structure)  
- [Prerequisites & Setup](#prerequisites--setup)  
- [Configuration](#configuration)  
- [Data Generation](#data-generation)  
- [Data Validation & Snowflake Load](#data-validation--snowflake-load)  
- [Source Code](#source-code)  
- [Dependencies](#dependencies)  
- [Future Directions](#future-directions)  

---

## Overview

This project generates a relational synthetic dataset modeled on TPC-DS, including:

- **Dimension tables** (e.g. customers, products, stores, promotions, dates, suppliers, employees, legacy_customers)  
- **Fact tables** (e.g. orders, transactions, inventory, returns, sales_targets)  
- **Schema metadata** (`schema_metadata.json`)  
- **ETL lineage** (`generation_lineage.yaml`, `labels.json`)  

All tables, domains, column types, and lineage steps are driven by YAML configs under `data_generation_config/`.

---

## Folder Structure

```

synthetic\_data\_generator/
│   .env
│   .gitignore
│   config.yaml
│   README.md
│   requirements.txt
│   setup\_project.bat
│
├── data/                     # Generated CSVs (varies per run)
│
├── data\_generation\_config/   # YAML configs for tables & lineage
│       tables.yml
│       lineage.yml
│
├── docs/                     # Project documentation
│       project\_documentation.md
│
├── metadata/                 # Generated metadata files
│       schema\_metadata.json
│       generation\_lineage.yaml
│       labels.json
│
├── notebooks/                # Notebooks for validation & exploration
│       data\_validation.ipynb
│
└── src/                      # Python source
generate\_data.py           # Legacy generator
enhanced\_generated\_data.py # YAML-driven generator

````

---

## Prerequisites & Setup

1. **Clone the repo** and navigate to its root:
   
```bash
   git clone <repo_url>
   cd synthetic_data_generator
```

2. **Create & activate Conda environment**:

```batch
   setup_project.bat
   conda activate synthetic_data_project
```

  This installs Python 3.11 and all packages in `requirements.txt`.

3. **Populate `.env`** in project root with any secrets (e.g. Snowflake credentials for validation):

   ```ini
   SNOWFLAKE_USER=...
   SNOWFLAKE_PASSWORD=...
   SNOWFLAKE_ACCOUNT=...
   SNOWFLAKE_ROLE=...
   SNOWFLAKE_WAREHOUSE=...
   SNOWFLAKE_DATABASE=...
   SNOWFLAKE_SCHEMA=...
   ```

---

## Configuration

* **`data_generation_config/tables.yml`**
  Defines each table’s row count, domain, column definitions (`int`, `float`, `faker`, `lookup`, `derived`, etc.), and anomaly settings.

* **`data_generation_config/lineage.yml`**
  Lists ETL steps with their inputs, outputs, and labels (e.g. `normal`, `job_failure`, `partial_write`, `mixed_operation`).

* **`config.yaml`** (optional)
  Override global parameters such as random seed or output directories.

---

## Data Generation

Two scripts:

* **Legacy** (hand-coded):

  ```bash
  python src/generate_data.py
  ```
* **Enhanced** (recommended, YAML-driven):

  ```bash
  python src/enhanced_generated_data.py
  ```

Both produce:

1. CSV files in `data/`
2. `metadata/schema_metadata.json`
3. `metadata/generation_lineage.yaml`
4. `metadata/labels.json`

---

## Data Validation & Snowflake Load

Run the Jupyter notebook:

```bash
jupyter notebook notebooks/data_validation.ipynb
```

It will:

1. Load CSVs (auto-detecting only `_date` columns)
2. Check primary key uniqueness
3. Check foreign key consistency (with date normalization)
4. Stage CSVs to Snowflake (using `.env` secrets)
5. Load tables into Snowflake according to `schema_metadata.json`

No secrets are hard-coded.

---

## Source Code

* **`src/generate_data.py`**
  A step-by-step, fixed-schema data generator.

* **`src/enhanced_generated_data.py`**
  A generic, YAML-driven generator supporting custom tables, columns, and lineage.

* **`notebooks/data_validation.ipynb`**
  Validates generated data integrity and loads into Snowflake.

* **`docs/project_documentation.md`**
  In-depth design, diagrams, and rationale.

---

## Dependencies

See `requirements.txt`:

* `pandas`, `numpy`, `Faker`, `PyYAML`
* `python-dotenv`
* `snowflake-connector-python`, `snowflake-snowpark-python`
* `jupyter`

---

## Future Directions

These enhancements are driven by the core goals of this thesis—developing and evaluating a hybrid GNN+LLM framework for data-warehouse lineage, silo discovery, and subgraph recovery:

1. **Scale & Temporal Depth**  
   - Expand the synthetic dataset from tens of thousands to millions of rows and span multiple years of dates.  
   - **Benefit:** Study how embedding quality, lineage-detection AUC/precision-recall, and clustering stability evolve as you add volume and long-range temporal patterns.

2. **Realistic Joint Distributions (SDV Integration)**  
   - Plug in SDV’s multi-table synthesizers to generate correlated customer-order-transaction data.  
   - **Benefit:** Compare GNN+LLM performance when trained on rule-based vs. statistically realistic data, and assess transferability to real enterprise pipelines.

3. **End-to-End CI/CD for Model Workflows**  
   - Automate the full cycle: YAML→CSV→metadata→PyG graph→GNN training→LLM lineage Q/A→evaluation dashboards.  
   - **Benefit:** Ensure any adjustment to table definitions or lineage steps triggers immediate retraining and metric reporting, supporting reproducible research and rapid iteration.

4. **Cross-Engine Benchmarking**  
   - Publish the same synthetic dataset to DuckDB, Spark, and Snowflake and measure how query planning differences impact downstream graph construction and LLM parsing.  
   - **Benefit:** Quantify the robustness of your hybrid pipeline across different data-warehouse backends.

5. **New Lineage & Anomaly Scenarios**  
   - Introduce more complex ETL patterns (e.g., branching merges, multi-stage transforms, schema drift) and anomaly types (e.g., delayed writes, partial joins).  
   - **Benefit:** Stress-test the GNN’s ability to detect subtle lineage errors and the LLM’s capacity to explain them in natural language.

By aligning each direction explicitly with your thesis tasks—**lineage detection**, **silo discovery**, and **lineage subgraph recovery**—you’ll build a comprehensive, reproducible suite of experiments that push the boundaries of GNN+LLM integration for data-warehouse intelligence.  
