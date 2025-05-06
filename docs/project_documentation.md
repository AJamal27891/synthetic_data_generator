# Project Documentation

This document provides a technical overview and detailed design of the synthetic data generation project. It explains the architecture, data generation methodology, and ETL lineage process. Additionally, it includes Mermaid diagrams to visually represent the repository structure, ETL flow, and schema relationships.

---

## 1. Project Architecture

The project is organized into several key components: synthetic data generation, metadata documentation, ETL lineage tracking, and validation. Below is a high-level view of the repository structure.

```mermaid
graph TD;
    A[synthetic_data_project/]
    A --> B[README.md]
    A --> C[setup_project.bat]
    A --> D[requirements.txt]
    A --> E[config.yaml]
    A --> F[data/]
    A --> G[metadata/]
    A --> H[src/]
    H --> H1[generate_data.py]
    H --> H2[enhanced_data_generator.py]
    A --> I[notebooks/]
    A --> J[docs/]
````

* **data/**: Contains the generated CSV files.
* **metadata/**: Contains JSON schema metadata and YAML ETL lineage files.
* **src/**:

  * **generate\_data.py** – original, hard-coded generator
  * **enhanced\_data\_generator.py** – new, fully-configurable generator driven by YAML
* **notebooks/**: Contains notebooks for data validation and analysis.
* **docs/**: Includes this documentation and other design notes.

---

## 2. Synthetic Data Generation Design

### 2.1 Overview of the Synthetic Schema

The synthetic dataset mimics a data warehouse modeled after the TPC-DS benchmark. It consists of:

* **Dimension Tables:**

  * **Customers** – Profiles including names, contact details, and loyalty status.
  * **Products** – Details like product name, category, sub-category, price, and cost.
  * **Stores** – Information about store locations and regions.
  * **Promotions** – Promotion details with discount rates and effective dates.
  * **Dates** – A date dimension covering one year for time-based analysis.
  * **Suppliers**, **Employees**, **Legacy\_Customers** – Additional operational dimensions.

* **Fact Tables:**

  * **Orders** – Order header information linking to customers and stores.
  * **Transactions** – Line-items linking to orders, products, promotions, and suppliers.
  * **Inventory** – Stock snapshots per store and product.
  * **Returns** – Records of returned items for a subset of orders.
  * **Sales\_Targets** – Monthly targets per store.

### 2.2 Data Generation Process

We have two generators:

1. **generate\_data.py** – a step-by-step, hand-coded generator.
2. **enhanced\_data\_generator.py** – a fully YAML-driven, generic approach that reads:

   * `data_generation_config/tables.yml` for all table definitions, domains, column types, and anomaly rates.
   * `data_generation_config/lineage.yml` for the ETL steps, inputs, outputs, and labels.

The enhanced generator ensures:

* **Configurability**: add new tables, columns, or lineage steps by editing YAML only.
* **Controlled noise**: anomaly rates, “ghost” outputs, and missing FKs are specified per-table.
* **Referential integrity**: lookups and derived columns sample only from the generated dimension data.

```mermaid
graph LR;
    A[Read tables.yml] --> B[Generate Dimension Tables];
    A --> C[Generate Fact Tables];
    B --> D[Inject Anomalies (missing FK, ghost)];
    E[Read lineage.yml] --> F[Build reads_from & writes_to edges];
    D --> G[Write CSVs];
    F --> H[Write lineage & labels YAML/JSON];
```

---

## 3. Schema Documentation

The schema metadata lives in `metadata/schema_metadata.json`. It defines each table’s:

* **row\_count**
* **columns** with data types and optional descriptions
* **domain** tags for clustering
* **primary\_key** (inferred or explicit)
* **foreign\_key** relationships for validation

### Sample Schema Metadata (Excerpt)

```json
{
  "tables": {
    "customers": {
      "row_count": 10000,
      "domain": "sales",
      "columns": {
        "customer_id":    { "type": "INTEGER" },
        "first_name":     { "type": "VARCHAR" },
        "last_name":      { "type": "VARCHAR" },
        "email":          { "type": "VARCHAR" },
        "phone":          { "type": "VARCHAR" },
        "city":           { "type": "VARCHAR" },
        "state":          { "type": "CHAR(2)" },
        "loyalty_status": { "type": "VARCHAR", "choices": ["Bronze","Silver","Gold"] }
      },
      "primary_key": "customer_id"
    }
    // … other tables …
  },
  "relationships": [
    { "table": "orders", "column": "customer_id", "references": "customers.customer_id" },
    { "table": "orders", "column": "store_id",    "references": "stores.store_id" }
    // … more …
  ]
}
```

---

## 4. ETL Lineage Documentation

The ETL steps are captured in `metadata/generation_lineage.yaml`, reflecting exactly the steps in your generator:

```yaml
steps:
  - name: extract_customers
    outputs: ["customers.csv"]
  - name: load_orders
    inputs:  ["customers.csv","stores.csv","dates.csv"]
    outputs: ["orders.csv"]
  - name: transform_inventory
    inputs:  ["stores.csv","products.csv","transactions.csv"]
    outputs: ["inventory.csv"]
  # … all other steps including archive_orders, transform_sales_data, merge_legacy_customers …
labels:
  archive_orders:        partial_write
  transform_sales_data:  job_failure
  merge_legacy_customers: mixed_operation
```

This lineage feed drives both:

* **Data validation** (which CSVs to expect)
* **Graph construction** for GNN training (`reads_from` / `writes_to` edges)

---

## 5. Validation & Snowflake Load

A set of Jupyter notebooks under **notebooks/** perform:

1. **Config‐driven CSV loading** (parse only `_date` columns)
2. **Primary key checks** – ensure each `*_id` column is unique
3. **Foreign key checks** – normalize dates, allow configurable noise
4. **Staging to Snowflake** – using credentials from `.env`
5. **Dynamic schema‐driven load** – leveraging `parse_header` and `schema_metadata.json`

---

## 6. Future Directions

* **Scale out** to larger row counts or multi-year date ranges.
* **Integrate SDV** multi-table synthesizers for more realistic joint distributions.
* **Hybrid GNN+LLM**: use this dataset to benchmark lineage extraction, silo detection, and subgraph recovery tasks.
* **CI/CD**: automate generation, validation, and Snowflake loading in a pipeline.

---

## 7. Getting Started

1. **Install dependencies**:

   ```bash
   conda env create -f environment.yml
   pip install -r requirements.txt
   ```
2. **Generate data**:

   ```bash
   python src/enhanced_data_generator.py
   ```
3. **Validate & load**:

   * Open `notebooks/data_validation.ipynb` and run all cells.
   * Snowflake credentials must be set in your project root `.env`.

```

