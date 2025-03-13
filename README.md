# Synthetic Data Warehouse Experimentation

This project generates a synthetic TPC-DS–like dataset to simulate a realistic data warehouse. The synthetic dataset is used to experiment with advanced data processing techniques, including integration with Graph Neural Networks (GNNs) and Large Language Models (LLMs) for enhancing natural language query responses and data lineage tracking.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Running the Data Generation Script](#running-the-data-generation-script)
- [Data Validation](#data-validation)
- [Dependencies](#dependencies)

## Overview

The goal of this project is to create a synthetic dataset that mimics a data warehouse based on the TPC-DS schema. This dataset includes:
- **Dimension Tables:** Customers, Products, Stores, Promotions, and Dates.
- **Fact Tables:** Orders, Transactions, Inventory, and Returns.
- **Metadata Files:** 
  - `schema_metadata.json` documents the schema, including table columns, data types, primary keys, and foreign key relationships.
  - `generation_lineage.yaml` details the ETL process, documenting each step from dimension generation to the final data output.

This synthetic data is designed to be used for experimentation with a hybrid GNN+LLM model for natural language query enhancement and data lineage tracking.

## Project Structure

```
synthetic_data_project/
├── README.md                 # Project overview and instructions
├── setup_project.bat         # Windows batch script for project setup and Conda environment creation
├── requirements.txt          # List of required Python libraries
├── config.yaml               # Optional configuration file for data generation parameters
├── data/                     # Folder for generated CSV files (customers.csv, orders.csv, etc.)
├── metadata/                 # Folder for schema metadata and ETL lineage files (JSON and YAML)
├── src/                      # Source code for data generation
│   ├── generate_data.py      # Main script to generate synthetic data and export CSV, JSON, and YAML
├── notebooks/                # Jupyter notebooks for data exploration and validation
│   └── data_validation.ipynb # Notebook to validate generated data and metadata
└── docs/                     # Additional project documentation
    └── project_documentation.md
```

## Setup Instructions

### Using the Batch Script (Windows)

1. **Open Command Prompt** in your main project directory.
2. **Run the setup script:**

```batch
   setup_project.bat
```

   This script will:
   - Create the required directory structure (`data/`, `metadata/`, `src/`, `notebooks/`, `docs/`).
   - Create placeholder files (`README.md`, `requirements.txt`, `config.yaml`).
   - Set up a Conda environment named `synthetic_data_project` with Python 3.11.
   - Install the required packages as specified in `requirements.txt`.

### Manual Setup (Alternative)

1. Create the directory structure manually as described in the **Project Structure** section.
2. Ensure you have Conda installed, then create and activate the environment:

   ```batch
   conda create -y -n synthetic_data_project python=3.11
   conda activate synthetic_data_project
   pip install -r requirements.txt
   ```

## Running the Data Generation Script

1. **Activate the Conda Environment:**

   ```batch
   conda activate synthetic_data_project
   ```

2. **Run the Python Script:**

   From the root directory, run:

   ```batch
   python src\generate_data.py
   ```

   This script will:
   - Generate synthetic data for dimension and fact tables.
   - Export CSV files to the `data/` folder.
   - Create a JSON file (`metadata/schema_metadata.json`) documenting the schema.
   - Create a YAML file (`metadata/generation_lineage.yaml`) detailing the ETL lineage.

3. **Verify the Output:**
   - Check the `data/` folder for CSV files.
   - Check the `metadata/` folder for JSON and YAML files.

## Data Validation

A Jupyter Notebook (`notebooks/data_validation.ipynb`) is provided to validate the integrity and consistency of the generated data.

### To Run the Notebook:

1. Launch Jupyter Notebook:

   ```batch
   jupyter notebook
   ```

2. Open the `data_validation.ipynb` file in the `notebooks/` folder.
3. Run all cells to perform the following checks:
   - Loading CSV files.
   - Primary key uniqueness.
   - Foreign key consistency.
   - Order total consistency.
   - Displaying schema metadata and ETL lineage.

## Dependencies

The project requires the following libraries (see `requirements.txt` for version details):

- **pandas**
- **numpy**
- **Faker**
- **PyYAML**
- **sdv**
- **snowflake-snowpark-python**
- **pyspark**
- **duckdb**
- **matplotlib** (optional)
- **seaborn** (optional)
- **scikit-learn** (optional)
- **jupyter**

These packages ensure that synthetic data generation, metadata documentation, and further experiments (including integration with Snowpark, PySpark, and DuckDB) can be conducted seamlessly with Python 3.11.


---

This README provides a comprehensive overview of your project, instructions for setup and running the scripts, and details on how the repository is organized. Feel free to customize and expand it as your project evolves.