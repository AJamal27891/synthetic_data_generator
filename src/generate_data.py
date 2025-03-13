#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from faker import Faker
import json
import yaml
from datetime import datetime, timedelta

# Initialize Faker and set random seed for reproducibility
fake = Faker()
np.random.seed(42)

# Define record counts
num_customers = 10000
num_products = 1000
num_stores = 100
num_promotions = 100
num_orders = 10000

# ----------------------------
# Dimension Tables Generation
# ----------------------------

# Customers Table
customers = pd.DataFrame({
    'customer_id': range(1, num_customers + 1),
    'first_name': [fake.first_name() for _ in range(num_customers)],
    'last_name': [fake.last_name() for _ in range(num_customers)],
    'email': [fake.email() for _ in range(num_customers)],
    'phone': [fake.phone_number() for _ in range(num_customers)],
    'address': [fake.address().replace("\n", ", ") for _ in range(num_customers)],
    'city': [fake.city() for _ in range(num_customers)],
    'state': [fake.state_abbr() for _ in range(num_customers)],
    'zip_code': [fake.zipcode() for _ in range(num_customers)],
    'loyalty_status': np.random.choice(['Bronze', 'Silver', 'Gold'], num_customers, p=[0.6, 0.3, 0.1])
})

# Products Table
categories = ['Electronics', 'Clothing', 'Home', 'Sports', 'Toys']
sub_categories = {
    'Electronics': ['Mobile', 'Laptop', 'Camera'],
    'Clothing': ['Men', 'Women', 'Kids'],
    'Home': ['Furniture', 'Kitchen', 'Decor'],
    'Sports': ['Fitness', 'Outdoor', 'Team'],
    'Toys': ['Educational', 'Action', 'Puzzle']
}

products_list = []
for i in range(1, num_products + 1):
    cat = np.random.choice(categories)
    sub_cat = np.random.choice(sub_categories[cat])
    price = round(np.random.uniform(5, 500), 2)
    cost = round(price * np.random.uniform(0.5, 0.9), 2)
    products_list.append({
        'product_id': i,
        'product_name': fake.catch_phrase(),
        'category': cat,
        'sub_category': sub_cat,
        'price': price,
        'cost': cost
    })
products = pd.DataFrame(products_list)

# Stores Table
stores = pd.DataFrame({
    'store_id': range(1, num_stores + 1),
    'store_name': [f"{fake.company()} Store" for _ in range(num_stores)],
    'address': [fake.address().replace("\n", ", ") for _ in range(num_stores)],
    'city': [fake.city() for _ in range(num_stores)],
    'state': [fake.state_abbr() for _ in range(num_stores)],
    'zip_code': [fake.zipcode() for _ in range(num_stores)],
    'region': np.random.choice(['North', 'South', 'East', 'West'], num_stores)
})

# Promotions Table
promotions = pd.DataFrame({
    'promo_id': range(1, num_promotions + 1),
    'promo_name': [f"Promo {i}" for i in range(1, num_promotions + 1)],
    'discount_rate': [round(np.random.uniform(0.05, 0.5), 2) for _ in range(num_promotions)],
    'start_date': [fake.date_between(start_date='-1y', end_date='today') for _ in range(num_promotions)],
    'end_date': [fake.date_between(start_date='today', end_date='+1y') for _ in range(num_promotions)]
})

# Dates Dimension Table (One year)
start_date = datetime.now() - timedelta(days=365)
dates_list = []
for i in range(365):
    current_date = start_date + timedelta(days=i)
    dates_list.append({
        'date_id': i + 1,
        'full_date': current_date.strftime('%Y-%m-%d'),
        'day': current_date.day,
        'month': current_date.month,
        'year': current_date.year,
        'weekday': current_date.strftime('%A')
    })
dates = pd.DataFrame(dates_list)

# ----------------------------
# Fact Tables Generation
# ----------------------------

# Raw Orders Generation: Minimal fields (to be enriched later)
order_dates = pd.to_datetime(np.random.choice(dates['full_date'], num_orders))
raw_orders = pd.DataFrame({
    'order_id': range(1, num_orders + 1),
    'order_date': order_dates,
    'customer_id': np.random.choice(customers['customer_id'], num_orders),
    'store_id': np.random.choice(stores['store_id'], num_orders)
})

# Raw Transactions Generation: Generate 1-5 transactions per order
transactions_list = []
transaction_id = 1
for order in raw_orders.itertuples():
    num_items = np.random.randint(1, 6)
    for _ in range(num_items):
        prod = products.sample(1).iloc[0]
        quantity = np.random.randint(1, 6)
        base_price = prod['price'] * quantity
        apply_promo = np.random.rand() < 0.2  # 20% chance
        promo_id = np.random.choice(promotions['promo_id']) if apply_promo else None
        discount = base_price * np.random.uniform(0.05, 0.3) if apply_promo else 0
        line_price = round(base_price - discount, 2)
        transactions_list.append({
            'transaction_id': transaction_id,
            'order_id': order.order_id,
            'product_id': prod['product_id'],
            'quantity': quantity,
            'price': line_price,
            'promo_id': promo_id
        })
        transaction_id += 1
transactions = pd.DataFrame(transactions_list)

# Aggregate Transactions: Compute order totals
order_totals = transactions.groupby('order_id')['price'].sum().reset_index()
orders = raw_orders.merge(order_totals, on='order_id', how='left')
orders.rename(columns={'price': 'order_total'}, inplace=True)
# Add order_status field
orders['order_status'] = np.random.choice(['Completed', 'Pending', 'Cancelled'], num_orders, p=[0.8, 0.15, 0.05])

# Inventory Generation: Each store gets inventory for 100 random products
inventory_list = []
for store in stores.itertuples():
    sampled_products = products.sample(100)
    for prod in sampled_products.itertuples():
        inventory_list.append({
            'store_id': store.store_id,
            'product_id': prod.product_id,
            'inventory_date': fake.date_between(start_date='-30d', end_date='today'),
            'stock_quantity': np.random.randint(0, 1000)
        })
inventory = pd.DataFrame(inventory_list)

# Returns Generation: 10% of orders have returns; select one transaction per order
returns_list = []
for order in orders.sample(frac=0.1).itertuples():
    related_trans = transactions[transactions['order_id'] == order.order_id]
    if not related_trans.empty:
        trans = related_trans.sample(1).iloc[0]
        returns_list.append({
            'return_id': len(returns_list) + 1,
            'order_id': order.order_id,
            'product_id': trans['product_id'],
            'return_date': fake.date_between(start_date=order.order_date, end_date='today'),
            'return_reason': np.random.choice(['Defective', 'Wrong Item', 'Customer Dissatisfaction'])
        })
returns = pd.DataFrame(returns_list)

# ----------------------------
# Export Data as CSV Files
# ----------------------------
customers.to_csv('data/customers.csv', index=False)
products.to_csv('data/products.csv', index=False)
stores.to_csv('data/stores.csv', index=False)
promotions.to_csv('data/promotions.csv', index=False)
dates.to_csv('data/dates.csv', index=False)
orders.to_csv('data/orders.csv', index=False)
transactions.to_csv('data/transactions.csv', index=False)
inventory.to_csv('data/inventory.csv', index=False)
returns.to_csv('data/returns.csv', index=False)

# ----------------------------
# Generate JSON Metadata for Schema Documentation
# ----------------------------
schema_metadata = {
    "tables": {
        "customers": {
            "row_count": num_customers,
            "columns": {
                "customer_id": {"type": "INTEGER", "description": "Unique customer identifier"},
                "first_name": {"type": "VARCHAR(50)", "description": "First name"},
                "last_name": {"type": "VARCHAR(50)", "description": "Last name"},
                "email": {"type": "VARCHAR(100)", "description": "Email address"},
                "phone": {"type": "VARCHAR(20)", "description": "Phone number"},
                "address": {"type": "VARCHAR(100)", "description": "Address"},
                "city": {"type": "VARCHAR(50)", "description": "City name"},
                "state": {"type": "CHAR(2)", "description": "State code"},
                "zip_code": {"type": "VARCHAR(10)", "description": "ZIP code"},
                "loyalty_status": {"type": "VARCHAR(10)", "description": "Loyalty status"}
            },
            "primary_key": "customer_id"
        },
        "products": {
            "row_count": num_products,
            "columns": {
                "product_id": {"type": "INTEGER", "description": "Unique product identifier"},
                "product_name": {"type": "VARCHAR(100)", "description": "Product name"},
                "category": {"type": "VARCHAR(50)", "description": "Product category"},
                "sub_category": {"type": "VARCHAR(50)", "description": "Product sub-category"},
                "price": {"type": "DECIMAL(7,2)", "description": "Product price"},
                "cost": {"type": "DECIMAL(7,2)", "description": "Product cost"}
            },
            "primary_key": "product_id"
        },
        "stores": {
            "row_count": num_stores,
            "columns": {
                "store_id": {"type": "INTEGER", "description": "Unique store identifier"},
                "store_name": {"type": "VARCHAR(100)", "description": "Store name"},
                "address": {"type": "VARCHAR(100)", "description": "Store address"},
                "city": {"type": "VARCHAR(50)", "description": "City name"},
                "state": {"type": "CHAR(2)", "description": "State code"},
                "zip_code": {"type": "VARCHAR(10)", "description": "ZIP code"},
                "region": {"type": "VARCHAR(10)", "description": "Store region"}
            },
            "primary_key": "store_id"
        },
        "promotions": {
            "row_count": num_promotions,
            "columns": {
                "promo_id": {"type": "INTEGER", "description": "Unique promotion identifier"},
                "promo_name": {"type": "VARCHAR(50)", "description": "Promotion name"},
                "discount_rate": {"type": "DECIMAL(3,2)", "description": "Discount rate"},
                "start_date": {"type": "DATE", "description": "Promotion start date"},
                "end_date": {"type": "DATE", "description": "Promotion end date"}
            },
            "primary_key": "promo_id"
        },
        "dates": {
            "row_count": 365,
            "columns": {
                "date_id": {"type": "INTEGER", "description": "Unique date identifier"},
                "full_date": {"type": "DATE", "description": "Full date in YYYY-MM-DD format"},
                "day": {"type": "INTEGER", "description": "Day of month"},
                "month": {"type": "INTEGER", "description": "Month number"},
                "year": {"type": "INTEGER", "description": "Year"},
                "weekday": {"type": "VARCHAR(10)", "description": "Weekday name"}
            },
            "primary_key": "date_id"
        },
        "orders": {
            "row_count": num_orders,
            "columns": {
                "order_id": {"type": "INTEGER", "description": "Unique order identifier"},
                "order_date": {"type": "DATE", "description": "Order date"},
                "customer_id": {"type": "INTEGER", "description": "Customer who placed the order"},
                "store_id": {"type": "INTEGER", "description": "Store fulfilling the order"},
                "order_total": {"type": "DECIMAL(10,2)", "description": "Total order amount"},
                "order_status": {"type": "VARCHAR(20)", "description": "Order status"}
            },
            "primary_key": "order_id",
            "foreign_keys": {
                "customer_id": {"references": "customers.customer_id"},
                "store_id": {"references": "stores.store_id"}
            }
        },
        "transactions": {
            "row_count": len(transactions),
            "columns": {
                "transaction_id": {"type": "INTEGER", "description": "Unique transaction identifier"},
                "order_id": {"type": "INTEGER", "description": "Order to which this transaction belongs"},
                "product_id": {"type": "INTEGER", "description": "Product sold"},
                "quantity": {"type": "INTEGER", "description": "Quantity purchased"},
                "price": {"type": "DECIMAL(7,2)", "description": "Line price (after discount)"},
                "promo_id": {"type": "INTEGER", "description": "Promotion applied (if any)"}
            },
            "primary_key": "transaction_id",
            "foreign_keys": {
                "order_id": {"references": "orders.order_id"},
                "product_id": {"references": "products.product_id"},
                "promo_id": {"references": "promotions.promo_id"}
            }
        },
        "inventory": {
            "row_count": len(inventory),
            "columns": {
                "store_id": {"type": "INTEGER", "description": "Store identifier"},
                "product_id": {"type": "INTEGER", "description": "Product identifier"},
                "inventory_date": {"type": "DATE", "description": "Date of inventory snapshot"},
                "stock_quantity": {"type": "INTEGER", "description": "Available stock quantity"}
            },
            "foreign_keys": {
                "store_id": {"references": "stores.store_id"},
                "product_id": {"references": "products.product_id"}
            }
        },
        "returns": {
            "row_count": len(returns),
            "columns": {
                "return_id": {"type": "INTEGER", "description": "Unique return identifier"},
                "order_id": {"type": "INTEGER", "description": "Associated order"},
                "product_id": {"type": "INTEGER", "description": "Returned product"},
                "return_date": {"type": "DATE", "description": "Date of return"},
                "return_reason": {"type": "VARCHAR(50)", "description": "Reason for return"}
            },
            "primary_key": "return_id",
            "foreign_keys": {
                "order_id": {"references": "orders.order_id"},
                "product_id": {"references": "products.product_id"}
            }
        }
    },
    "relationships": [
        {"table": "orders", "column": "customer_id", "references": "customers.customer_id"},
        {"table": "orders", "column": "store_id", "references": "stores.store_id"},
        {"table": "transactions", "column": "order_id", "references": "orders.order_id"},
        {"table": "transactions", "column": "product_id", "references": "products.product_id"},
        {"table": "transactions", "column": "promo_id", "references": "promotions.promo_id"},
        {"table": "inventory", "column": "store_id", "references": "stores.store_id"},
        {"table": "inventory", "column": "product_id", "references": "products.product_id"},
        {"table": "returns", "column": "order_id", "references": "orders.order_id"},
        {"table": "returns", "column": "product_id", "references": "products.product_id"}
    ]
}

with open('metadata/schema_metadata.json', 'w') as json_file:
    json.dump(schema_metadata, json_file, indent=4)

# ----------------------------
# Generate YAML for ETL Lineage Documentation
# ----------------------------
etl_lineage = {
    "steps": [
        {
            "name": "generate_dimensions",
            "description": "Generate base dimension tables using Faker: customers, products, stores, promotions, and dates.",
            "outputs": ["data/customers.csv", "data/products.csv", "data/stores.csv", "data/promotions.csv", "data/dates.csv"]
        },
        {
            "name": "generate_raw_orders",
            "description": "Generate raw orders data by randomly selecting customer, store, and date values.",
            "inputs": ["data/customers.csv", "data/stores.csv", "data/dates.csv"],
            "output": "data/raw_orders.csv"
        },
        {
            "name": "generate_raw_transactions",
            "description": "For each order, generate 1–5 raw transactions referencing products and promotions.",
            "inputs": ["data/raw_orders.csv", "data/products.csv", "data/promotions.csv"],
            "output": "data/transactions_raw.csv"
        },
        {
            "name": "aggregate_transactions",
            "description": "Aggregate transactions to compute order totals.",
            "inputs": ["data/transactions_raw.csv"],
            "output": "data/order_aggregates.csv"
        },
        {
            "name": "process_orders",
            "description": "Merge raw orders with aggregated totals and assign order statuses.",
            "inputs": ["data/raw_orders.csv", "data/order_aggregates.csv"],
            "output": "data/orders.csv"
        },
        {
            "name": "update_inventory",
            "description": "Update inventory based on sales data.",
            "inputs": ["data/stores.csv", "data/products.csv", "data/transactions_raw.csv"],
            "output": "data/inventory.csv"
        },
        {
            "name": "generate_returns",
            "description": "Generate returns for a subset of orders.",
            "inputs": ["data/orders.csv", "data/transactions_raw.csv", "data/products.csv"],
            "output": "data/returns.csv"
        },
        {
            "name": "compile_full_lineage",
            "description": "Compile all steps into a comprehensive ETL lineage document.",
            "inputs": [
                "data/customers.csv", "data/products.csv", "data/stores.csv",
                "data/promotions.csv", "data/dates.csv", "data/raw_orders.csv",
                "data/transactions_raw.csv", "data/order_aggregates.csv",
                "data/orders.csv", "data/inventory.csv", "data/returns.csv"
            ],
            "output": "metadata/generation_lineage.yaml"
        }
    ]
}

with open('metadata/generation_lineage.yaml', 'w') as yaml_file:
    yaml.dump(etl_lineage, yaml_file, sort_keys=False)

print("Synthetic data generation complete. CSV files saved in 'data/', metadata in 'metadata/' folder.")
