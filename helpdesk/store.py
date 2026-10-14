"""The mock store backend: customers, products, orders, refunds, tickets.

Stands in for a real e-commerce backend (Shopify, an internal ERP). Only the
MCP server reads or writes it, so the agent can only do what the MCP tools
allow, the same boundary you'd have with a real system.

`seed()` builds a deterministic dataset with Faker so tests and the eval
always see the same customers and orders.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL, tier TEXT DEFAULT 'standard', created_on TEXT
);
CREATE TABLE IF NOT EXISTS products (
    sku TEXT PRIMARY KEY, name TEXT NOT NULL, price REAL NOT NULL, final_sale INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id), placed_on TEXT, status TEXT,
    delivered_on TEXT, carrier TEXT, tracking TEXT
);
CREATE TABLE IF NOT EXISTS order_items (
    order_id INTEGER REFERENCES orders(id), sku TEXT REFERENCES products(sku), qty INTEGER, unit_price REAL,
    PRIMARY KEY (order_id, sku)
);
CREATE TABLE IF NOT EXISTS refunds (
    id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER REFERENCES orders(id), amount REAL, reason TEXT,
    approved_by TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY, customer_email TEXT, subject TEXT, body TEXT, status TEXT DEFAULT 'open',
    category TEXT, resolution TEXT, updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

PRODUCTS = [
    ("SKU-HEAD-01", "Wireless Headphones", 89.0, 0),
    ("SKU-KEYB-02", "Mechanical Keyboard", 129.0, 0),
    ("SKU-MOUS-03", "Ergonomic Mouse", 49.0, 0),
    ("SKU-MONI-04", "27-inch Monitor", 329.0, 0),
    ("SKU-CHAI-05", "Office Chair", 249.0, 0),
    ("SKU-CABL-06", "USB-C Cable 2m", 15.0, 0),
    ("SKU-LAMP-07", "Desk Lamp", 39.0, 0),
    ("SKU-GIFT-08", "Gift Card 50", 50.0, 1),
    ("SKU-CLRN-09", "Clearance Webcam", 25.0, 1),
    ("SKU-LAPT-10", "Laptop Stand", 59.0, 0),
]
STATUSES = ["processing", "shipped", "delivered", "delivered", "delivered", "cancelled"]
CARRIERS = ["BlueDart", "Delhivery", "DHL", "UPS"]


def connect(path: Path | str) -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def seed(conn: sqlite3.Connection, today: date, n_customers: int = 40, seed: int = 42) -> None:
    from faker import Faker

    rng = random.Random(seed)
    fake = Faker()
    Faker.seed(seed)
    conn.executescript(SCHEMA)
    if conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]:
        return  # already seeded
    conn.executemany("INSERT INTO products VALUES (?,?,?,?)", PRODUCTS)
    order_id = 1000
    for cid in range(1, n_customers + 1):
        name = fake.name()
        email = f"{name.lower().replace(' ', '.').replace('..', '.')}{cid}@example.com"
        tier = "gold" if rng.random() < 0.15 else "standard"
        conn.execute("INSERT INTO customers VALUES (?,?,?,?,?)", (cid, name, email, tier, str(today - timedelta(days=rng.randint(60, 900)))))
        for _ in range(rng.randint(1, 4)):
            order_id += 1
            placed = today - timedelta(days=rng.randint(1, 90))
            status = rng.choice(STATUSES)
            delivered = placed + timedelta(days=rng.randint(2, 7)) if status == "delivered" else None
            if delivered and delivered > today:
                delivered = today
            carrier = rng.choice(CARRIERS) if status in ("shipped", "delivered") else None
            tracking = f"{carrier[:2].upper()}{rng.randint(10**8, 10**9 - 1)}" if carrier else None
            conn.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?)",
                         (order_id, cid, str(placed), status, str(delivered) if delivered else None, carrier, tracking))
            for sku, _, price, _ in rng.sample(PRODUCTS, rng.randint(1, 3)):
                conn.execute("INSERT INTO order_items VALUES (?,?,?,?)", (order_id, sku, rng.randint(1, 2), price))
    conn.commit()


def init_db(path: Path | str, today: date | None = None) -> sqlite3.Connection:
    conn = connect(path)
    seed(conn, today or date.today())
    return conn
