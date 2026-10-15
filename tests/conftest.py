from datetime import date, timedelta

import pytest

from helpdesk.store import connect, seed

TODAY = date(2026, 10, 14)


@pytest.fixture
def conn(monkeypatch):
    monkeypatch.setenv("HELPDESK_TODAY", TODAY.isoformat())
    c = connect(":memory:")
    seed(c, TODAY)
    # Fixed fixtures on top of the random seed, so tests don't depend on Faker output
    c.execute("INSERT INTO customers VALUES (900, 'Asha Rao', 'asha@example.com', 'standard', '2025-01-01')")
    c.execute("INSERT INTO customers VALUES (901, 'Ravi Iyer', 'ravi@example.com', 'gold', '2025-01-01')")
    d = lambda n: str(TODAY - timedelta(days=n))
    rows = [  # id, customer, placed, status, delivered
        (5001, 900, d(12), "delivered", d(10)),   # in window
        (5002, 900, d(50), "delivered", d(45)),   # outside window
        (5003, 900, d(2), "processing", None),    # cancellable
        (5004, 900, d(3), "shipped", None),       # in transit
        (5005, 901, d(8), "delivered", d(5)),     # someone else's order
        (5006, 900, d(9), "delivered", d(6)),     # contains a final-sale item
    ]
    for oid, cid, placed, status, delivered in rows:
        c.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?)", (oid, cid, placed, status, delivered, None, None))
    items = [(5001, "SKU-MONI-04", 1, 329.0), (5002, "SKU-HEAD-01", 1, 89.0), (5003, "SKU-CABL-06", 2, 15.0),
             (5004, "SKU-MOUS-03", 1, 49.0), (5005, "SKU-CHAI-05", 1, 249.0), (5006, "SKU-LAMP-07", 1, 39.0), (5006, "SKU-GIFT-08", 1, 50.0)]
    c.executemany("INSERT INTO order_items VALUES (?,?,?,?)", items)
    c.commit()
    return c


