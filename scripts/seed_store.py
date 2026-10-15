"""Create and seed the demo store (data/store.db) and print a few customers to try."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpdesk.config import settings  # noqa: E402
from helpdesk.store import init_db  # noqa: E402

conn = init_db(settings.store_db)
print(f"Store at {settings.store_db}")
for row in conn.execute(
    "SELECT c.email, o.id, o.status, o.delivered_on FROM customers c JOIN orders o ON o.customer_id = c.id ORDER BY c.id LIMIT 8"
):
    print(f"  {row['email']:40s} order {row['id']}  {row['status']:10s} delivered {row['delivered_on']}")
