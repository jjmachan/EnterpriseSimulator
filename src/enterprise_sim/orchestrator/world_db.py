"""SQLite world database: schema creation and seed data."""

import json
import sqlite3
from pathlib import Path

SHARED_DIR = Path(__file__).resolve().parent.parent / "shared"
DEFAULT_DB_PATH = SHARED_DIR / "world.db"


def get_db_path() -> Path:
    import os

    return Path(os.environ.get("ENTERPRISE_SIM_DB_PATH", str(DEFAULT_DB_PATH)))


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db(db_path: Path | None = None) -> None:
    conn = get_connection(db_path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS customers (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            phone TEXT,
            address TEXT,
            persona_traits TEXT,
            patience_level REAL DEFAULT 0.5,
            satisfaction_score REAL DEFAULT 0.7,
            vip_status BOOLEAN DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS products (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            category TEXT,
            price REAL NOT NULL,
            stock_level INTEGER DEFAULT 0,
            restock_eta TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS orders (
            id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            total REAL NOT NULL,
            shipping_address TEXT,
            tracking_number TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );

        CREATE TABLE IF NOT EXISTS order_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT NOT NULL,
            product_id TEXT NOT NULL,
            quantity INTEGER DEFAULT 1,
            unit_price REAL NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );

        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            order_id TEXT,
            subject TEXT NOT NULL,
            status TEXT DEFAULT 'open',
            priority TEXT DEFAULT 'normal',
            assigned_agent TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            resolved_at DATETIME,
            FOREIGN KEY (customer_id) REFERENCES customers(id),
            FOREIGN KEY (order_id) REFERENCES orders(id)
        );

        CREATE TABLE IF NOT EXISTS ticket_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id INTEGER NOT NULL,
            sender_id TEXT NOT NULL,
            sender_role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ticket_id) REFERENCES tickets(id)
        );

        CREATE TABLE IF NOT EXISTS knowledge_base (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            category TEXT,
            tags TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT NOT NULL,
            type TEXT NOT NULL,
            amount REAL NOT NULL,
            reason TEXT,
            approved_by TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (order_id) REFERENCES orders(id)
        );

        CREATE TABLE IF NOT EXISTS callbacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            ticket_id INTEGER,
            scheduled_time DATETIME NOT NULL,
            reason TEXT,
            status TEXT DEFAULT 'scheduled',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );

        CREATE TABLE IF NOT EXISTS channels (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            members TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS channel_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id TEXT NOT NULL,
            sender_id TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (channel_id) REFERENCES channels(id)
        );
        """
    )
    # FTS table for knowledge base search
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts
        USING fts5(title, content, tags, content=knowledge_base, content_rowid=id)
        """
    )
    conn.commit()
    conn.close()


def seed_db(db_path: Path | None = None) -> None:
    conn = get_connection(db_path)

    # Check if already seeded
    count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    if count > 0:
        conn.close()
        return

    # --- Customers (12) ---
    customers = [
        ("customer_001", "Sarah Chen", "sarah.chen@email.com", "415-555-0101",
         "742 Evergreen Terrace, San Francisco, CA 94110",
         json.dumps(["direct", "impatient", "detail-oriented"]), 0.4, 0.7, False),
        ("customer_002", "Marcus Johnson", "marcus.j@email.com", "415-555-0102",
         "1234 Oak Ave, San Francisco, CA 94102",
         json.dumps(["friendly", "patient", "verbose"]), 0.8, 0.8, False),
        ("customer_003", "Maria Lopez", "maria.lopez@email.com", "650-555-0103",
         "567 Palm Dr, Palo Alto, CA 94301",
         json.dumps(["assertive", "impatient", "concise"]), 0.3, 0.6, False),
        ("customer_004", "David Kim", "david.kim@email.com", "408-555-0104",
         "890 Maple St, San Jose, CA 95112",
         json.dumps(["polite", "analytical", "thorough"]), 0.7, 0.75, True),
        ("customer_005", "Emily Watson", "emily.w@email.com", "510-555-0105",
         "321 Cedar Ln, Oakland, CA 94612",
         json.dumps(["emotional", "expressive", "loyal"]), 0.5, 0.65, False),
        ("customer_006", "James O'Brien", "james.ob@email.com", "415-555-0106",
         "456 Pine St, San Francisco, CA 94108",
         json.dumps(["sarcastic", "demanding", "tech-savvy"]), 0.35, 0.55, False),
        ("customer_007", "Priya Patel", "priya.p@email.com", "650-555-0107",
         "789 Birch Way, Mountain View, CA 94043",
         json.dumps(["calm", "methodical", "fair"]), 0.7, 0.8, True),
        ("customer_008", "Robert Taylor", "robert.t@email.com", "408-555-0108",
         "234 Elm Ct, Sunnyvale, CA 94086",
         json.dumps(["gruff", "no-nonsense", "busy"]), 0.3, 0.5, False),
        ("customer_009", "Lisa Chang", "lisa.chang@email.com", "510-555-0109",
         "678 Willow Ave, Berkeley, CA 94704",
         json.dumps(["cheerful", "understanding", "detail-oriented"]), 0.8, 0.85, False),
        ("customer_010", "Tom Martinez", "tom.m@email.com", "415-555-0110",
         "901 Spruce Blvd, San Francisco, CA 94115",
         json.dumps(["frustrated", "repeat-caller", "skeptical"]), 0.25, 0.4, False),
        ("customer_011", "Amanda Foster", "amanda.f@email.com", "650-555-0111",
         "543 Ash Dr, Redwood City, CA 94063",
         json.dumps(["professional", "concise", "reasonable"]), 0.6, 0.7, True),
        ("customer_012", "Kevin Nguyen", "kevin.n@email.com", "408-555-0112",
         "876 Poplar St, Santa Clara, CA 95054",
         json.dumps(["quiet", "patient", "passive"]), 0.9, 0.75, False),
    ]
    conn.executemany(
        "INSERT INTO customers VALUES (?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
        customers,
    )

    # --- Products (25) ---
    products = [
        ("prod_001", "ErgoDesk Pro Standing Desk", "Height-adjustable standing desk with programmable presets", "desks", 549.99, 45, None),
        ("prod_002", "ErgoDesk Basic", "Manual crank standing desk", "desks", 299.99, 120, None),
        ("prod_003", "ComfortMax Ergonomic Chair", "Full mesh ergonomic chair with lumbar support", "chairs", 449.99, 30, None),
        ("prod_004", "ComfortMax Executive Chair", "Premium leather executive chair", "chairs", 699.99, 15, None),
        ("prod_005", "FlexiArm Monitor Mount - Single", "Single monitor arm, 17-32 inch displays", "accessories", 89.99, 200, None),
        ("prod_006", "FlexiArm Monitor Mount - Dual", "Dual monitor arm, 17-27 inch displays", "accessories", 149.99, 85, None),
        ("prod_007", "CableHub Pro", "Under-desk cable management tray", "accessories", 39.99, 300, None),
        ("prod_008", "ErgoMat Anti-Fatigue Mat", "Standing desk anti-fatigue mat", "accessories", 59.99, 150, None),
        ("prod_009", "LuminaTask Desk Lamp", "LED desk lamp with adjustable color temperature", "lighting", 79.99, 90, None),
        ("prod_010", "KeyComfort Ergonomic Keyboard", "Split ergonomic mechanical keyboard", "peripherals", 169.99, 60, None),
        ("prod_011", "GlideTrack Ergonomic Mouse", "Vertical ergonomic mouse", "peripherals", 69.99, 110, None),
        ("prod_012", "DeskPad Premium", "Large desk pad, vegan leather", "accessories", 44.99, 250, None),
        ("prod_013", "FootRest Adjustable", "Adjustable under-desk footrest", "accessories", 49.99, 75, None),
        ("prod_014", "PrivacyScreen 27\"", "Monitor privacy filter, 27 inch", "accessories", 54.99, 40, None),
        ("prod_015", "ErgoDesk L-Shape", "L-shaped standing desk with dual motors", "desks", 799.99, 20, None),
        ("prod_016", "FileCab Mobile", "3-drawer mobile filing cabinet", "storage", 179.99, 55, None),
        ("prod_017", "ShelfUnit Modular", "Modular desk shelf organizer", "storage", 99.99, 70, None),
        ("prod_018", "WhiteBoard Desktop", "Small desktop whiteboard with markers", "accessories", 29.99, 180, None),
        ("prod_019", "Webcam ProHD", "1080p webcam with privacy shutter", "peripherals", 89.99, 95, None),
        ("prod_020", "Headset ComfortCall", "Noise-canceling USB headset", "peripherals", 129.99, 65, None),
        ("prod_021", "BookStand Adjustable", "Adjustable book/tablet stand", "accessories", 34.99, 140, None),
        ("prod_022", "DeskDrawer Attachable", "Under-desk sliding drawer", "storage", 44.99, 100, None),
        ("prod_023", "PlantPot Desktop Set", "Set of 3 desktop planters", "decor", 24.99, 200, None),
        ("prod_024", "AirPurifier Desktop", "Small HEPA desktop air purifier", "wellness", 119.99, 35, None),
        ("prod_025", "WristRest Gel Combo", "Keyboard and mouse gel wrist rest set", "accessories", 29.99, 160, None),
    ]
    conn.executemany(
        "INSERT INTO products VALUES (?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
        products,
    )

    # --- Orders (40) ---
    orders = [
        # Sarah Chen - 3 orders
        ("ord_001", "customer_001", "delivered", 549.99, "742 Evergreen Terrace, San Francisco, CA 94110", "FEDX-1001", "2026-01-15"),
        ("ord_002", "customer_001", "delivered", 449.99, "742 Evergreen Terrace, San Francisco, CA 94110", "FEDX-1002", "2026-01-20"),
        ("ord_003", "customer_001", "shipped", 89.99, "742 Evergreen Terrace, San Francisco, CA 94110", "FEDX-1003", "2026-03-01"),
        # Marcus Johnson - 2 orders
        ("ord_004", "customer_002", "delivered", 299.99, "1234 Oak Ave, San Francisco, CA 94102", "UPS-2001", "2026-02-01"),
        ("ord_005", "customer_002", "pending", 169.99, "1234 Oak Ave, San Francisco, CA 94102", None, "2026-03-05"),
        # Maria Lopez - 2 orders
        ("ord_006", "customer_003", "shipped", 449.99, "567 Palm Dr, Palo Alto, CA 94301", "FEDX-3001", "2026-02-20"),
        ("ord_007", "customer_003", "delivered", 59.99, "567 Palm Dr, Palo Alto, CA 94301", "FEDX-3002", "2026-01-10"),
        # David Kim - 3 orders (VIP)
        ("ord_008", "customer_004", "delivered", 799.99, "890 Maple St, San Jose, CA 95112", "UPS-4001", "2026-01-05"),
        ("ord_009", "customer_004", "delivered", 149.99, "890 Maple St, San Jose, CA 95112", "UPS-4002", "2026-02-10"),
        ("ord_010", "customer_004", "shipped", 699.99, "890 Maple St, San Jose, CA 95112", "UPS-4003", "2026-03-02"),
        # Emily Watson - 2 orders
        ("ord_011", "customer_005", "delivered", 299.99, "321 Cedar Ln, Oakland, CA 94612", "FEDX-5001", "2026-02-15"),
        ("ord_012", "customer_005", "returned", 79.99, "321 Cedar Ln, Oakland, CA 94612", "FEDX-5002", "2026-02-20"),
        # James O'Brien - 3 orders
        ("ord_013", "customer_006", "delivered", 549.99, "456 Pine St, San Francisco, CA 94108", "UPS-6001", "2025-12-20"),
        ("ord_014", "customer_006", "delivered", 89.99, "456 Pine St, San Francisco, CA 94108", "UPS-6002", "2026-01-25"),
        ("ord_015", "customer_006", "pending", 449.99, "456 Pine St, San Francisco, CA 94108", None, "2026-03-06"),
        # Priya Patel - 4 orders (VIP)
        ("ord_016", "customer_007", "delivered", 549.99, "789 Birch Way, Mountain View, CA 94043", "FEDX-7001", "2025-11-15"),
        ("ord_017", "customer_007", "delivered", 449.99, "789 Birch Way, Mountain View, CA 94043", "FEDX-7002", "2026-01-08"),
        ("ord_018", "customer_007", "delivered", 149.99, "789 Birch Way, Mountain View, CA 94043", "FEDX-7003", "2026-02-01"),
        ("ord_019", "customer_007", "shipped", 169.99, "789 Birch Way, Mountain View, CA 94043", "FEDX-7004", "2026-03-03"),
        # Robert Taylor - 1 order
        ("ord_020", "customer_008", "delivered", 299.99, "234 Elm Ct, Sunnyvale, CA 94086", "UPS-8001", "2026-02-25"),
        # Lisa Chang - 2 orders
        ("ord_021", "customer_009", "delivered", 549.99, "678 Willow Ave, Berkeley, CA 94704", "FEDX-9001", "2026-01-12"),
        ("ord_022", "customer_009", "shipped", 129.99, "678 Willow Ave, Berkeley, CA 94704", "FEDX-9002", "2026-03-04"),
        # Tom Martinez - 3 orders (repeat caller / frustrated)
        ("ord_023", "customer_010", "returned", 449.99, "901 Spruce Blvd, San Francisco, CA 94115", "UPS-0001", "2026-01-18"),
        ("ord_024", "customer_010", "delivered", 549.99, "901 Spruce Blvd, San Francisco, CA 94115", "UPS-0002", "2026-02-05"),
        ("ord_025", "customer_010", "shipped", 89.99, "901 Spruce Blvd, San Francisco, CA 94115", "UPS-0003", "2026-03-01"),
        # Amanda Foster - 3 orders (VIP)
        ("ord_026", "customer_011", "delivered", 799.99, "543 Ash Dr, Redwood City, CA 94063", "FEDX-1101", "2025-12-10"),
        ("ord_027", "customer_011", "delivered", 69.99, "543 Ash Dr, Redwood City, CA 94063", "FEDX-1102", "2026-02-14"),
        ("ord_028", "customer_011", "pending", 449.99, "543 Ash Dr, Redwood City, CA 94063", None, "2026-03-06"),
        # Kevin Nguyen - 2 orders
        ("ord_029", "customer_012", "delivered", 299.99, "876 Poplar St, Santa Clara, CA 95054", "UPS-1201", "2026-02-08"),
        ("ord_030", "customer_012", "shipped", 44.99, "876 Poplar St, Santa Clara, CA 95054", "UPS-1202", "2026-03-05"),
        # Extra orders for variety
        ("ord_031", "customer_001", "returned", 169.99, "742 Evergreen Terrace, San Francisco, CA 94110", "FEDX-1004", "2026-02-10"),
        ("ord_032", "customer_003", "pending", 129.99, "567 Palm Dr, Palo Alto, CA 94301", None, "2026-03-06"),
        ("ord_033", "customer_005", "shipped", 549.99, "321 Cedar Ln, Oakland, CA 94612", "FEDX-5003", "2026-03-02"),
        ("ord_034", "customer_006", "returned", 69.99, "456 Pine St, San Francisco, CA 94108", "UPS-6003", "2026-02-18"),
        ("ord_035", "customer_008", "pending", 449.99, "234 Elm Ct, Sunnyvale, CA 94086", None, "2026-03-07"),
        ("ord_036", "customer_010", "delivered", 79.99, "901 Spruce Blvd, San Francisco, CA 94115", "UPS-0004", "2026-02-22"),
        ("ord_037", "customer_002", "shipped", 549.99, "1234 Oak Ave, San Francisco, CA 94102", "UPS-2002", "2026-03-03"),
        ("ord_038", "customer_004", "pending", 89.99, "890 Maple St, San Jose, CA 95112", None, "2026-03-07"),
        ("ord_039", "customer_009", "delivered", 44.99, "678 Willow Ave, Berkeley, CA 94704", "FEDX-9003", "2026-02-15"),
        ("ord_040", "customer_012", "pending", 169.99, "876 Poplar St, Santa Clara, CA 95054", None, "2026-03-07"),
    ]
    conn.executemany(
        "INSERT INTO orders VALUES (?,?,?,?,?,?,?)",
        orders,
    )

    # --- Order Items ---
    order_items = [
        ("ord_001", "prod_001", 1, 549.99),
        ("ord_002", "prod_003", 1, 449.99),
        ("ord_003", "prod_005", 1, 89.99),
        ("ord_004", "prod_002", 1, 299.99),
        ("ord_005", "prod_010", 1, 169.99),
        ("ord_006", "prod_003", 1, 449.99),
        ("ord_007", "prod_008", 1, 59.99),
        ("ord_008", "prod_015", 1, 799.99),
        ("ord_009", "prod_006", 1, 149.99),
        ("ord_010", "prod_004", 1, 699.99),
        ("ord_011", "prod_002", 1, 299.99),
        ("ord_012", "prod_009", 1, 79.99),
        ("ord_013", "prod_001", 1, 549.99),
        ("ord_014", "prod_005", 1, 89.99),
        ("ord_015", "prod_003", 1, 449.99),
        ("ord_016", "prod_001", 1, 549.99),
        ("ord_017", "prod_003", 1, 449.99),
        ("ord_018", "prod_006", 1, 149.99),
        ("ord_019", "prod_010", 1, 169.99),
        ("ord_020", "prod_002", 1, 299.99),
        ("ord_021", "prod_001", 1, 549.99),
        ("ord_022", "prod_020", 1, 129.99),
        ("ord_023", "prod_003", 1, 449.99),
        ("ord_024", "prod_001", 1, 549.99),
        ("ord_025", "prod_005", 1, 89.99),
        ("ord_026", "prod_015", 1, 799.99),
        ("ord_027", "prod_011", 1, 69.99),
        ("ord_028", "prod_003", 1, 449.99),
        ("ord_029", "prod_002", 1, 299.99),
        ("ord_030", "prod_012", 1, 44.99),
        ("ord_031", "prod_010", 1, 169.99),
        ("ord_032", "prod_020", 1, 129.99),
        ("ord_033", "prod_001", 1, 549.99),
        ("ord_034", "prod_011", 1, 69.99),
        ("ord_035", "prod_003", 1, 449.99),
        ("ord_036", "prod_009", 1, 79.99),
        ("ord_037", "prod_001", 1, 549.99),
        ("ord_038", "prod_005", 1, 89.99),
        ("ord_039", "prod_012", 1, 44.99),
        ("ord_040", "prod_010", 1, 169.99),
    ]
    conn.executemany(
        "INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (?,?,?,?)",
        order_items,
    )

    # --- Knowledge Base (18 articles) ---
    kb_articles = [
        ("Return Policy", "Items can be returned within 30 days of delivery for a full refund. Items must be in original packaging and unused condition. Opened items may be subject to a 15% restocking fee. Mattresses and custom orders are final sale.", "returns", json.dumps(["return", "refund", "policy", "30 days"])),
        ("Shipping Policy", "Standard shipping is free on orders over $100. Standard delivery takes 5-7 business days. Express shipping ($29.99) delivers in 2-3 business days. Oversized items (desks, chairs) may require additional 2-3 days.", "shipping", json.dumps(["shipping", "delivery", "free shipping", "express"])),
        ("Warranty Information", "All furniture products come with a 5-year limited warranty covering manufacturing defects. Accessories have a 1-year warranty. Warranty does not cover normal wear and tear or damage from misuse.", "product", json.dumps(["warranty", "defect", "coverage", "5 year"])),
        ("How to Track Your Order", "Once your order ships, you'll receive a tracking number via email. You can also check order status using your order number on our website or by contacting support.", "shipping", json.dumps(["tracking", "order status", "shipped"])),
        ("Refund Processing Times", "Refunds are processed within 5-7 business days after we receive the returned item. Credit card refunds may take an additional 3-5 business days to appear on your statement.", "billing", json.dumps(["refund", "processing", "credit card", "timeline"])),
        ("Assembly Instructions", "All desks and chairs come with detailed assembly instructions. Average assembly time is 30-60 minutes. If you need help, contact support to schedule a virtual assembly assistance session.", "product", json.dumps(["assembly", "instructions", "setup"])),
        ("Discount Codes", "We offer seasonal discount codes. Current active codes: WELCOME10 (10% off first order), ERGOSAVE20 (20% off ergonomic chairs), BUNDLE15 (15% off orders with 3+ items), SORRY15 (15% courtesy discount for service issues).", "billing", json.dumps(["discount", "coupon", "code", "promo"])),
        ("Damaged Item Policy", "If you receive a damaged item, contact support within 48 hours with photos. We will arrange a free replacement or full refund. Do not discard the packaging until the claim is resolved.", "returns", json.dumps(["damaged", "broken", "replacement", "claim"])),
        ("Exchange Policy", "Exchanges can be made within 30 days. The replacement item will be shipped once we receive the original. If the replacement costs more, you'll be charged the difference.", "returns", json.dumps(["exchange", "swap", "replacement"])),
        ("Contact Hours", "Customer support is available Monday-Friday 8AM-8PM PT, Saturday 9AM-5PM PT. Average response time is under 2 hours during business hours.", "general", json.dumps(["hours", "contact", "availability", "response time"])),
        ("VIP Customer Benefits", "VIP customers receive: priority support (response within 30 minutes), free express shipping on all orders, extended 60-day return window, and exclusive early access to new products.", "general", json.dumps(["VIP", "priority", "benefits", "loyalty"])),
        ("Ergonomic Setup Guide", "For optimal ergonomics: monitor at eye level, arms at 90 degrees, feet flat on floor or footrest. Our standing desk presets: sitting 28-30 inches, standing 42-48 inches.", "product", json.dumps(["ergonomic", "setup", "height", "posture"])),
        ("Bulk Order Discounts", "Orders of 5+ units of the same item qualify for a 10% bulk discount. Orders of 10+ qualify for 15%. Contact sales for custom quotes on larger orders.", "billing", json.dumps(["bulk", "volume", "discount", "corporate"])),
        ("International Shipping", "We currently ship to US and Canada only. Canadian orders may be subject to customs duties and taxes. Delivery to Canada takes 7-14 business days.", "shipping", json.dumps(["international", "canada", "customs"])),
        ("Product Care Instructions", "Wipe surfaces with a damp cloth. Avoid harsh chemicals. Tighten bolts every 6 months. Lubricate desk motor mechanisms annually. Leather chairs should be conditioned every 3 months.", "product", json.dumps(["care", "maintenance", "cleaning"])),
        ("Payment Methods", "We accept Visa, Mastercard, Amex, Discover, PayPal, and Apple Pay. Financing available via Affirm for orders over $200 (0% APR for 6 months).", "billing", json.dumps(["payment", "credit card", "financing", "affirm"])),
        ("Order Cancellation", "Orders can be cancelled for free within 2 hours of placement. After 2 hours, a $25 cancellation fee applies. Orders that have shipped cannot be cancelled — please initiate a return instead.", "billing", json.dumps(["cancel", "cancellation", "order"])),
        ("Standing Desk Troubleshooting", "If your standing desk motor isn't working: 1) Check power connection, 2) Reset by holding UP+DOWN for 10 seconds, 3) Ensure weight doesn't exceed 300lbs. If still not working, contact support for warranty service.", "product", json.dumps(["troubleshooting", "motor", "standing desk", "fix"])),
    ]
    conn.executemany(
        "INSERT INTO knowledge_base (title, content, category, tags) VALUES (?,?,?,?)",
        kb_articles,
    )
    # Populate FTS index
    conn.execute(
        "INSERT INTO kb_fts(rowid, title, content, tags) SELECT id, title, content, tags FROM knowledge_base"
    )

    # --- Pre-existing tickets (8, mostly resolved) ---
    tickets = [
        ("customer_001", "ord_001", "Standing desk wobbles when raised", "resolved", "normal", "employee_support_01", "2026-02-01", "2026-02-03"),
        ("customer_003", "ord_007", "Anti-fatigue mat arrived torn", "resolved", "high", "employee_support_01", "2026-01-15", "2026-01-16"),
        ("customer_006", "ord_013", "Desk motor stopped working after 2 months", "resolved", "high", "employee_support_01", "2026-02-15", "2026-02-20"),
        ("customer_010", "ord_023", "Chair arrived with broken armrest", "resolved", "urgent", "employee_support_01", "2026-01-25", "2026-01-28"),
        ("customer_010", "ord_024", "Standing desk height presets not saving", "resolved", "normal", "employee_support_01", "2026-02-10", "2026-02-12"),
        ("customer_005", "ord_012", "Desk lamp flickers on highest setting", "resolved", "normal", "employee_support_01", "2026-02-22", "2026-02-25"),
        ("customer_007", "ord_016", "Request to change desk color", "closed", "low", "employee_support_01", "2026-01-20", "2026-01-20"),
        ("customer_002", None, "General question about bulk ordering", "closed", "low", "employee_support_01", "2026-02-28", "2026-02-28"),
    ]
    for t in tickets:
        conn.execute(
            "INSERT INTO tickets (customer_id, order_id, subject, status, priority, assigned_agent, created_at, resolved_at) VALUES (?,?,?,?,?,?,?,?)",
            t,
        )

    # --- Ticket messages for resolved tickets ---
    ticket_msgs = [
        # Ticket 1 - desk wobble
        (1, "customer_001", "customer", "My standing desk wobbles quite a bit when raised to standing height. Is this normal?"),
        (1, "employee_support_01", "agent", "I'm sorry to hear that. Could you check if all bolts are fully tightened? Sometimes they loosen during shipping."),
        (1, "customer_001", "customer", "Tightened everything, still wobbles. It's on carpet though."),
        (1, "employee_support_01", "agent", "On carpet, the desk can be unstable. I'm sending you free stabilizer pads. They should arrive in 2-3 days."),
        (1, "customer_001", "customer", "Got the pads, desk is stable now. Thanks!"),
        # Ticket 2 - torn mat
        (2, "customer_003", "customer", "The anti-fatigue mat I received has a big tear on one edge. This is unacceptable."),
        (2, "employee_support_01", "agent", "I sincerely apologize. I've arranged a replacement to ship today via express. You don't need to return the damaged one."),
        (2, "customer_003", "customer", "Replacement received, looks good. Thank you for the quick resolution."),
        # Ticket 4 - broken armrest
        (4, "customer_010", "customer", "The chair I just received has a completely broken armrest. Right out of the box. This is the second issue I've had."),
        (4, "employee_support_01", "agent", "I'm very sorry about this, especially given your previous experience. I'm processing a full refund and sending a replacement chair with express shipping at no charge."),
        (4, "customer_010", "customer", "Fine. I expect the replacement to be in better condition."),
    ]
    conn.executemany(
        "INSERT INTO ticket_messages (ticket_id, sender_id, sender_role, content) VALUES (?,?,?,?)",
        ticket_msgs,
    )

    # --- Transactions (payments for delivered/shipped orders + refunds for returns) ---
    txns = [
        ("ord_001", "payment", 549.99, None, None),
        ("ord_002", "payment", 449.99, None, None),
        ("ord_003", "payment", 89.99, None, None),
        ("ord_004", "payment", 299.99, None, None),
        ("ord_006", "payment", 449.99, None, None),
        ("ord_007", "payment", 59.99, None, None),
        ("ord_008", "payment", 799.99, None, None),
        ("ord_009", "payment", 149.99, None, None),
        ("ord_010", "payment", 699.99, None, None),
        ("ord_011", "payment", 299.99, None, None),
        ("ord_012", "payment", 79.99, None, None),
        ("ord_012", "refund", 79.99, "Item returned - lamp flickering", "employee_support_01"),
        ("ord_013", "payment", 549.99, None, None),
        ("ord_023", "payment", 449.99, None, None),
        ("ord_023", "refund", 449.99, "Chair arrived with broken armrest", "employee_support_01"),
        ("ord_024", "payment", 549.99, None, None),
        ("ord_034", "payment", 69.99, None, None),
        ("ord_034", "refund", 69.99, "Customer returned - wrong item ordered", None),
    ]
    conn.executemany(
        "INSERT INTO transactions (order_id, type, amount, reason, approved_by) VALUES (?,?,?,?,?)",
        txns,
    )

    # --- Channels (internal messaging) ---
    channels = [
        ("#support", "public", json.dumps(["employee_support_01", "employee_support_02", "employee_support_03", "employee_manager_01"])),
        ("#escalations", "public", json.dumps(["employee_manager_01"])),
        ("dm_support01_manager01", "dm", json.dumps(["employee_support_01", "employee_manager_01"])),
        ("dm_support02_manager01", "dm", json.dumps(["employee_support_02", "employee_manager_01"])),
        ("dm_support03_manager01", "dm", json.dumps(["employee_support_03", "employee_manager_01"])),
    ]
    conn.executemany(
        "INSERT INTO channels (id, type, members) VALUES (?,?,?)",
        channels,
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    db_path = get_db_path()
    print(f"Initializing database at {db_path}")
    init_db(db_path)
    print("Seeding database...")
    seed_db(db_path)
    print("Done.")

    # Quick verification
    conn = get_connection(db_path)
    for table in ["customers", "products", "orders", "order_items", "tickets", "ticket_messages", "knowledge_base", "transactions"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")
    conn.close()
