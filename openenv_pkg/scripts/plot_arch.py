"""Generate architecture diagram for EnterpriseSim as SVG.

Usage:
  uv run python openenv_pkg/scripts/plot_arch.py
  # Output: openenv_pkg/outputs/architecture.svg
"""

from pathlib import Path


# ── SVG helpers ──────────────────────────────────────────────────────────

def svg_rect(x, y, w, h, rx=12, fill="#fff", stroke="#ccc", stroke_width=1.5,
             opacity=1, filter_id=None):
    extra = f' filter="url(#{filter_id})"' if filter_id else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" '
            f'opacity="{opacity}"{extra}/>')


def svg_text(x, y, text, size=14, fill="#333", weight="normal", anchor="middle",
             family="Inter, -apple-system, sans-serif", dy="0.35em", style=""):
    st = f' font-style="{style}"' if style else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}" '
            f'font-family="{family}" dy="{dy}"{st}>{text}</text>')


def svg_line(x1, y1, x2, y2, stroke="#999", width=1.5, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{width}"{d}/>')


def svg_arrow(x1, y1, x2, y2, stroke="#666", width=2):
    """Line with arrowhead."""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{width}" marker-end="url(#arrow-{stroke.replace("#","")})"/>')


def svg_group(content, transform=""):
    t = f' transform="{transform}"' if transform else ""
    return f"<g{t}>\n{content}\n</g>"


# ── Component builders ───────────────────────────────────────────────────

def docker_card(x, y, w, h, title, subtitle, lines, badge_color="#4a90d9",
                accent_light="#e8f0fe"):
    """A Docker container card with title badge, pi-mono badge, and detail lines."""
    parts = []

    # Shadow + card
    parts.append(svg_rect(x+3, y+3, w, h, rx=10, fill="#00000010", stroke="none"))
    parts.append(svg_rect(x, y, w, h, rx=10, fill="white", stroke="#d0d0d0"))

    # Title badge
    bh = 32
    parts.append(svg_rect(x+6, y+6, w-12, bh, rx=6, fill=badge_color, stroke="none"))

    # Docker whale icon (simple)
    ix, iy = x + 18, y + 14
    parts.append(f'<text x="{ix}" y="{iy+9}" font-size="14" fill="white" opacity="0.9">&#x1F433;</text>')
    parts.append(svg_text(x + w/2 + 6, y + 6 + bh/2, title,
                          size=11, fill="white", weight="bold"))

    # Subtitle
    parts.append(svg_text(x + w/2, y + 48, subtitle, size=11, fill="#666"))

    # pi-mono badge
    pi_y = y + 64
    parts.append(svg_rect(x + 12, pi_y, w - 24, 24, rx=5,
                          fill="#eef7ee", stroke="#7bc67b", stroke_width=1))
    parts.append(svg_text(x + w/2, pi_y + 12, "pi-mono agent  (Node.js)",
                          size=9.5, fill="#2a7d2a", weight="600"))

    # Detail lines
    ly = pi_y + 38
    for line in lines:
        is_header = not line.startswith(" ")
        parts.append(svg_text(x + 18, ly, line, size=10,
                              fill="#444" if is_header else "#555",
                              weight="600" if is_header else "normal",
                              anchor="start",
                              family="'SF Mono', 'Fira Code', monospace"))
        ly += 17

    return "\n".join(parts)


def db_box(x, y, w, h):
    """The world.db database box with cylinder top."""
    parts = []

    # Shadow
    parts.append(svg_rect(x+3, y+3, w, h, rx=10, fill="#00000012", stroke="none"))
    # Main
    parts.append(svg_rect(x, y, w, h, rx=10, fill="#fffbeb", stroke="#d4a017", stroke_width=2))

    # Cylinder top ellipse
    cy = y + 8
    parts.append(f'<ellipse cx="{x+w/2}" cy="{cy+14}" rx="{w/2-8}" ry="10" '
                 f'fill="#d4a017" opacity="0.15"/>')

    # Title
    parts.append(svg_rect(x+8, y+6, w-16, 30, rx=6, fill="#d4a017", stroke="none"))
    parts.append(svg_text(x + w/2, y + 21, "world.db  (SQLite)",
                          size=13, fill="white", weight="bold"))

    # Table pills
    tables = [
        ["customers (12)", "orders", "order_items"],
        ["products", "tickets", "ticket_messages"],
        ["knowledge_base", "transactions", "sim_traces"],
    ]
    ty = y + 50
    for row in tables:
        tx = x + 16
        for tbl in row:
            tw = len(tbl) * 7.5 + 14
            parts.append(svg_rect(tx, ty, tw, 20, rx=4,
                                  fill="#fef3c7", stroke="#d4a01740", stroke_width=0.8))
            parts.append(svg_text(tx + tw/2, ty + 10, tbl,
                                  size=8.5, fill="#7a6200",
                                  family="'SF Mono', 'Fira Code', monospace"))
            tx += tw + 6
        ty += 26

    return "\n".join(parts)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    W, H = 1100, 950

    # Arrow markers for different colors
    arrow_colors = {"4a90d9": "#4a90d9", "e67e22": "#e67e22", "2c3e50": "#2c3e50",
                    "666666": "#666666"}
    markers = "\n".join(
        f'<marker id="arrow-{cid}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{c}"/></marker>'
        for cid, c in arrow_colors.items()
    )

    # Drop shadow filter
    filters = '''
    <filter id="shadow" x="-5%" y="-5%" width="115%" height="115%">
      <feDropShadow dx="2" dy="2" stdDeviation="4" flood-opacity="0.08"/>
    </filter>
    '''

    parts = []

    # ── Background ──
    parts.append(svg_rect(0, 0, W, H, rx=16, fill="#f7f8fa", stroke="#dee2e6", stroke_width=2))

    # ── Title ──
    parts.append(svg_text(W/2, 36, "EnterpriseSim  —  Architecture",
                          size=26, fill="#1a1a2e", weight="bold"))
    parts.append(svg_text(W/2, 62, "Multi-Agent Customer Support Simulation Environment",
                          size=13, fill="#888"))

    # ══════════════════════════════════════════════════════════════
    # Legend — horizontal row between subtitle and engine
    # ══════════════════════════════════════════════════════════════
    legend = [
        ("#e8f0fe", "#4a90d9", "Inside (Company)"),
        ("#fef5ec", "#e67e22", "Outside (Customers)"),
        ("#fffbeb", "#d4a017", "Shared State (SQLite)"),
        ("#eef7ee", "#7bc67b", "pi-mono Runtime"),
    ]
    lg_y = 86
    lg_total_w = sum(len(l) * 7 + 30 for _, _, l in legend) + 20
    lg_x = W/2 - lg_total_w/2
    cx = lg_x
    for fc, sc, label in legend:
        parts.append(svg_rect(cx, lg_y - 6, 12, 12, rx=3,
                              fill=fc, stroke=sc, stroke_width=1))
        parts.append(svg_text(cx + 18, lg_y, label,
                              size=9, fill="#888", anchor="start"))
        cx += len(label) * 7 + 30

    # ══════════════════════════════════════════════════════════════
    # SIMULATION ENGINE
    # ══════════════════════════════════════════════════════════════
    eng_x, eng_y, eng_w, eng_h = 50, 112, W - 100, 64
    parts.append(svg_rect(eng_x+3, eng_y+3, eng_w, eng_h, rx=10,
                          fill="#00000015", stroke="none"))
    parts.append(svg_rect(eng_x, eng_y, eng_w, eng_h, rx=10,
                          fill="#2c3e50", stroke="#1a252f", stroke_width=1.5))
    parts.append(svg_text(eng_x + eng_w/2, eng_y + 22,
                          "Simulation Engine",
                          size=15, fill="white", weight="bold"))
    parts.append(svg_text(eng_x + eng_w/2, eng_y + 46,
                          "Tick Processor (48 ticks x 5 min)  ·  "
                          "Phase 1: Customers  &#x2192;  Phase 2: Routing  &#x2192;  "
                          "Phase 3: Employees  &#x2192;  Phase 4: Managers",
                          size=10, fill="#bdc3c7"))

    # ══════════════════════════════════════════════════════════════
    # INSIDE — Employee Region
    # ══════════════════════════════════════════════════════════════
    in_x, in_y, in_w, in_h = 50, 200, W - 100, 270
    parts.append(svg_rect(in_x, in_y, in_w, in_h, rx=12,
                          fill="#e8f0fe", stroke="#4a90d9", stroke_width=1.5, opacity=0.7))
    parts.append(svg_text(in_x + 18, in_y + 20, "INSIDE  —  Company Employees",
                          size=13, fill="#2c5aa0", weight="bold", anchor="start"))

    # Employee cards
    card_w, card_h = 290, 230
    card_gap = 30
    card_y = in_y + 32
    start_x = in_x + (in_w - 3*card_w - 2*card_gap) / 2

    # Support Agent 01
    parts.append(docker_card(start_x, card_y, card_w, card_h,
                             "employee_support_01", "Support Agent",
                             ["Tools:",
                              "  lookup_customer",
                              "  check_order",
                              "  send_reply",
                              "  update_ticket",
                              "  request_escalation"],
                             badge_color="#4a90d9"))

    # Support Agent 02
    parts.append(docker_card(start_x + card_w + card_gap, card_y, card_w, card_h,
                             "employee_support_02", "Support Agent",
                             ["Tools:",
                              "  lookup_customer",
                              "  check_order",
                              "  send_reply",
                              "  update_ticket"],
                             badge_color="#4a90d9"))

    # Manager Agent
    parts.append(docker_card(start_x + 2*(card_w + card_gap), card_y, card_w, card_h,
                             "employee_manager_01", "Manager Agent",
                             ["Tools:",
                              "  lookup_customer",
                              "  check_order",
                              "  send_reply",
                              "  update_ticket",
                              "  approve_refund"],
                             badge_color="#6c5ce7"))

    # ══════════════════════════════════════════════════════════════
    # WORLD.DB
    # ══════════════════════════════════════════════════════════════
    db_w_val, db_h_val = 380, 130
    db_x_val = W/2 - db_w_val/2
    db_y_val = 498
    parts.append(db_box(db_x_val, db_y_val, db_w_val, db_h_val))

    # ══════════════════════════════════════════════════════════════
    # Arrows: Engine → Inside, Inside → DB, DB → Outside
    # ══════════════════════════════════════════════════════════════
    # Engine → Inside
    parts.append(svg_arrow(W/2 - 150, eng_y + eng_h, W/2 - 150, in_y,
                           stroke="#2c3e50", width=2))
    parts.append(svg_text(W/2 - 135, (eng_y + eng_h + in_y)/2, "orchestrates",
                          size=9, fill="#2c3e50", style="italic", anchor="start"))
    parts.append(svg_arrow(W/2 + 150, eng_y + eng_h, W/2 + 150, in_y,
                           stroke="#2c3e50", width=2))

    # Inside → DB
    parts.append(svg_arrow(W/2, in_y + in_h, W/2, db_y_val,
                           stroke="#4a90d9", width=2))
    parts.append(svg_text(W/2 + 12, (in_y + in_h + db_y_val)/2, "SQL read / write",
                          size=10, fill="#4a90d9", style="italic", anchor="start"))

    # ══════════════════════════════════════════════════════════════
    # OUTSIDE — Customer Region
    # ══════════════════════════════════════════════════════════════
    out_x, out_y, out_w, out_h = 50, 658, W - 100, 260
    parts.append(svg_rect(out_x, out_y, out_w, out_h, rx=12,
                          fill="#fef5ec", stroke="#e67e22", stroke_width=1.5, opacity=0.7))
    parts.append(svg_text(out_x + 18, out_y + 20, "OUTSIDE  —  Customers",
                          size=13, fill="#c0612b", weight="bold", anchor="start"))

    # Customer cards
    cust_w, cust_h = 260, 220
    cust_y = out_y + 32
    cust_start_x = out_x + (out_w - 3*cust_w - 2*card_gap) / 2

    parts.append(docker_card(cust_start_x, cust_y, cust_w, cust_h,
                             "customer_001", '"Sarah Chen"',
                             ["persona.json:",
                              "  patience: 0.7",
                              "  style: formal",
                              "  VIP: true",
                              "  issue: order tracking"],
                             badge_color="#e67e22"))

    parts.append(docker_card(cust_start_x + cust_w + card_gap, cust_y, cust_w, cust_h,
                             "customer_002", '"Mike Johnson"',
                             ["persona.json:",
                              "  patience: 0.5",
                              "  style: casual",
                              "  VIP: false",
                              "  issue: damaged product"],
                             badge_color="#e67e22"))

    # "... + 10 more" card
    c3x = cust_start_x + 2*(cust_w + card_gap)
    parts.append(svg_rect(c3x+3, cust_y+3, cust_w, cust_h, rx=10,
                          fill="#00000010", stroke="none"))
    parts.append(svg_rect(c3x, cust_y, cust_w, cust_h, rx=10,
                          fill="white", stroke="#d0d0d0"))
    parts.append(svg_rect(c3x+6, cust_y+6, cust_w-12, 32, rx=6,
                          fill="#e67e22", stroke="none"))
    parts.append(svg_text(c3x + cust_w/2, cust_y + 22, "... + 10 more",
                          size=11, fill="white", weight="bold"))
    parts.append(svg_text(c3x + cust_w/2, cust_y + 80,
                          "12 customers total",
                          size=13, fill="#555", weight="600"))
    more_lines = [
        "Each with unique persona,",
        "order history, and",
        "active support issues"
    ]
    for i, ml in enumerate(more_lines):
        parts.append(svg_text(c3x + cust_w/2, cust_y + 108 + i*20,
                              ml, size=11, fill="#888"))

    # DB → Outside
    parts.append(svg_arrow(W/2, db_y_val + db_h_val, W/2, out_y,
                           stroke="#e67e22", width=2))
    parts.append(svg_text(W/2 + 12, (db_y_val + db_h_val + out_y)/2,
                          "SQL read (tickets)",
                          size=10, fill="#e67e22", style="italic", anchor="start"))

    # ══════════════════════════════════════════════════════════════
    # Footer
    # ══════════════════════════════════════════════════════════════
    parts.append(svg_text(W/2, H - 16,
                          "Communication: ticket_messages table in world.db (async, tick-based)   ·   "
                          "LLM Backend: OpenAI-compatible API (Qwen3-8B / gpt-4o-mini)",
                          size=10, fill="#999", style="italic"))

    # ══════════════════════════════════════════════════════════════
    # Assemble SVG
    # ══════════════════════════════════════════════════════════════
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
  <defs>
    {markers}
    {filters}
    <style>
      text {{ font-family: Inter, -apple-system, Segoe UI, sans-serif; }}
    </style>
  </defs>
  {"".join(parts)}
</svg>"""

    out = Path(__file__).resolve().parent.parent / "outputs" / "architecture.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
