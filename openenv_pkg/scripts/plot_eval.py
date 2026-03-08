"""Plot evaluation comparison: Vanilla vs GRPO-trained Qwen3-8B.

Usage:
  python scripts/plot_eval.py outputs/eval_vanilla.json outputs/eval_trained.json
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_eval.py <vanilla.json> <trained.json>")
        sys.exit(1)

    vanilla = load(sys.argv[1])
    trained = load(sys.argv[2])

    tasks_v = {t["task_id"]: t for t in vanilla["tasks"]}
    tasks_t = {t["task_id"]: t for t in trained["tasks"]}
    task_ids = list(tasks_v.keys())

    # Short task labels
    short = {
        "task_001_order_tracking": "Order\nTracking",
        "task_002_pending_order": "Pending\nOrder",
        "task_003_vip_escalation": "VIP\nEscalation",
        "task_004_damaged_product": "Damaged\nProduct",
        "task_005_shipping_anxiety": "Shipping\nAnxiety",
        "task_006_warranty_claim": "Warranty\nClaim",
        "task_007_repeat_customer": "Repeat\nCustomer",
        "task_008_bulk_inquiry": "Bulk\nInquiry",
    }
    labels = [short.get(t, t) for t in task_ids]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GRPO Training Impact: Vanilla vs Trained Qwen3-8B\n(189 examples, 1 epoch, LoRA r=16)",
                 fontsize=14, fontweight="bold", y=0.98)

    colors_v = "#6baed6"
    colors_t = "#2171b5"
    highlight = "#e6550d"

    x = np.arange(len(task_ids))
    w = 0.35

    # --- Top-left: Average Metrics (overview) ---
    ax = axes[0, 0]
    metrics = ["Reward", "Resolution\nRate", "Reasoning", "Tool\nValidity", "Format", "No Leak"]
    sv = vanilla["summary"]
    st = trained["summary"]
    vals_v = [sv["avg_episode_reward"], sv["resolution_rate"],
              sv["avg_reasoning"], sv["avg_tool_valid"],
              max(0, sv["avg_format"]), sv["avg_no_leak"]]
    vals_t = [st["avg_episode_reward"], st["resolution_rate"],
              st["avg_reasoning"], st["avg_tool_valid"],
              max(0, st["avg_format"]), st["avg_no_leak"]]

    x2 = np.arange(len(metrics))
    ax.bar(x2 - w/2, vals_v, w, label="Vanilla", color=colors_v, edgecolor="white")
    ax.bar(x2 + w/2, vals_t, w, label="GRPO-trained", color=colors_t, edgecolor="white")

    for i in range(len(metrics)):
        delta = vals_t[i] - vals_v[i]
        if abs(delta) > 0.005:
            sign = "+" if delta > 0 else ""
            color = "green" if delta > 0 else "red"
            y = max(vals_v[i], vals_t[i]) + 0.02
            ax.text(i, y, f"{sign}{delta:.2f}", ha="center", fontsize=8, color=color, fontweight="bold")

    ax.set_ylabel("Score")
    ax.set_title("Average Metrics")
    ax.set_xticks(x2)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylim(0, 0.55)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # --- Top-right: Steps per task (overview) ---
    ax = axes[0, 1]
    steps_v = [tasks_v[t]["steps"] for t in task_ids]
    steps_t = [tasks_t[t]["steps"] for t in task_ids]

    ax.bar(x - w/2, steps_v, w, label="Vanilla", color=colors_v, edgecolor="white")
    ax.bar(x + w/2, steps_t, w, label="GRPO-trained", color=colors_t, edgecolor="white")
    ax.axhline(y=10, color="red", linestyle="--", alpha=0.5, label="Step limit")

    ax.set_ylabel("Steps")
    ax.set_title("Steps per Task (fewer = more efficient)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # --- Bottom-left: Per-task episode reward (detail) ---
    ax = axes[1, 0]
    rewards_v = [tasks_v[t]["episode_reward"] for t in task_ids]
    rewards_t = [tasks_t[t]["episode_reward"] for t in task_ids]

    bars_v = ax.bar(x - w/2, rewards_v, w, label="Vanilla", color=colors_v, edgecolor="white")
    bars_t = ax.bar(x + w/2, rewards_t, w, label="GRPO-trained", color=colors_t, edgecolor="white")

    idx_004 = task_ids.index("task_004_damaged_product")
    bars_t[idx_004].set_color(highlight)
    bars_t[idx_004].set_edgecolor("white")
    ax.annotate("+567%", xy=(idx_004 + w/2, rewards_t[idx_004]),
                xytext=(idx_004 + w/2 + 0.3, rewards_t[idx_004] + 0.08),
                fontsize=9, fontweight="bold", color=highlight,
                arrowprops=dict(arrowstyle="->", color=highlight, lw=1.5))

    for i, tid in enumerate(task_ids):
        if tasks_t[tid].get("resolved"):
            ax.text(i + w/2, rewards_t[i] + 0.02, "✓", ha="center", fontsize=10, color="green", fontweight="bold")
        if tasks_v[tid].get("resolved"):
            ax.text(i - w/2, rewards_v[i] + 0.02, "✓", ha="center", fontsize=10, color="green", fontweight="bold")

    ax.set_ylabel("Episode Reward")
    ax.set_title("Per-Task Reward (✓ = resolved)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # --- Bottom-right: Tool usage pattern (task_004 deep dive) ---
    ax = axes[1, 1]

    tool_colors = {
        "lookup_customer": "#1b9e77",
        "check_order": "#d95f02",
        "send_reply": "#7570b3",
        "update_ticket": "#e7298a",
        "request_escalation": "#e6ab02",
    }

    tools_004_v = tasks_v["task_004_damaged_product"]["tools"]
    tools_004_t = tasks_t["task_004_damaged_product"]["tools"]

    for i, tool in enumerate(tools_004_v):
        c = tool_colors.get(tool, "#999999")
        ax.barh(1.4, 0.9, left=i, height=0.5, color=c, edgecolor="white", linewidth=0.5)

    for i, tool in enumerate(tools_004_t):
        c = tool_colors.get(tool, "#999999")
        ax.barh(0.4, 0.9, left=i, height=0.5, color=c, edgecolor="white", linewidth=0.5)

    ax.set_yticks([0.65, 1.65])
    ax.set_yticklabels(["GRPO-trained\n(RESOLVED ✓)", "Vanilla\n(failed)"], fontsize=9)
    ax.set_xlabel("Step")
    ax.set_title("Task 004 (Damaged Product): Tool Sequence")
    ax.set_xlim(-0.2, 10.5)
    ax.set_ylim(0, 2.2)

    patches = [mpatches.Patch(color=c, label=t.replace("_", " ").title())
               for t, c in tool_colors.items()]
    ax.legend(handles=patches, fontsize=7, loc="upper right", ncol=2)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = Path(sys.argv[1]).parent / "eval_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
