"""Compare two evaluation runs side-by-side.

Usage:
  python scripts/compare_evals.py outputs/eval_vanilla.json outputs/eval_trained.json
"""

import json
import sys
from pathlib import Path


METRICS = [
    ("episode_reward", "Reward", ".3f"),
    ("resolved", "Resolved", ""),
    ("steps", "Steps", ".0f"),
    ("format", "Format", ".2f"),
    ("tool_valid", "Tool Valid", ".2f"),
    ("reasoning", "Reasoning", ".2f"),
    ("no_leak", "No Leak", ".2f"),
]


def fmt_val(val, spec):
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if spec:
        return f"{val:{spec}}"
    return str(val)


def fmt_delta(delta, spec):
    if isinstance(delta, bool):
        return ""
    sign = "+" if delta > 0 else ""
    if spec:
        return f"{sign}{delta:{spec}}"
    return f"{sign}{delta}"


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_evals.py <baseline.json> <trained.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        baseline = json.load(f)
    with open(sys.argv[2]) as f:
        trained = json.load(f)

    label_a = baseline.get("label", "baseline")
    label_b = trained.get("label", "trained")

    # Build task lookup
    tasks_a = {t["task_id"]: t for t in baseline["tasks"]}
    tasks_b = {t["task_id"]: t for t in trained["tasks"]}
    all_task_ids = list(tasks_a.keys())

    # Header
    col_w = 14
    task_col = 28
    print()
    print(f"{'':>{task_col}} | {label_a:>{col_w}} | {label_b:>{col_w}} | {'Delta':>{col_w}}")
    print("-" * (task_col + 3 * (col_w + 3) + 1))

    # Per-task comparison
    for task_id in all_task_ids:
        ta = tasks_a.get(task_id, {})
        tb = tasks_b.get(task_id, {})
        print(f"\n  {task_id}")

        for key, name, spec in METRICS:
            va = ta.get(key, 0)
            vb = tb.get(key, 0)
            if isinstance(va, bool) or isinstance(vb, bool):
                delta_str = ""
            else:
                delta = vb - va
                delta_str = fmt_delta(delta, spec)
            print(f"    {name:<{task_col - 4}} | {fmt_val(va, spec):>{col_w}} | {fmt_val(vb, spec):>{col_w}} | {delta_str:>{col_w}}")

    # Summary comparison
    sum_a = baseline.get("summary", {})
    sum_b = trained.get("summary", {})

    summary_metrics = [
        ("avg_episode_reward", "Avg Reward", ".3f"),
        ("resolution_rate", "Resolution Rate", ".0%"),
        ("avg_steps", "Avg Steps", ".1f"),
        ("avg_format", "Avg Format", ".2f"),
        ("avg_tool_valid", "Avg Tool Valid", ".2f"),
        ("avg_reasoning", "Avg Reasoning", ".2f"),
        ("avg_no_leak", "Avg No Leak", ".2f"),
    ]

    print()
    print("=" * (task_col + 3 * (col_w + 3) + 1))
    print(f"  SUMMARY")

    for key, name, spec in summary_metrics:
        va = sum_a.get(key, 0)
        vb = sum_b.get(key, 0)
        delta = vb - va
        delta_str = fmt_delta(delta, spec)
        print(f"    {name:<{task_col - 4}} | {fmt_val(va, spec):>{col_w}} | {fmt_val(vb, spec):>{col_w}} | {delta_str:>{col_w}}")

    print()


if __name__ == "__main__":
    main()
