"""Task quality analysis: distribution, rubric coverage, complexity, gaps."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from enterprise_sim.task_miner.schema import Task


def task_distribution(tasks_dir: Path) -> dict:
    """Analyze task distribution across categories and difficulty levels."""
    tasks = _load_tasks(tasks_dir)

    by_category = Counter(t.category for t in tasks)
    by_difficulty = Counter(t.difficulty for t in tasks)

    # Cross-tabulation
    cross = {}
    for t in tasks:
        key = f"{t.category}/{t.difficulty}"
        cross[key] = cross.get(key, 0) + 1

    # Expected categories and difficulties
    all_categories = ["information_retrieval", "communication", "reasoning", "multi_step"]
    all_difficulties = ["easy", "medium", "hard"]

    # Find gaps
    gaps = []
    for cat in all_categories:
        for diff in all_difficulties:
            key = f"{cat}/{diff}"
            if key not in cross:
                gaps.append(key)

    return {
        "total_tasks": len(tasks),
        "by_category": dict(by_category),
        "by_difficulty": dict(by_difficulty),
        "cross_tabulation": cross,
        "missing_combinations": gaps,
        "gap_count": len(gaps),
    }


def rubric_coverage(tasks_dir: Path) -> dict:
    """Analyze rubric criteria types and coverage."""
    tasks = _load_tasks(tasks_dir)

    type_counts = Counter()
    type_weights = {}
    ground_truth_count = 0
    total_criteria = 0

    for t in tasks:
        for r in t.rubric:
            type_counts[r.type] += 1
            total_criteria += 1
            if r.type not in type_weights:
                type_weights[r.type] = []
            type_weights[r.type].append(r.weight)
            if r.ground_truth is not None:
                ground_truth_count += 1

    # Average weight per type
    avg_weights = {
        k: round(sum(v) / len(v), 3) for k, v in type_weights.items()
    }

    return {
        "total_criteria": total_criteria,
        "by_type": dict(type_counts),
        "avg_weight_by_type": avg_weights,
        "with_ground_truth": ground_truth_count,
        "ground_truth_pct": round(ground_truth_count / total_criteria * 100, 1) if total_criteria else 0,
    }


def task_complexity(tasks_dir: Path) -> dict:
    """Analyze per-task complexity: tools, criteria, context entities."""
    tasks = _load_tasks(tasks_dir)

    per_task = []
    for t in tasks:
        context_entities = len([v for v in t.context.values() if v is not None])
        per_task.append({
            "id": t.id,
            "category": t.category,
            "difficulty": t.difficulty,
            "tools_required": len(t.tools),
            "tools": t.tools,
            "criteria_count": len(t.rubric),
            "context_entities": context_entities,
            "has_ground_truth": any(r.ground_truth is not None for r in t.rubric),
        })

    tools_counts = [p["tools_required"] for p in per_task]
    criteria_counts = [p["criteria_count"] for p in per_task]

    return {
        "per_task": per_task,
        "avg_tools": round(sum(tools_counts) / len(tools_counts), 1) if tools_counts else 0,
        "avg_criteria": round(sum(criteria_counts) / len(criteria_counts), 1) if criteria_counts else 0,
    }


def coverage_gaps(tasks_dir: Path) -> dict:
    """Identify what's not covered by tasks."""
    tasks = _load_tasks(tasks_dir)

    # All tools used
    all_tools_used = set()
    for t in tasks:
        all_tools_used.update(t.tools)

    # All available tools
    available_tools = {"lookup_customer", "check_order", "send_reply", "update_ticket",
                       "send_msg", "read_msgs", "list_channels"}
    unused_tools = available_tools - all_tools_used

    # All customers referenced
    referenced_customers = set()
    referenced_orders = set()
    for t in tasks:
        if t.context.get("customer_id"):
            referenced_customers.add(t.context["customer_id"])
        if t.context.get("order_id"):
            referenced_orders.add(t.context["order_id"])

    # Rubric types used
    rubric_types_used = set()
    for t in tasks:
        for r in t.rubric:
            rubric_types_used.add(r.type)
    missing_types = {"tool_use", "correctness", "constraint", "format"} - rubric_types_used

    return {
        "tools_used": sorted(all_tools_used),
        "tools_unused": sorted(unused_tools),
        "customers_referenced": sorted(referenced_customers),
        "customers_count": len(referenced_customers),
        "orders_referenced": sorted(referenced_orders),
        "orders_count": len(referenced_orders),
        "rubric_types_missing": sorted(missing_types),
    }


def _load_tasks(tasks_dir: Path) -> list[Task]:
    """Load all task files from directory."""
    task_files = sorted(tasks_dir.glob("task_*.json"))
    return [Task.load(f) for f in task_files]
