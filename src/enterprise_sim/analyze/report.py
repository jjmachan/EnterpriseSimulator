"""Combined report generation: merges all analyses into one output."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click

from enterprise_sim.analyze import world, tasks, simulation


def generate_report(
    db_path: Path | None = None,
    tasks_dir: Path | None = None,
) -> dict:
    """Generate a combined analysis report."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "inputs": {
            "db_path": str(db_path) if db_path else None,
            "tasks_dir": str(tasks_dir) if tasks_dir else None,
        },
    }

    if db_path:
        report["world"] = world.entity_statistics(db_path)
        report["coherence"] = world.coherence_checks(db_path)
        report["interconnectedness"] = world.interconnectedness(db_path)
        report["tickets"] = simulation.ticket_patterns(db_path)
        report["agent_behavior"] = simulation.agent_behavior(db_path)
        report["conversations"] = simulation.conversation_quality(db_path)
        report["resolution"] = simulation.resolution_metrics(db_path)

    if tasks_dir:
        report["task_distribution"] = tasks.task_distribution(tasks_dir)
        report["rubric_coverage"] = tasks.rubric_coverage(tasks_dir)
        report["task_complexity"] = tasks.task_complexity(tasks_dir)
        report["coverage_gaps"] = tasks.coverage_gaps(tasks_dir)

    return report


def print_world_report(stats: dict, coherence_result: dict, inter: dict) -> None:
    """Print world analysis to console."""
    click.echo("=" * 60)
    click.echo("  World Quality Analysis")
    click.echo("=" * 60)

    # Entity counts
    click.echo("\nEntity Counts:")
    counts = stats["entity_counts"]
    for table, count in sorted(counts.items()):
        new = stats["new_entities"].get(table, 0)
        new_str = f" (+{new} new)" if new > 0 else ""
        click.echo(f"  {table:<20} {count:>6}{new_str}")
    click.echo(f"  {'TOTAL':<20} {stats['total_entities']:>6} (+{stats['total_new']} new)")

    # Sim tables
    if any(v > 0 for v in stats["sim_counts"].values()):
        click.echo("\nSimulation Tables:")
        for table, count in stats["sim_counts"].items():
            if count > 0:
                click.echo(f"  {table:<20} {count:>6}")

    # Relationship density
    click.echo("\nRelationship Density:")
    for rel, s in stats["relationship_density"].items():
        click.echo(f"  {rel:<25} mean={s['mean']:.1f}  min={s['min']}  max={s['max']}")

    # Coherence
    click.echo(f"\nCoherence Checks: {'PASSED' if coherence_result['passed'] else 'FAILED'}")
    if coherence_result["issues"]:
        click.echo("  Issues:")
        for issue in coherence_result["issues"]:
            click.echo(f"    - {issue}")
    if coherence_result["warnings"]:
        click.echo("  Warnings:")
        for w in coherence_result["warnings"]:
            click.echo(f"    - {w}")

    # Interconnectedness
    click.echo("\nEntity Interconnectedness:")
    summary = inter["summary"]
    if summary:
        click.echo(f"  Avg reach per customer: {summary['mean']:.1f} entities")
        click.echo(f"  Min/Max reach: {summary['min']}/{summary['max']}")
    click.echo(f"  Total relationships: {inter['total_unique_relationships']}")


def print_tasks_report(dist: dict, rubric: dict, complexity: dict, gaps: dict) -> None:
    """Print task analysis to console."""
    click.echo("\n" + "=" * 60)
    click.echo("  Task Quality Analysis")
    click.echo("=" * 60)

    click.echo(f"\nTotal tasks: {dist['total_tasks']}")

    click.echo("\nBy Category:")
    for cat, count in sorted(dist["by_category"].items()):
        click.echo(f"  {cat:<25} {count}")

    click.echo("\nBy Difficulty:")
    for diff, count in sorted(dist["by_difficulty"].items()):
        click.echo(f"  {diff:<25} {count}")

    if dist["missing_combinations"]:
        click.echo(f"\nMissing category/difficulty combinations ({dist['gap_count']}):")
        for g in dist["missing_combinations"]:
            click.echo(f"  - {g}")

    click.echo(f"\nRubric Coverage ({rubric['total_criteria']} criteria):")
    for rtype, count in sorted(rubric["by_type"].items()):
        avg_w = rubric["avg_weight_by_type"].get(rtype, 0)
        click.echo(f"  {rtype:<15} {count:>3} criteria  avg_weight={avg_w:.3f}")
    click.echo(f"  With ground truth: {rubric['with_ground_truth']} ({rubric['ground_truth_pct']}%)")

    click.echo(f"\nComplexity: avg {complexity['avg_tools']:.0f} tools, {complexity['avg_criteria']:.0f} criteria per task")

    if gaps["tools_unused"]:
        click.echo(f"\nUnused tools: {', '.join(gaps['tools_unused'])}")
    click.echo(f"  Customers referenced: {gaps['customers_count']}/12")
    click.echo(f"  Orders referenced: {gaps['orders_count']}/40")


def print_simulation_report(tickets: dict, behavior: dict, convos: dict, resolution: dict) -> None:
    """Print simulation analysis to console."""
    click.echo("\n" + "=" * 60)
    click.echo("  Simulation Quality Analysis")
    click.echo("=" * 60)

    # Tickets
    click.echo(f"\nTickets: {tickets['total_tickets']} total ({tickets['new_tickets']} new)")
    click.echo("  By status:")
    for status, count in sorted(tickets["by_status"].items()):
        click.echo(f"    {status:<15} {count}")
    click.echo("  By priority:")
    for pri, count in sorted(tickets["by_priority"].items()):
        click.echo(f"    {pri:<15} {count}")
    click.echo(f"  Unique subjects: {tickets['unique_subjects']}")

    if tickets["per_tick"]:
        click.echo("  Tickets per tick:")
        for tick, count in sorted(tickets["per_tick"].items()):
            click.echo(f"    tick {tick:>2}: {'#' * count} ({count})")

    # Agent behavior
    if behavior.get("available") and behavior.get("total_traces", 0) > 0:
        click.echo(f"\nAgent Behavior ({behavior['total_traces']} traces):")
        if behavior.get("by_phase"):
            click.echo("  By phase:")
            for phase, info in sorted(behavior["by_phase"].items()):
                click.echo(f"    {phase:<25} {info['count']:>3} traces  avg={info['avg_duration_ms']}ms")
        if behavior.get("tool_usage"):
            click.echo("  Tool usage:")
            for tool, count in sorted(behavior["tool_usage"].items(), key=lambda x: -x[1]):
                click.echo(f"    {tool:<25} {count}")
    elif not behavior.get("available"):
        click.echo(f"\nAgent Behavior: {behavior.get('reason', 'N/A')}")

    # Conversations
    click.echo(f"\nConversations:")
    click.echo(f"  Total messages: {convos['total_messages']} ({convos['new_messages']} new)")
    click.echo(f"  Avg messages/ticket: {convos['avg_messages_per_ticket']}")
    click.echo(f"  Avg message length: {convos['avg_message_length']} chars")
    click.echo(f"  Avg turns/ticket: {convos['avg_turns_per_ticket']}")
    click.echo(f"  No messages: {convos['tickets_with_no_messages']} | Single: {convos['tickets_with_single_message']} | Multi-turn: {convos['tickets_with_multi_turn']}")

    # Resolution
    click.echo(f"\nResolution (new tickets only):")
    click.echo(f"  Resolution rate: {resolution['resolution_rate']}%")
    click.echo(f"  Escalation rate: {resolution['escalation_rate']}%")
    click.echo(f"  Open rate: {resolution['open_rate']}%")
    if resolution["avg_days_to_resolve"] is not None:
        click.echo(f"  Avg days to resolve: {resolution['avg_days_to_resolve']}")
    click.echo(f"  Unassigned: {resolution['unassigned_new']}")
