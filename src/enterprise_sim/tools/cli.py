"""Main CLI entry point for esim."""

import json
import os
import subprocess
from pathlib import Path

import click

from enterprise_sim.tools.employee_tools import (
    check_order,
    list_channels,
    lookup_customer,
    read_msgs,
    send_msg,
    send_reply,
    update_ticket,
)

AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"
SHARED_DIR = Path(__file__).resolve().parent.parent / "shared"


@click.group()
def cli():
    """EnterpriseSim CLI tools."""
    pass


cli.add_command(lookup_customer)
cli.add_command(check_order)
cli.add_command(send_reply)
cli.add_command(update_ticket)
cli.add_command(send_msg)
cli.add_command(read_msgs)
cli.add_command(list_channels)


# --- agent subgroup ---


@cli.group()
def agent():
    """Manage and interact with customer agents."""
    pass


def _detect_agent(agent_dir: Path) -> tuple[dict, str] | None:
    """Load agent config and detect type. Returns (config, type) or None."""
    role_path = agent_dir / "role.json"
    persona_path = agent_dir / "persona.json"
    if role_path.exists():
        with open(role_path) as f:
            return json.load(f), "employee"
    if persona_path.exists():
        with open(persona_path) as f:
            return json.load(f), "customer"
    return None


@agent.command("list")
def agent_list():
    """List all available agents with persona summary."""
    agent_dirs = sorted(
        d for d in AGENTS_DIR.iterdir()
        if d.is_dir() and _detect_agent(d) is not None
    )
    if not agent_dirs:
        click.echo("No agents found.")
        return

    # Header
    click.echo(f"{'ID':<24} {'Type':<10} {'Name':<20} {'Role/Style'}")
    click.echo("-" * 90)

    for d in agent_dirs:
        data, agent_type = _detect_agent(d)
        if agent_type == "customer":
            role_style = data.get("communication_style", "")
        else:
            role_style = data.get("role", "")
        click.echo(
            f"{d.name:<24} {agent_type:<10} {data['name']:<20} {role_style}"
        )


@agent.command("chat")
@click.argument("agent_id")
@click.option("--provider", default="openai", help="LLM provider")
@click.option("--model", default="gpt-5-mini", help="Model name")
def agent_chat(agent_id, provider, model):
    """Open pi-mono TUI to chat with an agent interactively."""
    from enterprise_sim.orchestrator.agent_manager import PiAgent

    agent_dir = AGENTS_DIR / agent_id
    if not agent_dir.exists() or _detect_agent(agent_dir) is None:
        click.echo(f"Agent '{agent_id}' not found. Run 'esim agent list' to see available agents.")
        raise SystemExit(1)

    # Build system prompt by instantiating PiAgent (without spawning)
    env = {}
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        env["OPENAI_API_KEY"] = api_key

    pi = PiAgent(agent_id, agent_dir, provider, model, env)
    system_prompt = pi._build_system_prompt()
    docker_image = pi._get_docker_image()
    extension_path = pi._get_extension_path()

    # Build docker command
    agent_dir_abs = agent_dir.resolve()
    cmd = [
        "docker", "run", "-it", "--rm",
        "-v", f"{agent_dir_abs}:/agent",
    ]
    # Employee agents get shared dir + work_context mounts
    if pi.agent_type == "employee":
        shared_dir = SHARED_DIR.resolve()
        cmd.extend([
            "-v", f"{shared_dir}:/shared",
            "-v", f"{shared_dir}/work_context:/work_context",
            "-e", "ENTERPRISE_SIM_DB_PATH=/shared/world.db",
        ])
    for key, value in env.items():
        cmd.extend(["-e", f"{key}={value}"])
    cmd.extend([
        docker_image,
        "--provider", provider,
        "--model", model,
        "--system-prompt", system_prompt,
        "--extension", extension_path,
    ])

    click.echo(f"Launching interactive session with {pi._persona['name']} ({agent_id})...")
    click.echo("Press Ctrl+C to exit.\n")
    subprocess.run(cmd)


@cli.command("benchmark")
@click.option("--tasks-dir", required=True, type=click.Path(exists=True), help="Directory containing task JSON files")
@click.option("--db", required=True, type=click.Path(exists=True), help="Path to world snapshot DB")
@click.option("--models", required=True, help="Comma-separated model names (e.g. gpt-5-mini,gpt-5-nano)")
@click.option("--provider", default="openai", help="LLM provider")
@click.option("--judge-model", default="gpt-5.4", help="LLM judge model for rubric evaluation")
@click.option("--timeout", default=180, type=int, help="Timeout per task in seconds")
@click.option("--output", default=None, type=click.Path(), help="Save full results JSON to this path")
def benchmark(tasks_dir, db, models, provider, judge_model, timeout, output):
    """Run the full benchmark: all tasks × all models, evaluated by LLM judge."""
    import json as _json
    from enterprise_sim.task_miner.schema import Task
    from enterprise_sim.task_miner.runner import run_benchmark

    # Load tasks
    tasks_path = Path(tasks_dir)
    task_files = sorted(tasks_path.glob("task_*.json"))
    if not task_files:
        click.echo(f"No task files found in {tasks_dir}")
        raise SystemExit(1)

    tasks = [Task.load(f) for f in task_files]
    model_list = [m.strip() for m in models.split(",")]

    click.echo(f"EnterpriseSim Benchmark")
    click.echo(f"  Tasks: {len(tasks)} | Models: {', '.join(model_list)} | Judge: {judge_model}")
    click.echo(f"  Total runs: {len(tasks) * len(model_list)}")
    click.echo("")

    def on_result(run_num, total, model, task, result):
        reward = result.get("reward", 0.0)
        err = result.get("error")
        duration = result.get("duration_ms", 0)
        tools = result.get("tool_calls", 0)
        if err:
            click.echo(f"  [{run_num}/{total}] {model} | {task.id} ({task.difficulty}) ERROR: {err[:60]}")
        else:
            click.echo(f"  [{run_num}/{total}] {model} | {task.id} ({task.difficulty}) -> {reward:.3f}  ({duration/1000:.0f}s, {tools} tools)")

    results = run_benchmark(
        tasks=tasks,
        world_db=Path(db),
        models=model_list,
        provider=provider,
        judge_model=judge_model,
        timeout=timeout,
        on_result=on_result,
    )

    # Print summary table
    click.echo("")
    click.echo("=" * 70)
    click.echo(f"  Benchmark Results (judge: {judge_model})")
    click.echo("=" * 70)

    # Column widths
    label_w = 36
    col_w = 12

    # Header
    header = " " * label_w + "".join(f"{m:>{col_w}}" for m in model_list)
    click.echo("")
    click.echo("Task Results:")
    click.echo(header)

    for task in tasks:
        cat_short = task.category[:8]
        label = f"  {task.id[:20]} ({cat_short}/{task.difficulty})"
        row = f"{label:<{label_w}}"
        for model in model_list:
            r = results["results"][model].get(task.id, {})
            reward = r.get("reward", 0.0)
            err = r.get("error")
            if err:
                row += f"{'ERROR':>{col_w}}"
            else:
                row += f"{reward:>{col_w}.3f}"
        click.echo(row)

    # By category
    click.echo("")
    click.echo("By Category:")
    click.echo(header)
    all_categories = sorted(set(t.category for t in tasks))
    for cat in all_categories:
        label = f"  {cat}"
        row = f"{label:<{label_w}}"
        for model in model_list:
            val = results["summary"][model]["by_category"].get(cat, 0.0)
            row += f"{val:>{col_w}.3f}"
        click.echo(row)

    # By difficulty
    click.echo("")
    click.echo("By Difficulty:")
    click.echo(header)
    for diff in ["easy", "medium", "hard"]:
        label = f"  {diff}"
        row = f"{label:<{label_w}}"
        for model in model_list:
            val = results["summary"][model]["by_difficulty"].get(diff, 0.0)
            row += f"{val:>{col_w}.3f}"
        click.echo(row)

    # Overall
    click.echo("")
    click.echo("Overall:")
    for model in model_list:
        click.echo(f"  {model:<{label_w - 2}}{results['summary'][model]['overall']:>{col_w}.3f}")

    click.echo("")

    # Save results
    if output:
        out_path = Path(output)
        with open(out_path, "w") as f:
            _json.dump(results, f, indent=2)
        click.echo(f"Full results saved to {out_path}")


@cli.command("run-task")
@click.argument("task_path", type=click.Path(exists=True))
@click.option("--db", required=True, type=click.Path(exists=True), help="Path to world snapshot DB")
@click.option("--provider", default="openai", help="LLM provider")
@click.option("--model", default="gpt-oss-20b", help="Trainee model name")
@click.option("--judge-model", default="gpt-5.4", help="LLM judge model for rubric evaluation")
@click.option("--timeout", default=180, type=int, help="Timeout in seconds")
def run_task(task_path, db, provider, model, judge_model, timeout):
    """Run a mined task against a trainee model, then evaluate with LLM judge."""
    from enterprise_sim.task_miner.schema import Task
    from enterprise_sim.task_miner.runner import run_task as _run_task, evaluate_rubric

    task = Task.load(Path(task_path))
    click.echo(f"Running task {task.id} ({task.category}/{task.difficulty}) against {model}...")

    trajectory = _run_task(task, Path(db), provider, model, timeout)

    if trajectory["success"]:
        click.echo(f"\nResponse ({trajectory['duration_ms']}ms, {len(trajectory['tool_calls'])} tool calls):")
        click.echo(trajectory["response"][:500])
        click.echo(f"\n--- Rubric Evaluation (judge: {judge_model}) ---")
        evaluation = evaluate_rubric(task, trajectory, judge_model=judge_model)
        for s in evaluation["scores"]:
            status = "PASS" if s["score"] >= 0.5 else "FAIL"
            click.echo(f"  [{status}] {s['criterion']}")
            click.echo(f"         score={s['score']:.1f}  weight={s['weight']:.1f}  | {s['reasoning']}")
        click.echo(f"\nTotal reward: {evaluation['reward']:.3f}")
    else:
        click.echo(f"FAILED: {trajectory['error']}")


@cli.command("simulate")
@click.option("--ticks", default=12, type=int, help="Number of simulation ticks")
@click.option("--ticket-prob", default=0.15, type=float, help="Per-customer ticket probability per tick")
@click.option("--provider", default="openai", help="LLM provider")
@click.option("--model", default="gpt-5-mini", help="Model name")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility")
@click.option("--output", default=None, type=click.Path(), help="Output directory path")
@click.option("--max-customers", default=None, type=int, help="Max customer agents to use")
@click.option("--max-employees", default=None, type=int, help="Max employee agents to use")
def simulate(ticks, ticket_prob, provider, model, seed, output, max_customers, max_employees):
    """Run a multi-agent world simulation."""
    from enterprise_sim.orchestrator.sim_config import WorldConfig
    from enterprise_sim.orchestrator.simulation_engine import SimulationEngine

    config = WorldConfig(
        num_ticks=ticks,
        ticket_probability=ticket_prob,
        provider=provider,
        model=model,
        seed=seed,
        output_dir=Path(output) if output else None,
        max_customers=max_customers,
        max_employees=max_employees,
    )
    engine = SimulationEngine(config)
    engine.run()


if __name__ == "__main__":
    cli()
