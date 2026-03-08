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
