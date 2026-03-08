"""Offline data collection: run Qwen 3.5-4B against EnterpriseSim environment.

Collects episode trajectories and formats them as a GRPO-compatible HuggingFace Dataset.

Prerequisites:
  - vLLM serving Qwen 3.5-4B on localhost:8001
  - OPENAI_API_KEY set (for customer agent LLM responses)

Usage:
  python scripts/collect_data.py --vllm-url http://localhost:8001/v1
"""

import argparse
import json
import re
import random
import sys
import time
from pathlib import Path

from openai import OpenAI
from datasets import Dataset

# Add parent to path so we can import server modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.environment import CustomerSupportEnvironment, SupportObservation
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# --- Tool call parsing (Qwen 3.5 XML format) ---

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", re.DOTALL
)
PARAM_RE = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)


def parse_tool_call(text: str) -> tuple[str | None, dict | None]:
    """Extract tool name and arguments from Qwen XML tool call format."""
    match = TOOL_CALL_RE.search(text)
    if not match:
        return None, None
    tool_name = match.group(1).strip()
    args = {}
    for pm in PARAM_RE.finditer(match.group(2)):
        key = pm.group(1).strip()
        val = pm.group(2).strip()
        if key == "ticket_id":
            try:
                val = int(val)
            except ValueError:
                pass
        args[key] = val
    return tool_name, args


# --- Prompt engineering ---


def format_tools(tools) -> str:
    """Format tool list into readable text for the system prompt."""
    lines = []
    for t in tools:
        lines.append(f"### {t.name}")
        lines.append(f"{t.description}")
        schema = t.input_schema
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        if props:
            lines.append("Parameters:")
            for pname, pinfo in props.items():
                req = " (required)" if pname in required else " (optional)"
                desc = pinfo.get("description", "")
                ptype = pinfo.get("type", "string")
                lines.append(f"  - {pname} ({ptype}{req}): {desc}")
        lines.append("")
    return "\n".join(lines)


def build_system_prompt(env: CustomerSupportEnvironment) -> str:
    """Build the agent system prompt with tools + work context."""
    handbook = (DATA_DIR / "work_context/handbook.md").read_text()
    escalation = (DATA_DIR / "work_context/escalation_policy.md").read_text()
    catalog = (DATA_DIR / "work_context/product_catalog.md").read_text()

    # Get tool schemas from the MCP environment
    tools_obs = env._handle_list_tools()
    tool_text = format_tools(tools_obs.tools)

    return f"""You are a Customer Support Representative at Office Furniture Co. Help customers by investigating their issues and providing concrete solutions.

## Available Tools

{tool_text}

## Company Policies

{handbook}

## Escalation Policy

{escalation}

## Product Catalog

{catalog}

## How to Respond

Use EXACTLY this XML format for tool calls:
<tool_call>
<function=TOOL_NAME>
<parameter=PARAM_NAME>value</parameter>
</function>
</tool_call>

Strategy:
1. Look up the customer profile first
2. Check their order details
3. Send a helpful reply with a concrete solution
4. Resolve the ticket when the issue is addressed

Always investigate before replying. Be professional and empathetic."""


def format_initial_obs(obs: SupportObservation) -> str:
    """Format the initial observation (from reset) as a user message."""
    return f"""New support ticket received.

{obs.ticket_context}

Customer message:
{obs.customer_message}

What tool would you like to use to help this customer?"""


def format_step_obs(obs: SupportObservation) -> str:
    """Format a step observation (after tool call) as a user message."""
    parts = []

    if obs.tool_name:
        parts.append(f'Tool "{obs.tool_name}" result:')
        parts.append(obs.tool_result if obs.tool_result else "(no result)")
        parts.append("")

    if obs.customer_message:
        parts.append("Customer responded:")
        parts.append(obs.customer_message)
        parts.append("")

    parts.append(f"Satisfaction: {obs.satisfaction:.0%} | Steps: {obs.step_count}/10")
    parts.append("")
    parts.append("What would you like to do next?")

    return "\n".join(parts)


# --- Episode runner ---


def run_episode(env, generate_fn, system_prompt, task_id=None, seed=None):
    """Run one full episode, returning list of step records."""
    reset_kwargs = {}
    if seed is not None:
        reset_kwargs["seed"] = seed
    if task_id:
        reset_kwargs["task_id"] = task_id

    obs = env.reset(**reset_kwargs)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": format_initial_obs(obs)},
    ]

    steps = []
    ticket_id = obs.ticket_id

    while not obs.done:
        # Snapshot the prompt at this decision point
        prompt_snapshot = [dict(m) for m in messages]

        # Generate with the trainee model
        try:
            response = generate_fn(messages)
        except Exception as e:
            print(f"    Generation error: {e}")
            break

        # Parse tool call from response
        tool_name, tool_args = parse_tool_call(response)

        if tool_name is None:
            # Fallback: treat raw text as a send_reply message
            tool_name = "send_reply"
            tool_args = {"ticket_id": ticket_id, "message": response[:500]}

        # Ensure ticket_id is set for tools that need it
        if tool_name in ("send_reply", "update_ticket") and "ticket_id" not in tool_args:
            tool_args["ticket_id"] = ticket_id

        # Execute in environment
        action = CallToolAction(tool_name=tool_name, arguments=tool_args)
        try:
            obs = env.step(action)
        except Exception as e:
            print(f"    Step error: {e}")
            steps.append({
                "prompt": prompt_snapshot,
                "completion": response,
                "error": str(e),
            })
            break

        steps.append({
            "prompt": prompt_snapshot,
            "completion": response,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": getattr(obs, "tool_result", ""),
            "customer_message": getattr(obs, "customer_message", ""),
            "satisfaction": getattr(obs, "satisfaction", 0.0),
            "satisfaction_delta": getattr(obs, "satisfaction_delta", 0.0),
            "done": obs.done,
            "reward": obs.reward,
            "resolved": getattr(obs, "resolved", False),
            "step_count": getattr(obs, "step_count", 0),
        })

        # Extend conversation for next turn
        messages.append({"role": "assistant", "content": response})
        if not obs.done:
            messages.append({"role": "user", "content": format_step_obs(obs)})

    # Backfill final episode reward to all steps
    final_reward = obs.reward if hasattr(obs, "reward") else 0.0
    resolved = getattr(obs, "resolved", False)
    for step in steps:
        step["episode_reward"] = final_reward
        step["episode_resolved"] = resolved
        step["task_id"] = task_id

    return steps


# --- Dataset formatting ---


def load_tasks(tasks_dir: Path) -> dict:
    """Load all task JSON files."""
    tasks = {}
    for f in sorted(tasks_dir.glob("task_*.json")):
        with open(f) as fh:
            data = json.load(fh)
            tasks[data["id"]] = data
    return tasks


def format_grpo_dataset(all_steps, tasks):
    """Convert collected steps into GRPO training dataset."""
    records = []
    for step in all_steps:
        if "error" in step:
            continue  # Skip failed steps

        task_data = tasks.get(step.get("task_id")) if step.get("task_id") else None
        ground_truths = []
        if task_data:
            ground_truths = [
                c.get("ground_truth")
                for c in task_data.get("rubric", [])
                if c.get("ground_truth")
            ]

        answer = json.dumps({
            "episode_reward": step["episode_reward"],
            "resolved": step["episode_resolved"],
            "task_id": step.get("task_id"),
            "ground_truth_values": ground_truths,
            "valid_tools": ["lookup_customer", "check_order", "send_reply", "update_ticket"],
        })

        records.append({"prompt": step["prompt"], "answer": answer})

    return Dataset.from_list(records)


# --- Main ---


def main():
    parser = argparse.ArgumentParser(description="Collect offline RL training data")
    parser.add_argument("--vllm-url", default="http://localhost:8001/v1", help="vLLM API URL")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="Model name for vLLM")
    parser.add_argument("--runs-per-task", type=int, default=8, help="Rollouts per task")
    parser.add_argument("--random-episodes", type=int, default=16, help="Random episodes (no task_id)")
    parser.add_argument("--output-dir", default="./data/trajectories", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # vLLM client
    client = OpenAI(base_url=args.vllm_url, api_key="none")

    def generate_fn(messages):
        resp = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )
        return resp.choices[0].message.content or ""

    # Verify vLLM is reachable
    try:
        models = client.models.list()
        print(f"Connected to vLLM. Available models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"ERROR: Cannot connect to vLLM at {args.vllm_url}: {e}")
        print("Start vLLM first: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.5-4B --port 8001")
        sys.exit(1)

    # Environment (in-process)
    print("Initializing environment...")
    env = CustomerSupportEnvironment()
    system_prompt = build_system_prompt(env)
    print(f"System prompt: {len(system_prompt)} chars")

    # Load tasks
    tasks = load_tasks(DATA_DIR / "tasks")
    print(f"Loaded {len(tasks)} tasks: {list(tasks.keys())}")

    # Collect episodes
    all_steps = []
    episode_count = 0
    total_episodes = len(tasks) * args.runs_per_task + args.random_episodes

    print(f"\n=== Collecting {total_episodes} episodes ===\n")

    # Task-based episodes
    for task_id in tasks:
        for run_idx in range(args.runs_per_task):
            seed = args.seed + run_idx
            episode_count += 1
            print(f"[{episode_count}/{total_episodes}] {task_id} (run {run_idx + 1})...", end=" ")

            t0 = time.time()
            steps = run_episode(env, generate_fn, system_prompt, task_id=task_id, seed=seed)
            elapsed = time.time() - t0

            all_steps.extend(steps)
            reward = steps[-1]["episode_reward"] if steps else 0.0
            resolved = steps[-1].get("episode_resolved", False) if steps else False
            print(f"{len(steps)} steps, reward={reward:.3f}, resolved={resolved}, {elapsed:.1f}s")

    # Random episodes
    for i in range(args.random_episodes):
        seed = args.seed + 1000 + i
        episode_count += 1
        print(f"[{episode_count}/{total_episodes}] random (seed={seed})...", end=" ")

        t0 = time.time()
        steps = run_episode(env, generate_fn, system_prompt, seed=seed)
        elapsed = time.time() - t0

        all_steps.extend(steps)
        reward = steps[-1]["episode_reward"] if steps else 0.0
        resolved = steps[-1].get("episode_resolved", False) if steps else False
        print(f"{len(steps)} steps, reward={reward:.3f}, resolved={resolved}, {elapsed:.1f}s")

    # Format and save
    print(f"\n=== Formatting {len(all_steps)} steps as GRPO dataset ===")
    dataset = format_grpo_dataset(all_steps, tasks)

    dataset.save_to_disk(str(output_dir / "grpo_dataset"))
    dataset.to_json(str(output_dir / "grpo_dataset.jsonl"))

    # Save raw episodes for debugging
    with open(output_dir / "episodes_raw.json", "w") as f:
        json.dump(all_steps, f, indent=2, default=str)

    # Summary stats
    rewards = [s["episode_reward"] for s in all_steps if "episode_reward" in s]
    resolved_count = sum(1 for s in all_steps if s.get("episode_resolved"))
    print(f"\nDone!")
    print(f"  Total training examples: {len(dataset)}")
    print(f"  Episodes: {total_episodes}")
    print(f"  Avg episode reward: {sum(rewards) / len(rewards):.3f}" if rewards else "  No rewards")
    print(f"  Steps with resolution: {resolved_count}/{len(all_steps)}")
    print(f"  Saved to: {output_dir}")

    env.close()


if __name__ == "__main__":
    main()
