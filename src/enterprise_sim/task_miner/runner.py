"""Task runner — execute a mined task against gpt-oss-20b via pi-mono,
then evaluate with gpt-5.4 LLM judge."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path

from openai import OpenAI

from enterprise_sim.orchestrator.agent_manager import PiAgent
from enterprise_sim.task_miner.schema import Task
from enterprise_sim.task_miner.snapshot import create_snapshot, reset_snapshot_for_task


AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"
# Use a support agent dir as base — the trainee acts as a support agent
DEFAULT_AGENT_DIR = AGENTS_DIR / "employee_support_01"


def run_task(
    task: Task,
    world_db: Path,
    provider: str = "openai",
    model: str = "gpt-oss-20b",
    timeout: int = 180,
) -> dict:
    """Run a single task against the trainee model.

    1. Creates a temporary DB snapshot reset to the task's starting state
    2. Spawns a pi-mono container with the trainee model + employee tools
    3. Sends the task prompt and collects the trajectory
    4. Returns trajectory + tool calls for rubric evaluation

    Args:
        task: The mined task to execute
        world_db: Path to the world snapshot DB
        provider: LLM provider (default: openai)
        model: Trainee model (default: gpt-oss-20b)
        timeout: Seconds before killing the agent

    Returns:
        dict with keys: response, tool_calls, duration_ms, success
    """
    # Create a temp snapshot for this run
    run_dir = Path(f"/tmp/esim_task_run_{task.id}")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    snapshot_path = create_snapshot(world_db, run_dir)
    # Rename to world.db — the Docker container expects /shared/world.db
    final_db = run_dir / "world.db"
    snapshot_path.rename(final_db)
    reset_snapshot_for_task(final_db, task.context)

    # Build env with API keys
    env = {}
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        env["OPENAI_API_KEY"] = api_key

    # Create agent — uses employee_support_01 as the base agent dir
    # (the trainee plays the role of a support agent)
    agent = PiAgent(
        agent_id=f"trainee_{task.id}",
        agent_dir=DEFAULT_AGENT_DIR,
        provider=provider,
        model=model,
        env=env,
    )
    agent.timeout = timeout

    # Override the sim DB directory so the container mounts the task snapshot
    agent._sim_db_dir = Path(run_dir).resolve()

    result = {
        "task_id": task.id,
        "model": model,
        "response": "",
        "tool_calls": [],
        "duration_ms": 0,
        "success": False,
        "error": None,
    }

    try:
        agent.spawn()

        # Send the task prompt (system prompt is built from agent dir,
        # but we prepend the task-specific context)
        full_prompt = f"""{task.user_message}

Use the CLI tools available to you (lookup-customer, check-order, send-reply, update-ticket, etc.) to handle this customer's request. Look up the relevant information before responding."""

        response = agent.send_message(full_prompt)

        trace = agent.last_trace or {}
        result["response"] = response
        result["tool_calls"] = trace.get("tool_calls", [])
        result["duration_ms"] = trace.get("duration_ms", 0)
        result["success"] = True

    except TimeoutError as e:
        result["error"] = str(e)
    except Exception as e:
        result["error"] = str(e)
    finally:
        agent.shutdown()

    return result


JUDGE_MODEL = "gpt-5.4"


def run_benchmark(
    tasks: list[Task],
    world_db: Path,
    models: list[str],
    provider: str = "openai",
    judge_model: str = JUDGE_MODEL,
    timeout: int = 180,
    on_result: callable = None,
) -> dict:
    """Run all tasks against all models, evaluate with LLM judge.

    Args:
        on_result: callback(run_num, total, model, task, result_dict) called after each run

    Returns a structured results dict with per-task scores, category/difficulty
    breakdowns, and overall scores per model.
    """
    total_runs = len(models) * len(tasks)
    run_num = 0

    results = {}  # model -> task_id -> evaluation
    task_meta = {}  # task_id -> {category, difficulty}

    for task in tasks:
        task_meta[task.id] = {
            "category": task.category,
            "difficulty": task.difficulty,
        }

    for model in models:
        results[model] = {}
        for task in tasks:
            run_num += 1

            trajectory = run_task(task, world_db, provider, model, timeout)

            if trajectory["success"]:
                evaluation = evaluate_rubric(task, trajectory, judge_model)
                results[model][task.id] = {
                    "reward": evaluation["reward"],
                    "scores": evaluation["scores"],
                    "tool_calls": len(trajectory["tool_calls"]),
                    "duration_ms": trajectory["duration_ms"],
                }
            else:
                results[model][task.id] = {
                    "reward": 0.0,
                    "scores": [],
                    "tool_calls": 0,
                    "duration_ms": 0,
                    "error": trajectory["error"],
                }

            if on_result:
                on_result(run_num, total_runs, model, task, results[model][task.id])

    # Build summary
    summary = {}
    for model in models:
        rewards = [results[model][tid]["reward"] for tid in results[model]]
        overall = sum(rewards) / len(rewards) if rewards else 0.0

        by_category: dict[str, list[float]] = {}
        by_difficulty: dict[str, list[float]] = {}
        for tid, meta in task_meta.items():
            if tid in results[model]:
                r = results[model][tid]["reward"]
                by_category.setdefault(meta["category"], []).append(r)
                by_difficulty.setdefault(meta["difficulty"], []).append(r)

        summary[model] = {
            "overall": round(overall, 3),
            "by_category": {
                k: round(sum(v) / len(v), 3) for k, v in sorted(by_category.items())
            },
            "by_difficulty": {
                k: round(sum(v) / len(v), 3) for k, v in sorted(by_difficulty.items())
            },
        }

    return {
        "models": models,
        "tasks": [t.id for t in tasks],
        "judge_model": judge_model,
        "results": results,
        "summary": summary,
    }


def evaluate_rubric(
    task: Task,
    trajectory: dict,
    judge_model: str = JUDGE_MODEL,
) -> dict:
    """Evaluate a trajectory against a task's rubric using an LLM judge.

    Sends all rubric criteria in a single call to gpt-5.4. The judge scores
    each criterion as 0.0 (fail), 0.5 (partial), or 1.0 (pass) with reasoning.
    Criteria are grounded in verifiable facts (DB values, tool names, policy terms)
    so the judge can make objective assessments.

    Returns per-criterion scores with reasoning and a weighted total reward.
    """
    client = OpenAI()

    response_text = trajectory.get("response", "")
    tool_calls = trajectory.get("tool_calls", [])

    # Build criteria list for the prompt
    criteria_block = ""
    for i, c in enumerate(task.rubric, 1):
        gt = f'\n   Ground truth: "{c.ground_truth}"' if c.ground_truth else ""
        criteria_block += f"""
{i}. [{c.type}] (weight={c.weight}) {c.criterion}{gt}"""

    judge_prompt = f"""You are an expert evaluator for a customer support training environment.
You are judging whether a trainee support agent correctly handled a customer request.

## Task Description
Category: {task.category} | Difficulty: {task.difficulty}
Customer message: {task.user_message}

## Agent's Response
{response_text}

## Tools Called by Agent
{json.dumps(tool_calls, indent=2)}

## Rubric Criteria to Evaluate
{criteria_block}

## Instructions
For EACH criterion above, determine if the agent satisfied it based on the response and tool calls.
- tool_use criteria: Check if the agent actually called the relevant tool (tools show as "bash" when using CLI commands like `esim lookup-customer`, `esim check-order`, etc.)
- correctness criteria: Check if the ground truth fact appears correctly in the response
- constraint criteria: Check if the agent applied the business rule correctly (e.g., escalated when required, stayed within refund limits)
- format criteria: Evaluate communication quality, professionalism, and completeness

Respond with a JSON object containing a "scores" array with one entry per criterion in order:
{{
  "scores": [
    {{"criterion_index": 1, "score": 0.0, "reasoning": "..."}},
    {{"criterion_index": 2, "score": 1.0, "reasoning": "..."}},
    ...
  ]
}}

You MUST include exactly {len(task.rubric)} entries in the scores array, one for each criterion.
Scores: 1.0 = fully satisfied, 0.5 = partially satisfied, 0.0 = not satisfied.
Be strict — partial credit (0.5) only when the agent made a genuine attempt but missed key details."""

    completion = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    # Parse judge response
    judge_raw = completion.choices[0].message.content
    try:
        judge_result = json.loads(judge_raw)
        # Handle both {"scores": [...]} and direct [...] formats
        if isinstance(judge_result, dict):
            judge_scores = judge_result.get("scores", judge_result.get("criteria", []))
            if not judge_scores:
                # Try to find any list value
                for v in judge_result.values():
                    if isinstance(v, list):
                        judge_scores = v
                        break
        elif isinstance(judge_result, list):
            judge_scores = judge_result
        else:
            judge_scores = []
    except json.JSONDecodeError:
        print(f"[Judge] Failed to parse response: {judge_raw[:200]}")
        judge_scores = []

    # Map judge scores back to rubric criteria
    scores = []
    for i, criterion in enumerate(task.rubric):
        judge_entry = judge_scores[i] if i < len(judge_scores) else {}
        score = float(judge_entry.get("score", 0.0))
        reasoning = judge_entry.get("reasoning", "No evaluation available")

        scores.append({
            "criterion": criterion.criterion,
            "type": criterion.type,
            "weight": criterion.weight,
            "score": score,
            "reasoning": reasoning,
        })

    total_reward = sum(s["score"] * s["weight"] for s in scores)

    return {
        "task_id": task.id,
        "judge_model": judge_model,
        "scores": scores,
        "reward": total_reward,
    }
