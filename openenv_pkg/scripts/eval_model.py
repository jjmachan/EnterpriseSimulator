"""Quick evaluation: run model on test tasks and score with reward functions.

Usage:
  # Evaluate base model via vLLM
  python scripts/eval_model.py --vllm-url http://localhost:8001/v1 --model Qwen/Qwen3-8B

  # Evaluate trained adapter (local inference)
  python scripts/eval_model.py --adapter-path ./outputs/grpo-support-agent/final_adapter

  # Run all 8 tasks and save results
  python scripts/eval_model.py --all-tasks --label vanilla --output-json ./outputs/eval_vanilla.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.environment import CustomerSupportEnvironment
from scripts.collect_data import (
    build_system_prompt, format_initial_obs, format_step_obs,
    parse_tool_call, run_episode, load_tasks, DATA_DIR,
)
from scripts.train_grpo import (
    format_reward, tool_validity_reward, reasoning_reward,
    no_reasoning_leak_reward, _get_text, TOOL_CALL_RE,
)


def eval_episode_rewards(steps):
    """Score an episode's steps with all reward functions."""
    scores = {"format": [], "tool_valid": [], "reasoning": [], "no_leak": []}
    for s in steps:
        comp = s.get("completion", "")
        comp_list = [[{"role": "assistant", "content": comp}]]
        scores["format"].extend(format_reward(comp_list))
        scores["tool_valid"].extend(tool_validity_reward(comp_list))
        scores["reasoning"].extend(reasoning_reward(comp_list))
        scores["no_leak"].extend(no_reasoning_leak_reward(comp_list))
    return {k: sum(v) / len(v) if v else 0 for k, v in scores.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-url", default="http://localhost:8001/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter-path", default=None, help="Path to LoRA adapter (if evaluating trained model)")
    parser.add_argument("--tasks", type=int, default=4, help="Number of tasks to evaluate")
    parser.add_argument("--all-tasks", action="store_true", help="Run all 8 tasks")
    parser.add_argument("--seed", type=int, default=77, help="Random seed")
    parser.add_argument("--label", default=None, help="Label for this eval run")
    parser.add_argument("--output-json", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    if args.all_tasks:
        args.tasks = 999  # will be capped by actual task count

    if args.adapter_path:
        # Load model with adapter for local inference
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch

        print(f"Loading base model {args.model}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        print(f"Loading adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model.eval()

        def generate_fn(messages):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=512, temperature=0.7,
                    do_sample=True, top_p=0.9,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True)
    else:
        # Use vLLM
        from openai import OpenAI
        client = OpenAI(base_url=args.vllm_url, api_key="none")
        try:
            models = client.models.list()
            print(f"Connected to vLLM: {[m.id for m in models.data]}")
        except Exception as e:
            print(f"Cannot connect to vLLM: {e}")
            sys.exit(1)

        def generate_fn(messages):
            resp = client.chat.completions.create(
                model=args.model, messages=messages,
                temperature=0.7, max_tokens=512,
            )
            return resp.choices[0].message.content or ""

    # Setup environment
    env = CustomerSupportEnvironment()
    system_prompt = build_system_prompt(env)
    tasks = load_tasks(DATA_DIR / "tasks")
    task_ids = list(tasks.keys())[:args.tasks]

    label = args.label or ("adapter" if args.adapter_path else "base")
    print(f"\n=== Evaluating [{label}] on {len(task_ids)} tasks ===\n")

    results = []
    for task_id in task_ids:
        t0 = time.time()
        steps = run_episode(env, generate_fn, system_prompt, task_id=task_id, seed=args.seed)
        elapsed = time.time() - t0

        final = steps[-1] if steps else {}
        episode_reward = final.get("episode_reward", 0)
        resolved = final.get("episode_resolved", final.get("resolved", False))
        reward_scores = eval_episode_rewards(steps)
        tools_used = [s.get("tool_name", "?") for s in steps if "tool_name" in s]

        result = {
            "task_id": task_id,
            "steps": len(steps),
            "episode_reward": episode_reward,
            "resolved": resolved,
            "tools": tools_used,
            "elapsed": round(elapsed, 1),
            **reward_scores,
        }
        results.append(result)

        status = "RESOLVED" if resolved else "not resolved"
        print(f"{task_id}:")
        print(f"  Steps: {len(steps)}, Reward: {episode_reward:.3f}, {status}")
        print(f"  Tools: {tools_used}")
        print(f"  Format: {reward_scores['format']:.2f}, Valid: {reward_scores['tool_valid']:.2f}, "
              f"Reasoning: {reward_scores['reasoning']:.2f}, No-leak: {reward_scores['no_leak']:.2f}")
        print(f"  Time: {elapsed:.1f}s")
        print()

    # Summary
    print(f"=== SUMMARY [{label}] ===")
    avg_reward = sum(r["episode_reward"] for r in results) / len(results)
    resolve_rate = sum(1 for r in results if r["resolved"]) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)
    avg_format = sum(r["format"] for r in results) / len(results)
    avg_valid = sum(r["tool_valid"] for r in results) / len(results)
    avg_reasoning = sum(r["reasoning"] for r in results) / len(results)
    avg_no_leak = sum(r["no_leak"] for r in results) / len(results)

    print(f"Avg episode reward: {avg_reward:.3f}")
    print(f"Resolution rate:    {resolve_rate:.0%}")
    print(f"Avg steps:          {avg_steps:.1f}")
    print(f"Avg format score:   {avg_format:.2f}")
    print(f"Avg tool validity:  {avg_valid:.2f}")
    print(f"Avg reasoning:      {avg_reasoning:.2f}")
    print(f"Avg no-leak:        {avg_no_leak:.2f}")

    # Save results
    if args.output_json:
        output = {
            "label": label,
            "model": args.model,
            "adapter_path": args.adapter_path,
            "seed": args.seed,
            "summary": {
                "avg_episode_reward": round(avg_reward, 4),
                "resolution_rate": round(resolve_rate, 4),
                "avg_steps": round(avg_steps, 2),
                "avg_format": round(avg_format, 4),
                "avg_tool_valid": round(avg_valid, 4),
                "avg_reasoning": round(avg_reasoning, 4),
                "avg_no_leak": round(avg_no_leak, 4),
            },
            "tasks": results,
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {args.output_json}")

    env.close()


if __name__ == "__main__":
    main()
