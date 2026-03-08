"""GRPO training for customer support agent.

Trains Qwen3.5-9B with LoRA using TRL's GRPOTrainer.
Uses pre-collected trajectory data as prompts, with online generation + reward functions.

Usage:
  python scripts/train_grpo.py --dataset-path ./data/quick_test/grpo_dataset
"""

import argparse
import json
import re
import sys
from pathlib import Path

from datasets import Dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

# --- Reward functions ---
# These score model-generated completions during training.
# Signature: func(completions, **kwargs) -> list[float]
# completions = list of [{"role": "assistant", "content": "..."}]

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", re.DOTALL
)
PARAM_RE = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)
VALID_TOOLS = {"lookup_customer", "check_order", "send_reply", "update_ticket"}


def _get_text(completion):
    if isinstance(completion, list):
        return completion[0]["content"] if completion else ""
    return str(completion)


def _parse_answer(ans):
    if isinstance(ans, str):
        try:
            return json.loads(ans)
        except json.JSONDecodeError:
            return {}
    return ans if isinstance(ans, dict) else {}


def format_reward(completions, **kwargs) -> list[float]:
    """Valid XML tool call format?

    +1.0  valid <tool_call> with function + params
    +0.25 partial structure
    -1.0  empty or no tool call
    """
    rewards = []
    for completion in completions:
        text = _get_text(completion)
        if not text.strip():
            rewards.append(-1.0)
            continue
        match = TOOL_CALL_RE.search(text)
        if match:
            params = PARAM_RE.findall(match.group(2))
            rewards.append(1.0 if params else 0.5)
        elif "<tool_call>" in text or "<function=" in text:
            rewards.append(0.25)
        else:
            rewards.append(-1.0)
    return rewards


def tool_validity_reward(completions, answer=None, **kwargs) -> list[float]:
    """Is the tool name valid?

    +1.0  valid tool name
    -0.5  parseable but wrong tool
     0.0  no tool call
    """
    rewards = []
    answers = answer or [None] * len(completions)
    for completion, ans in zip(completions, answers):
        text = _get_text(completion)
        ans_data = _parse_answer(ans) if ans else {}
        valid = set(ans_data.get("valid_tools", VALID_TOOLS))

        match = TOOL_CALL_RE.search(text)
        if match:
            tool_name = match.group(1).strip()
            rewards.append(1.0 if tool_name in valid else -0.5)
        else:
            rewards.append(0.0)
    return rewards


def reasoning_reward(completions, **kwargs) -> list[float]:
    """Does the completion have reasoning before the tool call?

    +1.0  reasoning text (>30 chars) before <tool_call>
    +0.3  has tool call but no reasoning
    -0.5  no tool call at all
    """
    rewards = []
    for completion in completions:
        text = _get_text(completion)
        tool_pos = text.find("<tool_call>")
        if tool_pos > 30:
            rewards.append(1.0)
        elif tool_pos >= 0:
            rewards.append(0.3)
        else:
            rewards.append(-0.5)
    return rewards


def no_reasoning_leak_reward(completions, **kwargs) -> list[float]:
    """Penalize leaking reasoning into send_reply message.

    The model should NOT send its internal reasoning as the customer message.
    +1.0  send_reply message doesn't start with reasoning patterns
    -1.0  send_reply message starts with "The customer", "I need to", etc.
     0.0  not a send_reply (neutral)
    """
    reasoning_starts = [
        "The customer", "I need to", "Let me", "From the", "Based on",
        "Looking at", "According to", "I should", "First,", "Now I",
        "The user", "I'll", "I will",
    ]
    rewards = []
    for completion in completions:
        text = _get_text(completion)
        match = TOOL_CALL_RE.search(text)
        if match and match.group(1).strip() == "send_reply":
            # Check if the message parameter starts with reasoning
            params = dict(PARAM_RE.findall(match.group(2)))
            msg = params.get("message", "").strip()
            if any(msg.startswith(p) for p in reasoning_starts):
                rewards.append(-1.0)
            else:
                rewards.append(1.0)
        else:
            rewards.append(0.0)  # Not a send_reply, neutral
    return rewards


def action_quality_reward(completions, answer=None, **kwargs) -> list[float]:
    """Heuristic quality: references ground truth values.

    +0.3 per ground truth referenced, capped at 1.0
    """
    rewards = []
    answers = answer or [None] * len(completions)
    for completion, ans in zip(completions, answers):
        text = _get_text(completion)
        ans_data = _parse_answer(ans) if ans else {}
        ground_truths = ans_data.get("ground_truth_values", [])
        score = 0.0
        for gt in ground_truths:
            if gt and str(gt).lower() in text.lower():
                score += 0.3
        rewards.append(min(1.0, score))
    return rewards


# --- Dataset preparation ---


def prepare_dataset(dataset_path: str) -> Dataset:
    """Load the collected trajectory dataset and format for GRPO.

    GRPO needs: prompt (list of message dicts), plus any extra columns
    for reward functions (like 'answer' with ground truth metadata).
    """
    ds = load_from_disk(dataset_path)
    print(f"Loaded dataset: {len(ds)} examples")
    print(f"Columns: {ds.column_names}")

    # The dataset has 'prompt' (list of messages) and 'answer' (JSON string)
    # GRPOTrainer expects 'prompt' to be the conversation so far
    # It will generate completions and evaluate with reward functions

    # Ensure prompt is in the right format
    # Each prompt should be a list of message dicts
    return ds


# --- Main ---


def main():
    parser = argparse.ArgumentParser(description="GRPO training for customer support agent")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Base model")
    parser.add_argument("--dataset-path", default="./data/quick_test/grpo_dataset", help="HF dataset path")
    parser.add_argument("--output-dir", default="./outputs/grpo-support-agent", help="Output directory")
    parser.add_argument("--num-epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num-generations", type=int, default=4, help="Completions per prompt")
    parser.add_argument("--max-completion-length", type=int, default=512, help="Max generation tokens")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    args = parser.parse_args()

    # Load dataset
    dataset = prepare_dataset(args.dataset_path)

    # Load tokenizer explicitly (Qwen3.5-9B is a VL model, so AutoProcessor
    # would try multimodal content blocks — we need text-only AutoTokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # GRPO config
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=0.7,
        beta=0.0,  # No KL penalty
        loss_type="grpo",
        scale_rewards="group",
        bf16=True,
        logging_steps=5,
        save_steps=100,
        log_completions=True,
        # Avoid OOM
        max_grad_norm=1.0,
        warmup_ratio=0.1,
    )

    # Reward functions (weighted)
    reward_funcs = [
        format_reward,          # Valid XML tool call format
        tool_validity_reward,   # Correct tool name
        reasoning_reward,       # Reasoning before action
        no_reasoning_leak_reward,  # Don't leak reasoning into replies
        action_quality_reward,  # References ground truth values
    ]
    reward_weights = [1.0, 0.8, 0.5, 0.7, 0.5]

    print(f"\n=== GRPO Training ===")
    print(f"Model: {args.model}")
    print(f"Dataset: {len(dataset)} examples")
    print(f"LoRA rank: {args.lora_r}")
    print(f"Batch size: {args.batch_size} x {args.grad_accum} grad accum")
    print(f"Generations per prompt: {args.num_generations}")
    print(f"Reward functions: {[f.__name__ for f in reward_funcs]}")
    print(f"Reward weights: {reward_weights}")
    print()

    trainer = GRPOTrainer(
        model=args.model,
        processing_class=tokenizer,  # Force text-only tokenizer (not VL processor)
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()

    # Save the LoRA adapter
    trainer.save_model(args.output_dir + "/final_adapter")
    print(f"\nTraining complete! Adapter saved to {args.output_dir}/final_adapter")


if __name__ == "__main__":
    main()
