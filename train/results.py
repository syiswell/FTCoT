#!/usr/bin/env python3
import os
import json
import argparse
from typing import List, Tuple, Optional


def find_trainer_state_jsons(root: str) -> List[Tuple[str, str]]:
    """
    Return list of (checkpoint_path, trainer_state_json_path) tuples under root.
    A checkpoint is any directory named like 'checkpoint*' that contains 'trainer_state.json'.
    """
    results = []
    if not os.path.isdir(root):
        return results

    for run_name in sorted(os.listdir(root)):
        run_dir = os.path.join(root, run_name)
        if not os.path.isdir(run_dir):
            continue

        for item in sorted(os.listdir(run_dir)):
            ckpt_dir = os.path.join(run_dir, item)
            if not os.path.isdir(ckpt_dir):
                continue
            if not item.startswith("checkpoint"):
                continue

            ts_path = os.path.join(ckpt_dir, "trainer_state.json")
            if os.path.isfile(ts_path):
                results.append((ckpt_dir, ts_path))
    return results


def loss_from_trainer_state(trainer_state_path: str) -> Optional[float]:
    """
    Load trainer_state.json and compute average of 'loss' values in 'log_history'.
    Returns None if no valid 'loss' entries found.
    """
    try:
        with open(trainer_state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    log_history = data.get("log_history", [])
    losses = []
    for entry in log_history:
        if not isinstance(entry, dict):
            continue
        val = entry.get("loss", None)
        if isinstance(val, (int, float)):
            losses.append(float(val))

    if not losses:
        return None
    return losses[-1]


def load_args_for_run(run_dir: str) -> dict:
    """
    Load args.json from a run directory and extract needed fields with fallbacks.
    Supports both flattened keys and nested 'peft_config'.
    """
    args_path = os.path.join(run_dir, "args.json")
    try:
        with open(args_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}

    def get_nested(d: dict, keys: List[str], default="N/A"):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    # Try flat keys first, then nested under peft_config
    epoch = data.get("epoch", "N/A")
    batch_size = data.get("batch_size", "N/A")
    grad_acc = data.get("gradient_accumulation_steps", "N/A")

    r = data.get("peft_config_r", None)
    alpha = data.get("peft_config_lora_alpha", None)
    dropout = data.get("peft_config_lora_dropout", None)

    if r is None:
        r = get_nested(data, ["peft_config", "r"])
    if alpha is None:
        alpha = get_nested(data, ["peft_config", "lora_alpha"])
    if dropout is None:
        dropout = get_nested(data, ["peft_config", "lora_dropout"])

    return {
        "epoch": epoch,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_acc,
        "peft_config_r": r if r is not None else "N/A",
        "peft_config_lora_alpha": alpha if alpha is not None else "N/A",
        "peft_config_lora_dropout": dropout if dropout is not None else "N/A",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Average loss from trainer_state.json across checkpoints and print args.json fields."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/home/sunyang/hlt/DebateTree/train/experiments/our_synthetic100",
        help="Root experiments directory containing run subdirectories.",
    )
    args = parser.parse_args()

    pairs = find_trainer_state_jsons(args.root)
    if not pairs:
        print(f"No checkpoints with trainer_state.json found under: {args.root}")
        return

    # Print header
    print(
        "run\tcheckpoint\tavg_loss\tepoch\tpeft_config_r\tpeft_config_lora_alpha\tpeft_config_lora_dropout\tbatch_size\tgradient_accumulation_steps"
    )

    # Cache args per run to avoid re-reading
    run_args_cache = {}

    for ckpt_dir, ts_path in pairs:
        run_dir = os.path.dirname(ckpt_dir)
        run_name = os.path.basename(run_dir)
        ckpt_name = os.path.basename(ckpt_dir)

        if run_name not in run_args_cache:
            run_args_cache[run_name] = load_args_for_run(run_dir)

        avg_loss = loss_from_trainer_state(ts_path)
        avg_str = "N/A" if avg_loss is None else f"{avg_loss:.6f}"

        run_args = run_args_cache[run_name]
        print(
            f"{run_name}\t{ckpt_name}\t{avg_str}\t"
            f"{run_args['epoch']}\t{run_args['peft_config_r']}\t{run_args['peft_config_lora_alpha']}\t"
            f"{run_args['peft_config_lora_dropout']}\t{run_args['batch_size']}\t{run_args['gradient_accumulation_steps']}"
        )


if __name__ == "__main__":
    main()
