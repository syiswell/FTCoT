import argparse
import os
import json
import time
from typing import List, Tuple, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm


# 顶部 imports 附近，新增一个轻量工具函数（不强依赖 peft，仅在用到时导入）
def _is_peft_adapter_dir(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def _load_tokenizer_for(path: str, fallback_base: Optional[str]):
    try:
        return AutoTokenizer.from_pretrained(path, use_fast=True)
    except Exception:
        if fallback_base is None:
            raise
        return AutoTokenizer.from_pretrained(fallback_base, use_fast=True)


def _read_adapter_base(path: str) -> Optional[str]:
    cfg_path = os.path.join(path, "adapter_config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k in ["base_model_name_or_path", "base_model_name", "model_name_or_path"]:
            v = cfg.get(k)
            if isinstance(v, str) and len(v.strip()) > 0:
                return v
    except Exception:
        pass
    return None


def sanitize_filename(name: str) -> str:
    return "".join([c if c.isalnum() else "_" for c in name])


def load_dataset(data_path):
    """
    Loads a JSON dataset from a file or a directory of files.

    Args:
        data_path (str): The path to a JSON file or a directory containing JSON files.

    Returns:
        list: A list of dictionaries, where each dictionary represents a JSON object from the dataset.
    """
    dataset = []

    if os.path.isdir(data_path):
        # If it's a directory, iterate through all files and load them
        for filename in os.listdir(data_path):
            if filename.endswith(".json"):
                file_path = os.path.join(data_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    dataset.append(json.load(f))
    elif os.path.isfile(data_path) and data_path.endswith(".json"):
        # If it's a single JSON file, load it
        with open(data_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    elif os.path.isfile(data_path) and data_path.endswith(".jsonl"):
        # If it's a single JSONL file, load it
        print(f"Loading JSONL file: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            dataset = [json.loads(line.strip()) for line in f]
        print(f"Loaded {len(dataset)} samples from JSONL file")
    else:
        raise ValueError(
            "Invalid data_path. It must be a JSON file or a directory containing JSON files."
        )

    return dataset


def save_topic_item(item: Dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    save_item = dict(item)
    save_item["last_updated"] = time.time()
    # 使用instruction作为文件名，如果没有则使用索引
    if "instruction" in save_item:
        fname = sanitize_filename(save_item["instruction"][:50]) + ".json"
    else:
        fname = f"item_{save_item.get('idx', time.time())}.json"
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(save_item, f, ensure_ascii=False, indent=2)


def load_model(
    model_path: str,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
    model_base: Optional[str] = None,
    peft_merge_and_unload: bool = False,
    model_devices: Optional[str] = None,  # NEW: "0,1"
) -> Tuple[
    AutoModelForCausalLM,
    AutoTokenizer,
    List[int],
]:
    torch_dtype = None
    if dtype:
        map_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = None if dtype is None else map_dtype.get(dtype)
        if dtype is not None and torch_dtype is None:
            raise ValueError("dtype must be one of: float16, bfloat16, float32")

    def _parse_devices(devs: Optional[str]) -> List[int]:
        if not devs:
            return (
                list(range(torch.cuda.device_count()))
                if torch.cuda.is_available()
                else []
            )
        return [int(x) for x in devs.split(",") if x.strip() != ""]

    def _mem_gib_str(i: int) -> str:
        props = torch.cuda.get_device_properties(i)
        gib = max(int(props.total_memory // (1024**3) - 1), 1)
        return f"{gib}GiB"

    devs = _parse_devices(model_devices)

    def _build_max_memory(dev_ids: List[int]) -> Dict[int, str]:
        if len(dev_ids) == 0:
            raise ValueError("No devices found")
        return {i: _mem_gib_str(i) for i in dev_ids}

    def _load_one(model_path: str, base_path: Optional[str], dev_ids: List[int]):
        is_adapter = _is_peft_adapter_dir(model_path)
        tok = _load_tokenizer_for(model_path, base_path)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        # Set left padding for decoder-only models (required for correct generation)
        tok.padding_side = "left"

        max_memory = _build_max_memory(dev_ids)
        device_map = "auto"

        if not is_adapter:
            mdl = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype if torch_dtype is not None else None,
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=True,
            )
            return mdl, tok

        if base_path is None:
            base_path = _read_adapter_base(model_path)
        if base_path is None:
            raise ValueError(
                f"Detected PEFT adapter at '{model_path}', please set corresponding base model path."
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch_dtype if torch_dtype is not None else None,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True,
        )

        mdl = PeftModel.from_pretrained(base_model, model_path)
        if peft_merge_and_unload:
            mdl = mdl.merge_and_unload()
        return mdl, tok

    model, tok = _load_one(model_path, model_base, devs)

    model.eval()

    # 检查模型参数的数据类型
    model_dtype = next(model.parameters()).dtype
    is_bf16 = model_dtype == torch.bfloat16
    is_fp16 = model_dtype == torch.float16
    is_fp32 = model_dtype == torch.float32

    print(f"\n{'='*60}")
    print(f"模型数据类型检查:")
    print(f"  数据类型: {model_dtype}")
    print(f"  是否使用 bfloat16: {is_bf16}")
    print(f"  是否使用 float16: {is_fp16}")
    print(f"  是否使用 float32: {is_fp32}")
    print(f"  Tokenizer: {type(tok).__name__} (tokenizer不涉及数据类型)")
    print(f"{'='*60}\n")

    return model, tok, devs


@torch.inference_mode()
def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    instructions: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.5,
    top_p: float = 0.9,
    do_sample: bool = True,
    stop_seqs: Optional[List[str]] = None,
    batch_size: int = 8,
) -> List[str]:
    """
    Generate predictions for a batch of instructions.

    Args:
        model: The language model
        tokenizer: The tokenizer
        instructions: List of instruction strings
        max_new_tokens: Maximum generation length
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        do_sample: Whether to use sampling
        stop_seqs: Optional stop sequences
        batch_size: Batch size for processing

    Returns:
        List of generated text strings
    """
    all_results = []

    # Calculate total number of batches
    total_batches = (len(instructions) + batch_size - 1) // batch_size

    # Create progress bar with batch info
    pbar = tqdm(
        total=len(instructions),
        desc=f"Generating (batches: {total_batches})",
        unit="item",
        ncols=100,
    )

    for i in range(0, len(instructions), batch_size):
        batch_instructions = instructions[i : i + batch_size]
        batch_num = i // batch_size + 1

        # Prepare messages for batch
        messages_list = [
            [{"role": "user", "content": instr}] for instr in batch_instructions
        ]

        # Apply chat template
        input_ids_list = []
        for messages in messages_list:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            input_ids_list.append(input_ids)

        # Pad to same length (LEFT PADDING for decoder-only models)
        max_len = max(ids.shape[1] for ids in input_ids_list)
        padded_ids = []
        attention_masks = []

        for input_ids in input_ids_list:
            pad_len = max_len - input_ids.shape[1]
            if pad_len > 0:
                # Left padding: padding tokens followed by input_ids
                padding = torch.full(
                    (1, pad_len),
                    tokenizer.pad_token_id,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                padded = torch.cat([padding, input_ids], dim=1)
            else:
                padded = input_ids
            padded_ids.append(padded)
            attention_masks.append((padded != tokenizer.pad_token_id).long())

        input_ids = torch.cat(padded_ids, dim=0)
        attention_mask = torch.cat(attention_masks, dim=0)

        # Determine target device
        target_device = None
        if hasattr(model, "hf_device_map"):
            dev_ids = set()
            for d in model.hf_device_map.values():
                if isinstance(d, int):
                    dev_ids.add(d)
                else:
                    s = str(d)
                    if s.startswith("cuda:"):
                        dev_ids.add(int(s.split(":")[1]))
            devs = sorted(dev_ids)
            if len(devs) > 0:
                target_device = torch.device(f"cuda:{devs[0]}")

        if target_device is None:
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_ids = input_ids.to(target_device)
        attention_mask = attention_mask.to(target_device)

        # Get original lengths for each item
        input_lens = [ids.shape[1] for ids in input_ids_list]

        # Generate
        gen_out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode each result (LEFT PADDING: [pad, pad, ..., original, generated])
        batch_results = []
        for j, output in enumerate(gen_out):
            # For left padding: output structure is [pad, pad, ..., original, generated]
            # We need to find where the actual input ends and generation begins
            original_len = input_lens[j]  # Original input length
            pad_len = max_len - original_len  # Padding length

            # In left-padded output, actual input+generation starts after padding
            # So generation starts at: pad_len + original_len
            actual_start = pad_len + original_len
            new_tokens = output[actual_start:]

            out_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if stop_seqs:
                out_text = truncate_at_stop_sequences(out_text, stop_seqs)
            batch_results.append(out_text.strip())

        all_results.extend(batch_results)

        # Update progress bar
        pbar.set_postfix({"batch": f"{batch_num}/{total_batches}"})
        pbar.update(len(batch_instructions))

    # Close progress bar
    pbar.close()

    return all_results


def truncate_at_stop_sequences(text: str, stop_seqs: List[str]) -> str:
    cut = len(text)
    for s in stop_seqs:
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].strip()


def save_results_to_jsonl(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save all results to a single JSONL file.

    Args:
        results: List of result dictionaries
        output_path: Path to the output JSONL file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Single model evaluator using instruction tuning format"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local path to model (can be LoRA adapter directory or full model)",
    )
    parser.add_argument(
        "--peft_merge_and_unload",
        action="store_true",
        help="Merge PEFT adapter into base weights for faster inference",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/sunyang/hlt/DebateTree/topic/debate_test_v1.json",
        help="Path to dataset (.jsonl or .json) with 'instruction' and optionally 'output' keys",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate (generation length)",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="One of: float16, bfloat16, float32 (or leave unset)",
    )
    parser.add_argument(
        "--model_devices",
        type=str,
        default=None,
        help="Comma-separated GPU IDs (e.g., '0,1')",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(args.dataset_path)
    print(f"Loaded {len(dataset)} items from dataset")

    # Load model
    print(f"Loading model from: {args.model_path}")
    model, tokenizer, devs = load_model(
        model_path=args.model_path,
        dtype=args.dtype,
        peft_merge_and_unload=args.peft_merge_and_unload,
        model_devices=args.model_devices,
    )
    print(f"Model loaded successfully. Using devices: {devs}")

    # Create output file
    output_dir = args.model_path
    output_filename = f"predictions_{args.max_new_tokens}.jsonl"
    output_path = os.path.join(output_dir, output_filename)
    print(f"Results will be saved to: {output_path}")

    # Prepare data for batch processing
    valid_items = []
    item_indices = []
    for idx, item in enumerate(dataset):
        # Ensure item has required format
        if "instruction" not in item:
            if isinstance(item, str):
                item = {"instruction": item}
            else:
                if args.verbose:
                    print(f"Warning: Item {idx} missing 'instruction' key, skipping")
                continue
        valid_items.append(item)
        item_indices.append(idx)

    if args.verbose:
        print(
            f"Processing {len(valid_items)} valid items in batches of {args.batch_size}"
        )

    # Extract all instructions
    instructions = [item["instruction"] for item in valid_items]

    # Batch generation
    print(f"Generating predictions in batches of {args.batch_size}...")
    predicted_outputs = generate_batch(
        model=model,
        tokenizer=tokenizer,
        instructions=instructions,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        stop_seqs=None,
        batch_size=args.batch_size,
    )

    # Combine results
    results = []
    for item, predicted_output in zip(valid_items, predicted_outputs):
        result = dict(item)
        result["predicted_output"] = predicted_output
        results.append(result)

    # Save all results to JSONL file
    save_results_to_jsonl(results, output_path)

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Processed {len(results)}/{len(dataset)} items")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
