import os
import sys

sys.path.append("..")
import json
import os
from typing import List, Dict, Any, Tuple
import random
import pandas as pd
import threading


def read_jsonl(filename: str) -> List[Dict]:
    """读取JSONL文件"""
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def split_dataset(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """将数据集按比例划分为训练集、验证集和测试集"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须等于1"

    # 设置随机种子以确保可重复性
    random.seed(random_seed)

    # 随机打乱数据
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    total_size = len(shuffled_data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    # 划分数据集
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size : train_size + val_size]
    test_data = shuffled_data[train_size + val_size :]

    print(f"数据集划分完成:")
    print(f"  训练集: {len(train_data)} 个对话 ({len(train_data)/total_size*100:.1f}%)")
    print(f"  验证集: {len(val_data)} 个对话 ({len(val_data)/total_size*100:.1f}%)")
    print(f"  测试集: {len(test_data)} 个对话 ({len(test_data)/total_size*100:.1f}%)")

    return train_data, val_data, test_data


def convert_to_instruction_following_format(
    item: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """将单个对话项转换为instruction-following格式"""
    topic = item.get("title", "")
    path = item.get("path", [])
    authors = item.get("authors", [])

    # 构建对话历史
    dialogue_history = []
    for i, (response, author) in enumerate(zip(path, authors)):
        role = "user" if i % 2 == 0 else "assistant"
        dialogue_history.append({"role": role, "content": response})

    # 为每个assistant回复创建instruction-following样本
    instruction_samples = []
    list_samples = []
    metadatas = []

    for i in range(1, len(dialogue_history), 2):  # 只处理assistant的回复
        if i >= len(dialogue_history):
            break

        # 构建对话上下文（包括当前assistant回复之前的所有对话）
        conversation_context = dialogue_history[: i + 1]

        # 构建instruction
        instruction = (
            f"Below is a debate dialogue history on the topic '{topic}':\n"
            + format_conversation_context(conversation_context[:-1])
        )

        # 构建期望的回复
        expected_response = dialogue_history[i]["content"]

        # 创建list样本
        list_sample = {
            "topic": topic,
            "dialogue_history": conversation_context[:-1],
            "output": expected_response,
        }

        # 创建instruction-following样本
        sample = {
            "instruction": instruction,
            "input": "",  # 在这个格式中，input通常为空
            "output": expected_response,
        }
        metadata = {
            "topic": topic,
            "turn": i // 2 + 1,  # assistant的轮次
            "total_turns": len(
                [d for d in dialogue_history if d["role"] == "assistant"]
            ),
            "author": authors[i] if i < len(authors) else "unknown",
        }

        instruction_samples.append(sample)
        metadatas.append(metadata)

        list_samples.append(list_sample)
    return instruction_samples, metadatas, list_samples


def format_conversation_context(conversation: List[Dict[str, str]]) -> str:
    """格式化对话上下文"""
    formatted = []
    for turn in conversation:
        formatted.append(f"{turn['role']}: {turn['content']}")
    return "\n".join(formatted)


def process_dialogue_data(input_file: str, output_dir: str):
    """处理对话数据并转换为instruction-following格式"""
    print(f"正在读取数据文件: {input_file}")
    data = read_jsonl(input_file)
    print(f"加载了 {len(data)} 个对话项")

    # 划分数据集
    train_data, val_data, test_data = split_dataset(data)

    # 处理每个数据集
    datasets = {"train": train_data, "val": val_data, "test": test_data}

    for split_name, split_data in datasets.items():
        print(f"\n处理 {split_name} 集...")
        all_instruction_samples = []
        all_metadatas = []
        all_list_samples = []
        for i, item in enumerate(split_data):
            if i % 100 == 0:  # 每100个打印一次进度
                print(f"  处理第 {i+1}/{len(split_data)} 个对话项")
            try:
                samples, metadatas, list_samples = (
                    convert_to_instruction_following_format(item)
                )
                all_instruction_samples.extend(samples)
                all_metadatas.extend(metadatas)
                all_list_samples.extend(list_samples)
            except Exception as e:
                print(f"  处理第 {i+1} 个对话项时出错: {e}")
                continue

        print(
            f"  {split_name} 集生成了 {len(all_instruction_samples)} 个instruction-following样本"
        )

        # 保存结果
        os.makedirs(output_dir, exist_ok=True)
        # output_file = os.path.join(output_dir, f"{split_name}.jsonl")
        # with open(output_file, "w", encoding="utf-8") as f:
        #     for sample in all_instruction_samples:
        #         f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        output_file = os.path.join(output_dir, f"{split_name}_list.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in all_list_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print(f"  {split_name} 集结果已保存到: {output_file}")

        # 打印统计信息
        print(f"\n{split_name} 集统计信息:")
        print(f"  总样本数: {len(all_instruction_samples)}")

        # 按轮次统计
        turn_counts = {}
        for metadata in all_metadatas:
            turn = metadata["turn"]
            turn_counts[turn] = turn_counts.get(turn, 0) + 1

        print("  按轮次分布:")
        for turn in sorted(turn_counts.keys()):
            print(f"    第{turn}轮: {turn_counts[turn]} 个样本")


def construct_sft_data(data):

    debate_dialogue_instruction = """You are an assistant in a debate dialogue. The user presents a strong opinion on a specific topic. Your goal is to respond persuasively and logically — challenging their stance with counterarguments, context, and ethical or emotional reasoning."""

    # 分析步骤的模板
    gen_inst_template = """Given the following task instruction and the corresponding argumentative input text, your need to complete the task.


## Task Instruction:

{task_instruction}


## Input Text:

{argument_text}


## Output Text:
"""

    new_data = []

    for item in data:
        # 从原始指令中提取任务指令和论证文本

        argument_text = item["instruction"]
        task_instruction = debate_dialogue_instruction

        gen_instruction = gen_inst_template.format(
            task_instruction=task_instruction,
            argument_text=argument_text,
        )
        gen_output = item.get("output", "")

        generation_item = {
            "instruction": gen_instruction,
            "input": "",
            "output": gen_output,
        }
        new_data.append(generation_item)

    return new_data


# 线程锁，用于保护共享资源
lock = threading.Lock()


def save_results(data, output_path):
    """保存结果到文件"""
    with lock:
        try:
            pd.DataFrame(data).to_json(
                output_path,
                orient="records",
                lines=True,
            )
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error saving results: {e}")


if __name__ == "__main__":
    # 输入和输出文件路径
    # input_file = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/all_data_evaluation_dialogue.jsonl"
    # output_file = (
    #     "/home/sunyang/hlt/new_cmv_dataset/final_dataset/instruction_following_format"
    # )

    # process_dialogue_data(input_file, output_file)

    input_file = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/all_data_evaluation_dialogue.jsonl"
    output_file = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/list_format"
    process_dialogue_data(input_file, output_file)

    # for data_split in ["train", "test"]:
    #     input_file = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/instruction_following_format/{data_split}.jsonl"
    #     data = read_jsonl(input_file)
    #     new_data = construct_sft_data(data)

    #     output_file = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/sft_data/{data_split}.jsonl"
    #     os.makedirs(os.path.dirname(output_file), exist_ok=True)
    #     save_results(new_data, output_file)
