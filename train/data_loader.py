from datasets import Dataset
from collections import Counter
from typing import List, Dict
import pandas as pd
import argparse
from sympy import false
import torch
import os, json
from transformers import AutoTokenizer
from copy import deepcopy
import random


# def sft_formatting_function(data, tokenizer) -> List[str]:
#     output_texts = []
#     for i in range(len(data["prompt"])):
#         converted_sample = [
#             {"role": "system", "content": data["prompt"][i]},
#             {"role": "user", "content": data["input"][i]},
#             {"role": "assistant", "content": data["label"][i]},
#         ]
#         output_texts.append(
#             tokenizer.apply_chat_template(converted_sample, tokenize=False)
#         )
#     return output_texts


def sft_formatting_function(
    examples: List[Dict], tokenizer: AutoTokenizer
) -> Dict[str, List]:
    """
    Formats the dataset for SFT training by applying the chat template and
    setting the labels for the prompt and input parts to -100.

    Args:
        examples: List of dictionaries with 'prompt', 'input', 'label' keys
        tokenizer: The tokenizer to use

    Returns:
        Dictionary with 'input_ids', 'attention_mask', 'labels' keys
    """
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    max_input_length = 0

    for example in examples:
        # 获取字符串格式的数据
        prompt = example.get("prompt", "")
        input_text = example.get("input", "")
        label = example.get("label", "")

        # 1. 构建完整的对话列表
        full_conversation = [
            {"role": "system", "content": prompt},
        ]

        if input_text != "":
            full_conversation.append({"role": "user", "content": input_text})
        # 注意：这里我们不添加 assistant 角色，因为我们只想对 assistant 的回复进行训练

        # 2. 对话模板化并分词，得到 prompt 部分的 token ids
        prompt_and_input_tokens = tokenizer.apply_chat_template(
            full_conversation,
            tokenize=True,
            add_generation_prompt=True,  # 添加生成提示符，告知模型开始生成
            return_tensors="pt",
        )[0]

        # 3. 对 label 进行模板化和分词，得到 label 部分的 token ids
        # 这部分通常不需要额外的模板，只需要将文本分词即可
        label_tokens = tokenizer(label, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ][0]

        if tokenizer.eos_token_id is not None:
            # 在末尾添加 EOS token
            label_tokens = torch.cat(
                [label_tokens, torch.tensor([tokenizer.eos_token_id])], dim=0
            )

        # 4. 合并 token ids
        full_input_ids = torch.cat([prompt_and_input_tokens, label_tokens], dim=0)

        # 5. 创建 labels
        # 复制一份 full_input_ids 作为初始 labels
        full_labels = full_input_ids.clone()
        full_input_ids = full_input_ids.tolist()
        max_input_length = max(max_input_length, len(full_input_ids))

        # 6. 将 prompt 和 input 对应的 labels 设置为 -100
        # 获取 prompt 部分的长度
        prompt_len = len(prompt_and_input_tokens)
        full_labels[:prompt_len] = -100
        full_attention_mask = [1] * len(full_input_ids)

        # 将处理好的 ids 和 labels 添加到列表中
        input_ids_list.append(full_input_ids)
        attention_mask_list.append(full_attention_mask)
        labels_list.append(full_labels.tolist())

        # print(len(full_input_ids))
        # print(len(full_attention_mask))
        # print(len(full_labels))

    print(f"max_input_length: {max_input_length}")

    # print("labels_list", labels_list)

    # # 验证数据格式 - 确保没有嵌套列表
    # def validate_data_format(data):
    #     for key in data:
    #         for item in data[key]:
    #             if not isinstance(item, list):
    #                 raise ValueError(f"Item in {key} is not a list")
    #             for element in item:
    #                 if not isinstance(element, int):
    #                     raise ValueError(
    #                         f"Element {element} in {key} is not an integer"
    #                     )

    # data = {
    #     "input_ids": input_ids_list,
    #     "attention_mask": attention_mask_list,
    #     "labels": labels_list,
    # }
    # validate_data_format(data)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


def instruction_tuning_formatting_function(
    examples: List[Dict], tokenizer: AutoTokenizer
) -> Dict[str, List]:
    """
    Formats the dataset for SFT training by applying the chat template and
    setting the labels for the prompt and input parts to -100.

    Args:
        examples: List of dictionaries with 'prompt', 'input', 'label' keys
        tokenizer: The tokenizer to use

    Returns:
        Dictionary with 'input_ids', 'attention_mask', 'labels' keys
    """
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    max_input_length = 0

    for example in examples:
        # 获取字符串格式的数据
        prompt = example.get("instruction", "")
        label = example.get("output", "")

        # 1. 构建完整的对话列表
        full_conversation = [
            {"role": "user", "content": prompt},
            # 注意：这里我们不添加 assistant 角色，因为我们只想对 assistant 的回复进行训练
        ]

        # 2. 对话模板化并分词，得到 prompt 部分的 token ids
        prompt_and_input_tokens = tokenizer.apply_chat_template(
            full_conversation,
            tokenize=True,
            add_generation_prompt=True,  # 添加生成提示符，告知模型开始生成
            return_tensors="pt",
        )[0]

        # 3. 对 label 进行模板化和分词，得到 label 部分的 token ids
        # 这部分通常不需要额外的模板，只需要将文本分词即可
        label_tokens = tokenizer(label, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ][0]

        if tokenizer.eos_token_id is not None:
            # 在末尾添加 EOS token
            label_tokens = torch.cat(
                [label_tokens, torch.tensor([tokenizer.eos_token_id])], dim=0
            )

        # 4. 合并 token ids
        full_input_ids = torch.cat([prompt_and_input_tokens, label_tokens], dim=0)

        # 5. 创建 labels
        # 复制一份 full_input_ids 作为初始 labels
        full_labels = full_input_ids.clone()
        full_input_ids = full_input_ids.tolist()
        max_input_length = max(max_input_length, len(full_input_ids))

        # 6. 将 prompt 和 input 对应的 labels 设置为 -100
        # 获取 prompt 部分的长度
        prompt_len = len(prompt_and_input_tokens)
        full_labels[:prompt_len] = -100
        full_attention_mask = [1] * len(full_input_ids)

        # 将处理好的 ids 和 labels 添加到列表中
        input_ids_list.append(full_input_ids)
        attention_mask_list.append(full_attention_mask)
        labels_list.append(full_labels.tolist())

        # print(len(full_input_ids))
        # print(len(full_attention_mask))
        # print(len(full_labels))

    print(f"max_input_length: {max_input_length}")

    # print("labels_list", labels_list)

    # # 验证数据格式 - 确保没有嵌套列表
    # def validate_data_format(data):
    #     for key in data:
    #         for item in data[key]:
    #             if not isinstance(item, list):
    #                 raise ValueError(f"Item in {key} is not a list")
    #             for element in item:
    #                 if not isinstance(element, int):
    #                     raise ValueError(
    #                         f"Element {element} in {key} is not an integer"
    #                     )

    # data = {
    #     "input_ids": input_ids_list,
    #     "attention_mask": attention_mask_list,
    #     "labels": labels_list,
    # }
    # validate_data_format(data)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


def map_preference_data_kto(data) -> Dataset:
    converted_data = []
    seen_completions = []
    for entry in data:
        prompt = entry["prompt"]
        chosen_completion = entry["chosen"]
        rejected_completion = entry["rejected"]

        chosen_entry = {
            "prompt": prompt,
            "completion": chosen_completion,
            "label": True,
        }
        rejected_entry = {
            "prompt": prompt,
            "completion": rejected_completion,
            "label": False,
        }

        if chosen_completion not in seen_completions:
            converted_data.append(chosen_entry)
            seen_completions.append(chosen_completion)
        converted_data.append(rejected_entry)
    return Dataset.from_dict(pd.DataFrame(converted_data))


def preference_data_formatting(data, tokenizer: AutoTokenizer) -> dict:
    prompt = data["prompt"]
    input_text = data["input"]
    chosen = data["chosen"]
    rejected = data["rejected"]

    full_conversation = [
        {"role": "system", "content": prompt},
    ]

    if input_text != "":
        full_conversation.append({"role": "user", "content": input_text})

    prompt_and_input = tokenizer.apply_chat_template(
        full_conversation,
        tokenize=False,
        add_generation_prompt=True,  # 添加生成提示符，告知模型开始生成
        # return_tensors="pt",
    )

    return {
        "prompt": prompt_and_input,
        "chosen": chosen + tokenizer.eos_token,
        "rejected": rejected + tokenizer.eos_token,
    }


prompt = """You are a seasoned debater representing the {role} side in a social-media persuasive dialogue on the topic: "{topic}". Based on the topic and debate dialogue, craft a persuasive response aimed at prompting your opponent to reconsider their stance."""


# def process_sft_data(data):

#     processed_data = []

#     pro_win_count = 0
#     con_win_count = 0

#     for dialogue in data:
#         topic = dialogue["topic"]
#         dialogue_history = dialogue["dialogue_history"]

#         # 存储当前对话历史，用于构建输入
#         current_dialogue = []

#         # 遍历每一轮对话，i表示当前轮次的索引
#         for i in range(0, len(dialogue_history), 2):
#             # 获取正反两方的对话
#             pro_turn = dialogue_history[i]
#             con_turn = dialogue_history[i + 1]

#             pro_role, pro_response, pro_result = pro_turn
#             con_role, con_response, con_result = con_turn

#             # 检查获胜方并构建样本
#             if pro_result.lower() == "win":
#                 # 如果正方获胜，输入为历史对话，输出为正方回答
#                 sample = {
#                     "topic": topic,
#                     "prompt": prompt.format(role=pro_role, topic=topic),
#                     "input": "\n".join(
#                         [f"{role}: {response}" for role, response in current_dialogue]
#                     ),
#                     "label": f"{pro_role}: {pro_response}",
#                 }
#                 processed_data.append(sample)
#                 pro_win_count += 1

#             elif con_result.lower() == "win":
#                 # 如果反方获胜，输入为历史对话 + 当前轮次正方对话，输出为反方回答
#                 input_with_pro = current_dialogue + [(pro_role, pro_response)]
#                 sample = {
#                     "topic": topic,
#                     "prompt": prompt.format(role=con_role, topic=topic),
#                     "input": "\n".join(
#                         [f"{role}: {response}" for role, response in input_with_pro]
#                     ),
#                     "label": f"{con_role}: {con_response}",
#                 }
#                 processed_data.append(sample)
#                 con_win_count += 1

#             # 更新对话历史，将当前轮次的对话添加到历史中
#             current_dialogue.append((pro_role, pro_response))
#             current_dialogue.append((con_role, con_response))

#     # 构建统计信息字典
#     stats = {"pro_as_label_count": pro_win_count, "con_as_label_count": con_win_count}
#     print(stats)

#     return processed_data


def process_sft_data(data):

    processed_data = []

    pro_count = 0
    con_count = 0

    for dialogue in data:
        topic = dialogue["topic"]
        dialogue_history = dialogue["dialogue_history"]

        # 存储当前对话历史，用于构建输入
        current_dialogue = []

        # 遍历每一轮对话，i表示当前轮次的索引

        print("len(dialogue_history): ", len(dialogue_history))
        for i in range(0, len(dialogue_history), 2):
            # 获取正反两方的对话
            pro_turn = dialogue_history[i]
            con_turn = dialogue_history[i + 1]

            pro_role, pro_response = pro_turn
            con_role, con_response = con_turn

            if i == 0:
                sample = {
                    "topic": topic,
                    "prompt": prompt.format(role=pro_role, topic=topic),
                    "input": "",
                    "label": f"{pro_role}: {pro_response}",
                }
                processed_data.append(sample)
            else:
                sample = {
                    "topic": topic,
                    "prompt": prompt.format(role=pro_role, topic=topic),
                    "input": "\n".join(
                        [f"{role}: {response}" for role, response in current_dialogue]
                    ),
                    "label": f"{pro_role}: {pro_response}",
                }
                processed_data.append(sample)
            pro_count += 1

            input_with_pro = current_dialogue + [(pro_role, pro_response)]
            sample = {
                "topic": topic,
                "prompt": prompt.format(role=con_role, topic=topic),
                "input": "\n".join(
                    [f"{role}: {response}" for role, response in input_with_pro]
                ),
                "label": f"{con_role}: {con_response}",
            }
            processed_data.append(sample)
            con_count += 1

            # 更新对话历史，将当前轮次的对话添加到历史中
            current_dialogue.append((pro_role, pro_response))
            current_dialogue.append((con_role, con_response))

    # 构建统计信息字典
    stats = {"pro_as_label_count": pro_count, "con_as_label_count": con_count}
    print(stats)

    return processed_data


def find_worse_response(chosen_score, chosen_response, score_responses):
    worse_responses = []
    for score_response in score_responses:
        cur_score = sum(score_response["value"]) / len(score_response["value"])
        if chosen_score > cur_score:
            worse_responses.append(score_response["solution"])
    if worse_responses:
        return random.choice(worse_responses)
    else:
        return None


def process_dpo_data(data):

    processed_data = []

    pro_count = 0
    con_count = 0

    for dialogue in data:
        topic = dialogue["topic"]
        dialogue_history = dialogue["dialogue_history"]
        scores = dialogue["scores"]

        assert len(scores) == len(
            dialogue_history
        ), "scores 与 dialogue_history 长度需一致"

        # 存储当前对话历史，用于构建输入
        current_dialogue = []

        # 遍历每一轮对话，i表示当前轮次的索引
        for i in range(0, len(dialogue_history), 2):
            # 获取正反两方的对话
            pro_turn = dialogue_history[i]
            con_turn = dialogue_history[i + 1]

            pro_role, pro_response = pro_turn
            con_role, con_response = con_turn

            pro_score = scores[i]
            con_score = scores[i + 1]

            pro_score_role = pro_score[0]
            assert pro_score_role == pro_role + f"_{i // 2}"
            pro_score_responses = pro_score[1]
            if len(pro_score_responses) > 1:
                pro_rejected_response = pro_score_responses[0]["solution"]
                if pro_rejected_response == pro_response:
                    vals = pro_score_responses[0].get("value", [])
                    if not vals:
                        pro_rejected_response = None
                    else:
                        chosen_score = sum(vals) / len(vals)
                        pro_rejected_response = find_worse_response(
                            chosen_score, pro_response, pro_score_responses[1:]
                        )

            else:
                pro_rejected_response = None

            con_score_role = con_score[0]
            assert con_score_role == con_role + f"_{i // 2}"
            con_score_responses = con_score[1]
            if len(con_score_responses) > 1:
                con_rejected_response = con_score_responses[0]["solution"]
                if con_rejected_response == con_response:
                    vals = con_score_responses[0].get("value", [])
                    if not vals:
                        con_rejected_response = None
                    else:
                        chosen_score = sum(vals) / len(vals)
                        con_rejected_response = find_worse_response(
                            chosen_score, con_response, con_score_responses[1:]
                        )

            else:
                con_rejected_response = None

            if (
                pro_rejected_response is not None
                and pro_rejected_response != pro_response
            ):
                if i == 0:
                    sample = {
                        "topic": topic,
                        "prompt": prompt.format(role=pro_role, topic=topic),
                        "input": "",
                        "chosen": f"{pro_role}: {pro_response}",
                        "rejected": f"{pro_role}: {pro_rejected_response}",
                    }
                    processed_data.append(sample)
                else:
                    sample = {
                        "topic": topic,
                        "prompt": prompt.format(role=pro_role, topic=topic),
                        "input": "\n".join(
                            [
                                f"{role}: {response}"
                                for role, response in current_dialogue
                            ]
                        ),
                        "chosen": f"{pro_role}: {pro_response}",
                        "rejected": f"{pro_role}: {pro_rejected_response}",
                    }
                    processed_data.append(sample)
                pro_count += 1

            if (
                con_rejected_response is not None
                and con_rejected_response != con_response
            ):
                input_with_pro = current_dialogue + [(pro_role, pro_response)]
                sample = {
                    "topic": topic,
                    "prompt": prompt.format(role=con_role, topic=topic),
                    "input": "\n".join(
                        [f"{role}: {response}" for role, response in input_with_pro]
                    ),
                    "chosen": f"{con_role}: {con_response}",
                    "rejected": f"{con_role}: {con_rejected_response}",
                }
                processed_data.append(sample)
                con_count += 1

            # 更新对话历史，将当前轮次的对话添加到历史中
            current_dialogue.append((pro_role, pro_response))
            current_dialogue.append((con_role, con_response))

    # 构建统计信息字典
    stats = {"pro_as_label_count": pro_count, "con_as_label_count": con_count}
    print(stats)

    return processed_data


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


class CustomSFTDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # 手动处理batch，确保填充正确
        max_len = max(len(feature["input_ids"]) for feature in features)

        batch = {"input_ids": [], "attention_mask": [], "labels": []}

        for feature in features:
            # 填充input_ids
            input_ids = feature["input_ids"] + [0] * (
                max_len - len(feature["input_ids"])
            )
            batch["input_ids"].append(input_ids)

            # 填充attention_mask
            attention_mask = feature["attention_mask"] + [0] * (
                max_len - len(feature["attention_mask"])
            )
            batch["attention_mask"].append(attention_mask)

            # 填充labels，用-100填充
            labels = feature["labels"] + [-100] * (max_len - len(feature["labels"]))
            batch["labels"].append(labels)

        # 转换为tensor
        return {
            "input_ids": torch.tensor(batch["input_ids"]),
            "attention_mask": torch.tensor(batch["attention_mask"]),
            "labels": torch.tensor(batch["labels"]),
        }


class DataLoader:
    def __init__(self, args: argparse.Namespace, tokenizer: AutoTokenizer) -> None:
        self.args = args
        if "sft" in args.train_using:
            if "it" in args.train_using:
                self.data = load_dataset(data_path=args.train_data)
            else:
                data = load_dataset(data_path=args.train_data)
                self.data = process_sft_data(data)
        elif "dpo" in args.train_using:
            data = load_dataset(data_path=args.train_data)
            self.data = process_dpo_data(data)
        else:
            raise ValueError(
                f"train_using must be one of ['sft', 'dpo', 'cpo', 'kto', 'ppo'] - got {args.train_using}"
            )

        self.config_kwargs = {}
        self.trainer_kwargs = {}
        self.tokenizer = tokenizer

    def load_data(self, train_using: str) -> None:

        if "sft" in train_using:
            if "it" in train_using:  # instruction tuning
                print("loading instruction tuning data")
                features = instruction_tuning_formatting_function(
                    self.data, self.tokenizer
                )
            else:
                features = sft_formatting_function(self.data, self.tokenizer)
            self.data = Dataset.from_dict(features)
            print(self.data)

        elif train_using in ["cpo", "dpo"]:
            self.data = Dataset.from_list(self.data)
            self.data = self.data.map(
                preference_data_formatting,
                fn_kwargs={"tokenizer": self.tokenizer},
            )

            # self.data = self.data.map(
            #     lambda ex: preference_data_formatting(ex, tokenizer=self.tokenizer)
            # )

        elif train_using == "kto":
            self.data = map_preference_data_kto(self.data)
            self.config_kwargs["desirable_weight"] = self.args.desirable_weight
            self.config_kwargs["undesirable_weight"] = self.args.undesirable_weight
            assert self.args.weighting_scheme in [
                "frequency",
                "uniform",
            ], "weighting scheme must be either 'frequency' or 'uniform'"
            self.data = self.data.map(preference_data_formatting)
            if self.args.weighting_scheme == "frequency":
                device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
                fallacy_frequencies = {
                    k: round(v / self.__len__(), 3)
                    for k, v in sorted(
                        Counter(self.__getitem__("fallacy_type")).items()
                    )
                }
                class_weights = torch.tensor(
                    [min(fallacy_frequencies.values())]
                    + list(fallacy_frequencies.values()),
                    device=device,
                )
            else:
                class_weights = None

            self.trainer_kwargs["custom_eval_steps"] = 500
        else:
            raise ValueError(
                f"train_using must be one of ['sft', 'dpo', 'cpo', 'kto', 'ppo'] - got {train_using}"
            )

    def __getdata__(self) -> Dataset:

        return self.data

    # # 检查数据格式并清理
    # def clean_sample(example):
    #     # 确保所有字段都是简单的整数列表
    #     input_ids = [int(x) for x in example["input_ids"]]
    #     attention_mask = [int(x) for x in example["attention_mask"]]
    #     labels = [int(x) if x != -100 else -100 for x in example["labels"]]

    #     return {
    #         "input_ids": input_ids,
    #         "attention_mask": attention_mask,
    #         "labels": labels,
    #     }

    # # 清理数据格式
    # cleaned_data = self.data.map(clean_sample)

    # # 打印一个样本检查格式
    # sample = cleaned_data[0]
    # print("Sample format check:")
    # print(
    #     f"input_ids type: {type(sample['input_ids'])}, length: {len(sample['input_ids'])}"
    # )
    # print(
    #     f"attention_mask type: {type(sample['attention_mask'])}, length: {len(sample['attention_mask'])}"
    # )
    # print(f"labels type: {type(sample['labels'])}, length: {len(sample['labels'])}")
    # print(f"First few input_ids: {sample['input_ids'][:5]}")
    # print(f"First few labels: {sample['labels'][:5]}")

    # # 添加这个设置来启用动态填充
    # return cleaned_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        return self.data[idx]

    # def __configkwargs__(self) -> dict:
    #     return self.config_kwargs

    # def __trainerkwargs__(self) -> dict:
    #     return self.trainer_kwargs
