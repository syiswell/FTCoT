import os
from call_llm import model_factory, Message
import json
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm


model = model_factory("gpt-4o-mini")


coherence_instruction = """You are a debate dialogue analysis assistant. Your task is to determine whether the current speaker’s statement responds to or refutes the opponent’s previous argument.

Analyze the logical coherence between the two statements and assign a coherence_score based on the following criteria:

- "directly_addresses": The current statement clearly and explicitly responds to or refutes the opponent’s argument.
- "partially_addresses": The current statement touches on or indirectly relates to the opponent’s argument but does not fully address it.
- "does_not_address": The current statement ignores, shifts away from, or fails to respond to the opponent’s argument.

Your output must strictly follow the JSON schema `coherence_evaluation` and include:
1. reason — a detailed reasoning or analysis for your judgment.
2. coherence_score — one of the three enumeration values above.

Example output:
{
  "reason": "The current statement directly refutes the opponent’s claim about economic costs, making it a direct response.",
  "coherence_score": "directly_addresses"
}"""

coherence_json_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "coherence_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Detailed reasoning or analysis for the coherence evaluation",
                },
                "coherence_score": {
                    "type": "string",
                    "enum": [
                        "directly_addresses",
                        "partially_addresses",
                        "does_not_address",
                    ],
                    "description": "Whether the current response directly addresses or responds to the previous opponent's arguments",
                },
            },
            "required": ["reason", "coherence_score"],
            "additionalProperties": False,
        },
    },
}


progressiveness_instruction = """You are given two consecutive utterances from the same speaker. 
Your task is to determine whether the current utterance introduces any new arguments, perspectives, or deeper reasoning compared to the previous one. 
If it does, it is considered "new". 
If it mainly repeats or rephrases previous ideas without meaningful advancement, it is "existing". 
If it partially adds new reasoning but also repeats earlier content, it is "mixed".  

Provide your reasoning and classification following the JSON schema below.

Output format:
{
  "reason": "Detailed explanation of why the utterance is classified this way.",
  "progressiveness": "new | existing | mixed"
}"""

progressiveness_json_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "progressiveness_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Detailed reasoning or analysis for the progressiveness evaluation",
                },
                "progressiveness": {
                    "type": "string",
                    "enum": ["new", "existing", "mixed"],
                    "description": "Whether the current response presents new arguments, new perspectives, or deeper reasoning compared to the previous response",
                },
            },
            "required": ["reason", "progressiveness"],
            "additionalProperties": False,
        },
    },
}

consistency_instruction = """You are a Debate Dialogue Stance Consistency Evaluator. Your task is to analyze the assistant’s stance across an entire multi-turn dialogue and assess its consistency.

Evaluation dimensions:
1) Polarity consistency: Determine whether the assistant’s stance polarity (support vs. oppose) toward the topic remains consistent throughout the dialogue.
2) Opposition to the user: Determine whether the assistant consistently opposes the user’s stance on the topic.

Instructions:
- Read the full dialogue and focus on the assistant’s stance with respect to the topic under discussion in each turn.
- Identify:
  - The topic being debated (if implicit, infer from context).
  - The user’s stance on that topic (supporting, opposing, or unclear/neutral).
  - The assistant’s stance in each relevant turn (supporting, opposing, neutral/mixed).
- Evaluate whether the assistant’s stance polarity remains consistent across turns. If the assistant’s stance clearly changes (e.g., from support to oppose), mark it as “inconsistent”. If it is mostly consistent but with minor hedging or ambiguity, you may mark it as “mixed”. If it remains the same throughout, mark it as “consistent”.
- Evaluate the assistant’s relationship to the user’s stance:
  - “strongly_opposes” if the assistant persistently and explicitly argues against the user’s stance.
  - “opposes” if the assistant generally argues against the user’s stance but with moderate or occasional hedging.
  - “neutral” if the assistant does not take a clear side relative to the user or remains balanced.
  - “supports” if the assistant aligns with or reinforces the user’s stance.
- If information is insufficient or ambiguous, make your best inference and clearly explain uncertainties in the reason.
- Output must be a single JSON object strictly following the schema below. Do not include any fields not defined by the schema. Do not add extra text outside the JSON.

Output format (strict):
{
  "reason": string,  // Detailed reasoning for your assessment across the dialogue, including references to specific turns and any uncertainties.
  "polarity_consistency": "consistent" | "inconsistent" | "mixed",
  "opposition_to_user": "strongly_opposes" | "opposes" | "neutral" | "supports"
}

Constraints:
- Use only the fields defined above. Do not add a score or any additional properties.
- Do not include explanations outside the JSON."""

stance_consistency_json_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "stance_consistency_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Detailed reasoning or analysis for the assistant's stance throughout the dialogue",
                },
                "polarity_consistency": {
                    "type": "string",
                    "enum": ["consistent", "inconsistent", "mixed"],
                    "description": "Whether the assistant's stance polarity is consistent throughout the dialogue",
                },
                "opposition_to_user": {
                    "type": "string",
                    "enum": ["strongly_opposes", "opposes", "neutral", "supports"],
                    "description": "Whether the assistant opposes the user's stance on the topic",
                },
            },
            "required": ["reason", "polarity_consistency", "opposition_to_user"],
            "additionalProperties": False,
        },
    },
}


def read_jsonl(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# 添加 Coherence 评估函数
@retry(
    wait=wait_random_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry_error_callback=lambda retry_state: None,
)
def evaluate_coherence(
    previous_claim, current_response, previous_arguments=None, current_arguments=None
):
    """评估当前发言是否回应了对手上一轮的论点"""

    messages = [
        Message(role="system", content=coherence_instruction),
        Message(
            role="user",
            content=f"[Previous Opponent Response]\n{previous_claim}\n\n[Previous Opponent Arguments]\n{previous_arguments}",
        ),
        Message(
            role="user",
            content=f"[Current Speaker Response]\n{current_response}\n\n[Current Speaker Arguments]\n{current_arguments}",
        ),
    ]

    # print("coherence messages: ", messages)

    response = model.generate_chat(
        messages=messages, temperature=0.0, response_format=coherence_json_format
    )

    # print("coherence response: ", response)

    return response


# 添加 Progressiveness 评估函数
@retry(
    wait=wait_random_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry_error_callback=lambda retry_state: None,
)
def evaluate_progressiveness(
    previous_self_utterance,
    current_self_utterance,
    previous_self_arguments=None,
    current_self_arguments=None,
    opponent_utterance=None,
    opponent_arguments=None,
):
    """评估当前发言相比上一次发言是否提出了新的论据或更深入的推理"""

    messages = [
        Message(role="system", content=progressiveness_instruction),
        Message(
            role="user",
            content=f"[Previous Self Utterance]\n{previous_self_utterance}\n\n[Arguments from Previous Self Utterance]\n{previous_self_arguments}",
        ),
    ]

    if opponent_utterance:
        messages.append(
            Message(
                role="user",
                content=f"[Opponent Utterance (Optional)]\n{opponent_utterance}\n\n[Opponent Arguments (Optional)]\n{opponent_arguments}",
            )
        )

    messages.append(
        Message(
            role="user",
            content=f"[Current Self Utterance]\n{current_self_utterance}\n\n[Arguments from Current Self Utterance]\n{current_self_arguments}",
        )
    )

    # print("progressiveness messages: ", messages)

    response = model.generate_chat(
        messages=messages,
        temperature=0.0,
        response_format=progressiveness_json_format,
    )

    # print("progressiveness response: ", response)

    return response


# 添加 Consistency 评估函数
@retry(
    wait=wait_random_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry_error_callback=lambda retry_state: None,
)
def evaluate_consistency(topic, dialogue):
    """评估assistant的立场是否在对话中保持一致，以及是否反对user的立场"""

    messages = [
        Message(role="system", content=consistency_instruction),
        Message(
            role="user",
            content=f"[Topic]\n{topic}\n\n[Dialogue History]\n{process_chat_one(dialogue)}",
        ),
    ]

    # print("consistency messages: ", messages)

    response = model.generate_chat(
        messages=messages,
        temperature=0.0,
        response_format=stance_consistency_json_format,
    )
    # print("consistency response: ", response)

    return response


def process_chat_one(dialogue):
    if len(dialogue) == 0:
        return "No Dialogue History"

    prompt_messages = ""

    for i, (role, utt, _) in enumerate(dialogue):
        content = f"{role}: {utt}".strip()
        prompt_messages += content + "\n"
    return prompt_messages.strip()


# 线程锁，用于保护共享资源
lock = threading.Lock()


# 添加一个辅助函数来格式化 arguments
def format_arguments(arguments):
    """将 arguments 格式化为可读的字符串"""
    if not arguments or not isinstance(arguments, dict):
        return None

    args_list = arguments.get("arguments", [])
    if not args_list:
        return None

    formatted_args = []
    for arg in args_list:
        claim = arg.get("Claim", "")
        evidence = arg.get("Evidence", [])

        if claim:
            formatted_arg = f"Claim: {claim}"
            if evidence and evidence != ["No evidence"]:
                evidence_str = "; ".join(evidence)
                formatted_arg += f"\nEvidence: {evidence_str}"
            formatted_args.append(formatted_arg)

    return "\n".join(formatted_args) if formatted_args else None


def process_item(item, index):
    """处理单个数据项的函数"""
    topic = item.get("title")
    print(f"Processing item {index}: {topic}")

    # 尝试从多种常见字段中抽取对话 [(role, utt), ...]
    path = item["path"]
    dialogue = []

    response_evaluations = item["response_evaluations"]

    role_map = {0: "user", 1: "assistant"}
    for idx, response in enumerate(path):
        dialogue.append(
            (
                role_map[idx % 2],
                response,
                json.loads(response_evaluations[idx]["evaluation"]),
            )
        )

    evaluations = []
    for turn_id, (role, utt, arguments) in enumerate(dialogue):
        history = dialogue[: turn_id + 1]  # 当前响应之前的对话历史

        # 新增：Coherence 评估
        coherence_eval = None
        if turn_id > 0:  # 不是第一轮
            # 获取上一轮对手的发言
            prev_role, prev_utt, prev_arguments = dialogue[turn_id - 1]
            if prev_role != role:  # 确保是不同角色的发言
                try:
                    # 格式化 arguments 为字符串
                    prev_args_str = (
                        format_arguments(prev_arguments) if prev_arguments else None
                    )
                    current_args_str = (
                        format_arguments(arguments) if arguments else None
                    )

                    coherence_eval = evaluate_coherence(
                        prev_utt, utt, prev_args_str, current_args_str
                    )
                except Exception as e:
                    coherence_eval = None

        # 新增：Progressiveness 评估
        progressiveness_eval = None
        if turn_id >= 2:  # 至少需要两轮该角色的发言
            # 找到该角色上一次发言
            prev_self_turn = None
            for i in range(turn_id - 1, -1, -1):
                if dialogue[i][0] == role:
                    prev_self_turn = i
                    break

            if prev_self_turn is not None:
                prev_self_utt, prev_self_arguments = (
                    dialogue[prev_self_turn][1],
                    dialogue[prev_self_turn][2],
                )
                # 可选：获取对手上一轮发言
                opponent_utt = (
                    dialogue[turn_id - 1][1]
                    if dialogue[turn_id - 1][0] != role
                    else None
                )
                opponent_arguments = (
                    dialogue[turn_id - 1][2]
                    if dialogue[turn_id - 1][0] != role
                    else None
                )

                try:
                    # 格式化 arguments 为字符串
                    prev_self_args_str = (
                        format_arguments(prev_self_arguments)
                        if prev_self_arguments
                        else None
                    )
                    current_self_args_str = (
                        format_arguments(arguments) if arguments else None
                    )
                    opponent_args_str = (
                        format_arguments(opponent_arguments)
                        if opponent_arguments
                        else None
                    )

                    progressiveness_eval = evaluate_progressiveness(
                        prev_self_utt,
                        utt,
                        prev_self_args_str,
                        current_self_args_str,
                        opponent_utt,
                        opponent_args_str,
                    )
                except Exception as e:
                    progressiveness_eval = None

        evaluations.append(
            {
                "turn": turn_id,
                "role": role,
                "coherence": coherence_eval,  # 新增
                "progressiveness": progressiveness_eval,  # 新增
            }
        )

    # 新增：Consistency 评估（对话级别，只对assistant进行评估）
    consistency_eval = None
    try:
        # 提取user的立场（第一轮user发言）
        consistency_eval = evaluate_consistency(topic, dialogue)
    except Exception as e:
        consistency_eval = None

    if evaluations:
        item["dialogue_evaluation"] = evaluations
        # 添加对话级别的consistency评估
        item["assistant_consistency_evaluation"] = consistency_eval
        print(f"Completed item {index}")
        return item
    else:
        print(f"Failed to process item {index}")
        return None


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


def split_by_dialogue(data):
    """根据每个对话的所有responses是否都有claim及其evidence来拆分数据"""
    kept = []
    removed = []

    error_count = 0

    for item in tqdm(data, desc="Processing items"):
        # 获取response_evaluations
        response_evaluations = item.get("response_evaluations", [])

        # 如果没有response_evaluations，则丢弃
        if not response_evaluations:
            removed.append(item)
            continue

        # 检查每个response是否都满足条件
        should_keep = True

        # 1. 检查assistant的立场一致性：必须是consistent和strongly_opposes或opposes
        assistant_consistency = item.get("assistant_consistency_evaluation")
        if assistant_consistency:
            try:
                if isinstance(assistant_consistency, str):
                    assistant_consistency = json.loads(assistant_consistency)

                polarity_consistency = assistant_consistency.get("polarity_consistency")
                opposition_to_user = assistant_consistency.get("opposition_to_user")

                # 检查立场一致性必须是consistent
                if polarity_consistency != "consistent":
                    should_keep = False
                    print(
                        f"Item removed: polarity_consistency is {polarity_consistency}, not consistent"
                    )

                # 检查对用户的反对程度必须是strongly_opposes或opposes
                if opposition_to_user not in ["strongly_opposes", "opposes"]:
                    should_keep = False
                    print(
                        f"Item removed: opposition_to_user is {opposition_to_user}, not strongly_opposes or opposes"
                    )

            except Exception as e:
                print(f"Error parsing assistant_consistency_evaluation: {e}")
                should_keep = False
        else:
            should_keep = False
            print("Item removed: no assistant_consistency_evaluation")

        if not should_keep:
            removed.append(item)
            continue

        # 2. 检查所有response的progressiveness：必须是new或mixed
        dialogue_evaluation = item.get("dialogue_evaluation", [])
        for eval_item in dialogue_evaluation:
            progressiveness_eval = eval_item.get("progressiveness")
            if progressiveness_eval:
                try:
                    if isinstance(progressiveness_eval, str):
                        progressiveness_eval = json.loads(progressiveness_eval)

                    progressiveness = progressiveness_eval.get("progressiveness")
                    if progressiveness not in ["new", "mixed"]:
                        should_keep = False
                        print(
                            f"Item removed: progressiveness is {progressiveness}, not new or mixed"
                        )
                        break

                except Exception as e:
                    print(f"Error parsing progressiveness: {e}")
                    should_keep = False
                    break

        if not should_keep:
            removed.append(item)
            continue

        # 3. 检查所有response的coherence：必须是directly_addresses或partially_addresses
        for eval_item in dialogue_evaluation:
            coherence_eval = eval_item.get("coherence")
            if coherence_eval:
                try:
                    if isinstance(coherence_eval, str):
                        coherence_eval = json.loads(coherence_eval)

                    coherence_score = coherence_eval.get("coherence_score")
                    if coherence_score not in [
                        "directly_addresses",
                        "partially_addresses",
                    ]:
                        should_keep = False
                        print(
                            f"Item removed: coherence_score is {coherence_score}, not directly_addresses or partially_addresses"
                        )
                        break

                except Exception as e:
                    print(f"Error parsing coherence: {e}")
                    should_keep = False
                    break

        # 根据检查结果决定保留还是丢弃
        if should_keep:
            kept.append(item)
        else:
            removed.append(item)

    print(f"Error count: {error_count}")

    return kept, removed


if __name__ == "__main__":
    # data_path = "cmv_dialog_flow_clean_user_multi_paths_round2_2000_4_all_6_all_8_all_short_all/response_evaluation/all_data_evaluation_argument.jsonl"
    # output_path = "cmv_dialog_flow_clean_user_multi_paths_round2_2000_4_all_6_all_8_all_short_all/dialogue_evaluation/all_data_evaluation.jsonl"

    # os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # print("Loading data...")
    # data = read_jsonl(data_path)
    # print(f"Loaded {len(data)} items")

    # # for index, item in enumerate(data):
    # #     topic = item["title"]
    # #     data[index] = process_item(item, index)
    # #     break

    # # 设置线程数，可以根据需要调整
    # max_workers = os.cpu_count()  # 建议根据API限制和机器性能调整, 设置为可用的最大值

    # print(f"Starting processing with {max_workers} threads...")
    # start_time = time.time()

    # # 使用ThreadPoolExecutor进行多线程处理
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     # 提交所有任务
    #     future_to_index = {
    #         executor.submit(process_item, item, i): i for i, item in enumerate(data)
    #     }

    #     # 收集结果
    #     completed_count = 0
    #     failed_count = 0
    #     for future in as_completed(future_to_index):
    #         index = future_to_index[future]
    #         try:
    #             result = future.result()
    #             if result:
    #                 data[index] = result
    #                 completed_count += 1

    #                 # 每处理10个item保存一次中间结果
    #                 if completed_count % 10 == 0:
    #                     print(
    #                         f"Progress: {completed_count}/{len(data)} items completed"
    #                     )
    #                     save_results(data, output_path)

    #         except Exception as e:
    #             print(f"Error processing item {index}: {e}")

    # # 保存最终结果
    # save_results(data, output_path)

    # end_time = time.time()
    # print(f"Processing completed in {end_time - start_time:.2f} seconds")
    # print(f"Successfully processed {completed_count}/{len(data)} items")

    data_path = "cmv_dialog_flow_clean_user_multi_paths_round2_2000_4_all_6_all_8_all_short_all/dialogue_evaluation/all_data_evaluation.jsonl"

    print("Loading data...")
    data = read_jsonl(data_path)
    print(f"Loaded {len(data)} items")

    # 处理完成后进行后处理与保存
    kept, removed = split_by_dialogue(data)

    print(f"Kept {len(kept)} items, removed {len(removed)} items")

    # 输出文件名：在原输出名基础上追加后缀
    base_out = "cmv_dialog_flow_clean_user_multi_paths_round2_2000_4_all_6_all_8_all_short_all/dialogue_evaluation/all_data_evaluation_dialogue.jsonl"
    # if base_out.endswith(".jsonl"):
    #     out_true = base_out.replace(".jsonl", "_debatable_true.jsonl")
    #     out_false = base_out.replace(".jsonl", "_debatable_false.jsonl")
    # else:
    #     out_true = base_out + "_debatable_true"
    #     out_false = base_out + "_debatable_false"

    save_results(kept, base_out)
