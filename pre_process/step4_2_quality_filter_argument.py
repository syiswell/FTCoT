import os

os.environ["OPENAI_API_KEY"] = "sk-Pr6Ye2wTLQ05aczR6riYQLjBT4znKzketM93YxfucfYcUZUI"
os.environ["OPENAI_BASE_URL"] = "https://pro.xiaoai.plus/v1"

from call_llm import model_factory, Message
import json
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm


model = model_factory("gpt-4o-mini")


instruction = """Extract the list of arguments from the Target Utterance.
1. For each core claim or main point, identify and extract the claim, if exists.
2. For each claim, list the supporting evidence, examples, data, or legal precedents associated with it, if exists.
3. Ignore non-argumentative content.
4. If multiple sentences express similar arguments, combine them into a single, comprehensive claim with all relevant evidence.
5. If there is no evidence, write "No evidence".

Present your output in the following format:
[
    {
        "Claim": "Claim 1",
        "Evidence": [
            "Evidence 1",
            ...
            "Evidence X"
            ]
    },
    ...
    {
        "Claim": "Claim X",
        "Evidence": [
            "Evidence 1",
            ...
            "Evidence X"
            ]
    },
...
]
"""


analyze_with_evidence_json_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "arguments_with_evidence_list",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "arguments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Claim": {
                                "type": "string",
                                "description": "The core claim or main point of the argument, or 'No claim' if not exists",
                            },
                            "Evidence": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 0,
                                "description": "Brief evidence or material supporting the claim , or 'No evidence'",
                            },
                        },
                        "required": ["Claim", "Evidence"],
                        "additionalProperties": False,
                    },
                    "minItems": 3,
                }
            },
            "required": ["arguments"],
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


@retry(
    wait=wait_random_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry_error_callback=lambda retry_state: None,
)
def evaluate_response(topic, dialogue_history, response):
    response_format = analyze_with_evidence_json_format
    messages = [
        Message(role="system", content=instruction),
        Message(
            role="user",
            content=f"[Topic]\n{topic}\n\n[Debate Dialogue]\n{proccess_chat_one(dialogue_history)}",
        ),
        Message(role="user", content=f"Target Utterance: {response}"),
    ]
    structure = model.generate_chat(
        messages=messages,
        temperature=0.0,
        response_format=response_format,
    )

    return structure


def proccess_chat_one(dialogue):
    if len(dialogue) == 0:
        return "No Dialogue History"

    prompt_messages = ""

    for i, (role, utt) in enumerate(dialogue):
        content = f"{role}: {utt}".strip()
        prompt_messages += content + "\n"
    return prompt_messages.strip()


# 线程锁，用于保护共享资源
lock = threading.Lock()


def process_item(item, index):
    """处理单个数据项的函数"""
    topic = item.get("title")
    print(f"Processing item {index}: {topic}")

    # 尝试从多种常见字段中抽取对话 [(role, utt), ...]
    path = item["path"]
    dialogue = []

    role_map = {0: "user", 1: "assistant"}
    for idx, response in enumerate(path):
        dialogue.append((role_map[idx % 2], response))

    evaluations = []
    for turn_id, (role, utt) in enumerate(dialogue):
        history = dialogue[: turn_id + 1]  # 当前响应之前的对话历史
        try:
            ev = evaluate_response(topic, history, utt)
        except Exception as e:
            ev = None
        evaluations.append(
            {
                "turn": turn_id,
                "role": role,
                "response": utt,
                "evaluation": ev,
            }
        )

    if evaluations:
        item["response_evaluations"] = evaluations
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


def split_by_argument(data):
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

        for eval_item in response_evaluations:
            evaluation = eval_item.get("evaluation")

            # # 如果evaluation为空或None，则丢弃
            # if not evaluation or not isinstance(evaluation, dict):
            #     should_keep = False
            #     break
            try:
                evaluation = json.loads(evaluation)
            except:
                error_count += 1
                continue

            arguments = evaluation.get("arguments", [])

            # 如果没有arguments或arguments为空，则丢弃（没有claim）
            if not arguments or len(arguments) == 0:
                should_keep = False
                break

            # 检查该response是否有至少一个有效的claim
            has_valid_claim = False
            has_any_evidence = False

            for arg in arguments:
                claim = arg.get("Claim", "").strip()
                evidence = arg.get("Evidence", [])

                # 如果claim不为空且不是"No claim"，则认为有有效claim
                if claim and claim.lower() != "no claim":
                    has_valid_claim = True

                    # save1: 检查evidence是否有效（不为空且不全是"No evidence"）
                    if evidence and not (
                        len(evidence) == 1 and evidence[0].lower() == "no evidence"
                    ):
                        has_any_evidence = True

                    # save2: 检查evidence是否有效（为空或是"No evidence"）
                    # if not evidence or (
                    #     len(evidence) == 1 and evidence[0].lower() == "no evidence"
                    # ):
                    #     has_any_evidence = False
                    #     break

            # 规则1：没有claim则丢弃
            if not has_valid_claim:
                should_keep = False
                break

            # 规则2：有claim但没有evidence则丢弃
            if has_valid_claim and not has_any_evidence:
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
    # data_path = "cmv_dialog_flow_clean_user_multi_paths_round2_2000_4_all_6_all_8_all_short_all/topic_evaluation/all_data_topic_evaluation_debatable_true.jsonl"
    # output_path = "cmv_dialog_flow_clean_user_multi_paths_round2_2000_4_all_6_all_8_all_short_all/response_evaluation/all_data_evaluation.jsonl"

    # os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # print("Loading data...")
    # data = read_jsonl(data_path)
    # print(f"Loaded {len(data)} items")

    # # for index, item in enumerate(data):
    # #     topic = item["title"]
    # #     response = evaluate_topic(topic)
    # #     if response:
    # #         item["topic_evaluation"] = response
    # #         print(f"Completed item {index}")
    # #     else:
    # #         print(f"Failed to process item {index}")

    # # 设置线程数，可以根据需要调整
    # max_workers = 5  # 建议根据API限制和机器性能调整

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

    data_path = "cmv_dialog_flow_clean_user_multi_paths_round2_2000_4_all_6_all_8_all_short_all/response_evaluation/all_data_evaluation.jsonl"

    print("Loading data...")
    data = read_jsonl(data_path)
    print(f"Loaded {len(data)} items")

    # 处理完成后进行后处理与保存
    kept, removed = split_by_argument(data)

    print(f"Kept {len(kept)} items, removed {len(removed)} items")

    # 输出文件名：在原输出名基础上追加后缀
    base_out = "cmv_dialog_flow_clean_user_multi_paths_round2_2000_4_all_6_all_8_all_short_all/response_evaluation/all_data_evaluation_argument.jsonl"
    # if base_out.endswith(".jsonl"):
    #     out_true = base_out.replace(".jsonl", "_debatable_true.jsonl")
    #     out_false = base_out.replace(".jsonl", "_debatable_false.jsonl")
    # else:
    #     out_true = base_out + "_debatable_true"
    #     out_false = base_out + "_debatable_false"

    save_results(kept, base_out)
