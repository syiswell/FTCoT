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


model = model_factory("gpt-4o-mini")


instruction = f"""You will play the role of a logical and thorough debater. Your task is to analyze and evaluate the following topic for its debatability. A reasonable debate topic should have the following characteristics: it should be clear, have opposing viewpoints, be controversial to some extent, and allow for multi-angle discussions.

Please follow these steps for your analysis:

Clarify the Topic: Read the topic statement in the provided dialogue sample and extract the core issue.

Evaluate Debatability Characteristics:

Clarity: Is the topic clear and easy to understand?
Opposability: Can clear support and opposition positions be identified?
Controversiality: Is the topic commonly associated with differing opinions or controversy?
Depth for Discussion: Does the topic allow for in-depth exploration and presentation of various perspectives?
Intermediate Reasoning:

For each characteristic, provide reasons supporting or opposing the topic's debatability.
Offer at least one positive and one negative argument illustrating typical debates the topic might spark.
Final Judgment: Based on the analysis above, judge whether the topic is "debatable," and provide a brief concluding statement.
"""


topic_evaluation_json_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "topic_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "clarity": {
                    "type": "object",
                    "properties": {
                        "assessment": {
                            "type": "string",
                            "enum": ["clear", "unclear"],
                            "description": "Assessment of clarity",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Reason for clarity assessment",
                        },
                    },
                    "required": ["assessment", "reasoning"],
                    "additionalProperties": False,
                },
                "opposability": {
                    "type": "object",
                    "properties": {
                        "assessment": {
                            "type": "string",
                            "enum": ["opposable", "unopposable"],
                            "description": "Assessment of opposability",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Reason for opposability assessment",
                        },
                    },
                    "required": ["assessment", "reasoning"],
                    "additionalProperties": False,
                },
                "controversiality": {
                    "type": "object",
                    "properties": {
                        "assessment": {
                            "type": "string",
                            "enum": ["controversial", "uncontroversial"],
                            "description": "Assessment of controversiality",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Reason for controversiality assessment",
                        },
                    },
                    "required": ["assessment", "reasoning"],
                    "additionalProperties": False,
                },
                "depth": {
                    "type": "object",
                    "properties": {
                        "assessment": {
                            "type": "string",
                            "enum": ["deep", "shallow"],
                            "description": "Assessment of depth",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Reason for depth assessment",
                        },
                    },
                    "required": ["assessment", "reasoning"],
                    "additionalProperties": False,
                },
                "final_judgment": {
                    "type": "object",
                    "properties": {
                        "debatable": {
                            "type": "boolean",
                            "description": "Whether the topic is debatable",
                        },
                        "summary": {
                            "type": "string",
                            "description": "Brief summary of the judgment",
                        },
                    },
                    "required": ["debatable", "summary"],
                    "additionalProperties": False,
                },
            },
            "required": [
                "clarity",
                "opposability",
                "controversiality",
                "depth",
                "final_judgment",
            ],
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
def evaluate_topic(topic):
    response = model.generate_chat(
        messages=[
            Message(role="system", content=instruction),
            Message(role="user", content=f"[Topic]\n{topic}"),
        ],
        temperature=0.0,
        response_format=topic_evaluation_json_format,
    )

    return response


# 线程锁，用于保护共享资源
lock = threading.Lock()


def process_item(item, index):
    """处理单个数据项的函数"""
    topic = item["title"]
    print(f"Processing item {index}: {topic}")

    response = evaluate_topic(topic)
    if response:
        item["topic_evaluation"] = response
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


def split_by_debatable(data):
    """根据 topic_evaluation.final_judgment.debatable 拆分数据"""
    kept = []
    removed = []

    for item in data:
        debatable = None
        te = item.get("topic_evaluation")

        # 兼容几种可能的类型：dict 或 JSON 字符串
        if isinstance(te, str):
            try:
                te = json.loads(te)
            except Exception:
                te = None

        if isinstance(te, dict):
            fj = te.get("final_judgment")
            if isinstance(fj, dict):
                debatable = fj.get("debatable")

        # debatable 为 False → 剔除；其他情况（True 或 None）默认保留
        if debatable is False:
            removed.append(item)
        else:
            kept.append(item)

    return kept, removed


if __name__ == "__main__":
    # data_path = "cmv_dialog_flow_clean_user_multi_paths_round2_2000_4_all_6_all_8_all_short_all/all_data.jsonl"
    # output_path = "cmv_dialog_flow_clean_user_multi_paths_round2_2000_4_all_6_all_8_all_short_all/all_data_topic_evaluation.jsonl"

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

    data_path = "cmv_dialog_flow_clean_user_multi_paths_round2_2000_4_all_6_all_8_all_short_all/all_data_topic_evaluation.jsonl"

    print("Loading data...")
    data = read_jsonl(data_path)
    print(f"Loaded {len(data)} items")

    # 处理完成后进行后处理与保存
    kept, removed = split_by_debatable(data)

    # 输出文件名：在原输出名基础上追加后缀
    base_out = "cmv_dialog_flow_clean_user_multi_paths_round2_2000_4_all_6_all_8_all_short_all/all_data_topic_evaluation_debatable_true.jsonl"
    # if base_out.endswith(".jsonl"):
    #     out_true = base_out.replace(".jsonl", "_debatable_true.jsonl")
    #     out_false = base_out.replace(".jsonl", "_debatable_false.jsonl")
    # else:
    #     out_true = base_out + "_debatable_true"
    #     out_false = base_out + "_debatable_false"

    save_results(kept, base_out)
