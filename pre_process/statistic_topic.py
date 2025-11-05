import os

os.environ["OPENAI_API_KEY"] = "sk-Pr6Ye2wTLQ05aczR6riYQLjBT4znKzketM93YxfucfYcUZUI"
os.environ["OPENAI_BASE_URL"] = "https://pro.xiaoai.plus/v1"
import sys

sys.path.append("..")
from call_llm import model_factory, Message
import json
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential


model = model_factory("gpt-4o-mini")
title_cache = {}  # 用于缓存已处理的title结果


domains = [
    "General reference",
    "Culture and the arts",
    "Geography and places",
    "Health and fitness",
    "History and events",
    "Human activities",
    "Mathematics and logic",
    "Natural and physical sciences",
    "People and self",
    "Philosophy and thinking",
    "Religion and belief systems",
    "Society and social sciences",
    "Technology and applied sciences",
]
domains_str = ", ".join(domains)

instruction = f"""You are an expert in topic classification familiar with Wikipedia categories.

Your task is to analyze the following text and identify its keywords and domain.
The text may belong to a specific domain ({domains_str}). 
If the domain is not directly apparent, make an educated guess based on the content. 


Steps:
1. Extract keywords or phrases from the given text.
2. Analyze the keywords and reason why this text belongs to a specific domain.
3. Provide the most likely domain (must be one of [{domains_str}]) with confidence score for the given text.
"""


topic_domain_json_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "topic_domain_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "List of extracted keywords",
                    },
                    "required": ["keywords"],
                    "additionalProperties": False,
                },
                "reason": {
                    "type": "string",
                    "description": "Detailed reasoning or analysis for the most likely domain",
                },
                "domain": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": f"The name of the domain, should be one of {domains_str}",
                        },
                        "confidence_score": {
                            "type": "number",
                            "description": f"The confidence score for the domain, between 0 and 1, the confidence score should be based on the keywords and the reasoning",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": ["name", "confidence_score"],
                    "additionalProperties": False,
                },
            },
            "required": ["keywords", "reason", "domain"],
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
def evaluate_topic_domain(topic):

    messages = [
        Message(role="system", content=instruction),
        Message(role="user", content=f"[Text]\n{topic}"),
    ]
    # print("messages: ", messages)
    response = model.generate_chat(
        messages, temperature=0.0, response_format=topic_domain_json_format
    )
    # print("response: ", response)
    return response


# 线程锁，用于保护共享资源
lock = threading.Lock()


def process_item(item, index):
    """处理单个数据项的函数"""
    topic = item["title"]
    print(f"Processing item {index}: {topic}")

    # 检查是否已经处理过这个title
    if topic in title_cache:
        print(f"Using cached result for item {index}: {topic}")
        item["topic_domain"] = title_cache[topic]
        return item

    response = evaluate_topic_domain(topic)
    if response:
        # 将结果存入缓存
        title_cache[topic] = response
        item["topic_domain"] = response
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


domain_mapping = {
    "Society and social sciences": "Society and social sciences",
    "Philosophy and thinking": "Philosophy and thinking",
    "Culture and the arts": "Culture and the arts",
    "Health and fitness": "Health and fitness",
    "Technology and applied sciences": "Technology and applied sciences",
    "People and self": "People and self",
    "Human activities": "Human activities",
    "Religion and belief systems": "Religion and belief systems",
    "History and events": "History and events",
    "Natural and physical sciences": "Natural and physical sciences",
    "Education": "Society and social sciences",
    "Mathematics and logic": "Mathematics and logic",
    "Economics and finance": "Society and social sciences",
    "Education and learning": "Society and social sciences",
    "Geography and places": "Geography and places",
    "Economics": "Society and social sciences",
    "General reference": "General reference",
    "Education and pedagogy": "Society and social sciences",
    "Politics and social sciences": "Society and social sciences",
    "Food and drink": "Human activities",
    "Politics and government": "Society and social sciences",
    "Education and social sciences": "Society and social sciences",
    "Sports and recreation": "Human activities",
    "Language and linguistics": "Culture and the arts",
    "Education and self": "People and self",
    "Law and legal studies": "Society and social sciences",
}


def statistic_topic_domain(data):
    """统计所有样本的主题类别和对应的数量"""
    domain_counts = {}
    domain_details = {}
    total_samples = len(data)
    processed_samples = 0

    print("开始统计主题类别...")

    for item in data:
        topic_domain = item.get("topic_domain")

        if topic_domain is None:
            continue

        # 处理topic_domain可能是字符串或字典的情况
        if isinstance(topic_domain, str):
            try:
                topic_domain = json.loads(topic_domain)
            except Exception:
                continue

        if isinstance(topic_domain, dict):
            domain_info = topic_domain.get("domain")
            if isinstance(domain_info, dict):
                domain_name = domain_info.get("name")
                confidence_score = domain_info.get("confidence_score", 0)
                domain_name = domain_mapping.get(domain_name, domain_name)

                if domain_name:
                    processed_samples += 1

                    # 统计数量
                    if domain_name in domain_counts:
                        domain_counts[domain_name] += 1
                    else:
                        domain_counts[domain_name] = 1

                    # 收集详细信息（包括置信度）
                    if domain_name not in domain_details:
                        domain_details[domain_name] = {
                            "count": 0,
                            "confidence_scores": [],
                            "avg_confidence": 0,
                        }

                    domain_details[domain_name]["count"] += 1
                    domain_details[domain_name]["confidence_scores"].append(
                        confidence_score
                    )

    # 计算平均置信度
    for domain_name, details in domain_details.items():
        if details["confidence_scores"]:
            details["avg_confidence"] = sum(details["confidence_scores"]) / len(
                details["confidence_scores"]
            )

    # 按数量排序
    sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)

    # 打印统计结果
    print(f"\n=== 主题类别统计结果 ===")
    print(f"总样本数: {total_samples}")
    print(f"成功处理的样本数: {processed_samples}")
    print(f"发现的主题类别数: {len(domain_counts)}")
    print(f"\n各主题类别分布:")
    print("-" * 60)
    print(f"{'主题类别':<25} {'数量':<8} {'占比':<8} {'平均置信度':<12}")
    print("-" * 60)

    for domain_name, count in sorted_domains:
        percentage = (count / processed_samples) * 100 if processed_samples > 0 else 0
        avg_confidence = domain_details[domain_name]["avg_confidence"]
        print(
            f"{domain_name:<25} {count:<8} {percentage:<7.2f}% {avg_confidence:<11.3f}"
        )


if __name__ == "__main__":
    # data_path = "/home/sunyang/hlt/new_cmv_dataset/cmv_dialog_flow_clean_user_multi_paths_round2_2000_4_all_6_all_8_all_short_all/dialogue_evaluation/all_data_evaluation_dialogue.jsonl"
    # output_path = "/home/sunyang/hlt/new_cmv_dataset/cmv_dialog_flow_clean_user_multi_paths_round2_2000_4_all_6_all_8_all_short_all/dialogue_evaluation/all_data_topic_domain.jsonl"

    # print("Loading data...")
    # data = read_jsonl(data_path)
    # print(f"Loaded {len(data)} items")

    # # 统计唯一title数量
    # unique_titles = set(item["title"] for item in data)
    # print(f"Found {len(unique_titles)} unique titles out of {len(data)} total items")

    # # for index, item in enumerate(data):
    # #     topic = item["title"]
    # #     response = evaluate_topic_domain(topic)
    # #     if response:
    # #         item["topic_domain"] = response
    # #         print(f"Completed item {index}")
    # #     else:
    # #         print(f"Failed to process item {index}")

    # # 设置线程数，可以根据需要调整
    # max_workers = os.cpu_count()  # 建议根据API限制和机器性能调整

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

    data_path = "/home/sunyang/hlt/new_cmv_dataset/cmv_dialog_flow_clean_user_multi_paths_round2_2000_4_all_6_all_8_all_short_all/dialogue_evaluation/all_data_topic_domain.jsonl"

    print("Loading data...")
    data = read_jsonl(data_path)
    print(f"Loaded {len(data)} items")

    # 处理完成后进行后处理与保存
    statistic_topic_domain(data)
