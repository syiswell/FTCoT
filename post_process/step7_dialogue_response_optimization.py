import os
import hashlib

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
from tqdm import tqdm
from typing import List, Dict, Tuple

from utils import load_jsonl, load_text
import re


system_instruction = "You are an experienced debater who can efficiently analyze the content of debate dialogue history and develop reasonable plans to accomplish debate-related tasks."

debate_dialogue_instruction = """You are an assistant in a debate dialogue. The user presents a strong opinion on a specific topic. Your goal is to respond persuasively and logically — challenging their stance with counterarguments, context, and ethical or emotional reasoning."""


model_name = "gpt-4o-mini"
model = model_factory(model_name)
prompt_inst_path = "/home/sunyang/hlt/new_cmv_dataset/post_process/response_optimization/response_optimization_prompt.txt"
response_optimization_prompt = load_text(prompt_inst_path)


def generate_data_hash(item):
    """为数据项生成唯一标识符"""
    # 使用instruction和output的内容生成hash
    content = f"{item.get('instruction', '')}{item.get('output', '')}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def load_processed_cache(output_dir, data_split):
    """加载已处理的数据缓存"""
    cache_file = os.path.join(output_dir, f"processed_cache_{data_split}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            print(f"Loaded cache with {len(cache_data)} processed items")
            return cache_data
        except Exception as e:
            print(f"Error loading cache: {e}")
            return {}
    return {}


def save_processed_cache(cache_data, output_dir, data_split):
    """保存处理缓存"""
    cache_file = os.path.join(output_dir, f"processed_cache_{data_split}.json")
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving cache: {e}")


def load_existing_results(output_dir, data_split):
    """加载已存在的结果文件"""
    result_file = os.path.join(output_dir, f"response_optimization_{data_split}.jsonl")
    if os.path.exists(result_file):
        try:
            existing_data = load_jsonl(result_file)
            print(f"Found existing results with {len(existing_data)} items")
            return existing_data
        except Exception as e:
            print(f"Error loading existing results: {e}")
            return []
    return []


@retry(
    wait=wait_random_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry_error_callback=lambda retry_state: None,
)
def response_analysis(input_query):
    try:
        messages = [
            Message(role="system", content=system_instruction),
            Message(role="user", content=input_query),
        ]
        response = model.generate_chat(
            messages=messages,
            temperature=0.2,
        )
        return response
    except Exception as e:
        print(f"Error in response_analysis: {e}")
        return None


# 添加response优化的第二步prompt
optimization_second_step_prompt = """Now, based on your analysis, please optimize this candidate response to better address the task instruction.

Please provide your response in the following JSON format:
{{
    "optimized_response": "...",
    "any_additional_information": "..."
}}
"""


@retry(
    wait=wait_random_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry_error_callback=lambda retry_state: None,
)
def response_optimization(messages):
    """基于分析结果优化response"""
    try:
        response = model.generate_chat(
            messages=messages, temperature=0.2, response_format={"type": "json_object"}
        )
        return response
    except Exception as e:
        print(f"Error in response_optimization: {e}")
        return None


def validate_input_data(item):
    """验证输入数据的完整性"""
    required_fields = ["instruction", "output"]
    for field in required_fields:
        if field not in item or not item[field]:
            return False, f"Missing or empty field: {field}"
    return True, "Valid"


def parse_optimization_result(optimization_result, max_retries=3):
    """解析优化结果，支持重试"""
    for attempt in range(max_retries):
        try:
            optimization_json = json.loads(optimization_result)
            if "optimized_response" not in optimization_json:
                raise ValueError("Missing 'optimized_response' field in JSON")
            return optimization_json
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON parsing attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                # 尝试修复常见的JSON格式问题
                optimization_result = optimization_result.strip()
                if not optimization_result.startswith("{"):
                    # 查找第一个{和最后一个}
                    start_idx = optimization_result.find("{")
                    end_idx = optimization_result.rfind("}")
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        optimization_result = optimization_result[
                            start_idx : end_idx + 1
                        ]
            else:
                print(
                    f"All JSON parsing attempts failed for result: {optimization_result[:200]}..."
                )
                return None
    return None


def process_response_optimization(item, index, processed_cache):
    """处理单个数据项的response优化"""
    # 生成数据项的唯一标识
    data_hash = generate_data_hash(item)

    # 检查是否已经处理过
    if data_hash in processed_cache:
        print(f"Item {index} already processed, skipping...")
        return processed_cache[data_hash]

    print(f"Processing response optimization for item {index}")

    try:

        # 第一步：分析response
        arg_text = item["instruction"]
        task_inst = debate_dialogue_instruction
        cand_resp = item.get("output", "")

        analysis_query = response_optimization_prompt.format(
            task_inst=task_inst, arg_text=arg_text, cand_resp=cand_resp
        )

        # 重试整个流程（包括分析和优化）
        for attempt in range(3):
            print(f"Processing attempt {attempt + 1} for item {index}")

            # 重试分析步骤
            analysis_result = None
            for analysis_attempt in range(2):  # 分析步骤内部重试2次
                analysis_result = response_analysis(analysis_query)
                if analysis_result:
                    break
                print(
                    f"Analysis attempt {analysis_attempt + 1} failed for item {index}"
                )

            if not analysis_result:
                print(f"Analysis failed for item {index}, retrying entire process...")
                continue

            # 第二步：基于分析结果优化response
            optimization_messages = [
                Message(role="system", content=system_instruction),
                Message(role="user", content=analysis_query),
                Message(role="assistant", content=analysis_result),
                Message(role="user", content=optimization_second_step_prompt),
            ]

            # 重试优化步骤
            optimization_result = None
            for optimization_attempt in range(2):  # 优化步骤内部重试2次
                optimization_result = response_optimization(optimization_messages)
                if optimization_result:
                    break
                print(
                    f"Optimization attempt {optimization_attempt + 1} failed for item {index}"
                )

            if not optimization_result:
                print(
                    f"Optimization failed for item {index}, retrying entire process..."
                )
                continue

            # 解析JSON结果
            optimization_json = parse_optimization_result(optimization_result)
            if not optimization_json:
                print(
                    f"JSON parsing failed for item {index}, retrying entire process..."
                )
                continue

            # 如果所有步骤都成功，更新数据项并返回
            item["optimized_response"] = optimization_json.get("optimized_response", "")
            item["response_analysis"] = analysis_result
            item["optimization_metadata"] = optimization_json.get(
                "any_additional_information", ""
            )
            item["data_hash"] = data_hash  # 添加hash标识

            print(f"Completed response optimization for item {index}")
            return item

        # 如果所有尝试都失败了
        print(f"Failed to process item {index} after 3 complete attempts")
        return None

    except Exception as e:
        print(f"Error processing response optimization for item {index}: {e}")
        return None


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


def save_progress(processed_data, output_path, completed_count, total_count):
    """保存中间进度"""
    try:
        # 过滤掉None值
        valid_data = [item for item in processed_data if item is not None]
        pd.DataFrame(valid_data).to_json(
            output_path,
            orient="records",
            lines=True,
        )
        print(f"Progress saved: {completed_count}/{total_count} items completed")
    except Exception as e:
        print(f"Error saving progress: {e}")


def main(
    data_split: str,
    orig_input_fp: str,
    output_dir: str,
) -> None:

    os.makedirs(output_dir, exist_ok=True)

    # Load input data and prompt instructions
    input_data_points = load_jsonl(orig_input_fp)
    print(f"Loaded {len(input_data_points)} items")

    # 加载已处理的数据缓存
    processed_cache = load_processed_cache(output_dir, data_split)

    # 加载已存在的结果
    existing_results = load_existing_results(output_dir, data_split)

    # 将已存在的结果添加到缓存中
    for item in existing_results:
        if "data_hash" in item:
            processed_cache[item["data_hash"]] = item

    # 统计需要处理的数据
    to_process = []
    skipped_count = 0

    for i, item in enumerate(input_data_points):
        data_hash = generate_data_hash(item)
        if data_hash in processed_cache:
            skipped_count += 1
        else:
            to_process.append((i, item))

    print(
        f"Found {skipped_count} already processed items, {len(to_process)} items to process"
    )

    if not to_process:
        print("All items already processed!")
        return

    # 处理response优化
    max_workers = os.cpu_count()  # 限制并发数避免API限制
    print(f"Starting response optimization with {max_workers} threads...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_response_optimization, item, i, processed_cache): i
            for i, item in to_process
        }

        completed_count = 0
        failed_count = 0
        processed_data = [None] * len(input_data_points)

        # 先填充已处理的数据
        for i, item in enumerate(input_data_points):
            data_hash = generate_data_hash(item)
            if data_hash in processed_cache:
                processed_data[i] = processed_cache[data_hash]

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                if result:
                    processed_data[index] = result
                    # 更新缓存
                    data_hash = result.get("data_hash")
                    if data_hash:
                        processed_cache[data_hash] = result
                    completed_count += 1

                    # 每处理10个item保存一次中间结果和缓存
                    if completed_count % 10 == 0:
                        print(
                            f"Progress: {completed_count}/{len(to_process)} items completed"
                        )
                        # 保存已处理的结果
                        completed_items = [
                            item for item in processed_data if item is not None
                        ]
                        if completed_items:
                            save_results(
                                completed_items,
                                os.path.join(
                                    output_dir,
                                    f"response_optimization_{data_split}.jsonl",
                                ),
                            )
                        # 保存缓存
                        save_processed_cache(processed_cache, output_dir, data_split)

            except Exception as e:
                print(f"Error processing item {index}: {e}")
                failed_count += 1

    # 过滤掉None值
    final_data = [item for item in processed_data if item is not None]

    # 保存最终结果
    output_file = os.path.join(output_dir, f"response_optimization_{data_split}.jsonl")
    save_results(final_data, output_file)

    # 保存最终缓存
    save_processed_cache(processed_cache, output_dir, data_split)

    end_time = time.time()
    print(f"Response optimization completed in {end_time - start_time:.2f} seconds")
    print(f"Successfully processed {completed_count}/{len(to_process)} new items")
    print(f"Total items in final result: {len(final_data)}")
    print(f"Failed: {failed_count} items")


def construct_one_step_data(data, optimized_responses, test_mode=False):
    """
    构造一步数据：分析结果作为中间推理过程，生成最终响应
    """

    # 分析步骤的模板
    gen_with_ana_inst_template = """You are an experienced debater who can efficiently analyze the content of debate dialogue history and develop reasonable plans to accomplish debate-related tasks.

Given the following **task instruction** and its corresponding **input text**, your goal is to:
1. **First**, analyze the text carefully according to the **Analysis Guidelines** provided at the end.
2. **Then**, based on your analysis, **complete the given task** as required by the **Task Instruction**.


## Task Instruction:

{task_instruction}


## Input Text:

{argument_text}


## Analysis Guidelines:

**Argument Structure**:
    - Analyze the text's argument structure. Identify the central thesis or main claim, key evidence, and how counterarguments are addressed.
    - Outline the main logical and reasoning flow.
**Argument Quality**:
    - Highlight the main strengths and primary weaknesses.
    - Overall Evaluation: Assess the overall quality and persuasiveness of the given argumentative text, and provide an overall rating as an integer between 1 (lowest) to 5 (highest).
**Key Considerations for Solving the Task**:
    - Based on the given argumentative text, explain in detail the key arguments, content, strategies, etc., that may need to be considered to solve the task.


## Output Format:
<Thinking>
[Your detailed analysis based on the Analysis Guidelines]
</Thinking>
<Response>
[Your final response to the task]
</Response>


## Output Text:
"""

    new_data = []

    # 创建optimized_responses的hash索引，用于快速查找
    optimized_responses_dict = {}
    for opt_item in optimized_responses:
        opt_hash = generate_data_hash(opt_item)
        optimized_responses_dict[opt_hash] = opt_item

    for item in data:
        current_hash = generate_data_hash(item)
        # 查找对应的优化响应
        optimized_item = optimized_responses_dict.get(current_hash)
        if optimized_item is None:
            print(
                f"Warning: No optimized response found for data item with hash {current_hash}"
            )
            optimized_item = item  # 回退到原始数据

        # 从原始指令中提取任务指令和论证文本
        argument_text = item["instruction"]
        task_instruction = debate_dialogue_instruction

        analysis_result = item.get("argument_analysis", "")
        gen_instruction = gen_with_ana_inst_template.format(
            task_instruction=task_instruction,
            argument_text=argument_text,
        )
        if test_mode is False:
            optimized_response = optimized_item.get(
                "optimized_response", item.get("output", "")
            )
            gen_output = (
                "<Thinking>\n"
                + analysis_result
                + "\n</Thinking>\n<Response>\n"
                + optimized_response
                + "\n</Response>"
            )
        else:
            optimized_response = optimized_item.get("optimized_response", "")
            gen_output = optimized_response

        generation_item = {
            "instruction": gen_instruction,
            "input": "",
            "output": gen_output,
        }
        new_data.append(generation_item)

    return new_data


def construct_two_step_data(data, optimized_responses, test_mode=False):
    """
    构造两步数据：第一步是分析，第二步是基于分析结果生成
    """

    # 分析步骤的模板
    ana_inst_template = """Given the following task instruction, the corresponding argumentative input text, first analyze the structure and quality of the input text, and then analyze the key points that may need to be considered to solve the task. The analysis guidelines are provided at the end.


## Task Instruction:

{task_instruction}


## Input Text:

{argument_text}


## Analysis Guidelines:

**Argument Structure**:
    - Analyze the text's argument structure. Identify the central thesis or main claim, key evidence, and how counterarguments are addressed.
    - Outline the main logical and reasoning flow.
**Argument Quality**:
    - Highlight the main strengths and primary weaknesses.
    - Overall Evaluation: Assess the overall quality and persuasiveness of the given argumentative text, and provide an overall rating as an integer between 1 (lowest) to 5 (highest).
**Key Considerations for Solving the Task**:
    - Based on the given argumentative text, explain in detail the key arguments, content, strategies, etc., that may need to be considered to solve the task.


## Note: 
    - You only need to analyze the following text according to the above requirements. You do not need to actually complete the task specified in the **Task Instruction**.
    - Throughout your analysis, provide specific examples from the argumentative text to support your judgments.
    - Please try to give a response of about 300 words.


## Output Text:
"""

    # 生成步骤的模板
    gen_with_ana_inst_template = """Given the following task instruction, the corresponding input text, and the analysis result of the input text, your need to complete the task according to the analysis result.


## Task Instruction:

{task_instruction}


## Input Text:

{argument_text}


## Analysis of the Input Text:

{analysis_result}


## Output Text:
"""

    analysis_data = []
    generation_data = []

    # 创建optimized_responses的hash索引，用于快速查找
    optimized_responses_dict = {}
    for opt_item in optimized_responses:
        if "data_hash" in opt_item:
            optimized_responses_dict[opt_item["data_hash"]] = opt_item
        else:
            # 如果没有data_hash，尝试生成一个
            opt_hash = generate_data_hash(opt_item)
            optimized_responses_dict[opt_hash] = opt_item

    for item in data:
        # 生成当前数据项的hash
        current_hash = generate_data_hash(item)

        # 查找对应的优化响应
        optimized_item = optimized_responses_dict.get(current_hash)
        if optimized_item is None:
            print(
                f"Warning: No optimized response found for data item with hash {current_hash}"
            )
            optimized_item = item  # 回退到原始数据

        # 从原始指令中提取任务指令和论证文本
        argument_text = item["instruction"]
        task_instruction = debate_dialogue_instruction

        # 第一步：分析数据
        ana_instruction = ana_inst_template.format(
            task_instruction=task_instruction, argument_text=argument_text
        )
        if test_mode is False:
            ana_output = item.get("argument_analysis", "")
        else:
            ana_output = ""
        analysis_item = {
            "instruction": ana_instruction,
            "input": "",
            "output": ana_output,
        }
        analysis_data.append(analysis_item)

        # 第二步：生成数据（使用分析结果）
        analysis_result = item.get("argument_analysis", "")
        gen_instruction = gen_with_ana_inst_template.format(
            task_instruction=task_instruction,
            argument_text=argument_text,
            analysis_result=analysis_result,
        )

        # 第二步：生成数据（使用分析结果）
        gen_output = optimized_item.get("optimized_response", "")

        if test_mode is False:
            analysis_result = item.get("argument_analysis", "")
        else:
            analysis_result = "{{analysis_result}}"

        gen_instruction = gen_with_ana_inst_template.format(
            task_instruction=task_instruction,
            argument_text=argument_text,
            analysis_result=analysis_result,
        )
        # 创建两步数据
        generation_item = {
            "instruction": gen_instruction,
            "input": "",
            "output": gen_output,
        }
        generation_data.append(generation_item)

    return analysis_data, generation_data


def construct_sft_data(data, optimized_responses):

    debate_dialogue_instruction = """You are an assistant in a debate dialogue. The user presents a strong opinion on a specific topic. Your goal is to respond persuasively and logically — challenging their stance with counterarguments, context, and ethical or emotional reasoning."""

    # 分析步骤的模板
    gen_with_optimized_response_template = """Given the following task instruction and the corresponding argumentative input text, your need to complete the task.


## Task Instruction:

{task_instruction}


## Input Text:

{argument_text}


## Output Text:
"""

    new_data = []

    # 创建optimized_responses的hash索引，用于快速查找
    optimized_responses_dict = {}
    for opt_item in optimized_responses:
        opt_hash = generate_data_hash(opt_item)
        optimized_responses_dict[opt_hash] = opt_item

    for item in data:
        # 从原始指令中提取任务指令和论证文本
        current_hash = generate_data_hash(item)
        optimized_item = optimized_responses_dict.get(current_hash)
        if optimized_item is None:
            print(
                f"Warning: No optimized response found for data item with hash {current_hash}"
            )
            optimized_item = item  # 回退到原始数据

        argument_text = item["instruction"]
        task_instruction = debate_dialogue_instruction

        gen_instruction = gen_with_optimized_response_template.format(
            task_instruction=task_instruction,
            argument_text=argument_text,
        )
        gen_output = optimized_item.get("optimized_response", "")

        generation_item = {
            "instruction": gen_instruction,
            "input": "",
            "output": gen_output,
        }
        new_data.append(generation_item)

    return new_data


if __name__ == "__main__":

    # for data_split in ["train", "test"]:
    #     orig_input_fp = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/instruction_following_format/{data_split}.jsonl"
    #     # orig_input_fp = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/gpt_analysis/train/argument_analysis_train.jsonl"
    #     output_dir = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/response_optimization/{data_split}"
    #     test_mode = False

    #     main(
    #         data_split,
    #         orig_input_fp,
    #         output_dir,
    #     )

    # 构造训练数据

    for data_split in ["test"]:  # "train",
        if data_split == "train":
            data_path = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/gpt_analysis/train/argument_analysis_train.jsonl"  # test_mode=False
            optimized_response_path = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/response_optimization/train/response_optimization_train.jsonl"
        else:
            data_path = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/instruction_following_format/test.jsonl"  # test_mode=True
            optimized_response_path = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/response_optimization/test/response_optimization_test.jsonl"

        print("Loading data...")
        data = load_jsonl(data_path)
        optimized_responses = load_jsonl(optimized_response_path)
        print(f"Loaded {len(data)} items")

        if data_split == "train":
            test_mode = False
        else:
            test_mode = True

        # # 处理完成后进行后处理与保存
        new_data = construct_one_step_data(
            data, optimized_responses, test_mode=test_mode
        )

        # 输出文件名：在原输出名基础上追加后缀
        base_out = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/response_optimization/one_step/{data_split}.jsonl"
        os.makedirs(os.path.dirname(base_out), exist_ok=True)
        save_results(new_data, base_out)

        # # two-step处理完成后进行后处理与保存
        # analysis_data, generation_data = construct_two_step_data(
        #     data, optimized_responses, test_mode=test_mode
        # )

        # # 输出文件名：在原输出名基础上追加后缀
        # base_out = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/response_optimization/two_step_1/{data_split}.jsonl"
        # os.makedirs(os.path.dirname(base_out), exist_ok=True)
        # save_results(analysis_data, base_out)

        # base_out = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/response_optimization/two_step_2/{data_split}.jsonl"
        # os.makedirs(os.path.dirname(base_out), exist_ok=True)
        # save_results(generation_data, base_out)

        # new_data = construct_sft_data(data, optimized_responses)
        # output_file = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/sft_data_with_optimized_resp/{data_split}.jsonl"
        # os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # save_results(new_data, output_file)
