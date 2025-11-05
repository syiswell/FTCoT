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
from tqdm import tqdm
from typing import List, Dict, Tuple
from utils import load_jsonl, load_text
import re
import hashlib


system_instruction = "You are an experienced debater who can efficiently analyze the content of debate dialogue history and develop reasonable plans to accomplish debate-related tasks."

debate_dialogue_instruction = """You are an assistant in a debate dialogue. The user presents a strong opinion on a specific topic. Your goal is to respond persuasively and logically — challenging their stance with counterarguments, context, and ethical or emotional reasoning."""

dialogue_analysis_prompt_instion = """Given a piece of debate dialogue text and a task instruction, analyze the argument structure of the debate dialogue text.
The specific analysis guidelines are provided below.

**Argument Structure**:

    - Analyze the text's argument structure. Identify the central thesis or main claim, key evidence, and how counterarguments are addressed.

    - Outline the main logical and reasoning flow.


**Note**: 

    - You only need to analyze the following text according to the above requirements. You do not need to actually complete the task specified in the **Task Instruction**.

    - Throughout your analysis, provide specific examples from the debate dialogue text to support your judgments.

    - Please try to give a response of about 300 words.

Now, the debate dialogue text and task instruction are provided below for your analysis. Please analyze the argument structure of the debate dialogue text:

```
**Task Instruction**:

{task_inst}


**debate dialogue Text**:

{arg_text}

```"""


model_name = "gpt-4o-mini"
model = model_factory(model_name)


@retry(
    wait=wait_random_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry_error_callback=lambda retry_state: None,
)
def argument_analysist(input_text):

    input_query = dialogue_analysis_prompt_instion.format(
        arg_text=input_text, task_inst=debate_dialogue_instruction
    )

    messages = [
        Message(role="system", content=system_instruction),
        Message(role="user", content=input_query),
    ]
    # print("argument_analysist:", messages)
    response = model.generate_chat(
        messages=messages,
        temperature=0.2,
    )
    # print("argument_analysist:", response)

    return response


utterance_analysis_instruction = """Extract the list of arguments from the Target Utterance.
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


def validate_argument_analysis_result(result):
    """
    验证论证分析结果是否可以正确解析

    Args:
        result: 要验证的结果（可能是字符串或字典）

    Returns:
        bool: 如果结果可以正确解析则返回True，否则返回False
    """
    if not result:
        return False

    try:
        # 如果输入是字符串，尝试解析为字典
        if isinstance(result, str):
            analysis_dict = json.loads(result)
        else:
            analysis_dict = result

        # 检查是否包含必需的字段
        if not isinstance(analysis_dict, dict):
            return False

        arguments = analysis_dict.get("arguments", [])
        if not isinstance(arguments, list):
            return False

        # 检查每个argument的结构
        for arg in arguments:
            if not isinstance(arg, dict):
                return False
            if "Claim" not in arg or "Evidence" not in arg:
                return False
            if not isinstance(arg["Evidence"], list):
                return False

        return True

    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return False


@retry(
    wait=wait_random_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry_error_callback=lambda retry_state: None,
)
def response_argument_analysis(dialogue_history, response):
    response_format = analyze_with_evidence_json_format
    messages = [
        Message(role="system", content=utterance_analysis_instruction),
        Message(role="user", content=dialogue_history),
        Message(role="user", content=f"Target Utterance: {response}"),
    ]
    # print("response_argument_analysis:", messages)
    structure = model.generate_chat(
        messages=messages,
        temperature=0.0,
        response_format=response_format,
    )
    # print("response_argument_analysis result:", structure)
    return structure


# 线程锁，用于保护共享资源
lock = threading.Lock()


def generate_item_hash(item):
    """生成数据项的唯一标识符"""
    # 使用topic、dialogue_history和output来生成hash
    content = {
        "topic": item["topic"],
        "output": item["output"],
    }
    content_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(content_str.encode("utf-8")).hexdigest()


def load_cache(cache_file_path):
    """加载缓存文件"""
    cache = {}
    if os.path.exists(cache_file_path):
        try:
            cached_data = load_jsonl(cache_file_path)
            for item in cached_data:
                item_hash = generate_item_hash(item)
                cache[item_hash] = item
            print(f"Loaded {len(cache)} items from cache")
        except Exception as e:
            print(f"Error loading cache: {e}")
            cache = {}
    else:
        print("Cache file does not exist, starting with empty cache")
    return cache


def process_item(item, index, cache):
    """处理单个数据项的函数"""
    item_hash = generate_item_hash(item)
    # 检查是否已在缓存中
    # 检查是否已在缓存中且结果有效
    if item_hash in cache:
        cached_item = cache[item_hash]
        if cached_item.get("output") is not None:
            print(f"Item {index} already in cache with valid results, skipping")
            return cached_item

    print(f"Processing item {index}")

    dialogue_history = "\n\n".join(
        [
            f"{utterance['role']}: {utterance['content']}"
            for utterance in item["dialogue_history"]
        ]
    )
    input_text = f"""Below is a dialogue history on the topic '{item["topic"]}':
{dialogue_history}
"""

    try:
        output_argument_analysis_result = response_argument_analysis(
            input_text + "\n\nassistant: " + item["output"],
            "assistant: " + item["output"],
        )

        # 验证结果
        cnt = 0
        while (
            not validate_argument_analysis_result(output_argument_analysis_result)
            and cnt < 10
        ):
            print(f"Item {index} output analysis result is invalid, retrying...")
            output_argument_analysis_result = response_argument_analysis(
                input_text + "\n\nassistant: " + item["output"],
                "assistant: " + item["output"],
            )
            cnt += 1
    except Exception as e:
        output_argument_analysis_result = None

    if output_argument_analysis_result:
        item["output_argument_analysis"] = output_argument_analysis_result
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


def format_conversation_context(conversation: List[Dict[str, str]]) -> str:
    """格式化对话上下文"""
    formatted = []
    for turn in conversation:
        formatted.append(f"{turn['role']}: {turn['content']}")
    return "\n".join(formatted)


def format_argument_analysis(argument_analysis, index):
    """
    将论证分析结果从字典格式转换为字符串格式

    Args:
        argument_analysis: 字典格式的论证分析结果，包含arguments列表

    Returns:
        str: 格式化后的字符串
    """
    if not argument_analysis:
        return ""

    try:
        # 如果输入是字符串，先解析为字典
        if isinstance(argument_analysis, str):
            analysis_dict = json.loads(argument_analysis)
        else:
            analysis_dict = argument_analysis

        # 提取arguments列表
        arguments = analysis_dict.get("arguments", [])

        if not arguments:
            return ""

        formatted_text = []
        for i, arg in enumerate(arguments, 1):
            claim = arg.get("Claim", "")
            evidence_list = arg.get("Evidence", [])

            # Format each argument
            arg_text = f"Claim {i}: {claim}"

            if evidence_list and evidence_list[0].lower() != "no evidence":
                evidence_text = "; ".join(evidence_list)
                arg_text += f"\nEvidence: {evidence_text}"
            else:
                arg_text += "\nEvidence: No evidence"

            formatted_text.append(arg_text)

        return "\n\n".join(formatted_text)

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error formatting argument analysis for item {index}: {e}")
        return str(argument_analysis) if argument_analysis else ""


def construct_one_step_data(data, test_mode=False):
    """
    构造两步数据：第一步是分析，第二步是基于分析结果生成
    """

    # 分析步骤的模板
    gen_with_ana_inst_template = """You are an experienced debater who can efficiently analyze the content of debate dialogue history and develop reasonable plans to accomplish debate-related tasks.


## Task Instruction:
{task_instruction}


## Input Text:
{argument_text}


## Thinking Workflow:
Step 1: Analyze the text's argument structure. Identify the central thesis or main claim, key evidence, and how counterarguments are addressed. Outline the main logical and reasoning flow.
Step 2: Identify the claims and all supporting evidences in the last utterance of the user.
Step 3: Plan the claims and supporting evidences in the assistant's next response to refute the user's stance.


## Output Format:
<Thinking>
[Your detailed analysis based on the Workflow]
</Thinking>
<Response>
[The Assistant's next response to refute the user's stance]
</Response>


## Output Text:
"""

    new_data = []

    for index, item in enumerate(data):
        # 从原始指令中提取任务指令和论证文本

        instruction = (
            f"Below is a debate dialogue history on the topic '{item['topic']}':\n"
            + format_conversation_context(item["dialogue_history"])
        )

        argument_text = instruction

        task_instruction = debate_dialogue_instruction

        if test_mode is False:
            analysis_result = (
                item.get("dialogue_argument_analysis", "")
                + "\n\n"
                + format_argument_analysis(
                    item.get("response_argument_analysis", ""), index
                )
                + "\n\n"
                + format_argument_analysis(
                    item.get("output_argument_analysis", ""), index
                )
            )

            gen_output = (
                "<Thinking>\n"
                + analysis_result
                + "\n</Thinking>\n<Response>\n"
                + item.get("output", "")
                + "\n</Response>"
            )
        else:
            gen_output = item.get("output", "")

        gen_instruction = gen_with_ana_inst_template.format(
            task_instruction=task_instruction,
            argument_text=argument_text,
        )

        generation_item = {
            "instruction": gen_instruction,
            "input": "",
            "output": gen_output,
        }
        new_data.append(generation_item)

    return new_data


def construct_two_step_data(data, test_mode=False):
    """
    构造两步数据：第一步是分析，第二步是基于分析结果生成
    """

    # 分析步骤的模板
    ana_inst_template = """Given the following task instruction and the corresponding argumentative dialogue text, conduct a detailed analytical reasoning process to prepare for generating the assistant’s next response. The analysis guidelines are provided at the end.


## Task Instruction:

{task_instruction}


## Input Text:

{argument_text}


## Analysis Guidelines:
**Argument Structure**:
- Analyze the text's argument structure. Identify the central thesis or main claim, key evidence, and how counterarguments are addressed. 
- Outline the main logical and reasoning flow.

**user’s Final Argument**:
- Identify the claims and all supporting evidences in the last utterance of the user.

**Planning the aistant’s Next Response**:
- Plan the claims and supporting evidences for the assistant's next response to refute the user's stance.


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

    for index, item in enumerate(data):
        # 从原始指令中提取任务指令和论证文本

        instruction = (
            f"Below is a debate dialogue history on the topic '{item['topic']}':\n"
            + format_conversation_context(item["dialogue_history"])
        )

        argument_text = instruction

        task_instruction = debate_dialogue_instruction

        # 第一步：分析数据
        ana_instruction = ana_inst_template.format(
            task_instruction=task_instruction, argument_text=argument_text
        )
        if test_mode is False:
            ana_output = (
                item.get("dialogue_argument_analysis", "")
                + "\n\n"
                + format_argument_analysis(
                    item.get("response_argument_analysis", ""), index
                )
                + "\n\n"
                + format_argument_analysis(
                    item.get("output_argument_analysis", ""), index
                )
            )
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
        if test_mode is False:
            analysis_result = (
                item.get("dialogue_argument_analysis", "")
                + "\n\n"
                + format_argument_analysis(
                    item.get("response_argument_analysis", ""), index
                )
                + "\n\n"
                + format_argument_analysis(
                    item.get("output_argument_analysis", ""), index
                )
            )
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
            "output": item.get("output", ""),
        }
        generation_data.append(generation_item)

    return analysis_data, generation_data


def replace_output_with_optimization_response(
    input_data_points, optimized_response_data
):
    # 将optimized_response_data转换为字典，以instruction为key
    optimized_response_data_dict = {}
    for item in optimized_response_data:
        instruction = item.get("instruction", "")
        optimized_response_data_dict[instruction] = item.get("optimized_response", "")

    # 遍历input_data_points，用optimized_response替换output
    for item in input_data_points:
        # 构造instruction（与optimized_response_data中的instruction格式一致）
        instruction = (
            f"Below is a debate dialogue history on the topic '{item['topic']}':\n"
            + format_conversation_context(item["dialogue_history"])
        )

        # 如果找到匹配的instruction，则替换output
        if instruction in optimized_response_data_dict:
            item["output"] = optimized_response_data_dict[instruction]
        else:
            print(
                f"Warning: No optimized response found for instruction: {instruction[:100]}..."
            )

    return input_data_points


def main(
    model_name: str,
    data_split: str,
    orig_input_fp: str,
    optimized_response_fp: str,
    output_dir: str,
) -> None:

    os.makedirs(output_dir, exist_ok=True)

    # Load input data and prompt instructions
    input_data_points = load_jsonl(orig_input_fp)
    print(f"Loaded {len(input_data_points)} items")

    optimized_response_data = load_jsonl(optimized_response_fp)
    print(f"Loaded {len(optimized_response_data)} items")

    input_data_points = replace_output_with_optimization_response(
        input_data_points, optimized_response_data
    )

    # 加载缓存
    cache_file_path = os.path.join(
        output_dir, f"ftcot_response_optimization_{data_split}.jsonl"
    )
    cache = load_cache(cache_file_path)

    # for index, item in enumerate(input_data_points):
    #     item = process_item(item, index, cache)
    #     break

    # 设置线程数，可以根据需要调整
    max_workers = os.cpu_count()  # 建议根据API限制和机器性能调整

    print(f"Starting processing with {max_workers} threads...")
    start_time = time.time()

    # 使用ThreadPoolExecutor进行多线程处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(process_item, item, i, cache): i
            for i, item in enumerate(input_data_points)
        }

        # 收集结果
        completed_count = 0
        failed_count = 0
        skipped_count = 0
        processed_data = [None] * len(input_data_points)

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                if result:
                    processed_data[index] = result
                    completed_count += 1

                    # 每处理10个item保存一次中间结果
                    if completed_count % 10 == 0:
                        print(
                            f"Progress: {completed_count}/{len(input_data_points)} items completed"
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
                                    f"ftcot_response_optimization_{data_split}.jsonl",
                                ),
                            )

            except Exception as e:
                print(f"Error processing item {index}: {e}")
                failed_count += 1

    # 过滤掉None值
    final_data = [item for item in processed_data if item is not None]

    # 保存最终结果
    output_file = os.path.join(
        output_dir, f"ftcot_response_optimization_{data_split}.jsonl"
    )
    save_results(final_data, output_file)

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    print(f"Successfully processed {completed_count}/{len(input_data_points)} items")
    print(f"Failed: {failed_count} items")


if __name__ == "__main__":
    model_name = "gpt-4o-mini-2024-07-18"

    # for data_split in ["train"]:
    #     orig_input_fp = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/ftcot/{data_split}/ftcot_{data_split}.jsonl"
    #     optimized_response_fp = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/response_optimization/{data_split}/response_optimization_{data_split}.jsonl"
    #     output_dir = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/ftcot_response_optimization/{data_split}"
    #     test_mode = False

    #     main(
    #         model_name,
    #         data_split,
    #         orig_input_fp,
    #         optimized_response_fp,
    #         output_dir,
    #     )

    for data_split in ["train", "test"]:
        if data_split == "train":
            data_path = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/ftcot_response_optimization/train/ftcot_response_optimization_train.jsonl"  # test_mode=False
        else:
            data_path = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/list_format/test_list.jsonl"  # test_mode=True

        if data_split == "train":
            test_mode = False
        else:
            test_mode = True

        print("Loading data...")
        data = load_jsonl(data_path)
        print(f"Loaded {len(data)} items")

        if test_mode:
            optimized_response_data_path = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/response_optimization/test/response_optimization_test.jsonl"
            optimized_response_data = load_jsonl(optimized_response_data_path)
            data = replace_output_with_optimization_response(
                data, optimized_response_data
            )

        # # 处理完成后进行后处理与保存
        new_data = construct_one_step_data(data, test_mode=test_mode)

        # 输出文件名：在原输出名基础上追加后缀
        base_out = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/ftcot_response_optimization/one_step/{data_split}.jsonl"
        os.makedirs(os.path.dirname(base_out), exist_ok=True)
        save_results(new_data, base_out)

        # # two-step处理完成后进行后处理与保存
        analysis_data, generation_data = construct_two_step_data(
            data, test_mode=test_mode
        )

        # 输出文件名：在原输出名基础上追加后缀
        base_out = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/ftcot_response_optimization/two_step_1/{data_split}.jsonl"
        os.makedirs(os.path.dirname(base_out), exist_ok=True)
        save_results(analysis_data, base_out)

        base_out = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/ftcot_response_optimization/two_step_2/{data_split}.jsonl"
        os.makedirs(os.path.dirname(base_out), exist_ok=True)
        save_results(generation_data, base_out)
