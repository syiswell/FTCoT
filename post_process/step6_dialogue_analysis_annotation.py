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


system_instruction = "You are an experienced debater who can efficiently analyze the content of debate dialogue history and develop reasonable plans to accomplish debate-related tasks."

debate_dialogue_instruction = """You are an assistant in a debate dialogue. The user presents a strong opinion on a specific topic. Your goal is to respond persuasively and logically — challenging their stance with counterarguments, context, and ethical or emotional reasoning."""


model_name = "gpt-4o-mini"
model = model_factory(model_name)
prompt_inst_path = "/home/sunyang/hlt/new_cmv_dataset/post_process/argument_analysis/analysis_prompt.txt"
prompt_inst = load_text(prompt_inst_path)


@retry(
    wait=wait_random_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry_error_callback=lambda retry_state: None,
)
def argument_analysist(input_query):

    messages = [
        Message(role="system", content=system_instruction),
        Message(role="user", content=input_query),
    ]
    # print("messages:", messages)
    response = model.generate_chat(
        messages=messages,
        temperature=0.2,
    )
    # print("response:", response)

    return response


# 线程锁，用于保护共享资源
lock = threading.Lock()


def process_item(item, index):
    """处理单个数据项的函数"""
    print(f"Processing item {index}")

    try:
        input_query = prompt_inst.format(
            arg_text=item["instruction"], task_inst=debate_dialogue_instruction
        )
        ev = argument_analysist(input_query)
    except Exception as e:
        ev = None

    if ev:
        item["argument_analysis"] = ev
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


def construct_one_step_data(data, test_mode=False):
    """
    构造两步数据：第一步是分析，第二步是基于分析结果生成
    """
    import re
    from typing import List, Dict

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

    for item in data:
        # 从原始指令中提取任务指令和论证文本

        argument_text = item["instruction"]
        task_instruction = debate_dialogue_instruction

        analysis_result = item.get("argument_analysis", "")
        gen_instruction = gen_with_ana_inst_template.format(
            task_instruction=task_instruction,
            argument_text=argument_text,
        )
        if test_mode is False:
            gen_output = (
                "<Thinking>\n"
                + analysis_result
                + "\n</Thinking>\n<Response>\n"
                + item.get("output", "")
                + "\n</Response>"
            )
        else:
            gen_output = item.get("output", "")
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
    import re
    from typing import List, Dict

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

    for item in data:
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
            "output": item.get("output", ""),
        }
        generation_data.append(generation_item)

    return analysis_data, generation_data


def main(
    model_name: str,
    data_split: str,
    orig_input_fp: str,
    prompt_inst_path: str,
    output_dir: str,
) -> None:

    os.makedirs(output_dir, exist_ok=True)

    # Load input data and prompt instructions
    input_data_points = load_jsonl(orig_input_fp)
    prompt_inst = load_text(prompt_inst_path)

    print(f"Loaded {len(input_data_points)} items")

    # for index, item in enumerate(input_data_points):
    #     item = process_item(item, index)

    # 设置线程数，可以根据需要调整
    max_workers = os.cpu_count()  # 建议根据API限制和机器性能调整

    print(f"Starting processing with {max_workers} threads...")
    start_time = time.time()

    # 使用ThreadPoolExecutor进行多线程处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(process_item, item, i): i
            for i, item in enumerate(input_data_points)
        }

        # 收集结果
        completed_count = 0
        failed_count = 0
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
                                    output_dir, f"argument_analysis_{data_split}.jsonl"
                                ),
                            )

            except Exception as e:
                print(f"Error processing item {index}: {e}")
                failed_count += 1

    # 过滤掉None值
    final_data = [item for item in processed_data if item is not None]

    # 保存最终结果
    output_file = os.path.join(output_dir, f"argument_analysis_{data_split}.jsonl")
    save_results(final_data, output_file)

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    print(f"Successfully processed {completed_count}/{len(input_data_points)} items")
    print(f"Failed: {failed_count} items")


if __name__ == "__main__":
    model_name = "gpt-4o-mini-2024-07-18"

    # for data_split in ["train"]:
    #     orig_input_fp = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/instruction_following_format/{data_split}.jsonl"
    #     prompt_inst_path = "/home/sunyang/hlt/new_cmv_dataset/post_process/argument_analysis/analysis_prompt.txt"
    #     output_dir = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/gpt_analysis/{data_split}"
    #     test_mode = False

    #     main(
    #         model_name,
    #         data_split,
    #         orig_input_fp,
    #         prompt_inst_path,
    #         output_dir,
    #     )

    for data_split in ["train", "test"]:
        if data_split == "train":
            data_path = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/gpt_analysis/train/argument_analysis_train.jsonl"  # test_mode=False
        else:
            data_path = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/instruction_following_format/test.jsonl"  # test_mode=True

        print("Loading data...")
        data = load_jsonl(data_path)
        print(f"Loaded {len(data)} items")

        if data_split == "train":
            test_mode = False
        else:
            test_mode = True

        # # 处理完成后进行后处理与保存
        new_data = construct_one_step_data(data, test_mode=test_mode)

        # 输出文件名：在原输出名基础上追加后缀
        base_out = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/gpt_analysis/one_step/{data_split}.jsonl"
        os.makedirs(os.path.dirname(base_out), exist_ok=True)
        save_results(new_data, base_out)

        # # two-step处理完成后进行后处理与保存
        # analysis_data, generation_data = construct_two_step_data(data, test_mode=test_mode)

        # # 输出文件名：在原输出名基础上追加后缀
        # base_out = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/gpt_analysis/two_step_1/{data_split}.jsonl"
        # os.makedirs(os.path.dirname(base_out), exist_ok=True)
        # save_results(analysis_data, base_out)

        # base_out = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/gpt_analysis/two_step_2/{data_split}.jsonl"
        # os.makedirs(os.path.dirname(base_out), exist_ok=True)
        # save_results(generation_data, base_out)

    # # data_path = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/gpt_analysis/train/argument_analysis_train.jsonl" # test_mode=False
    # data_path = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/instruction_following_format/test.jsonl"  # test_mode=True

    # print("Loading data...")
    # data = load_jsonl(data_path)
    # print(f"Loaded {len(data)} items")

    # # # 处理完成后进行后处理与保存
    # new_data = construct_one_step_data(data, test_mode=True)

    # # 输出文件名：在原输出名基础上追加后缀
    # # base_out = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/gpt_analysis/one_step/train.jsonl"
    # base_out = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/gpt_analysis/one_step/test.jsonl"
    # os.makedirs(os.path.dirname(base_out), exist_ok=True)

    # save_results(new_data, base_out)

    # # two-step处理完成后进行后处理与保存
    # analysis_data, generation_data = construct_two_step_data(data, test_mode=True)

    # # 输出文件名：在原输出名基础上追加后缀
    # # base_out = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/gpt_analysis/two_step_1/train.jsonl"
    # base_out = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/gpt_analysis/two_step_1/test.jsonl"
    # os.makedirs(os.path.dirname(base_out), exist_ok=True)
    # save_results(analysis_data, base_out)

    # # base_out = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/gpt_analysis/two_step_2/train.jsonl"
    # base_out = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/gpt_analysis/two_step_2/test.jsonl"
    # os.makedirs(os.path.dirname(base_out), exist_ok=True)
    # save_results(generation_data, base_out)
