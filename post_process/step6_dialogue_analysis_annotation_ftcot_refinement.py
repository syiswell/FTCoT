import os
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


model_name = "gpt-4o-mini"
model = model_factory(model_name)


judge_instruction = """You are an expert evaluator for argument analysis. Your task is to evaluate the quality of an argument analysis result by checking three dimensions:

1. **Claim Identification Correctness**: Are the claims correctly identified? Each claim should be a clear, distinct core argument or main point from the text.

2. **Evidence Extraction Correctness**: Are the evidence items correctly extracted? Each evidence should be specific, relevant, and accurately represent supporting material from the text.

3. **Claim-Evidence Consistency**: Does each evidence item logically support its corresponding claim? There should be a clear connection between claims and their supporting evidence.

Given the original text and the argument analysis result, evaluate each dimension and return a binary pass/fail judgment for each check.

**Original Text**:
{original_text}

**Argument Analysis Result**:
{analysis_result}

**Evaluation**: For each dimension, answer "pass" or "fail" with a brief reason.
"""


judge_json_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "judgment_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "claim_identification_correct": {
                    "type": "string",
                    "enum": ["pass", "fail"],
                    "description": "Whether claim identification is correct",
                },
                "claim_identification_reason": {
                    "type": "string",
                    "description": "Reason for claim identification judgment",
                },
                "evidence_extraction_correct": {
                    "type": "string",
                    "enum": ["pass", "fail"],
                    "description": "Whether evidence extraction is correct",
                },
                "evidence_extraction_reason": {
                    "type": "string",
                    "description": "Reason for evidence extraction judgment",
                },
                "claim_evidence_consistency": {
                    "type": "string",
                    "enum": ["pass", "fail"],
                    "description": "Whether claim-evidence consistency is correct",
                },
                "claim_evidence_consistency_reason": {
                    "type": "string",
                    "description": "Reason for claim-evidence consistency judgment",
                },
            },
            "required": [
                "claim_identification_correct",
                "claim_identification_reason",
                "evidence_extraction_correct",
                "evidence_extraction_reason",
                "claim_evidence_consistency",
                "claim_evidence_consistency_reason",
            ],
            "additionalProperties": False,
        },
    },
}


refine_instruction = """You are an expert in argument analysis refinement. Given an argument analysis result that has been flagged for revision, along with the original text and context from other related analyses, refine the analysis to correct the identified issues.

**Original Text**:
{original_text}

**Current Argument Analysis Result** (needs refinement):
{current_analysis}

**Issues Identified**:
{issues}

**Refinement Guidelines**:
1. Correct any claim identification errors - ensure each claim is a clear, distinct core argument.
2. Correct any evidence extraction errors - ensure evidence is specific, relevant, and accurately extracted.
3. Ensure claim-evidence consistency - each evidence should logically support its claim.
4. Maintain the original format and structure of the analysis result.
5. Use the context from other analyses to ensure consistency and coherence.

Please provide the refined argument analysis result in the same format as the original.
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


@retry(
    wait=wait_random_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry_error_callback=lambda retry_state: None,
)
def judge_argument_analysis(
    original_text: str, analysis_result: str, element_name: str
):
    """
    判断论证分析结果的准确性

    Args:
        original_text: 原始文本
        analysis_result: 论证分析结果（可能是字符串或字典）
        element_name: 元素名称（用于区分不同类型的分析）

    Returns:
        dict: 包含三个维度判断结果的字典，如果判断失败返回None
    """
    try:
        # 如果analysis_result是字典，转换为字符串格式
        if isinstance(analysis_result, dict):
            analysis_str = json.dumps(analysis_result, ensure_ascii=False, indent=2)
        elif isinstance(analysis_result, str):
            # 尝试解析为JSON以验证格式
            try:
                json.loads(analysis_result)
                analysis_str = analysis_result
            except json.JSONDecodeError:
                analysis_str = analysis_result
        else:
            print(f"Warning: Invalid analysis_result type for {element_name}")
            return None

        # 构造判断提示
        prompt = judge_instruction.format(
            original_text=original_text, analysis_result=analysis_str
        )

        messages = [
            Message(
                role="system",
                content="You are an expert evaluator for argument analysis.",
            ),
            Message(role="user", content=prompt),
        ]
        print("judge_argument_analysis messages:", messages)
        response = model.generate_chat(
            messages=messages,
            temperature=0.0,
            response_format=judge_json_format,
        )
        print("judge_argument_analysis response:", response)

        # 解析响应
        if isinstance(response, str):
            judgment = json.loads(response)
        else:
            judgment = response

        return judgment

    except Exception as e:
        print(f"Error judging {element_name}: {e}")
        return None


@retry(
    wait=wait_random_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry_error_callback=lambda retry_state: None,
)
def refine_argument_analysis(
    original_text: str,
    current_analysis: str,
    issues: str,
    element_name: str,
):
    """
    对论证分析结果进行refinement

    Args:
        original_text: 原始文本
        current_analysis: 当前需要refine的分析结果
        issues: 识别出的问题描述
        element_name: 元素名称

    Returns:
        str或dict: 修正后的分析结果，如果失败返回None
    """
    try:
        # 格式化当前分析结果
        if isinstance(current_analysis, dict):
            current_analysis_str = json.dumps(
                current_analysis, ensure_ascii=False, indent=2
            )
        elif isinstance(current_analysis, str):
            current_analysis_str = current_analysis
        else:
            print(f"Warning: Invalid current_analysis type for {element_name}")
            return None

        # 构造refinement提示
        prompt = refine_instruction.format(
            original_text=original_text,
            current_analysis=current_analysis_str,
            issues=issues,
        )

        messages = [
            Message(
                role="system",
                content="You are an expert in argument analysis refinement.",
            ),
            Message(role="user", content=prompt),
        ]
        print("refine_argument_analysis messages:", messages)
        # 使用JSON格式要求返回修正后的结果
        if element_name in ["UA", "AA"]:
            refined_result = model.generate_chat(
                messages=messages,
                temperature=0.2,
                response_format=analyze_with_evidence_json_format,
            )
        else:
            refined_result = model.generate_chat(messages=messages, temperature=0.2)

        print("refine_argument_analysis response:", refined_result)
        return refined_result

    except Exception as e:
        print(f"Error refining {element_name}: {e}")
        return None


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


def iterative_refinement(
    item: dict,
    index: int,
    max_iter: int = 3,
):
    """
    迭代refinement循环

    Args:
        item: 包含三个分析结果的数据项
        index: 数据项索引
        max_iter: 最大迭代次数

    Returns:
        dict: 包含refined分析结果的数据项
    """
    # 准备原始文本
    dialogue_history = "\n\n".join(
        [
            f"{utterance['role']}: {utterance['content']}"
            for utterance in item["dialogue_history"]
        ]
    )
    input_text = f"""Below is a dialogue history on the topic '{item["topic"]}':
{dialogue_history}
"""

    last_user, last_utterance = (
        item["dialogue_history"][-1]["role"],
        item["dialogue_history"][-1]["content"],
    )

    # 初始化三个元素
    SU = item.get("dialogue_argument_analysis")  # 历史对话的论辩结构
    UA = item.get("response_argument_analysis")  # user的response的论点论据
    AA = item.get("output_argument_analysis")  # output的response的论点论据

    # 准备原始文本映射
    original_texts = {
        "SU": input_text,
        "UA": input_text,
        "AA": input_text + "\n\nassistant: " + item["output"],
    }

    # 跟踪已通过所有检查的元素（不再需要检查）
    passed_elements = set()

    iter = 0
    while iter < max_iter:
        print(f"Item {index} - Refinement iteration {iter + 1}/{max_iter}")
        flags = {}  # 跟踪需要revision的元素

        # 对每个元素进行检查（只检查尚未通过的元素）
        elements = {
            "SU": (SU, "Dialogue Structure Analysis"),
            "UA": (UA, "User Response Analysis"),
            "AA": (AA, "Assistant Output Analysis"),
        }

        for elem_key, (elem_value, elem_name) in elements.items():
            # 跳过已经通过所有检查的元素
            if elem_key in passed_elements:
                print(f"Item {index} - {elem_name} already passed all checks, skipping")
                continue

            if not elem_value:
                continue

            # 判断三个维度
            judgment = judge_argument_analysis(
                original_texts[elem_key], elem_value, elem_name
            )

            if judgment is None:
                print(f"Warning: Failed to judge {elem_name}, skipping refinement")
                continue

            # 检查是否有任何维度失败
            checks = [
                (
                    "claim_identification_correct",
                    judgment.get("claim_identification_correct"),
                ),
                (
                    "evidence_extraction_correct",
                    judgment.get("evidence_extraction_correct"),
                ),
                (
                    "claim_evidence_consistency",
                    judgment.get("claim_evidence_consistency"),
                ),
            ]

            failed_checks = [
                check_name
                for check_name, check_result in checks
                if check_result == "fail"
            ]

            if failed_checks:
                # 收集失败原因
                issues = []
                if "claim_identification_correct" in failed_checks:
                    issues.append(
                        f"Claim Identification: {judgment.get('claim_identification_reason', 'N/A')}"
                    )
                if "evidence_extraction_correct" in failed_checks:
                    issues.append(
                        f"Evidence Extraction: {judgment.get('evidence_extraction_reason', 'N/A')}"
                    )
                if "claim_evidence_consistency" in failed_checks:
                    issues.append(
                        f"Claim-Evidence Consistency: {judgment.get('claim_evidence_consistency_reason', 'N/A')}"
                    )

                flags[elem_key] = {
                    "element": elem_value,
                    "issues": "\n".join(issues),
                    "name": elem_name,
                }
                print(
                    f"Item {index} - {elem_name} flagged for refinement: {', '.join(failed_checks)}"
                )
            else:
                # 所有检查都通过，标记为已通过
                passed_elements.add(elem_key)
                print(
                    f"Item {index} - {elem_name} passed all checks, will skip in future iterations"
                )

        # 如果所有元素都通过，退出循环
        if not flags:
            print(
                f"Item {index} - All elements passed judgment, exiting refinement loop"
            )
            break

        # 对标记的元素进行refinement
        for elem_key, flag_info in flags.items():
            # 准备上下文（其他元素的分析结果）

            # 执行refinement
            refined_elem = refine_argument_analysis(
                original_texts[elem_key],
                flag_info["element"],
                flag_info["issues"],
                flag_info["name"],
            )

            if refined_elem:
                # 替换元素
                if elem_key == "SU":
                    SU = refined_elem
                elif elem_key == "UA":
                    UA = refined_elem
                elif elem_key == "AA":
                    AA = refined_elem
                print(f"Item {index} - Refined {flag_info['name']}")
            else:
                print(
                    f"Item {index} - Failed to refine {flag_info['name']}, keeping original"
                )

        iter += 1

    # 更新item
    item["dialogue_argument_analysis"] = SU
    item["response_argument_analysis"] = UA
    item["output_argument_analysis"] = AA

    return item


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


def process_item(item, index):
    """处理单个数据项的函数"""

    print(f"Processing item {index}")

    # 添加refinement步骤
    try:
        item = iterative_refinement(item, index, max_iter=3)
        print(f"Completed refinement for item {index}")
    except Exception as e:
        print(f"Error in refinement for item {index}: {e}")
        # 即使refinement失败，也返回原始结果

    print(f"Completed item {index}")
    return item


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


def main(
    model_name: str,
    data_split: str,
    orig_input_fp: str,
    output_dir: str,
) -> None:

    os.makedirs(output_dir, exist_ok=True)

    # Load input data and prompt instructions
    input_data_points = load_jsonl(orig_input_fp)
    print(f"Loaded {len(input_data_points)} items")

    for index, item in enumerate(input_data_points):
        item = process_item(item, index)
        break

    # 设置线程数，可以根据需要调整
    # max_workers = os.cpu_count()  # 建议根据API限制和机器性能调整

    # print(f"Starting processing with {max_workers} threads...")
    # start_time = time.time()

    # # 使用ThreadPoolExecutor进行多线程处理
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     # 提交所有任务
    #     future_to_index = {
    #         executor.submit(process_item, item, i): i
    #         for i, item in enumerate(input_data_points)
    #     }

    #     # 收集结果
    #     completed_count = 0
    #     failed_count = 0
    #     skipped_count = 0
    #     processed_data = [None] * len(input_data_points)

    #     for future in as_completed(future_to_index):
    #         index = future_to_index[future]
    #         try:
    #             result = future.result()
    #             if result:
    #                 processed_data[index] = result
    #                 completed_count += 1

    #                 # 每处理10个item保存一次中间结果
    #                 if completed_count % 10 == 0:
    #                     print(
    #                         f"Progress: {completed_count}/{len(input_data_points)} items completed"
    #                     )
    #                     # 保存已处理的结果
    #                     completed_items = [
    #                         item for item in processed_data if item is not None
    #                     ]
    #                     if completed_items:
    #                         save_results(
    #                             completed_items,
    #                             os.path.join(output_dir, f"ftcot_{data_split}.jsonl"),
    #                         )

    #         except Exception as e:
    #             print(f"Error processing item {index}: {e}")
    #             failed_count += 1

    # # 过滤掉None值
    # final_data = [item for item in processed_data if item is not None]

    # # 保存最终结果
    # output_file = os.path.join(output_dir, f"ftcot_{data_split}.jsonl")
    # save_results(final_data, output_file)

    # end_time = time.time()
    # print(f"Processing completed in {end_time - start_time:.2f} seconds")
    # print(f"Successfully processed {completed_count}/{len(input_data_points)} items")
    # print(f"Failed: {failed_count} items")


if __name__ == "__main__":
    model_name = "gpt-4o-mini-2024-07-18"

    for data_split in ["train"]:
        orig_input_fp = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/ftcot/{data_split}/ftcot_{data_split}.jsonl"
        output_dir = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/ftcot/{data_split}_refine"
        test_mode = False

        main(
            model_name,
            data_split,
            orig_input_fp,
            output_dir,
        )

    # for data_split in ["train", "test"]:
    #     if data_split == "train":
    #         data_path = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/ftcot/train/ftcot_train.jsonl"  # test_mode=False
    #     else:
    #         data_path = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/list_format/test_list.jsonl"  # test_mode=True

    #     print("Loading data...")
    #     data = load_jsonl(data_path)
    #     print(f"Loaded {len(data)} items")

    #     if data_split == "train":
    #         test_mode = False
    #     else:
    #         test_mode = True

    #     # # 处理完成后进行后处理与保存
    #     new_data = construct_one_step_data(data, test_mode=test_mode)

    #     # 输出文件名：在原输出名基础上追加后缀
    #     base_out = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/ftcot/one_step/{data_split}.jsonl"
    #     os.makedirs(os.path.dirname(base_out), exist_ok=True)
    #     save_results(new_data, base_out)

    #     # # two-step处理完成后进行后处理与保存
    #     analysis_data, generation_data = construct_two_step_data(
    #         data, test_mode=test_mode
    #     )

    #     # 输出文件名：在原输出名基础上追加后缀
    #     base_out = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/ftcot/two_step_1/{data_split}.jsonl"
    #     os.makedirs(os.path.dirname(base_out), exist_ok=True)
    #     save_results(analysis_data, base_out)

    #     base_out = f"/home/sunyang/hlt/new_cmv_dataset/final_dataset/post_process/ftcot/two_step_2/{data_split}.jsonl"
    #     os.makedirs(os.path.dirname(base_out), exist_ok=True)
    #     save_results(generation_data, base_out)
