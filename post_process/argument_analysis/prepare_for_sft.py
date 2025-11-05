import sys
import os
import json

sys.path.append(".")
from my_src_refactored.utils import load_json, load_jsonl, save_json, load_text
from get_gpt_analysis_res import prepare_batch_data

import re
from typing import List, Dict, Optional


def process_analysis_data(
    input_data_points: List[Dict],
    batch_input_data: List[Dict],
    ana_output_fp: Optional[str],
) -> List[Dict]:

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
**Note**: 
    - You only need to analyze the following text according to the above requirements. You do not need to actually complete the task specified in the **Task Instruction**.
    - If the given argumentative text is too short or insufficient to support a complete structure and quality analysis, then these two analyses should not be conducted. Simply provide a brief explanation of the reason. Then, proceed to analyze only the key points to consider when solving the task.
    - Throughout your analysis, provide specific examples from the argumentative text to support your judgments.
    - Please try to give a response of about 300 words.


## Output Text:
"""

    ana_input_list = []
    for d in input_data_points:
        orig_input_text = d["instruction"]
        task_instruction_match = re.search(
            r"## Task Instruction:\n\n(.+?)\n\n\n##", orig_input_text, re.DOTALL
        )
        argument_text_match = re.search(
            r"## Input Text:\n\n(.+?)\n\n\n##", orig_input_text, re.DOTALL
        )
        # task_instruction_match = re.search(r"\*\*Task Instruction\*\*:\n\n(.+?)\n\n\n", text, re.DOTALL)
        # argument_text_match = re.search(r"\*\*Argumentative Text\*\*:\n\n(.+)$", text, re.DOTALL)
        argument_text = (
            argument_text_match.group(1).strip() if argument_text_match else None
        )
        task_instruction = (
            task_instruction_match.group(1).strip() if task_instruction_match else None
        )
        assert argument_text is not None and task_instruction is not None

        ana_input_list.append(
            ana_inst_template.format(
                task_instruction=task_instruction, argument_text=argument_text
            )
        )

    ana_output_list: List[str] = []
    if ana_output_fp is not None:
        ana_output_list = [
            d["response"]["body"]["choices"][0]["message"]["content"]
            for d in load_jsonl(ana_output_fp)
        ]
        assert len(ana_input_list) == len(
            ana_output_list
        ), "the length of ana_input and ana_output should be the same"

    return [
        {
            "instruction": ana_in,
            "input": "",
            "output": ana_out if ana_output_fp is not None else "",
        }
        for ana_in, ana_out in zip(ana_input_list, ana_output_list or ana_input_list)
    ]


def prepare_generation_data(
    with_placeholder: bool,
    ana_input: List[str],
    ana_output: List[str],
    orig_data: List[Dict],
    dataset_name: str,
) -> List[Dict]:
    output_start_prompt_dict = {
        "ADSP": "Selected Facts and Response",
        "ArgTersely": "Counter-argument",
        "CAG_CMV": "Counter-argument",
        "AEG": "Essay",
        "DebateSum": "Summary",
    }

    gen_with_ana_inst_template = """Given the following task instruction, the corresponding input text, and the analysis result of the input text, your need to complete the task according to the analysis result.


## Task Instruction:

{task_instruction}


## Input Text:

{argument_text}


## Analysis of the Input Text:

{analysis_result}


## Output Text:
"""

    gen_with_ana_data_list = []
    for ana_in, ana_out, orig_data_item in zip(ana_input, ana_output, orig_data):
        text = ana_in
        task_instruction_match = re.search(
            r"## Task Instruction:\n\n(.+?)\n\n\n##", text, re.DOTALL
        )
        argument_text_match = re.search(
            r"## Input Text:\n\n(.+?)\n\n\n## Analysis Guidelines:", text, re.DOTALL
        )
        # task_instruction_match = re.search(r"\*\*Task Instruction\*\*:\n\n(.+?)\n\n\n", text, re.DOTALL)
        # argument_text_match = re.search(r"\*\*Argumentative Text\*\*:\n\n(.+)$", text, re.DOTALL)
        argument_text = (
            argument_text_match.group(1).strip() if argument_text_match else None
        )
        task_instruction = (
            task_instruction_match.group(1).strip() if task_instruction_match else None
        )
        assert argument_text is not None and task_instruction is not None

        gen_with_ana_inst = gen_with_ana_inst_template.format(
            task_instruction=task_instruction,
            argument_text=argument_text,
            analysis_result="{{analysis_result}}" if with_placeholder else ana_out,
            output_start_prompt=output_start_prompt_dict[dataset_name],
        )

        gen_with_ana_data = {
            "instruction": gen_with_ana_inst,
            "input": "",
            "output": orig_data_item["output"],
        }
        gen_with_ana_data_list.append(gen_with_ana_data)

    return gen_with_ana_data_list


def main(
    dataset_name: str,
    data_split: str,
    orig_input_fp: str,
    prompt_inst_path: str,
    ana_output_fp: str,
    save_dir: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # Load input data and prompt instructions
    input_data_points = load_json(orig_input_fp)
    prompt_inst = load_text(prompt_inst_path)

    # Prepare batch data
    batch_input_data = prepare_batch_data(
        input_data_points,
        prompt_inst,
        dataset_name,
        data_split,
        "gpt-4o-mini-2024-07-18",
    )

    # Debug
    # if True:
    #     if ana_output_fp is not None:
    #         ana_input_fp = os.path.join(os.path.dirname(ana_output_fp), "batch_input.jsonl")
    #         if os.path.exists(ana_input_fp) and dataset_name != "ADSP":
    #             ana_input_data = load_jsonl(ana_input_fp)
    #             assert len(batch_input_data) == len(ana_input_data)
    #             for b_in, a_in in zip(batch_input_data, ana_input_data):
    #                 assert json.dumps(b_in) == json.dumps(a_in)
    #             print("Batch input data and previous batch input data are the same")

    # Process analysis data
    ana_data = process_analysis_data(input_data_points, batch_input_data, ana_output_fp)
    save_json(ana_data, f"{save_dir}/ana-{data_split}.json")
    os.makedirs(f"{save_dir}/sample_data", exist_ok=True)
    save_json(ana_data[:10], f"{save_dir}/sample_data/ana-{data_split}-sample.json")

    # Prepare generation data
    orig_data = load_json(orig_input_fp)
    ana_input = [d["instruction"] for d in ana_data]
    ana_output = [d["output"] for d in ana_data]

    os.makedirs(f"{save_dir}/sample_data", exist_ok=True)
    if data_split == "test":
        gen_with_ana_data_list = prepare_generation_data(
            with_placeholder=True,
            ana_input=ana_input,
            ana_output=ana_output,
            orig_data=orig_data,
            dataset_name=dataset_name,
        )
        save_json(
            gen_with_ana_data_list,
            f"{save_dir}/gen_with_ana-{data_split}-w_placeholder.json",
        )
        save_json(
            gen_with_ana_data_list[:10],
            f"{save_dir}/sample_data/gen_with_ana-{data_split}-w_placeholder-sample.json",
        )
    elif data_split == "train":
        gen_with_ana_data_list = prepare_generation_data(
            with_placeholder=False,
            ana_input=ana_input,
            ana_output=ana_output,
            orig_data=orig_data,
            dataset_name=dataset_name,
        )
        save_json(gen_with_ana_data_list, f"{save_dir}/gen_with_ana-{data_split}.json")
        save_json(
            gen_with_ana_data_list[:10],
            f"{save_dir}/sample_data/gen_with_ana-{data_split}-sample.json",
        )
        gen_with_ana_data_list = prepare_generation_data(
            with_placeholder=True,
            ana_input=ana_input,
            ana_output=ana_output,
            orig_data=orig_data,
            dataset_name=dataset_name,
        )
        save_json(
            gen_with_ana_data_list,
            f"{save_dir}/gen_with_ana-{data_split}-for_infer-w_placeholder.json",
        )
        save_json(
            gen_with_ana_data_list[:10],
            f"{save_dir}/sample_data/gen_with_ana-{data_split}-for_infer-w_placeholder-sample.json",
        )
    else:
        raise ValueError(f"Invalid data split: {data_split}")


if __name__ == "__main__":
    prompt_inst_path = "my_src_refactored/gpt_analysis/analysis_prompt.txt"

    for dataset_name in ["ArgTersely", "CAG_CMV", "AEG", "DebateSum", "ADSP"]:
        for data_split in ["train", "test"]:
            orig_input_fp = (
                f"data/my_data/datasets/sft_dataset_5k/{dataset_name}/{data_split}.json"
            )
            if data_split == "train":
                ana_output_fp = f"data/my_data/intermediate_data/gpt_analysis/sft_dataset_5k/{dataset_name}/{data_split}/batch_output.jsonl"
            elif data_split == "test":
                ana_output_fp = None
            save_dir = f"data/my_data/datasets/sft_dataset_5k/{dataset_name}/generate_with_analysis"
            print(f"Processing {dataset_name} {data_split}")
            main(
                dataset_name,
                data_split,
                orig_input_fp,
                prompt_inst_path,
                ana_output_fp,
                save_dir,
            )

    # Upload to Hugging Face
    # os.system("huggingface-cli upload benjpau/arg_gen data/my_data/datasets/sft_dataset_5k sft_dataset_5k --repo-type dataset")
