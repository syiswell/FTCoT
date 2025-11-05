import json
import os, sys
import re
from typing import List, Dict
sys.path.append('.')
from my_src_refactored.utils import load_json, load_jsonl, save_json

def prepare_ref_data(input_data_points: List[Dict], ref_input: List[str], ref_output: List[str]) -> List[Dict]:


    ref_inst_template = \
'''Given the following task instruction, the corresponding argumentative input text, and a candidate response, your task is to first analyze the argument structure and quality of the candidate response. Then, evaluate whether the candidate response effectively address the task, and provide suggestions for improvement. The analysis guidelines are provided at the end.


## Task Instruction:

{task_instruction}


## Input Text:

{argument_text}


## Candidate Response:

{candidate_response}


## Analysis Guidelines:

**Argument Structure**:
    - Analyze the candidate response's argument structure. Identify the central thesis or main claim, key evidence, and how counterarguments are addressed.
    - Outline the main logical and reasoning flow.
**Argument Quality**:
    - Highlight the main strengths and primary weaknesses.
    - Overall Evaluation: Assess the overall quality and persuasiveness of the candidate response, and provide an overall rating as an integer between 1 (lowest) to 5 (highest).
**Task Effectiveness**:
    - Assess how well the candidate response addresses the task instruction. Highlight any areas where the response falls short or excels.
    - Give suggestions to enhance the candidate response to better address the task.
**Note**: 
    - You only need to analyze the following text according to the above requirements. You do not need to actually complete the task specified in the **Task Instruction**.
    - If the candidate response is too short or insufficient to support a complete structure and quality analysis, then these two analyses should not be conducted. Simply provide a brief explanation of the reason. Then, proceed to other analyses.
    - Throughout your analysis, provide specific examples from the candidate response and the argumentative text to support your judgments.
    - Specifically, for the Argument Summarization task and the Sentence-Level Counter-Argument Generation task, the expected responses are inherently brief. Therefore, when evaluating candidate responses, do not dismiss them solely based on their brevity.
    - Please try to give a response of about 100 words.


## Output Text:
'''


    ref_data = []
    for d, ref_in, ref_out in zip(input_data_points, ref_input, ref_output):
        orig_input_text = d['instruction']
        task_instruction_match = re.search(r"## Task Instruction:\n\n(.+?)\n\n\n##", orig_input_text, re.DOTALL)
        argument_text_match = re.search(r"## Input Text:\n\n(.+?)\n\n\n##", orig_input_text, re.DOTALL)
        argument_text = argument_text_match.group(1).strip() if argument_text_match else None
        task_instruction = task_instruction_match.group(1).strip() if task_instruction_match else None
        assert argument_text is not None and task_instruction is not None

        # # check if the argument text is the same
        # arg_match_from_orig = re.search(r"Argument:\n\n(.+)", argument_text, re.DOTALL)
        # arg_from_orig = arg_match_from_orig.group(1).strip() if arg_match_from_orig else None
        # arg_match_from_batchinput = re.search(r"Argument:\n(.+)\n\n\n\*\*", ref_in, re.DOTALL)
        # arg_from_batchinput = arg_match_from_batchinput.group(1).strip() if arg_match_from_batchinput else None
        # assert arg_from_orig is not None and arg_from_batchinput is not None
        # assert arg_from_orig == arg_from_batchinput


        candidate_response_match = re.search(r"\*\*Candidate Response\*\*:\n\n(.+)\n\n```", ref_in, re.DOTALL)
        candidate_response = candidate_response_match.group(1).strip() if candidate_response_match else None
        assert candidate_response is not None

        ref_data.append({
            "instruction": ref_inst_template.format(task_instruction=task_instruction, argument_text=argument_text, candidate_response=candidate_response),
            "input": "",
            "output": ref_out
        })
    return ref_data

def save_ref_data(data_split: str, ref_data: List[Dict], save_dir: str) -> None:
    sample_dir = os.path.join(save_dir, 'sample_data')
    os.makedirs(sample_dir, exist_ok=True)
    if data_split == "train":
        save_file_name = f"ref-train"
    elif data_split == "test":
        save_file_name = f"ref-test-w_placeholder"
    else:
        raise ValueError(f"Invalid data split: {data_split}")
    save_json(ref_data, f"{save_dir}/{save_file_name}.json")
    save_json(ref_data[:10], f"{sample_dir}/{save_file_name}-sample.json")

def prepare_optimize_data(dataset_name: str, data_split: str, ref_data: List[Dict], ref_output: List[str], optimized_outputs: List[str]) -> List[Dict]:
    
    optimize_template = \
'''Given the following task instruction, the corresponding argumentative input text, the corresponding candidate response, and the self-reflective analysis result of the candidate response, your task is to optimize the candidate response based on the self-reflective analysis result.


## Task Instruction:

{task_instruction}


## Input Text:

{argument_text}


## Candidate Response:

{candidate_response}


## Self-reflective Analysis:

{reflective_analysis}


## Output Text:
'''

    optimize_data_list = []
    for ref_in, ref_out, optimized_output in zip(ref_data, ref_output, optimized_outputs):
        if dataset_name == "ADSP" and (not optimized_output.startswith("Agent:")):
            optimized_output = "Agent: " + optimized_output
        ref_in = ref_in['instruction']

        task_instruction_match = re.search(r"## Task Instruction:\n\n(.+?)\n\n\n## Input Text:", ref_in, re.DOTALL)
        argument_text_match = re.search(r"## Input Text:\n\n(.+?)\n\n\n## Candidate Response:", ref_in, re.DOTALL)
        argument_text = argument_text_match.group(1).strip() if argument_text_match else None
        task_instruction = task_instruction_match.group(1).strip() if task_instruction_match else None
        candidate_response_match = re.search(r"## Candidate Response:\n\n(.+?)\n\n\n## Analysis Guidelines:", ref_in, re.DOTALL)
        candidate_response = candidate_response_match.group(1).strip() if candidate_response_match else None
        
        assert argument_text is not None and task_instruction is not None and candidate_response is not None
        
        reflective_analysis = ref_out
        
        if data_split == "train":   
            inst = optimize_template.format(
                task_instruction=task_instruction,
                argument_text=argument_text,
                candidate_response=candidate_response,
                reflective_analysis=reflective_analysis
            )
            if dataset_name == "ADSP":
                pattern = r'(Selected Facts:.*?Response:)'
                match = re.search(pattern, candidate_response, re.DOTALL)
                if match is not None:
                    selected_facts = match.group(0)
                    optimized_output = selected_facts + '\n' + optimized_output
                else:
                    optimized_output = "Selected Facts:\n\n\nResponse:\n" + optimized_output
                    print(f"Can not parse selected facts from candidate response: {candidate_response}")
        else:
            inst = optimize_template.format(
                task_instruction=task_instruction,
                argument_text=argument_text,
                candidate_response="{{candidate_response}}",
                reflective_analysis="{{reflective_analysis}}"
            )
        
        output = optimized_output
        data_dict = {
            "instruction": inst,
            "input": "",
            "output": output
        }
        optimize_data_list.append(data_dict)
    
    return optimize_data_list

def save_optimize_data(data_split: str, optimize_data_list: List[Dict], save_dir: str) -> None:
    sample_dir = os.path.join(save_dir, 'sample_data')
    os.makedirs(sample_dir, exist_ok=True)
    if data_split == "train":
        save_file_name = f"gen_with_ref-train"
    elif data_split == "test":
        save_file_name = f"gen_with_ref-test-w_placeholder"
    else:
        raise ValueError(f"Invalid data split: {data_split}")
    save_json(optimize_data_list, f"{save_dir}/{save_file_name}.json")
    save_json(optimize_data_list[:10], f"{sample_dir}/{save_file_name}-sample.json")

def main(dataset_name: str, data_split: str, orig_data_fp: str, save_dir: str, ref_input_fp: str, ref_output_fp: str, optimized_output_fp: str) -> None:
    print(f"Processing {dataset_name} {data_split} data...")

    # Load data
    input_data_points = load_json(orig_data_fp)

    with open(ref_input_fp, "r") as f:
        ref_input = [json.loads(line) for line in f]
        ref_input = [d['body']['messages'][1]['content'] for d in ref_input]
    
    if data_split == "train":
        with open(ref_output_fp, "r") as f:
            ref_output = [json.loads(line) for line in f]
            ref_output = [d['response']['body']['choices'][0]['message']['content'] for d in ref_output]
    else:
        ref_output = [""] * len(ref_input)

    # Prepare and save ref data
    ref_data = prepare_ref_data(input_data_points, ref_input, ref_output)
    save_ref_data(data_split,ref_data, save_dir)

    # Load optimized outputs
    if data_split == "train":
        with open(optimized_output_fp, "r") as f:
            optimized_outputs = [json.loads(line) for line in f]
            optimized_outputs = [d['response']['body']['choices'][0]['message']['content'] for d in optimized_outputs]
            optimized_outputs = [json.loads(d)['optimized_response'] for d in optimized_outputs]
    else:
        optimized_outputs = [""] * len(ref_input)

    assert len(ref_input) == len(ref_output) == len(optimized_outputs)

    # Prepare and save optimize data
    optimize_data_list = prepare_optimize_data(dataset_name, data_split, ref_data, ref_output, optimized_outputs)
    save_optimize_data(data_split, optimize_data_list, save_dir)

    print(f"Saved {len(ref_data)} {data_split} data to {save_dir}")

if __name__ == "__main__":
    for dataset_name in ["ArgTersely", "CAG_CMV", "AEG", "DebateSum", "ADSP"]:
        for data_split in ["train", "test"]:
            ref_input_fp = f"my_scripts/scripts_with_ADSP/new_format_data/gen_with_ana-epoch_2/output/predict/{data_split}/step_2-gen_with_ana/self_reflection/{dataset_name}/batchinput.jsonl"
            if data_split == "train":
                ref_output_fp = f"my_scripts/scripts_with_ADSP/new_format_data/gen_with_ana-epoch_2/output/predict/{data_split}/step_2-gen_with_ana/self_reflection/{dataset_name}/batchoutput.jsonl"
                optimized_output_fp = f"my_scripts/scripts_with_ADSP/new_format_data/gen_with_ana-epoch_2/output/predict/{data_split}/step_2-gen_with_ana/self_reflection/{dataset_name}/second_batchoutput.jsonl"
            elif data_split == "test":
                ref_output_fp = None
                optimized_output_fp = None
            save_dir = f"data/my_data/datasets/sft_dataset_5k/{dataset_name}/generate_with_analysis_and_reflection"
            orig_data_fp = f"data/my_data/datasets/sft_dataset_5k/{dataset_name}/{data_split}.json"
            main(dataset_name, data_split, orig_data_fp, save_dir, ref_input_fp, ref_output_fp, optimized_output_fp)