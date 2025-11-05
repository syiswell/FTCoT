import json
import os
import sys
import time
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
sys.path.append('.')
from my_src_refactored.utils import load_json, load_text, load_jsonl, save_jsonl

def prepare_first_batch_data(input_data: List[Dict], output_data: List[Dict], prompt_inst: str, dataset_name: str, data_split: str, model_name: str, word_cnt: int) -> List[Dict]:
    batch_data = []
    for i, (dp, outputs) in enumerate(zip(input_data, output_data)):
        arg_text = '\n\n\n'.join(dp['instruction'].split('\n\n\n')[1:-1])
        task_inst = dp['instruction'].split('\n\n\n')[0]
        try:
            pred = outputs['predict']
        except:
            pred = "{{candidate_response}}"
        ref_resp = dp['output']
        input_query = prompt_inst.format(arg_text=arg_text, task_inst=task_inst, cand_resp=pred, ref_resp=ref_resp, word_cnt=word_cnt)
        idx = f"{dataset_name}_{data_split}_{i:05d}"
        
        batch_item = {
            "custom_id": idx,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are an experienced debater who can efficiently analyze the content of argumentative texts and develop reasonable plans to accomplish debate-related tasks."},
                    {"role": "user", "content": input_query}
                ],
                # "max_tokens": 1000
            }
        }
        batch_data.append(batch_item)
    return batch_data

def prepare_second_batch_data(first_batch_data: List[Dict], first_output_data: List[Dict], dataset_name: str, data_split: str, model_name: str, second_inst: str) -> List[Dict]:
    second_batch_data = []
    for i, (inputs, outputs) in enumerate(zip(first_batch_data, first_output_data)):
        input_text = inputs['body']['messages'][1]['content']
        try:
            output_text = outputs['response']['body']['choices'][0]['message']['content']
        except:
            output_text = ""
        input_query = \
'''Now, based on your analysis, please optimize this candidate response to better address the task instruction.
{second_inst}
Please provide your response in the following JSON format:
{{
    "optimized_response": "...",
    "any_additional_information": "..."
}}
'''
        input_query = input_query.format(second_inst=second_inst)
        idx = f"{dataset_name}_{data_split}_second_request_{i:05d}"
        
        batch_item = {
            "custom_id": idx,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are an experienced debater who can efficiently analyze the content of argumentative texts and develop reasonable plans to accomplish debate-related tasks."},
                    {"role": "user", "content": input_text},
                    {"role": "assistant", "content": output_text},
                    {"role": "user", "content": input_query}
                ],
                "response_format": {"type": "json_object"},
                # "max_tokens": 1000
            }
        }
        second_batch_data.append(batch_item)
    return second_batch_data

def create_batch_job(client: OpenAI, input_file_path: str, dataset_name: str, file_name: str) -> Dict:
    batch_file_info = client.files.create(
        file=open(input_file_path, "rb"),
        purpose="batch"
    )

    batch_job_info = client.batches.create(
        input_file_id=batch_file_info.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
          "description": f"Task: {dataset_name} {file_name}, nightly eval job"
        }
    )

    return batch_job_info

def wait_for_batch_completion(client: OpenAI, batch_id: str) -> Dict:
    while True:
        check_info = client.batches.retrieve(batch_id)
        print(check_info)
        if check_info.status == "completed":
            return check_info
        time.sleep(10)

def main(dataset_name: str, data_split: str, sample_test: bool = False):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    client = OpenAI(api_key=api_key)

    dataset_name2word_cnt = {
        'AEG': 300,
        'ArgTersely': 100,
        'CAG_CMV': 300,
        'DebateSum': 50,
        'ADSP': 100,
    }
    dataset_name2second_inst = {
        'AEG': '',
        'ArgTersely': '**Note**: Ensure your optimized response does not exceed 40 words.',
        'CAG_CMV': '',
        'DebateSum': '**Note**: Ensure your optimized response does not exceed 20 words.',
        'ADSP': '**Note**: Only optimize the response, not the selected facts. Ensure that all content in the optimized response comes exclusively from the selected facts, **do not add any new information**. Additionally, ensure your optimized response does not exceed 80 words.',
    }
    model_name = "gpt-4o-mini-2024-07-18"
    prompt_template_fp = 'my_src_refactored/gpt_self_reflection/self_reflection_prompt.txt'

    prompt_inst = load_text(prompt_template_fp)

    word_cnt = dataset_name2word_cnt[dataset_name]
    input_data_fp = f'data/my_data/datasets/sft_dataset_5k/{dataset_name}/{data_split}.json'
    if data_split == "train":
        output_fp = f'my_scripts/scripts_with_ADSP/new_format_data/gen_with_ana-epoch_2/output/predict/train/step_2-gen_with_ana/generated_predictions_{dataset_name}.jsonl'
        save_dir = f'my_scripts/scripts_with_ADSP/new_format_data/gen_with_ana-epoch_2/output/predict/train/step_2-gen_with_ana/self_reflection/{dataset_name}'
    elif data_split == "test":
        output_fp = f'my_scripts/scripts_with_ADSP/new_format_data/gen_with_ana-epoch_2/output/predict/test/step_2-gen_with_ana/generated_predictions_{dataset_name}.jsonl'
        save_dir = f'my_scripts/scripts_with_ADSP/new_format_data/gen_with_ana-epoch_2/output/predict/test/step_2-gen_with_ana/self_reflection/{dataset_name}'
    else:
        raise ValueError(f"Invalid data split: {data_split}")
    os.makedirs(save_dir, exist_ok=True)

    input_data = load_json(input_data_fp)
    if output_fp is not None:
        output_data = load_jsonl(output_fp)
    else:
        output_data = [""] * len(input_data)

    # First batch job
    first_batch_data = prepare_first_batch_data(input_data, output_data, prompt_inst, dataset_name, data_split, model_name, word_cnt)
    
    save_jsonl(first_batch_data, os.path.join(save_dir, 'batchinput.jsonl'))
    save_jsonl(first_batch_data[:10], os.path.join(save_dir, 'batchinput_sample.jsonl'))

    target_input_batch_file = 'batchinput_sample.jsonl' if sample_test else 'batchinput.jsonl'
    target_output_batch_file = 'batchoutput_sample.jsonl' if sample_test else 'batchoutput.jsonl'

    # if data_split == "train":
    batch_job_info = create_batch_job(client, os.path.join(save_dir, target_input_batch_file), dataset_name, target_input_batch_file)
    print(batch_job_info)

    check_info = wait_for_batch_completion(client, batch_job_info.id)

    file_response = client.files.content(check_info.output_file_id)
    with open(os.path.join(save_dir, target_output_batch_file), 'w') as f:
        f.write(file_response.text)

    # Second batch job
    # if data_split == "train":
    first_output_data = load_jsonl(os.path.join(save_dir, target_output_batch_file))
    # else:
    #     first_output_data = [""] * len(first_batch_data)
    second_batch_data = prepare_second_batch_data(first_batch_data, first_output_data, dataset_name, data_split, model_name, dataset_name2second_inst[dataset_name])

    save_jsonl(second_batch_data, os.path.join(save_dir, 'second_batchinput.jsonl'))
    save_jsonl(second_batch_data[:10], os.path.join(save_dir, 'second_batchinput_sample.jsonl'))

    target_input_batch_file = 'second_batchinput_sample.jsonl' if sample_test else 'second_batchinput.jsonl'
    target_output_batch_file = 'second_batchoutput_sample.jsonl' if sample_test else 'second_batchoutput.jsonl'

    # if data_split == "train":
    batch_job_info = create_batch_job(client, os.path.join(save_dir, target_input_batch_file), dataset_name, target_input_batch_file)
    print(batch_job_info)

    check_info = wait_for_batch_completion(client, batch_job_info.id)

    file_response = client.files.content(check_info.output_file_id)
    with open(os.path.join(save_dir, target_output_batch_file), 'w') as f:
        f.write(file_response.text)

if __name__ == "__main__":
    # for dataset_name in ["AEG", "ArgTersely", "CAG_CMV", "DebateSum", "ADSP"]:
    # only process ADSP for now, others are reused from my_scripts/train-gen_with_ana-epoch_5/outputs/predict_gen_with_ana-train_set/self-reflection
    # for dataset_name in ["ADSP"]:
    #     for data_split in ["train"]:
    #         sample_test = False
    #         main(dataset_name, data_split, sample_test)

    # for dataset_name in ["AEG", "ArgTersely", "CAG_CMV", "DebateSum", "ADSP"]:
    for dataset_name in ["ADSP"]:
        for data_split in ["test"]:
            sample_test = False
            main(dataset_name, data_split, sample_test)


# file-kRZq3eEP4QFCBFFuwUiXEemf