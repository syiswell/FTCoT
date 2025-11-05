import json
import os, sys
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI
import time
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
sys.path.append('.')
from my_src_refactored.utils import load_json, load_text, load_jsonl, save_jsonl

def prepare_batch_data(input_data_points: List[Dict], prompt_inst: str, dataset_name: str, data_split: str, model_name: str) -> List[Dict]:
    # Prepare batch data
    batch_data = []
    for i, dp in tqdm(enumerate(input_data_points), total=len(input_data_points)):
        # Process input data
        arg_text = '\n\n\n'.join(dp['instruction'].split('\n\n\n')[1:-1])
        task_inst = dp['instruction'].split('\n\n\n')[0]
        input_query = prompt_inst.format(arg_text=arg_text, task_inst=task_inst)
        idx = f"{dataset_name}_{data_split}_{i:05d}"
        
        # Create batch item
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
            }
        }
        batch_data.append(batch_item)
    return batch_data

def create_batch_job(client: OpenAI, input_file_path: str, dataset_name: str) -> Dict:
    # Upload input file
    batch_file_info = client.files.create(
        file=open(input_file_path, "rb"),
        purpose="batch"
    )

    # Create batch job
    batch_job_info = client.batches.create(
        input_file_id=batch_file_info.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
          "description": f"Task: {dataset_name}, nightly eval job"
        }
    )

    return batch_job_info

def main(
        model_name: str, 
        dataset_name: str, 
        data_split: str, 
        orig_input_fp: str, 
        prompt_inst_path: str, 
        output_dir: str, 
        test_mode: bool = False
    ) -> None:
    # Main function
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    # Set up output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load input data and prompt instructions
    input_data_points = load_json(orig_input_fp)
    prompt_inst = load_text(prompt_inst_path)

    # Prepare batch data
    batch_data = prepare_batch_data(input_data_points, prompt_inst, dataset_name, data_split, model_name)

    # Save batch input data
    save_jsonl(batch_data, os.path.join(output_dir, 'batch_input.jsonl'))
    save_jsonl(batch_data[:10], os.path.join(output_dir, 'batch_input_sample.jsonl'))

    client = OpenAI(api_key=api_key)

    if test_mode:
        print("Running in test mode. Using batch_input_sample.jsonl")
        input_file = os.path.join(output_dir, 'batch_input_sample.jsonl')
        output_file = os.path.join(output_dir, 'batch_output_sample.jsonl')
    else:
        print("Running in full mode. Using batch_input.jsonl")
        input_file = os.path.join(output_dir, 'batch_input.jsonl')
        output_file = os.path.join(output_dir, 'batch_output.jsonl')

    batch_job_info = create_batch_job(client, input_file, dataset_name)
    print(batch_job_info)

    while True:
        check_info = client.batches.retrieve(batch_job_info.id)
        print(f"Batch job status: {check_info.status}")
        print(f"Request counts: {check_info.request_counts}")
        if check_info.status == "completed":
            break
        time.sleep(30)
            
    file_response = client.files.content(check_info.output_file_id)
    with open(output_file, 'w') as f:
        f.write(file_response.text)

if __name__ == "__main__":
    model_name = "gpt-4o-mini-2024-07-18"

    # for dataset_name in ["AEG", "ArgTersely", "CAG_CMV", "DebateSum", "ADSP"]:
    for dataset_name in ["ADSP"]:
        for data_split in ["train"]:
            orig_input_fp = f"data/my_data/datasets/sft_dataset_5k/{dataset_name}/{data_split}.json"
            prompt_inst_path = "my_src_refactored/gpt_analysis/analysis_prompt.txt"
            output_dir = f"data/my_data/intermediate_data/gpt_analysis/sft_dataset_5k/{dataset_name}/{data_split}"
            test_mode = False

            main(model_name, dataset_name, data_split, orig_input_fp, prompt_inst_path, output_dir, test_mode)

    # get train no synthetic res from train res for ADSP
    train_w_synthetic_gpt_input_fp = "data/my_data/intermediate_data/gpt_analysis/sft_dataset_5k/ADSP/train/batch_input.jsonl"
    train_w_synthetic_gpt_output_fp = "data/my_data/intermediate_data/gpt_analysis/sft_dataset_5k/ADSP/train/batch_output.jsonl"
    train_orig_data_fp = "data/my_data/datasets/sft_dataset_5k/ADSP/train_no_synthetic.json"

    train_gpt_save_dir = "data/my_data/intermediate_data/gpt_analysis/sft_dataset_5k/ADSP/train_no_synthetic"
    os.makedirs(train_gpt_save_dir, exist_ok=True)
    train_gpt_output_fp = os.path.join(train_gpt_save_dir, "batch_output.jsonl")
    train_gpt_input_fp = os.path.join(train_gpt_save_dir, "batch_input.jsonl")

    train_w_synthetic_gpt_input = load_jsonl(train_w_synthetic_gpt_input_fp)
    train_w_synthetic_gpt_output = load_jsonl(train_w_synthetic_gpt_output_fp)
    train_orig_data = load_json(train_orig_data_fp)

    train_gpt_input = train_w_synthetic_gpt_input[-len(train_orig_data):]
    train_gpt_output = train_w_synthetic_gpt_output[-len(train_orig_data):]

    save_jsonl(train_gpt_input, train_gpt_input_fp)
    save_jsonl(train_gpt_output, train_gpt_output_fp)


    # for dataset_name in ["AEG", "ArgTersely", "CAG_CMV", "DebateSum", "ADSP"]:
    #     for data_split in ["test"]:
    #         orig_input_fp = f"data/my_data/datasets/sft_dataset_5k/{dataset_name}/{data_split}.json"
    #         prompt_inst_path = "my_src_refactored/gpt_analysis/analysis_prompt.txt"
    #         output_dir = f"data/my_data/intermediate_data/gpt_analysis/sft_dataset_5k/{dataset_name}/{data_split}"
    #         test_mode = True

    #         main(model_name, dataset_name, data_split, orig_input_fp, prompt_inst_path, output_dir, test_mode)

        