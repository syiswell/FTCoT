import sys

sys.path.append("..")

import argparse
import csv
import json
import logging
import os
import random
import re
from collections import OrderedDict
from tqdm import tqdm
from nlgeval import calc_nlg_metrics


# os.environ['HF_ENDPOINT']='hf-mirror.com'

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--eval_file", type=str, required=True)
parser.add_argument("--save_file", type=str, default="results.csv")
args = parser.parse_args()


def compute_metrics(ref_list, hyp_list):
    return calc_nlg_metrics(decoder_preds=hyp_list, decoder_labels=ref_list)


def read_jsonl(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_args_json(eval_file_path):
    """从评估文件所在目录加载args.json"""
    eval_dir = os.path.dirname(eval_file_path)
    args_json_path = os.path.join(eval_dir, "args.json")

    if os.path.exists(args_json_path):
        try:
            with open(args_json_path, "r", encoding="utf-8") as f:
                args_data = json.load(f)
            return args_data
        except Exception as e:
            logger.warning(f"无法加载args.json: {e}")
            return {}
    else:
        logger.warning(f"未找到args.json文件: {args_json_path}")
        return {}


def remove_duplicate_sentences(text):
    # 分割句子
    sentences = text.split(". ")

    # 使用列表和集合去重，保持顺序
    unique_sentences = ""

    for sentence in sentences:
        if sentence not in unique_sentences:
            unique_sentences = unique_sentences + sentence + ". "

    # 合并回文本
    return unique_sentences


def save_to_csv(filename, eval_file_path, metrics, args_data):
    """将评估结果保存到CSV文件（追加模式）"""
    file_exists = os.path.isfile(filename)

    # 从eval_file_path中提取文件名（不包含路径）
    eval_file_name = os.path.basename(eval_file_path)

    # 准备CSV行数据
    row_data = {"file_name": eval_file_name, "eval_file_path": eval_file_path}

    # 将metrics字典添加到row_data中
    if isinstance(metrics, dict):
        row_data.update(metrics)
    else:
        row_data["metrics"] = str(metrics)

    # 添加args.json中的参数（例如学习率和epoch）
    if args_data:
        # 提取感兴趣的参数，例如学习率和epoch
        # 可以根据需要调整要保存的字段
        param_fields = [
            "learning_rate",
            "n_epochs",
        ]
        for field in param_fields:
            if field in args_data:
                row_data[field] = args_data[field]

    # 获取所有字段名（如果文件不存在，需要从第一行数据中提取）
    if file_exists:
        # 读取现有文件以获取字段名
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
    else:
        # 文件不存在，从当前数据中提取字段名
        fieldnames = list(row_data.keys())

    # 追加模式写入CSV
    with open(filename, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # 如果是新文件，写入表头
        if not file_exists:
            writer.writeheader()

        # 确保所有字段都有值（缺失的字段填充为空字符串）
        row_to_write = {field: row_data.get(field, "") for field in fieldnames}
        writer.writerow(row_to_write)


def main():
    eval_datas = read_jsonl(args.eval_file)

    print("len(eval_datas): ", len(eval_datas))

    # 加载args.json
    args_data = load_args_json(args.eval_file)
    if args_data:
        print(f"成功加载args.json，包含 {len(args_data)} 个参数")
    else:
        print("警告：未能加载args.json，将不保存训练参数")

    ref_list = []
    hyp_list = []
    for sample in tqdm(eval_datas, total=len(eval_datas)):
        gold_response = sample["output"]
        predicted_response = sample["predicted_output"]

        if "<Response>" in predicted_response:
            predicted_response = (
                predicted_response.split("<Response>")[-1].strip().strip("</Response>")
            )

        # predicted_response = remove_duplicate_sentences(predicted_response)

        # print(predicted_response)
        # print("*"*50)

        ref_list.append(gold_response)
        hyp_list.append(predicted_response)

    metrics = compute_metrics(ref_list, hyp_list)
    print(f"Metrics on the test set: {metrics}")

    # 保存结果到CSV
    save_to_csv(args.save_file, args.eval_file, metrics, args_data)
    print(f"Results saved to {args.save_file}")


if __name__ == "__main__":
    main()
