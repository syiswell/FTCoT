from datetime import datetime
import shutil
import time
import pandas as pd
from tqdm import tqdm
import logging
import os
import argparse
import json
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def read_jsonl(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def statistic_samples(datas):
    short_samples = []

    utterance_len_list = []
    sample_len_list = []
    utterance_len_dic = defaultdict(int)
    sample_len_dic = defaultdict(int)
    round_dic = defaultdict(int)

    topic_list = []

    all_uteerence_num = 0

    for data in tqdm(datas):
        sample_utterance_len_list = []
        path = data["path"]
        longer_path_flag = 0

        topic_list.append(data["title"])

        round = len(path)
        round_dic[round] += 1

        all_uteerence_num += round

        path_length = 0
        for utterance in path:
            utterance_len_list.append(len(utterance.split()))
            sample_utterance_len_list.append(len(utterance.split()))
            path_length += len(utterance.split())
            utterance_len_dic[len(utterance.split())] += 1

        data["utterance_len"] = sample_utterance_len_list
        sample_len_list.append(path_length)
        sample_len_dic[path_length] += 1

        if longer_path_flag == 0:
            short_samples.append(data)

    print("len(selected_samples): ", len(datas))
    print("topic_list: ", len(list(set(topic_list))))
    print("all_utterance_num: ", all_uteerence_num)
    print("utterance_len_list: ", sum(utterance_len_list) / len(utterance_len_list))
    print("sample_len_list: ", sum(sample_len_list) / len(sample_len_list))
    print("utterance_len_dic: ", utterance_len_dic)
    print("sample_len_dic: ", sample_len_dic)
    print("round_dic: ", round_dic)

    plot_bar(utterance_len_dic, "utterance")
    plot_bar(sample_len_dic, "sample")


def plot_bar(example_data, text):
    # Sort dictionary by text length
    sorted_data = dict(sorted(example_data.items()))

    # Create a bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(list(sorted_data.keys()), list(sorted_data.values()), color="blue")
    plt.xlabel(f"{text} Length")
    plt.ylabel(f"Number of {text}")
    plt.title(f"{text} Length")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Setting x-axis ticks using MultipleLocator
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(500))

    plt.show()


if __name__ == "__main__":
    data_path = "/home/sunyang/hlt/new_cmv_dataset/final_dataset/all_data_evaluation_dialogue.jsonl"
    dataset = read_jsonl(data_path)

    statistic_samples(dataset)
