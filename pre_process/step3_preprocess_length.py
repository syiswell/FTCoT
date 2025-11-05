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


def select_samples(dataset):
    selected_samples = []
    condidate_samples = []
    round2_samples = []
    len_list = []
    len_dic = defaultdict(int)
    round_dic = defaultdict(int)
    longer_uteerence_num = 0
    longer_path_num = 0

    for sample in tqdm(dataset):
        if sample["text"] == []:
            # print(sample["op_info"]["title"])
            continue

        post_list = []
        author_list = []

        temp_len_list = []
        temp_len_dic = defaultdict(int)

        pre_author = sample["text"][0]["author"]
        post_list.append(" ".join(sample["text"][0]["body"]).strip())
        author_list.append(pre_author)

        temp_len = len(post_list[-1].split())

        temp_len_list.append(temp_len)
        temp_len_dic[temp_len_list[-1]] = 1

        round = 1
        exist_reply = []

        for post in sample["text"][1:]:
            reply = " ".join(post["body"]).strip()
            if reply not in exist_reply:
                if post["author"] == pre_author:
                    post_list[-1] += "\n" + reply
                    temp_len_dic[temp_len_list[-1]] -= 1
                    temp_len_list[-1] += len(reply.split())
                    temp_len_dic[temp_len_list[-1]] += 1
                else:
                    post_list.append(reply)
                    temp_len_dic[len(reply.split())] += 1
                    temp_len_list.append(len(reply.split()))
                    round += 1

                    author_list.append(post["author"])
                    pre_author = post["author"]

                exist_reply.append(reply)

        continue_flag = 0
        for tl in temp_len_list:
            if tl < 50:
                continue_flag = 1
                break

        if continue_flag == 1:
            continue

        longer_path_flag = 0
        for tl in temp_len_list:
            if tl > 500 and longer_path_flag == 0:
                longer_path_num += 1
                longer_path_flag = 1

            if tl > 500:
                longer_uteerence_num += 1

        round_dic[round] += 1

        len_list.extend(temp_len_list)
        for k, v in temp_len_dic.items():
            len_dic[k] += v

        if round % 2 == 0 and post_list != []:
            new_sample = {}
            new_sample["title"] = sample["op_info"]["title"]
            new_sample["path"] = post_list
            new_sample["authors"] = author_list
            selected_samples.append(new_sample)
            if round in [4, 6, 8]:
                condidate_samples.append(new_sample)

            if round in [2]:
                round2_samples.append(new_sample)

    # print("len(selected_samples): ", len(selected_samples))
    # print("longer_path_num ", longer_path_num)
    # print("longer_uteerence_num ", longer_uteerence_num)
    # print(len_dic)
    # print(sum(len_list) / len(len_list))
    # print(round_dic)

    condidate_samples = condidate_samples + random_select(round2_samples, 2000)

    return (
        selected_samples,
        condidate_samples,
    )


def random_select(datas, num):
    return random.sample(datas, num)


def statistic_samples(datas):
    print_samples = []

    short_samples = []

    utterance_len_list = []
    sample_len_list = []
    utterance_len_dic = defaultdict(int)
    sample_len_dic = defaultdict(int)
    round_dic = defaultdict(int)
    longer_uteerence_num = 0
    longer_path_num = 0

    all_uteerence_num = 0

    for data in tqdm(datas):
        sample_utterance_len_list = []
        path = data["path"]
        authors = data["authors"]
        longer_path_flag = 0

        round = len(path)
        round_dic[round] += 1

        all_uteerence_num += round

        path_length = 0
        for utterance in path:
            utterance_len_list.append(len(utterance.split()))
            sample_utterance_len_list.append(len(utterance.split()))
            path_length += len(utterance.split())
            utterance_len_dic[len(utterance.split())] += 1
            if len(utterance.split()) > 500 and longer_path_flag == 0:
                longer_path_num += 1
                longer_path_flag = 1
                print_samples.append(data)

            if len(utterance.split()) > 500:
                longer_uteerence_num += 1

        data["utterance_len"] = sample_utterance_len_list
        sample_len_list.append(path_length)
        sample_len_dic[path_length] += 1

        if longer_path_flag == 0 and len(list(set(authors))) == 2:
            short_samples.append(data)

    print("len(selected_samples): ", len(datas))
    print("longer_path_num: ", longer_path_num)
    print("longer_path percent: ", longer_path_num / len(datas))
    print("longer_uteerence_num: ", longer_uteerence_num)
    print("longer_uteerence percent: ", longer_uteerence_num / all_uteerence_num)
    print("utterance_len_list: ", sum(utterance_len_list) / len(utterance_len_list))
    print("sample_len_list: ", sum(sample_len_list) / len(sample_len_list))
    print("utterance_len_dic: ", utterance_len_dic)
    print("max utterance length: ", max(utterance_len_dic.keys()))
    print("sample_len_dic: ", sample_len_dic)
    print("round_dic: ", round_dic)

    print("90th percentile:", np.percentile(utterance_len_list, 90))
    print("95th percentile:", np.percentile(utterance_len_list, 95))

    plot_bar(utterance_len_dic, "utterance")
    plot_bar(sample_len_dic, "sample")

    return (
        random_select(print_samples, 10),
        short_samples,
    )


def statistic_samples2(datas):
    short_samples = []

    utterance_len_list = []
    sample_len_list = []
    utterance_len_dic = defaultdict(int)
    sample_len_dic = defaultdict(int)
    round_dic = defaultdict(int)
    longer_uteerence_num = 0
    longer_path_num = 0

    all_uteerence_num = 0

    for data in tqdm(datas):
        sample_utterance_len_list = []
        path = data["path"]
        longer_path_flag = 0

        round = len(path)
        round_dic[round] += 1

        all_uteerence_num += round

        path_length = 0
        for utterance in path:
            utterance_len_list.append(len(utterance.split()))
            sample_utterance_len_list.append(len(utterance.split()))
            path_length += len(utterance.split())
            utterance_len_dic[len(utterance.split())] += 1
            if len(utterance.split()) > 500 and longer_path_flag == 0:
                longer_path_num += 1
                longer_path_flag = 1
                print_samples.append(data)

            if len(utterance.split()) > 500:
                longer_uteerence_num += 1

        data["utterance_len"] = sample_utterance_len_list
        sample_len_list.append(path_length)
        sample_len_dic[path_length] += 1

        if longer_path_flag == 0:
            short_samples.append(data)

    print("len(selected_samples): ", len(datas))
    print("longer_path_num: ", longer_path_num)
    print("longer_path percent: ", longer_path_num / len(datas))
    print("longer_uteerence_num: ", longer_uteerence_num)
    print("longer_uteerence percent: ", longer_uteerence_num / all_uteerence_num)
    print("utterance_len_list: ", sum(utterance_len_list) / len(utterance_len_list))
    print("sample_len_list: ", sum(sample_len_list) / len(sample_len_list))
    print("utterance_len_dic: ", utterance_len_dic)
    print("sample_len_dic: ", sample_len_dic)
    print("round_dic: ", round_dic)

    print("90th percentile:", np.percentile(utterance_len_list, 90))
    print("95th percentile:", np.percentile(utterance_len_list, 95))

    plot_bar(utterance_len_dic, "utterance")
    plot_bar(sample_len_dic, "sample")


def plot_bar(example_data, text):
    # Sort dictionary by text length
    sorted_data = dict(sorted(example_data.items()))

    # Create a bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(sorted_data.keys(), sorted_data.values(), color="blue")
    plt.xlabel(f"{text} Length")
    plt.ylabel(f"Number of {text}")
    plt.title(f"{text} Length")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Setting x-axis ticks using MultipleLocator
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(500))

    plt.show()


if __name__ == "__main__":
    data_path = "result/cmv_dialog_flow_clean_user_multi_paths.jsonl"
    dataset = read_jsonl(data_path)

    random.seed(3407)

    dataset, condidate_dataset = select_samples(dataset)

    # statistic_samples(dataset)

    print_samples, short_samples = statistic_samples(condidate_dataset)

    statistic_samples2(short_samples)

    # pd.DataFrame(short_samples).to_json(
    #     f"cmv_dialog_flow_clean_user_multi_paths_round2_2000_4_all_6_all_8_all_short_all/all_data.jsonl",
    #     orient="records",
    #     lines=True,
    # )
