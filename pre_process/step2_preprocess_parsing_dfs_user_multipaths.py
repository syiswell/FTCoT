"""
python preprocess_parsing.py "input file" "output folder"

"""

import glob
import codecs
import json
import os
from nltk.tokenize import word_tokenize
import xml.etree.ElementTree as ET
import copy
import re
import multiprocessing
import math
import time
import copy
import collections
import sys
import nltk
from tqdm import tqdm
import concurrent.futures


def parsing(in_data):
    # we need to remove useless data here
    total = []  # use this to mantain all child
    temp = {}

    for key in ["title", "num_comments", "author", "id", "url"]:
        temp[key] = in_data[key]

    temp["body"] = in_data["selftext"]

    temp["child"] = []

    total.append(temp)
    count = 0

    for data in in_data["comments"]:
        child_temp = {}
        parent_id = data["parent_id"]

        if isinstance(parent_id, int):
            continue

        for key in ["author", "id", "body"]:
            child_temp[key] = data[key]
        child_temp["child"] = []

        if "author" in data and data["author"] == "DeltaBot":
            count += 1

        for past in total:
            if "id" not in past:
                print(in_data)
                print("*" * 10)
                print(past)

            if past["id"] == parent_id[3:]:
                past["child"].append(child_temp)
                break
        total.append(child_temp)

    temp["count"] = count

    return temp


# find for all pair, if there are multiple choice,
# choose the middle one
def dfs(data, past):
    temp = {}
    for key in ["author", "id", "body"]:
        if key in data:
            temp[key] = data[key]

    if "body" in data:
        past.append(temp)
        if len(data["child"]) == 0:
            yield (0, copy.deepcopy(past))
        else:

            for _ in data["child"]:
                if (
                    ("author" in _)
                    and (_["author"] == "DeltaBot")
                    and ("Confirmed:" in _["body"])
                ):
                    past.append(
                        {"author": _["author"], "id": _["id"], "body": _["body"]}
                    )

                    yield (1, copy.deepcopy(past))
                    past.pop()
                    break
            else:
                for _ in data["child"]:
                    yield from dfs(_, past)

        past.pop()


def find_pair(data):
    count = [[], []]
    op_info = {}
    for key in ["title", "num_comments", "author", "id", "url", "body"]:
        op_info[key] = data[key]

    root = {"author": data["author"], "id": data["id"], "body": data["body"]}
    for c, child in enumerate(data["child"]):
        temp = [[], []]
        for index, flow in dfs(child, [root]):
            temp[index].append(flow)

        if len(temp[1]) > 0:
            count[1].extend(temp[1])

    # 检查正样本的正确性，过滤掉包含已删除内容的路径
    temp = []
    for flow in count[1]:
        for _ in flow:
            if "[deleted]" == _["body"]:
                break
        else:
            temp.append(flow)
    count[1] = temp

    # 处理positive样本
    count_positive = []
    pos_author = [[], []]  # 仍然保留数组结构，但只使用 pos_author[1]

    for path in count[1]:
        # 保存delta信息
        last = (path[-2]["author"], (path[0]["author"], path[-3]["author"]))

        # 去掉delta相关的最后两个评论，因为都是deltabot的评论
        path_trimmed = path[:-2]

        if len(path_trimmed):
            count_positive.append(path_trimmed)
            pos_author[1].append([(data["author"], last[0]), last[1]])

    count[1] = count_positive

    if len(count[1]) > 1:
        print(f"Found {len(count[1])} delta paths")

    if "Society is apathetic to child abuse and it's victims" in data["title"]:
        print(data["title"])
        print(len(count[1]))

    return count, op_info, pos_author


mispell_dict = {
    "didn't": "did not",
    "doesn't": "does not",
    "isn't": "is not",
    "shouldn't": "should not",
    "wasn't": "was not",
    "hasn't": "has not",
    "won't": "wont",
    "theatre": "theater",
    "cancelled": "canceled",
    "organisation": "organization",
    "labour": "labor",
    "favourite": "favorite",
    "travelling": "traveling",
    "washingtons": "washington",
    "marylands": "maryland",
    "chinas": "china",
    "russias": "russia",
    "‘the": "the",
    "irans": "iran",
    "dulles": "dulle",
    "commuincation": "communication",
    "parantage": "parentage",
    "gorvernment": "government",
    "&#8710;": "∆",
}
punct_dict = {
    "--": "",
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "``": '"',
    "''": '"',
    "]": "",
    "[": "",
    "_____": "",
    ") )": ")",
    "( (": "(",
    "//": "",
}


def rem_url(text):
    def check(text, sym_pair):
        if "http" in text:
            temp = ""
            while True:
                lindex = text.find(sym_pair[0] + "http")
                if lindex == -1:
                    temp += text
                    break
                temp += text[:lindex]
                rindex = text.find(sym_pair[-1], lindex)
                if rindex == -1:
                    temp += text[lindex:]
                    break
                # temp += '<link>'
                text = text[rindex + 1 :]
            return temp
        return text

    text = check(text, "()")
    text = check(text, "[]")

    def complex_regex(text, queue):
        """复杂的正则表达式操作"""
        try:
            pattern = re.compile(
                r"""(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))"""
            )
            result = pattern.sub("", text)
            queue.put(result)
        except Exception as e:
            queue.put(str(e))

    def simple_regex(text):
        """简单的正则表达式操作"""
        return re.sub(r"https?://\S+|www\.\S+", "", text)

    def replace_text(text):
        """处理文本，具备超时回退功能"""
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=complex_regex, args=(text, queue))
        p.start()

        # 等待3秒超时
        p.join(timeout=3)
        if p.is_alive():
            p.terminate()
            p.join()
            print("Complex regex timed out, using simpler regex.")
            return simple_regex(text)
        else:
            return queue.get()

    if "http" in text:
        # text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', "<link>", text)
        # print("text: ", text)
        # text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', "", text)
        # text = re.sub(r'https?://\S+|www\.\S+', '', text)

        text = replace_text(text)

    return text


def rem_sym(text):
    index = len(text) - 1
    while (
        (index >= 1)
        and (text[index] == text[index - 1])
        and (
            not (text[index].isalpha() or text[index] == "~" or text[index].isdecimal())
        )
    ):
        index -= 1
    text = text[: index + 1]
    index = 0
    while (index < len(text)) and (
        not (text[index].isalpha() or text[index].isdecimal() or text[index] == "~")
    ):
        if text[index] == ">" or text[index] == "∆":
            break
        index += 1
    text = text[index:]
    return text


def rem_italic(text):
    sent, buffer = [], []
    flag = False
    try:
        for word in text.split(" "):
            if word[0] == "_":
                flag = True

            if flag:
                buffer.append(word)
            else:
                sent.append(word)

            if word[-1] == "_":
                flag = False
                sent.append(" ".join(buffer)[1:-1])
                buffer.clear()
        else:
            sent.extend(buffer)
    except:
        print([text])
        None + 1

    return " ".join(sent)


def rem_delete(text):
    sent = []
    buffer = []
    flag = False
    for word in text.split(" "):
        if word[:2] == "~~":
            word = word[2:]
            flag = True
        if word[-2:] == "~~":
            buffer.clear()
            flag = False
            continue

        if flag:
            buffer.append(word)
        else:
            sent.append(word)
    else:
        sent.extend(buffer)

    return " ".join(sent)


def clean(text):
    # build up inverted file
    # text = text.lower().strip()
    text = text.strip()

    if text == "":
        return ""

    text = rem_url(text)

    for punct_c, punct_r in punct_dict.items():
        text = text.replace(punct_c, punct_r)

    for key, data in mispell_dict.items():
        text = text.replace(key, data)

    """
    text = rem_sym(text)
    if(text==''):
        return ''
    """

    for _ in ["***", "**", "*", "^"]:
        text = text.replace(_, "")

    text = " ".join(text.split())
    sent = []
    for word in text.split(" "):
        if len(word) > 1:
            if word[0] == "." and word[1] != ".":
                sent.append(".")
                word = word[1:]
        if word[:5] == "/user":
            # word = '<user>'
            word = word.replace("user", "u")
            print(text)
        # elif('/u/' in word and word[0]=='/'):
        #     word = '<user>'
        # elif('/r/' in word and word[0]=='/'):
        #     word = '<reply>'
        sent.append(word)

    sent = " ".join(sent)
    if len(sent):
        sent = rem_delete(sent)
    sent = " ".join(sent.split())
    if len(sent):
        sent = rem_italic(sent)
    sent = " ".join(sent.split())

    for key in ["link", "edit", "cite"]:
        # sent = sent.replace('< {} >'.format(key), '<{}>'.format(key))
        sent = sent.replace("< {} >".format(key), "")
        sent = sent.replace("<{}>".format(key), "")

    for punct_c, punct_r in punct_dict.items():
        sent = sent.replace(punct_c, punct_r)
    sent = " ".join(sent.split()).strip()

    return sent


# for here, we need to partition post into paragraph
def job(q, base, total_text):
    def clean_post(text):
        data = []
        for para in text.split("\n\n"):

            num = 1
            temp_para = []
            for sent in para.split("\n"):
                sent = " ".join(sent.split())

                if len(sent) and sent[0] == "*":
                    # check if dot at fromt
                    sent = "{}. {}.".format(num, sent)
                    num += 1
                else:
                    num = 1
                    # check if copy from front
                    if len(sent) and sent[:4].lower() == "edit":
                        sent = ""
                    elif sent.startswith("&gt;") or sent.startswith(">"):
                        sent = ""

                temp_para.append(sent)
            para = clean(" ".join(temp_para))
            if len(para):
                data.append(para)

        return data

    for index, text in enumerate(total_text):
        index += base

        for side in [1]:
            for path_index, path in enumerate(text[0][side]):
                arr = {
                    "op_info": copy.deepcopy(text[1]),
                    "text": [],
                }
                arr["op_info"]["body"] = clean_post(arr["op_info"]["body"])
                for re_index, _ in enumerate(path):
                    arr["text"].append(
                        {
                            "body": clean_post(_["body"]),
                            "author": _["author"],
                            "id": _["id"],
                        }
                    )
                q.put(copy.deepcopy(arr))


def write_out(f, q):
    while not q.empty():
        arr = q.get()

        # arr['op_info']['title'] = arr['op_info']['title']
        for _ in [
            "cmv:",
            "cmv :",
            "cmw:",
            "cmv",
            "/r/changemyview",
            "CMV:",
            "CMV :",
            "CMW:",
            "CMV",
        ]:
            arr["op_info"]["title"] = arr["op_info"]["title"].replace(_, "")
        arr["op_info"]["title"] = " ".join(arr["op_info"]["title"].split())

        f.write(json.dumps(arr))
        f.write("\n")
    f.flush()


def checkandwait(f, process, delay_time=5, end=False):
    while (len(process) >= num_cpu) or end:
        print(pair_index, len(process))
        if len(process) == 0:
            break
        #
        # temp_process = []
        # for _ in process:
        #     _.join(timeout = delay_time)
        #
        # count = 0
        # for _ in process:
        #     if(_.exitcode == None):
        #         temp_process.append(_)
        #     elif(_.exitcode == 0):
        #         #process[i].close()
        #         pass
        #     else:
        #         print('error')
        #
        # process = temp_process

        still_running = []
        # 检查每个进程是否仍在运行
        for proc in process:
            proc.join(timeout=delay_time)  # 给每个进程一定时间来完成
            if proc.is_alive():
                still_running.append(proc)  # 如果仍在运行，则保持在列表中
            else:
                if proc.exitcode != 0:
                    print(f"Process exited with error code {proc.exitcode}")

        process = still_running  # 更新正在运行的进程列表
        write_out(f, q)  # 写出队列中的数据

    else:
        write_out(f, q)
    return process


def clean_sent(sent):
    for a in [",", ".", "?"]:
        for b in [",", ".", "?"]:
            sent = sent.replace("{} {}".format(a, b), " {} ".format(b))
    sent = sent.replace("ca n't", "can't")
    sent = sent.replace(" n't", "n't")
    return sent.strip()


if __name__ == "__main__":
    # check for all data, and search for delta data
    # make output folder

    # input_file = "/home/sunyang/hlt/reddit/subreddits23/changemyview_datas.jsonl"

    # datas = []
    # with open(input_file, 'r') as f:
    #     for i, line in enumerate(f):
    #         data = json.loads(line)
    #         datas.append(parsing(data))

    # print(len(datas))

    # output_file = "/home/sunyang/hlt/reddit/subreddits23/changemyview_datas_parsed.jsonl"
    # with open(output_file, 'w', encoding="utf-8") as f:
    #     for i, sample in enumerate(datas):
    #         f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    input_file = "/home/sunyang/hlt/reddit/subreddits23/changemyview_datas_parsed.jsonl"
    datas = []
    with open(input_file, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            datas.append(data)
    print(len(datas))

    pairs = []
    for index, _ in enumerate(datas):
        if _["count"] >= 1:
            count, op_info, pos_author = find_pair(_)

            if len(count[1]) == 0:
                continue
            pairs.append((count, op_info, pos_author))
    print("len(pairs): ", len(pairs))

    # use multi-process for cleaning
    num_cpu = multiprocessing.cpu_count()
    q = multiprocessing.Queue()
    f = open(
        "result/cmv_dialog_flow_clean_user_multi_paths.jsonl", "w", encoding="utf-8"
    )
    base = 256
    process = []
    for pair_index in range(math.ceil(len(pairs) / base)):
        process.append(
            multiprocessing.Process(
                target=job,
                args=(
                    q,
                    pair_index * base,
                    pairs[pair_index * base : (pair_index + 1) * base],
                ),
            )
        )
        process[-1].start()
        process = checkandwait(f, process)

    else:
        process = checkandwait(f, process, 5, True)
        f.close()
    for _ in process:
        _.terminate()
