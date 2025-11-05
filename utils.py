import json
from typing import List, Dict


def load_json(file_path: str) -> List[Dict]:
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data: List[Dict], file_path: str) -> None:
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def load_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data: List[Dict], file_path: str) -> None:
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_text(file_path: str) -> str:
    with open(file_path, "r") as fp:
        return fp.read()
