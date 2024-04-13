import json
from pathlib import Path
from typing import Any


def dump_jsonl(filepath: str | Path, objs: list[Any]):
    with open(filepath, "w") as f:
        for entry in objs:
            json.dump(entry, f)
            f.write("\n")


def load_jsonl(filepath: str | Path) -> list[Any]:
    objs = []
    with open(filepath) as f:
        for line in f:
            objs.append(json.loads(line))
    return objs
