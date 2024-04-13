import json
import urllib.parse
import webbrowser
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


def open_neuronpedia(layer: int, features: list[int], name: str = "temporary_list"):
    url = "https://neuronpedia.org/quick-list/"
    name = urllib.parse.quote(name)
    url = url + "?name=" + name
    list_feature = [
        {"modelId": "gpt2-small", "layer": f"{layer}-res-jb", "index": str(feature)}
        for feature in features
    ]
    url = url + "&features=" + urllib.parse.quote(json.dumps(list_feature))
    webbrowser.open(url)
