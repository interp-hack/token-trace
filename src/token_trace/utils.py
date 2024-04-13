import json
from pathlib import Path
from typing import Any

import numpy as np

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

def summary_features(data: list[Any]) -> dict:
    pass

def histogram_features(data: list[Any]) -> list:
    """
        used for a single block.
    """
    feat_ids = [int(item[0]) for item in data]
    feat_values = [float(item[1]) for item in data]

    v_min = np.floor(min(feat_values))
    v_max = np.ceil(max(feat_values))
    n_bins = int(v_max - v_min)

    feat_locations = dict()
    for i in range(len(feat_ids)):
        feat_locations[feat_ids[i]] = np.floor(feat_values[i])

    return [
        np.histogram(
            feat_values, 
            range=(v_min, v_max), 
            bins=n_bins
        ),
        feat_locations
    ]