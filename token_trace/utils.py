import json
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any

import torch
from transformer_lens import HookedTransformer


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


def get_neuronpedia_url(
    layer: int, features: list[int], name: str = "temporary_list"
) -> str:
    url = "https://neuronpedia.org/quick-list/"
    name = urllib.parse.quote(name)
    url = url + "?name=" + name
    list_feature = [
        {"modelId": "gpt2-small", "layer": f"{layer}-res-jb", "index": str(feature)}
        for feature in features
    ]
    url = url + "&features=" + urllib.parse.quote(json.dumps(list_feature))
    return url


def open_neuronpedia(layer: int, features: list[int], name: str = "temporary_list"):
    url = get_neuronpedia_url(layer, features, name)
    webbrowser.open(url)


def dense_to_sparse(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a dense tensor to a sparse tensor of the same shape"""
    indices = torch.nonzero(tensor).t()
    values = tensor[*indices]
    return torch.sparse_coo_tensor(
        indices,
        values,
        tensor.size(),
        device=tensor.device,
        dtype=tensor.dtype,
    )


def last_token_loss(model: HookedTransformer, prompt: str) -> torch.Tensor:
    loss = model(prompt, return_type="loss", loss_per_token=True)
    return loss[0, -1]
