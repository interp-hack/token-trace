import json
import urllib.parse
import webbrowser
from typing import cast

import torch
from transformer_lens import HookedTransformer

from token_trace.load_pretrained_model import load_model


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


def get_token_strs(model_name: str, text: str) -> list[str]:
    model = load_model(model_name)
    return cast(list[str], model.to_str_tokens(text))


def last_token_prediction_loss(model: HookedTransformer, text: str) -> torch.Tensor:
    """Compute the prediction loss of the last token in the text"""
    loss = model(text, return_type="loss", loss_per_token=True)
    return loss[0, -1]
