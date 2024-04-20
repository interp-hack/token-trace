import json
import logging
import urllib.parse
import webbrowser
from typing import cast

import pandas as pd
import torch
from pandera import DataFrameModel
from transformer_lens import HookedTransformer

from token_trace.load_pretrained_model import load_model
from token_trace.types import ModuleName


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


def get_layer_from_module_name(module_name: ModuleName) -> int:
    # NOTE: currently hardcoded to Joseph's naming convention
    # e.g. "blocks.0.hook_resid_pre" -> 0
    return int(module_name.split(".")[1])


def get_token_strs(model_name: str, text: str) -> list[str]:
    model = load_model(model_name)
    return cast(list[str], model.to_str_tokens(text))


def last_token_prediction_loss(model: HookedTransformer, text: str) -> torch.Tensor:
    """Compute the prediction loss of the last token in the text"""
    loss = model(text, return_type="loss", loss_per_token=True)
    return loss[0, -1]


def setup_logger(name: str, level: int = logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.hasHandlers():
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def get_empty_dataframe_from_pa_model(model: DataFrameModel) -> pd.DataFrame:
    schema = model.to_schema()
    column_names = list(schema.columns.keys())
    data_types = {
        column_name: column_type.dtype.type.name
        for column_name, column_type in schema.columns.items()
    }
    return pd.DataFrame(columns=column_names).astype(data_types)
