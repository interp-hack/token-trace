from typing import cast

import pandas as pd
import torch
from transformer_lens import HookedTransformer

from token_trace.load_pretrained_model import load_model, load_sae_dict
from token_trace.sae_circuit import (
    get_sae_cache_dict,
)

pd.options.mode.chained_assignment = None  # default='warn'
DEFAULT_MODEL_NAME = "gpt2-small"
DEFAULT_REPO_ID = "jbloom/GPT2-Small-SAEs"
DEFAULT_PROMPT = "When John and Mary went to the shops, John gave the bag to"
DEFAULT_ANSWER = " Mary"
DEFAULT_TEXT = DEFAULT_PROMPT + DEFAULT_ANSWER
DEVICE = "cpu"
# if torch.cuda.is_available():
#     DEVICE = "cuda"


def loss_helper(
    model: HookedTransformer,
    text: str,
) -> torch.Tensor:
    """Compute the loss of the model when predicting the last token."""
    loss = model(text, return_type="loss", loss_per_token=True)
    return loss[0, -1]


def get_token_strs(model_name: str, text: str) -> list[str]:
    model = load_model(model_name)
    return cast(list[str], model.to_str_tokens(text))


def compute_node_attribution(
    model_name: str,
    text: str,
    *,
    metric: str = "loss",
) -> pd.DataFrame:
    model = load_model(model_name)
    sae_dict = load_sae_dict(model_name)
    if metric == "loss":
        # Last-token loss
        metric_fn = lambda model: loss_helper(model, text)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Get the token strings.
    text_tokens = model.to_str_tokens(text)
    prompt_tokens: list[str] = text_tokens[:-1]  # pyright: ignore
    response_token: list[str] = text_tokens[-1]  # pyright: ignore

    prompt_str = "".join(prompt_tokens)
    response_str = response_token
    sae_cache_dict = get_sae_cache_dict(model, sae_dict, metric_fn)

    # Construct dataframe.
    rows = []
    for module_name, module_activations in sae_cache_dict.items():
        print(f"Processing module {module_name}")
        layer = int(module_name)
        acts = module_activations.activations.coalesce()
        grads = module_activations.gradients.coalesce()
        effects = acts * grads
        effects = effects.coalesce()

        for index, ie_atp, act, grad in zip(
            effects.indices().t(), effects.values(), acts.values(), grads.values()
        ):
            example_idx, token_idx, node_idx = index
            assert example_idx == 0

            rows.append(
                {
                    "layer": layer,
                    "module_name": module_name,
                    "feature": node_idx.item(),
                    "token": token_idx.item(),
                    "token_str": text_tokens[token_idx],
                    "value": act.item(),
                    "grad": grad.item(),
                    "indirect_effect": ie_atp.item(),
                    "prompt_str": prompt_str,
                    "response_str": response_str,
                    "node_type": "feature" if node_idx < 24576 else "error",
                }
            )

    df = pd.DataFrame(rows)
    # Filter out zero indirect effects
    df = df[df["indirect_effect"] != 0]
    print(f"{len(df)} non-zero indirect effects found.")
    return df
