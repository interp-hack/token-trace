from typing import cast

import pandas as pd
import pandera as pa
from transformer_lens import HookedTransformer

from token_trace.load_pretrained_model import load_model
from token_trace.sae_activation_cache import get_sae_activation_cache
from token_trace.types import MetricFunction, SAEDict

node_df_schema = pa.DataFrameSchema(
    {
        "layer": pa.Column(int),
        "example_idx": pa.Column(int),
        "node_idx": pa.Column(int),
        "token_idx": pa.Column(int),
        "value": pa.Column(float),
        "grad": pa.Column(float),
        "indirect_effect": pa.Column(float),
        "node_type": pa.Column(str),
        "module_name": pa.Column(str),
        "token_str": pa.Column(str),
        "example_str": pa.Column(str),
    }
)


def get_token_strs(model_name: str, text: str) -> list[str]:
    model = load_model(model_name)
    return cast(list[str], model.to_str_tokens(text))


def compute_node_attribution(
    model: HookedTransformer,
    sae_dict: SAEDict,
    metric_fn: MetricFunction,
    text: str,
) -> pd.DataFrame:
    # Get the token strings.
    text_tokens = model.to_str_tokens(text)
    prompt_tokens: list[str] = text_tokens[:-1]  # pyright: ignore
    response_token: list[str] = text_tokens[-1]  # pyright: ignore

    prompt_str = "".join(prompt_tokens)
    response_str = response_token
    sae_cache_dict = get_sae_activation_cache(model, sae_dict, metric_fn, text)

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
                    "node_idx": node_idx.item(),
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
