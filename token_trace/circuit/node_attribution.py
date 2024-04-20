from typing import cast, get_args

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
from transformer_lens import HookedTransformer

from token_trace.load_pretrained_model import load_model
from token_trace.sae_activation_cache import get_sae_activation_cache
from token_trace.types import MetricFunction, ModuleType, NodeType, SAEDict
from token_trace.utils import get_layer_from_module_name


class NodeAttributionSchema(pa.SchemaModel):
    layer: Series[int]
    module_type: Series[str] = pa.Field(isin=get_args(ModuleType), nullable=False)
    module_name: Series[str]
    example_idx: Series[int]
    example_str: Series[str]
    node_idx: Series[int]
    node_type: Series[str] = pa.Field(isin=get_args(NodeType), nullable=False)
    token_idx: Series[int]
    token_str: Series[str]
    value: Series[float]
    grad: Series[float]
    indirect_effect: Series[float]


NodeAttributionDataFrame = DataFrame[NodeAttributionSchema]


def get_token_strs(model_name: str, text: str) -> list[str]:
    model = load_model(model_name)
    return cast(list[str], model.to_str_tokens(text))


def validate_node_attribution(node_df: pd.DataFrame) -> NodeAttributionDataFrame:
    validated_df = NodeAttributionSchema.validate(node_df)
    return cast(NodeAttributionDataFrame, validated_df)


def compute_node_attribution(
    model: HookedTransformer,
    sae_dict: SAEDict,
    metric_fn: MetricFunction,
    text: str,
) -> NodeAttributionDataFrame:
    # Get the token strings.
    text_tokens = model.to_str_tokens(text)
    sae_cache_dict = get_sae_activation_cache(model, sae_dict, metric_fn, text)

    # Construct dataframe.
    rows = []
    for module_name, module_activations in sae_cache_dict.items():
        print(f"Processing module {module_name}")
        layer = get_layer_from_module_name(module_name)
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
                    "module_type": "resid",
                    "example_idx": example_idx.item(),
                    "example_str": text,
                    "node_idx": node_idx.item(),
                    "token_idx": token_idx.item(),
                    "token_str": text_tokens[token_idx],
                    "value": act.item(),
                    "grad": grad.item(),
                    "indirect_effect": ie_atp.item(),
                    "node_type": "feature" if node_idx < 24576 else "error",
                }
            )

    df = pd.DataFrame(rows)
    # Filter out zero indirect effects
    df = df[df["indirect_effect"] != 0]
    print(f"{len(df)} non-zero indirect effects found.")
    validated_df = NodeAttributionSchema.validate(df)
    return cast(NodeAttributionDataFrame, validated_df)
