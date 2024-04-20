import logging
from typing import cast, get_args

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series

from token_trace.circuit.node_attribution import (
    NodeAttributionDataFrame,
    get_nodes_in_module,
)
from token_trace.sae_activation_cache import SAEActivationCache
from token_trace.types import ModuleName, ModuleType
from token_trace.utils import setup_logger


class EdgeAttributionSchema(pa.SchemaModel):
    # Downstream node
    d_layer: Series[int]
    d_module_type: Series[str] = pa.Field(isin=get_args(ModuleType), nullable=False)
    d_module_name: Series[str]
    d_example_idx: Series[int]
    d_token_idx: Series[int]
    d_act_idx: Series[int]
    # Upstream node
    u_layer: Series[int]
    u_module_type: Series[str] = pa.Field(isin=get_args(ModuleType), nullable=False)
    u_module_name: Series[str]
    u_example_idx: Series[int]
    u_token_idx: Series[int]
    u_act_idx: Series[int]
    # Value
    ie: Series[float]
    abs_ie: Series[float] = pa.Field(
        gt=0,
    )


EdgeAttributionDataFrame = DataFrame[EdgeAttributionSchema]

logger = setup_logger(__name__, logging.INFO)


def validate_edge_attribution(edge_df: pd.DataFrame) -> EdgeAttributionDataFrame:
    validated_df = EdgeAttributionSchema.validate(edge_df)
    return cast(EdgeAttributionDataFrame, validated_df)


def filter_edges(
    edge_ie_df: EdgeAttributionDataFrame,
    *,
    min_edge_ie: float = 0.0,
    max_n_edges: int = -1,
):
    edge_index_cols = [
        "d_module_name",
        "d_act_idx",
        "d_token_idx",
        "u_module_name",
        "u_act_idx",
        "u_token_idx",
    ]

    # Mean across examples
    edge_ie_df["edge_abs_ie"] = edge_ie_df.groupby(edge_index_cols)["abs_ie"].transform(
        "mean"
    )

    # Filter out nodes with low indirect effect
    df = edge_ie_df[edge_ie_df.edge_abs_ie > min_edge_ie]

    if max_n_edges == -1:
        # Return all nodes
        return validate_edge_attribution(df)
    else:
        # Select top edges
        df = df[edge_index_cols + ["edge_abs_ie"]].drop_duplicates()
        df = df.sort_values("edge_abs_ie", ascending=False)
        df = df.head(max_n_edges)
        df = edge_ie_df.merge(df, on=edge_index_cols, how="inner")
        return validate_edge_attribution(df)


def compute_edge_attribution(
    node_ie_df: NodeAttributionDataFrame,
    *,
    sae_acts_clean: SAEActivationCache,
    sae_acts_patch: SAEActivationCache | None = None,  # noqa
) -> EdgeAttributionDataFrame:
    rows = []
    for layer in range(11, 0, -1):
        curr_layer = layer
        prev_layer = layer - 1
        curr_module_name = ModuleName(f"blocks.{curr_layer}.hook_resid_pre")
        prev_module_name = ModuleName(f"blocks.{prev_layer}.hook_resid_pre")
        nodes_in_curr_layer: NodeAttributionDataFrame = get_nodes_in_module(
            node_ie_df, module_name=curr_module_name
        )
        nodes_in_prev_layer: NodeAttributionDataFrame = get_nodes_in_module(
            node_ie_df, module_name=prev_module_name
        )

        logger.info(f"Finding edges between {curr_module_name} and {prev_module_name}")

        for _, downstream_node in nodes_in_curr_layer.iterrows():
            logger.debug(f"Processing downstream node {downstream_node}")
            # backprop the downstream node activation
            d_example_idx = downstream_node.example_idx
            d_token_idx = downstream_node.token_idx
            d_act_idx = downstream_node.act_idx

            d_grad = sae_acts_clean[curr_module_name].gradients[
                d_example_idx, d_token_idx, d_act_idx
            ]
            # Save the gradient for later
            d_grad = d_grad.detach()
            d_act = sae_acts_clean[curr_module_name].activations[
                d_example_idx, d_token_idx, d_act_idx
            ]
            d_act.backward(retain_graph=True)
            d_act = d_act.detach()

            # get the upstream node gradient
            upstream_module_acts = sae_acts_clean[prev_module_name]
            # We only want upstream nodes with the same example_idx and not-more-than token_idx
            upstream_nodes = nodes_in_prev_layer[
                (nodes_in_prev_layer.example_idx == d_example_idx)
                & (nodes_in_prev_layer.token_idx <= d_token_idx)
            ]
            for _, upstream_node in upstream_nodes.iterrows():
                u_example_idx = upstream_node.example_idx
                assert u_example_idx == d_example_idx
                u_token_idx = upstream_node.token_idx
                assert u_token_idx <= d_token_idx
                u_act_idx = upstream_node.act_idx
                u_act = upstream_module_acts.activations[
                    u_example_idx, u_token_idx, u_act_idx
                ]
                u_grad = upstream_module_acts.gradients[
                    u_example_idx, u_token_idx, u_act_idx
                ]

                # TODO: Implement u_patch
                edge_ie = (d_grad * u_grad * u_act).item()

                rows.append(
                    {
                        # Indexing
                        "d_layer": curr_layer,
                        "d_module_type": "resid",
                        "d_module_name": curr_module_name,
                        "d_example_idx": d_example_idx,
                        "d_token_idx": d_token_idx,
                        "d_act_idx": d_act_idx,
                        "u_layer": prev_layer,
                        "u_module_type": "resid",
                        "u_module_name": prev_module_name,
                        "u_example_idx": u_example_idx,
                        "u_token_idx": u_token_idx,
                        "u_act_idx": u_act_idx,
                        # Value
                        "ie": edge_ie,
                        "abs_ie": abs(edge_ie),
                    }
                )

    edge_df = pd.DataFrame(rows)
    edge_df = edge_df[edge_df.abs_ie > 0.0]
    return validate_edge_attribution(edge_df)
