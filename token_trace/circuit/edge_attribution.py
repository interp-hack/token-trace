# import pandas as pd
# from transformer_lens import HookedTransformer

# from token_trace.types import MetricFunction, SAEDict


# def compute_edge_attributions(
#     model: HookedTransformer,
#     sae_dict: SAEDict,
#     metric_fn: MetricFunction,
#     text: str,
#     node_df: pd.DataFrame,
# ):
#     # sae_activation_cache = get_sae_activation_cache(model, sae_dict, metric_fn)
#     # # TODO: need to test if we can get the gradient of intermediate activations.

#     # TODO: un-hardcode

#     rows = []
#     # The edge effects for the last layer to the loss
#     # are the same as the node effects for the last layer.
#     # So we can just copy them over.
#     last_layer_nodes = node_df[node_df.layer == 11]
#     for _, row in last_layer_nodes.iterrows():
#         rows.append(
#             {
#                 "upstream_layer": row["layer"],
#                 "upstream_example_idx": 0,
#                 "upstream_feature_idx": row["node_idx"],
#                 "upstream_token_idx": row["token"],
#                 "upstream_node_type": row["node_type"],
#                 "downstream_layer": 12,
#                 "downstream_example_idx": 0,
#                 "downstream_feature_idx": row["node_idx"],
#                 "downstream_token_idx": row["token"],
#                 "value": row["value"],
#                 "grad": row["grad"],
#                 "indirect_effect": row["indirect_effect"],
#                 "module_name": row["module_name"],
#                 "token_str": row["token_str"],
#                 "example_str": row["prompt_str"] + row["response_str"],
#             }
#         )

#     for layer in range(11, 0, -1):
#         prev_layer_nodes = node_df[node_df.layer == layer]["node_idx"]
#         print(f"Layer {layer}: {len(prev_layer_nodes)} nodes")
