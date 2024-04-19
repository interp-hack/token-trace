# def get_circuit(
#     model: HookedTransformer,
#     sae_dict: dict[ModuleName, SparseAutoencoder],
#     metric_fn: MetricFunction,
#     node_threshold: float = 0.1,
#     # edge_threshold: float = 0.01,
# ):
#     sae_cache_dict: dict[ModuleName, ModuleActivations] = get_sae_cache_dict(
#         model, sae_dict, metric_fn
#     )

#     # Compute node indirect effects
#     node_indirect_effects: dict[str, SparseTensor] = {}
#     for module_name, sae_cache in sae_cache_dict.items():
#         feature_act = sae_cache.activations
#         feature_grad = sae_cache.gradients
#         # NOTE: currently the zero-patch ablation is hardcoded
#         # TODO: support patching with other activations
#         feature_act_patch = torch.zeros_like(feature_act)
#         indirect_effect = feature_grad * (feature_act - feature_act_patch)

#         # Sum across token positions.
#         # TODO: make this a parameter.
#         indirect_effect = sparse.sum(indirect_effect, dim=1)

#         # Take the mean across examples.
#         # TODO: make this a parameter.
#         indirect_effect = sparse.sum(indirect_effect, dim=0)

#         # Convert back to dense tensor
#         indirect_effect = indirect_effect.coalesce().to_dense()
#         node_indirect_effects[module_name] = indirect_effect

#     nodes: dict[str, list[int]] = {
#         module_name: [] for module_name in sae_cache_dict.keys()
#     }
#     # Filter by node threshold
#     for module_name, indirect_effect in node_indirect_effects.items():
#         nodes[module_name] = (
#             torch.nonzero(indirect_effect.abs() > node_threshold).squeeze().tolist()
#         )

#     edges: dict[str, list[tuple[int, int]]] = {
#         module_name: [] for module_name in sae_cache_dict.keys()
#     }

#     return nodes, edges
