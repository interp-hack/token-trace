"""Code and utilities to find sparse feature circuits"""

from dataclasses import dataclass
from typing import NewType, Protocol

import torch
from sae_lens import SparseAutoencoder
from torch import sparse, sparse_coo_tensor
from transformer_lens import HookedTransformer

from token_trace.sae_patcher import SAEPatcher


class MetricFunction(Protocol):
    def __call__(self, model: HookedTransformer) -> torch.Tensor: ...


ModuleName = NewType("ModuleName", str)
# NOTE: I can't believe torch doesn't have a type for sparse tensors
SparseTensor = torch.Tensor


def dense_to_sparse(tensor: torch.Tensor, sparse_dim: int = -1) -> torch.Tensor:
    """Convert a dense tensor to a sparse tensor of the same shape"""
    indices = torch.nonzero(tensor).t()
    values = tensor[*indices]
    return sparse_coo_tensor(
        indices,
        values,
        tensor.size(),
        device=tensor.device,
        dtype=tensor.dtype,
    )


@dataclass
class ModuleActivations:
    module_name: ModuleName
    activations: SparseTensor
    gradients: SparseTensor


def last_token_loss(model: HookedTransformer, prompt: str) -> torch.Tensor:
    loss = model(prompt, return_type="loss", loss_per_token=True)
    return loss[0, -1]


def get_sae_cache_dict(
    model: HookedTransformer,
    sae_dict: dict[ModuleName, SparseAutoencoder],
    metric_fn: MetricFunction,
) -> dict[ModuleName, ModuleActivations]:
    sae_patcher_dict = {name: SAEPatcher(sae) for name, sae in sae_dict.items()}

    # Patch the SAEs into the computational graph
    # NOTE: problem, we're running out of CUDA memory here...
    with model.hooks(
        fwd_hooks=[
            sae_patcher.get_forward_hook() for sae_patcher in sae_patcher_dict.values()
        ],
        bwd_hooks=[
            sae_patcher.get_backward_hook() for sae_patcher in sae_patcher_dict.values()
        ],
    ):
        metric = metric_fn(model)
        metric.backward()

    sae_cache_dict = {}
    for name, patcher in sae_patcher_dict.items():
        sae_cache_dict[name] = ModuleActivations(
            module_name=ModuleName(name),
            # NOTE: Convert dense tensors to sparse tensors
            activations=dense_to_sparse(patcher.get_node_values()).detach(),
            gradients=dense_to_sparse(patcher.get_node_grads()).detach(),
        )
    return sae_cache_dict


def get_circuit(
    model: HookedTransformer,
    sae_dict: dict[ModuleName, SparseAutoencoder],
    metric_fn: MetricFunction,
    node_threshold: float = 0.1,
    edge_threshold: float = 0.01,
):
    sae_cache_dict: dict[ModuleName, ModuleActivations] = get_sae_cache_dict(
        model, sae_dict, metric_fn
    )

    # Compute node indirect effects
    node_indirect_effects: dict[str, SparseTensor] = {}
    for module_name, sae_cache in sae_cache_dict.items():
        feature_act = sae_cache.activations
        feature_grad = sae_cache.gradients
        # NOTE: currently the zero-patch ablation is hardcoded
        # TODO: support patching with other activations
        feature_act_patch = torch.zeros_like(feature_act)
        indirect_effect = feature_grad * (feature_act - feature_act_patch)

        # Sum across token positions.
        # TODO: make this a parameter.
        indirect_effect = sparse.sum(indirect_effect, dim=1)

        # Take the mean across examples.
        # TODO: make this a parameter.
        indirect_effect = sparse.sum(indirect_effect, dim=0)

        # Convert back to dense tensor
        indirect_effect = indirect_effect.coalesce().to_dense()
        node_indirect_effects[module_name] = indirect_effect

    nodes: dict[str, list[int]] = {
        module_name: [] for module_name in sae_cache_dict.keys()
    }
    # Filter by node threshold
    for module_name, indirect_effect in node_indirect_effects.items():
        nodes[module_name] = (
            torch.nonzero(indirect_effect.abs() > node_threshold).squeeze().tolist()
        )

    edges: dict[str, list[tuple[int, int]]] = {
        module_name: [] for module_name in sae_cache_dict.keys()
    }

    return nodes, edges
