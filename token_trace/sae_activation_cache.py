"""Activation cache for SAEs"""

from sae_lens import SparseAutoencoder
from transformer_lens import HookedTransformer

from token_trace.sae_patcher import SAEPatcher
from token_trace.types import MetricFunction, ModuleActivations, ModuleName
from token_trace.utils import dense_to_sparse

SAEActivationCache = dict[ModuleName, ModuleActivations]


def get_sae_activation_cache(
    model: HookedTransformer,
    sae_dict: dict[ModuleName, SparseAutoencoder],
    metric_fn: MetricFunction,
    text: str,
    retain_graph: bool = True,
) -> SAEActivationCache:
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
        metric = metric_fn(model, text)
        metric.backward(retain_graph=retain_graph)

    sae_cache_dict = {}
    for name, patcher in sae_patcher_dict.items():
        n_features = patcher.sae.cfg.d_sae
        assert n_features is not None
        sae_cache_dict[name] = ModuleActivations(
            module_name=ModuleName(name),
            # NOTE: Convert dense tensors to sparse tensors
            n_features=n_features,
            activations=dense_to_sparse(patcher.get_node_values()),
            gradients=dense_to_sparse(patcher.get_node_grads()),
        )
    return sae_cache_dict
