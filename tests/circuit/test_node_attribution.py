from token_trace.circuit.node_attribution import (
    compute_node_attribution,
    filter_nodes,
)
from token_trace.sae_activation_cache import SAEActivationCache
from token_trace.types import (
    HookedTransformer,
)


def test_compute_node_attribution(
    model: HookedTransformer,
    sae_activation_cache: SAEActivationCache,
    text: str,
):
    node_df = compute_node_attribution(
        model=model,
        sae_activation_cache=sae_activation_cache,
        text=text,
    )
    assert not node_df.empty

    node_df = filter_nodes(node_df, max_n_nodes=10)
    assert not node_df.empty
    assert len(node_df) <= 10
