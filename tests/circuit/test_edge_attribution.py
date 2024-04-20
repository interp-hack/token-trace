import pytest

from token_trace.circuit.edge_attribution import (
    compute_edge_attribution,
    filter_edges,
)
from token_trace.circuit.node_attribution import NodeAttributionDataFrame
from token_trace.sae_activation_cache import SAEActivationCache


# NOTE: currently failing due to SAE dict having only one SAE
@pytest.mark.xfail
def test_compute_node_attribution(
    node_attribution_df: NodeAttributionDataFrame,
    sae_activation_cache: SAEActivationCache,
):
    edge_df = compute_edge_attribution(
        node_attribution_df,
        sae_acts_clean=sae_activation_cache,
    )
    assert not edge_df.empty

    node_df = filter_edges(edge_df, max_n_edges=10)
    assert not node_df.empty
    assert len(edge_df) <= 10
