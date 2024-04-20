from token_trace.circuit.node_attribution import compute_node_attribution
from token_trace.types import (
    HookedTransformer,
    MetricFunction,
    SAEDict,
)


def test_compute_node_attribution(
    model: HookedTransformer,
    sae_dict: SAEDict,
    metric_fn: MetricFunction,
    text: str,
):
    node_df = compute_node_attribution(
        model=model,
        sae_dict=sae_dict,
        metric_fn=metric_fn,
        text=text,
    )
    assert not node_df.empty
