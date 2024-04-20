import pytest
from sae_lens import SparseAutoencoder
from transformer_lens import HookedTransformer

from tests.helpers import TINYSTORIES_MODEL, build_sae_cfg, load_model_cached
from token_trace.circuit.node_attribution import (
    NodeAttributionDataFrame,
    compute_node_attribution,
    filter_nodes,
)
from token_trace.sae_activation_cache import (
    SAEActivationCache,
    get_sae_activation_cache,
)
from token_trace.types import MetricFunction, ModuleName
from token_trace.utils import last_token_prediction_loss


@pytest.fixture()
def model() -> HookedTransformer:
    return load_model_cached(TINYSTORIES_MODEL)


@pytest.fixture()
def sae() -> SparseAutoencoder:
    return SparseAutoencoder(build_sae_cfg())


@pytest.fixture()
def sae_dict(sae: SparseAutoencoder) -> dict[ModuleName, SparseAutoencoder]:
    sae_dict = {ModuleName(sae.cfg.hook_point): sae}
    return sae_dict


@pytest.fixture()
def sae_activation_cache(
    model: HookedTransformer,
    sae_dict: dict[ModuleName, SparseAutoencoder],
    metric_fn: MetricFunction,
    text: str,
) -> SAEActivationCache:
    return get_sae_activation_cache(model, sae_dict, metric_fn, text)


@pytest.fixture()
def metric_fn() -> MetricFunction:
    return last_token_prediction_loss


@pytest.fixture()
def text() -> str:
    return "Hello world"


@pytest.fixture()
def node_attribution_df(
    model: HookedTransformer,
    sae_activation_cache: SAEActivationCache,
    text: str,
) -> NodeAttributionDataFrame:
    node_df = compute_node_attribution(
        model=model,
        sae_activation_cache=sae_activation_cache,
        text=text,
    )
    node_df = filter_nodes(node_df, max_n_nodes=100)
    return node_df
