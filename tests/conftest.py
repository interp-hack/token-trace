import pytest
from sae_lens import SparseAutoencoder
from transformer_lens import HookedTransformer

from tests.helpers import TINYSTORIES_MODEL, build_sae_cfg, load_model_cached
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
def metric_fn() -> MetricFunction:
    return last_token_prediction_loss


@pytest.fixture()
def text() -> str:
    return "Hello world"
