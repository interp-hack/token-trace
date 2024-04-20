import pytest
from sae_lens import SparseAutoencoder
from transformer_lens import HookedTransformer

from tests.helpers import TINYSTORIES_MODEL, build_sae_cfg, load_model_cached


@pytest.fixture()
def model() -> HookedTransformer:
    return load_model_cached(TINYSTORIES_MODEL)


@pytest.fixture()
def sae() -> SparseAutoencoder:
    return SparseAutoencoder(build_sae_cfg())


@pytest.fixture()
def text() -> str:
    return "Hello world"
