import pytest
import torch
from sae_lens import SparseAutoencoder
from transformer_lens import HookedTransformer

from token_trace.sae_activation_cache import get_sae_activation_cache
from token_trace.types import (
    MetricFunction,
    ModuleName,
)
from token_trace.utils import last_token_prediction_loss


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")


def test_get_sae_cache_dict(
    model: HookedTransformer, sae_dict: dict[ModuleName, SparseAutoencoder], text: str
):
    metric_fn: MetricFunction = last_token_prediction_loss

    sae_cache_dict = get_sae_activation_cache(
        model=model, sae_dict=sae_dict, metric_fn=metric_fn, text=text
    )

    for name, module_activations in sae_cache_dict.items():
        assert module_activations.module_name == name
        assert module_activations.activations is not None
        assert module_activations.gradients is not None
