from functools import partial

import pytest
import torch
from sae_lens import SparseAutoencoder
from transformer_lens import HookedTransformer

from token_trace.sae_activation_cache import get_sae_activation_cache
from token_trace.types import (
    MetricFunction,
    ModuleName,
)
from token_trace.utils import (
    last_token_loss,
)


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture()
def sae_dict(sae: SparseAutoencoder) -> dict[ModuleName, SparseAutoencoder]:
    sae_dict = {ModuleName(sae.cfg.hook_point): sae}
    return sae_dict


def test_get_sae_cache_dict(
    model: HookedTransformer, sae_dict: dict[ModuleName, SparseAutoencoder], prompt: str
):
    metric_fn: MetricFunction = partial(last_token_loss, prompt=prompt)

    sae_cache_dict = get_sae_activation_cache(
        model=model,
        sae_dict=sae_dict,
        metric_fn=metric_fn,
    )

    for name, module_activations in sae_cache_dict.items():
        assert module_activations.module_name == name
        assert module_activations.activations is not None
        assert module_activations.gradients is not None
