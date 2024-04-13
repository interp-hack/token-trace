import functools
import pickle
from functools import partial
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sae_lens import SparseAutoencoder
from sae_lens.training.utils import BackwardsCompatibleUnpickler
from transformer_lens import HookedTransformer

DEFAULT_MODEL_NAME = "gpt2-small"
DEFAULT_REPO_ID = "jbloom/GPT2-Small-SAEs"
DEFAULT_TEXT = "When John and Mary went to the shops, John gave the bag to Mary"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@functools.lru_cache(maxsize=1)
def load_model(model_name: str) -> HookedTransformer:
    if model_name != DEFAULT_MODEL_NAME:
        raise ValueError(f"Unknown model: {model_name}")
    return cast(
        HookedTransformer, HookedTransformer.from_pretrained(model_name).to(DEVICE)
    )


def load_sae(layer: int) -> SparseAutoencoder:
    filename = (
        f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.hook_resid_pre_24576.pt"
    )
    path = hf_hub_download(repo_id=DEFAULT_REPO_ID, filename=filename)
    # Hacky way to get torch to unpickle an old version of SAELens model
    fake_pickle = SimpleNamespace()
    fake_pickle.Unpickler = BackwardsCompatibleUnpickler
    fake_pickle.__name__ = pickle.__name__
    data = torch.load(path, map_location=torch.device("cpu"), pickle_module=fake_pickle)
    sparse_autoencoder = SparseAutoencoder(cfg=data["cfg"])
    sparse_autoencoder.load_state_dict(data["state_dict"])
    return sparse_autoencoder.to(DEVICE)


def loss_helper(
    model: HookedTransformer,
    text: str,
) -> torch.Tensor:
    """Compute the loss of the model when predicting the last token."""
    loss = model(text, return_type="loss", loss_per_token=True)
    return loss[0, -1]


def get_token_strs(model_name: str, text: str) -> list[str]:
    model = load_model(model_name)
    return cast(list[str], model.to_str_tokens(text))


@functools.lru_cache(maxsize=100)
def compute_node_attribution(
    model_name: str,
    text: str,
    *,
    metric: str = "neg_loss",
    max_k: int = 10_000,
) -> pd.DataFrame:
    model = load_model(model_name)
    if metric == "neg_loss":
        # Last-token loss
        metric_fn = lambda model: -loss_helper(model, text)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Pre-compute cache.
    _, cache = model.run_with_cache(text)

    # Get the token strings.
    text_tokens = model.to_str_tokens(text)
    prompt_tokens: list[str] = text_tokens[:-1]  # pyright: ignore
    response_token: list[str] = text_tokens[-1]  # pyright: ignore

    prompt_str = "".join(prompt_tokens)
    response_str = response_token

    # Define hook to patch features into model.
    def patch_hook_sae_reconstruction(
        a_orig: torch.Tensor,
        hook: Any,  # noqa: ARG001
        a_rec: torch.Tensor,
    ) -> torch.Tensor:
        assert torch.allclose(a_orig, a_rec, rtol=1e-3, atol=1e-3)
        return a_rec

    # TODO: un-hardcode this
    layers = list(range(12))
    rows = []

    for layer in layers:
        print("Layer", layer)
        sae = load_sae(layer)
        hook_point = sae.cfg.hook_point

        # Turn on gradients
        sae.train()

        # Compute a_rec
        a_orig = cache[sae.cfg.hook_point]
        a_rec, z = sae(a_orig)[:2]
        assert z.requires_grad
        z.retain_grad()
        a_err = a_orig - a_rec.detach()
        # Add the SAE error so we exactly match the original computational graph
        a_err.requires_grad = True
        a_err.retain_grad()
        a_rec = a_rec + a_err

        # Patch the SAE into the computational graph so it receives grad
        hook = (hook_point, partial(patch_hook_sae_reconstruction, a_rec=a_rec))
        with model.hooks(fwd_hooks=[hook]):
            patched_loss = metric_fn(model)
            patched_loss.backward()

            grad_loss_z = z.grad
            assert grad_loss_z is not None
            assert not torch.isclose(grad_loss_z, torch.zeros_like(grad_loss_z)).all()
            # Indirect effect relative to zero ablation = gradient * magnitude
            # NOTE: Directly computing this is not accurate, we need integrated gradients...
            indirect_effects = (grad_loss_z * (0 - z)).sum(dim=0).sum(dim=0)
            for feature_idx, ie_atp in enumerate(indirect_effects):
                rows.append(
                    {
                        "layer": layer,
                        "feature": feature_idx,
                        "indirect_effect": ie_atp.item(),
                        "prompt_str": prompt_str,
                        "response_str": response_str,
                        "node_type": "feature",
                    }
                )

            grad_loss_err = a_err.grad
            assert grad_loss_err is not None
            assert not torch.isclose(
                grad_loss_err, torch.zeros_like(grad_loss_err)
            ).all()
            indirect_effects = (grad_loss_err * (0 - a_err)).sum(dim=0).sum(dim=0)
            for feature_idx, ie_atp in enumerate(indirect_effects):
                rows.append(
                    {
                        "layer": layer,
                        "feature": feature_idx,
                        "indirect_effect": ie_atp.item(),
                        "prompt_str": prompt_str,
                        "response_str": response_str,
                        "node_type": "error",
                    }
                )

    df = pd.DataFrame(rows)
    # Filter out zero indirect effects
    df = df[df["indirect_effect"] != 0]
    df["absolute_ie"] = df["indirect_effect"].abs()
    df = df.sort_values("absolute_ie", ascending=False).head(max_k)
    return df
