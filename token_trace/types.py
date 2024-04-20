from dataclasses import dataclass
from typing import Literal, NewType, Protocol

import torch
from sae_lens import SparseAutoencoder
from transformer_lens import HookedTransformer


class MetricFunction(Protocol):
    def __call__(self, model: HookedTransformer, text: str) -> torch.Tensor: ...


ActType = Literal["feature", "error"]
ModuleType = Literal["resid", "mlp", "attn"]
ModuleName = NewType("ModuleName", str)
# NOTE: I can't believe torch doesn't have a type for sparse tensors
SparseTensor = torch.Tensor

SAEDict = dict[ModuleName, SparseAutoencoder]


@dataclass
class ModuleActivations:
    module_name: ModuleName
    activations: SparseTensor
    gradients: SparseTensor
    n_features: int
