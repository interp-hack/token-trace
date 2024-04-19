from dataclasses import dataclass
from typing import NewType, Protocol

import torch
from transformer_lens import HookedTransformer


class MetricFunction(Protocol):
    def __call__(self, model: HookedTransformer) -> torch.Tensor: ...


ModuleName = NewType("ModuleName", str)
# NOTE: I can't believe torch doesn't have a type for sparse tensors
SparseTensor = torch.Tensor


@dataclass
class ModuleActivations:
    module_name: ModuleName
    activations: SparseTensor
    gradients: SparseTensor
