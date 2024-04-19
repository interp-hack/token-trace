import pytest
import torch

from token_trace.utils import dense_to_sparse


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")


def test_dense_to_sparse_device(device: torch.device):
    dense_tensor = torch.tensor([[3, 0, 0], [0, 4, 5], [0, 0, 0]], device=device)
    sparse_tensor = dense_to_sparse(dense_tensor)
    assert sparse_tensor.device.type == device.type


def test_dense_to_sparse_dim_2():
    dense_tensor = torch.tensor([[3, 0, 0], [0, 4, 5], [0, 0, 0]])
    sparse_tensor = dense_to_sparse(dense_tensor)
    dense_reconstructed = sparse_tensor.to_dense()
    assert torch.allclose(dense_tensor, dense_reconstructed)


def test_dense_to_sparse_dim_3():
    dense_tensor = torch.tensor(
        [
            [[0, 0, 0], [0, 8, 0], [0, 0, 0]],
            [[5, 0, 0], [0, 0, 7], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 6, 0]],
        ]
    )
    sparse_tensor = dense_to_sparse(dense_tensor)
    dense_reconstructed = sparse_tensor.to_dense()
    assert torch.allclose(dense_tensor, dense_reconstructed)
