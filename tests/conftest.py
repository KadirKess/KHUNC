import pytest
import torch

from khunc import KHUNC, ModelConfig


@pytest.fixture(autouse=True)
def seed():
    torch.set_num_threads(2)
    torch.manual_seed(42)


@pytest.fixture
def small_model():
    return KHUNC(ModelConfig(base_channels=2)).eval()
