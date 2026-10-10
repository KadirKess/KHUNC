import pytest
import torch

from khunc import load_model, save_model


@pytest.mark.parametrize("prefix", ["", "model.", "model._orig_mod."])
def test_original_and_lightning_state_dicts_load(small_model, tmp_path, prefix):
    path = tmp_path / "checkpoint.ckpt"
    state = {prefix + key: value for key, value in small_model.state_dict().items()}
    torch.save({"state_dict": state, "epoch": 25}, path)
    loaded = load_model(path, device="cpu")
    assert loaded.config == small_model.config
    image = torch.rand(1, 3, 32, 48)
    with torch.inference_mode():
        assert torch.equal(loaded(image), small_model(image))


def test_save_and_reload_preserves_weights_and_refuses_overwrite(small_model, tmp_path):
    path = save_model(small_model, tmp_path / "model.pt")
    loaded = load_model(path, device="cpu")
    assert all(
        torch.equal(value, loaded.state_dict()[name])
        for name, value in small_model.state_dict().items()
    )
    with pytest.raises(FileExistsError):
        save_model(small_model, path)
