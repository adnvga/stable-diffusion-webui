import importlib
from types import SimpleNamespace

import pytest
import torch

from modules import sd_vae_tiling


class FakeConvOut:
    out_channels = 3


class FakeDecoder:
    num_resolutions = 2
    conv_out = FakeConvOut()


class FakeVAE:
    decoder = FakeDecoder()

    def __init__(self):
        self.decode_calls = 0
        self.decode_shapes = []

    def decode(self, samples):
        self.decode_calls += 1
        self.decode_shapes.append(tuple(samples.shape))
        decoded = samples.repeat(1, 3, 1, 1)
        return decoded.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)


def test_tiled_decode_preserves_shape_and_covers_edges():
    vae = FakeVAE()
    samples = torch.ones(1, 1, 5, 7)

    output = sd_vae_tiling.decode_tiled(vae, samples, tile_size=3, overlap=1, passes=1)

    assert output.shape == (1, 3, 10, 14)
    assert output.dtype == torch.float32
    assert torch.allclose(output, torch.ones_like(output))
    assert vae.decode_calls == 12


def test_tiled_decode_never_decodes_the_full_latent():
    vae = FakeVAE()
    samples = torch.ones(1, 1, 17, 14)

    sd_vae_tiling.decode_tiled(vae, samples, tile_size=8, overlap=2, passes=3)

    assert vae.decode_calls > 3
    assert all(shape[-2:] != samples.shape[-2:] for shape in vae.decode_shapes)


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_enabled_accepts_true_environment_values(monkeypatch, value):
    monkeypatch.setenv("SD_WEBUI_VAE_TILING", value)

    assert sd_vae_tiling.enabled()


def test_enabled_is_off_by_default(monkeypatch):
    monkeypatch.delenv("SD_WEBUI_VAE_TILING", raising=False)

    assert not sd_vae_tiling.enabled()


def test_report_configuration_prints_effective_settings(monkeypatch, capsys):
    monkeypatch.setenv("SD_WEBUI_VAE_TILING", "1")
    monkeypatch.setenv("SD_WEBUI_VAE_TILE_SIZE", "48")
    monkeypatch.setenv("SD_WEBUI_VAE_TILE_OVERLAP", "12")
    monkeypatch.setenv("SD_WEBUI_VAE_TILE_PASSES", "2")

    sd_vae_tiling.report_configuration()

    assert capsys.readouterr().out == "[VAE][tiling] enabled tile=48 overlap=12 passes=2\n"


def test_sdxl_decode_first_stage_uses_tiling(monkeypatch, capsys):
    import modules.paths  # noqa: F401
    diffusion = importlib.import_module("sgm.models.diffusion")

    monkeypatch.setenv("SD_WEBUI_VAE_TILING", "1")
    monkeypatch.setenv("SD_WEBUI_VAE_TILE_SIZE", "3")
    monkeypatch.setenv("SD_WEBUI_VAE_TILE_OVERLAP", "1")
    monkeypatch.setenv("SD_WEBUI_VAE_TILE_PASSES", "1")
    monkeypatch.setattr(diffusion, "AutoencoderKL", FakeVAE)

    vae = FakeVAE()
    model = SimpleNamespace(scale_factor=1.0, disable_first_stage_autocast=True, first_stage_model=vae)
    samples = torch.ones(1, 1, 5, 7)

    output = diffusion.DiffusionEngine.decode_first_stage(model, samples)

    assert output.shape == (1, 3, 10, 14)
    assert all(shape[-2:] != samples.shape[-2:] for shape in vae.decode_shapes)
    assert "[VAE][tiling] active tile=3 overlap=1 passes=1 input=(1, 1, 5, 7)" in capsys.readouterr().out