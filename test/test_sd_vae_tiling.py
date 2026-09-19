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
    monkeypatch.setenv("SD_WEBUI_VAE_CLEAR_CACHE", "1")

    sd_vae_tiling.report_configuration()

    assert capsys.readouterr().out == "[VAE][tiling] enabled tile=48 overlap=12 passes=2 clear_cache=on\n"


def test_trim_cuda_cache_reports_released_memory(monkeypatch, capsys):
    gib = 1024 ** 3
    samples = SimpleNamespace(is_cuda=True, device=torch.device("cuda:0"))
    reserved = iter([10 * gib, 7 * gib])
    allocated = iter([6 * gib, 6 * gib])
    timer = iter([1.0, 1.0125])
    empty_cache_calls = []
    monkeypatch.setenv("SD_WEBUI_VAE_TILING", "1")
    monkeypatch.setenv("SD_WEBUI_VAE_CLEAR_CACHE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: next(reserved))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: next(allocated))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True))
    monkeypatch.setattr(sd_vae_tiling.time, "perf_counter", lambda: next(timer))

    sd_vae_tiling.trim_cuda_cache(samples, "before", 6)

    assert empty_cache_calls == [True]
    assert capsys.readouterr().out == (
        "[VAE][tiling][CACHE] phase=before batch=6 device=cuda:0 "
        "reserved_before=10.000 GiB cached_before=4.000 GiB "
        "reserved_after=7.000 GiB released=3.000 GiB "
        "cached_after=1.000 GiB allocated=6.000 GiB elapsed=12.5 ms\n"
    )


def test_trim_cuda_cache_is_disabled_by_default(monkeypatch):
    samples = SimpleNamespace(is_cuda=True, device=torch.device("cuda:0"))
    monkeypatch.delenv("SD_WEBUI_VAE_CLEAR_CACHE", raising=False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: pytest.fail("empty_cache should not be called"))

    sd_vae_tiling.trim_cuda_cache(samples, "before", 1)


def test_report_vram_measurement_prints_compact_peak(monkeypatch, capsys):
    gib = 1024 ** 3
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 7.25 * gib)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 8.5 * gib)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 6.5 * gib)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 8 * gib)

    sd_vae_tiling._report_vram_measurement((torch.device("cuda:0"), 6 * gib, 7 * gib))

    assert capsys.readouterr().out == (
        "[VAE][tiling][VRAM] device=cuda:0 allocated_start=6.000 GiB "
        "allocated_peak=7.250 GiB allocated_delta=1.250 GiB allocated_end=6.500 GiB "
        "reserved_start=7.000 GiB reserved_peak=8.500 GiB reserved_end=8.000 GiB\n"
    )


def test_decode_context_labels_batch_image_and_resets():
    with sd_vae_tiling.decode_context(2, 6):
        assert sd_vae_tiling.log_context() == " image=2/6"

    assert sd_vae_tiling.log_context() == ""


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

    with sd_vae_tiling.decode_context(2, 6):
        output = diffusion.DiffusionEngine.decode_first_stage(model, samples)

    assert output.shape == (1, 3, 10, 14)
    assert all(shape[-2:] != samples.shape[-2:] for shape in vae.decode_shapes)
    assert "[VAE][tiling] active image=2/6 tile=3 overlap=1 passes=1 input=(1, 1, 5, 7)" in capsys.readouterr().out


@pytest.mark.parametrize("environment_value", [None, "0"])
def test_disabled_features_preserve_original_sdxl_decode(monkeypatch, capsys, environment_value):
    import modules.paths  # noqa: F401
    monkeypatch.setenv("GIT_PYTHON_GIT_EXECUTABLE", r"C:\ai\git\bin\git.exe")
    monkeypatch.setenv("GIT_PYTHON_REFRESH", "quiet")
    memory_debug = importlib.import_module("modules.memory_debug")
    diffusion = importlib.import_module("sgm.models.diffusion")
    legacy_ddpm = importlib.import_module("ldm.models.diffusion.ddpm")

    for name in ["SD_WEBUI_VAE_TILING", "SD_WEBUI_MEMORY_DEBUG", "SD_WEBUI_VAE_CLEAR_CACHE"]:
        if environment_value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, environment_value)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: pytest.fail("empty_cache should not be called"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: pytest.fail("peak stats should not be reset"))

    sdxl_vae = FakeVAE()
    sdxl_model = SimpleNamespace(scale_factor=1.0, disable_first_stage_autocast=True, first_stage_model=sdxl_vae)
    legacy_vae = FakeVAE()
    legacy_model = SimpleNamespace(scale_factor=1.0, first_stage_model=legacy_vae)
    samples = torch.ones(1, 1, 5, 7)

    sd_vae_tiling.report_configuration()
    sd_vae_tiling.trim_cuda_cache(samples, "before", 1)
    memory_debug.reset_peak()
    memory_debug.snapshot("disabled")
    with memory_debug.vae_decode_trace(sdxl_model):
        sdxl_output = diffusion.DiffusionEngine.decode_first_stage(sdxl_model, samples)
        legacy_output = legacy_ddpm.LatentDiffusion.decode_first_stage(legacy_model, samples)

    assert sdxl_output.shape == legacy_output.shape == (1, 3, 10, 14)
    assert sdxl_vae.decode_calls == legacy_vae.decode_calls == 1
    assert sdxl_vae.decode_shapes == legacy_vae.decode_shapes == [(1, 1, 5, 7)]
    assert capsys.readouterr().out == ""