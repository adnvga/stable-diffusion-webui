import itertools
import os

import torch


_TRUE_VALUES = {"1", "true", "yes", "on"}
DEFAULT_TILE_SIZE = 64
DEFAULT_TILE_OVERLAP = 16
DEFAULT_TILE_PASSES = 3


def enabled():
    return os.environ.get("SD_WEBUI_VAE_TILING", "").lower() in _TRUE_VALUES


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def settings():
    tile_size = max(2, _env_int("SD_WEBUI_VAE_TILE_SIZE", DEFAULT_TILE_SIZE))
    overlap = max(0, _env_int("SD_WEBUI_VAE_TILE_OVERLAP", DEFAULT_TILE_OVERLAP))
    overlap = min(overlap, tile_size - 1)
    passes = max(1, min(3, _env_int("SD_WEBUI_VAE_TILE_PASSES", DEFAULT_TILE_PASSES)))
    return tile_size, overlap, passes


def report_configuration():
    if not enabled():
        return

    tile_size, overlap, passes = settings()
    print(f"[VAE][tiling] enabled tile={tile_size} overlap={overlap} passes={passes}", flush=True)


@torch.inference_mode()
def tiled_scale_multidim(samples, function, tile=(64, 64), overlap=16, upscale_amount=8, out_channels=3, output_device=None):
    dims = len(tile)
    if samples.dim() != dims + 2:
        raise ValueError(f"expected {dims + 2}D samples, got {samples.dim()}D")
    if any(size <= overlap for size in tile):
        raise ValueError("tile dimensions must be greater than overlap")

    if output_device is None:
        output_device = samples.device

    output_shape = [samples.shape[0], out_channels] + [round(size * upscale_amount) for size in samples.shape[2:]]
    output = torch.empty(output_shape, device=output_device, dtype=torch.float32)

    for batch_index in range(samples.shape[0]):
        sample = samples[batch_index:batch_index + 1]
        sample_shape = [sample.shape[0], out_channels] + [round(size * upscale_amount) for size in sample.shape[2:]]
        out = torch.zeros(sample_shape, device=output_device, dtype=output.dtype)
        out_div = torch.zeros(sample_shape, device=output_device, dtype=output.dtype)

        steps = [size - overlap for size in tile]
        ranges = [range(0, sample.shape[dimension + 2], steps[dimension]) for dimension in range(dims)]
        for positions in itertools.product(*ranges):
            sample_tile = sample
            upscaled_positions = []

            for dimension in range(dims):
                position = max(0, min(sample.shape[dimension + 2] - overlap, positions[dimension]))
                length = min(tile[dimension], sample.shape[dimension + 2] - position)
                sample_tile = sample_tile.narrow(dimension + 2, position, length)
                upscaled_positions.append(round(position * upscale_amount))

            tile_output = function(sample_tile)
            if not torch.is_tensor(tile_output):
                raise TypeError("tiled decode function must return a tensor")
            tile_output = tile_output.to(device=output_device, dtype=output.dtype)
            mask = torch.ones_like(tile_output)
            feather = round(overlap * upscale_amount)

            for dimension in range(2, dims + 2):
                axis_feather = min(feather, mask.shape[dimension])
                for offset in range(axis_feather):
                    mask.narrow(dimension, offset, 1).mul_((offset + 1) / axis_feather)
                    mask.narrow(dimension, mask.shape[dimension] - 1 - offset, 1).mul_((offset + 1) / axis_feather)

            tile_out = out
            tile_out_div = out_div
            for dimension in range(dims):
                tile_out = tile_out.narrow(dimension + 2, upscaled_positions[dimension], mask.shape[dimension + 2])
                tile_out_div = tile_out_div.narrow(dimension + 2, upscaled_positions[dimension], mask.shape[dimension + 2])

            tile_out.add_(tile_output * mask)
            tile_out_div.add_(mask)

        output[batch_index:batch_index + 1] = out / out_div.clamp_min(1e-12)

    return output


def decode_tiled(first_stage_model, samples, tile_size=None, overlap=None, passes=None):
    if tile_size is None or overlap is None or passes is None:
        configured_tile_size, configured_overlap, configured_passes = settings()
        tile_size = configured_tile_size if tile_size is None else tile_size
        overlap = configured_overlap if overlap is None else overlap
        passes = configured_passes if passes is None else passes

    tile_size = max(2, int(tile_size))
    overlap = max(0, min(int(overlap), tile_size - 1))
    passes = max(1, min(3, int(passes)))
    downscale_ratio = 2 ** (first_stage_model.decoder.num_resolutions - 1)
    out_channels = first_stage_model.decoder.conv_out.out_channels

    decode_fn = first_stage_model.decode
    variants = [
        (tile_size, tile_size),
        (tile_size // 2, tile_size * 2),
        (tile_size * 2, tile_size // 2),
    ]
    output = None

    for tile_x, tile_y in variants[:passes]:
        pass_overlap = min(overlap, tile_x - 1, tile_y - 1)
        tiled_output = tiled_scale_multidim(
            samples,
            decode_fn,
            tile=(tile_y, tile_x),
            overlap=pass_overlap,
            upscale_amount=downscale_ratio,
            out_channels=out_channels,
            output_device=samples.device,
        )
        output = tiled_output if output is None else output + tiled_output

    return output / passes