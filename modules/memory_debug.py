import os
from contextlib import contextmanager

import torch

from modules import shared


_TRUE_VALUES = {"1", "true", "yes", "on"}


def enabled():
    option = getattr(getattr(shared, "opts", None), "memory_debug", False)
    environment = os.environ.get("SD_WEBUI_MEMORY_DEBUG", "").lower()
    return bool(option) or environment in _TRUE_VALUES


def reset_peak():
    if enabled() and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _is_vae_trace_module(name):
    if name in {"post_quant_conv", "decoder.conv_in", "decoder.norm_out", "decoder.conv_out"}:
        return True

    if name.startswith("decoder.mid."):
        return len(name.split(".")) == 3

    parts = name.split(".")
    if len(parts) == 4 and parts[:2] == ["decoder", "up"] and parts[2].isdigit():
        return parts[3] == "upsample"

    return len(parts) == 5 and parts[:2] == ["decoder", "up"] and parts[2].isdigit() and parts[3] in {"block", "attn"} and parts[4].isdigit()


@contextmanager
def vae_decode_trace(model):
    if not enabled() or not torch.cuda.is_available():
        yield
        return

    first_stage_model = getattr(model, "first_stage_model", None)
    if first_stage_model is None:
        yield
        return

    handles = []

    def make_hook(module_name):
        def hook(module, inputs, output):
            snapshot(f"vae.{module_name}.after", {"output": output})

        return hook

    for name, module in first_stage_model.named_modules():
        if _is_vae_trace_module(name):
            handles.append(module.register_forward_hook(make_hook(name)))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _format_bytes(value):
    value = int(value)
    return f"{value} bytes ({value / (1024 ** 3):.3f} GiB)"


def _tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()

    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())

    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)

    return 0


def _iter_tensors(value):
    if torch.is_tensor(value):
        yield value
        return

    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _describe_tensor(name, value):
    tensors = list(_iter_tensors(value))
    if not tensors:
        return

    if len(tensors) == 1 and torch.is_tensor(value):
        tensor = tensors[0]
        print(
            f"[VRAM][tensor] {name}: shape={tuple(tensor.shape)} "
            f"dtype={tensor.dtype} device={tensor.device} size={_format_bytes(_tensor_bytes(tensor))}",
            flush=True,
        )
        return

    total_bytes = sum(_tensor_bytes(tensor) for tensor in tensors)
    devices = sorted({str(tensor.device) for tensor in tensors})
    dtypes = sorted({str(tensor.dtype) for tensor in tensors})
    print(
        f"[VRAM][tensor] {name}: tensors={len(tensors)} "
        f"dtypes={','.join(dtypes)} devices={','.join(devices)} size={_format_bytes(total_bytes)}",
        flush=True,
    )


def snapshot(label, tensors=None):
    if not enabled() or not torch.cuda.is_available():
        return None

    device = torch.cuda.current_device()
    stats = torch.cuda.memory_stats(device)
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    active = stats.get("active_bytes.all.current", allocated)

    print(
        f"[VRAM][{label}] device=cuda:{device} "
        f"allocated={_format_bytes(allocated)} "
        f"active={_format_bytes(active)} "
        f"reserved={_format_bytes(reserved)} "
        f"peak_allocated={_format_bytes(peak_allocated)} "
        f"peak_reserved={_format_bytes(peak_reserved)} "
        f"free={_format_bytes(free)} total={_format_bytes(total)}",
        flush=True,
    )

    if tensors:
        for name, value in tensors.items():
            _describe_tensor(name, value)

    return {
        "allocated": allocated,
        "active": active,
        "reserved": reserved,
        "peak_allocated": peak_allocated,
        "peak_reserved": peak_reserved,
        "free": free,
        "total": total,
    }
