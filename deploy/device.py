# Copyright (C) 2026 Xiaomi Corporation.
"""Device and attention backend selection for the XPU, CUDA and CPU paths.

Everything that used to hard-code ``.cuda()`` or ``flash_attention_2`` goes through
this module. Detection order is XPU, then CUDA, then CPU, and an explicit device
always wins over detection. Flash Attention 2 needs a CUDA build of PyTorch plus the
``flash_attn`` package, so every other backend uses PyTorch SDPA and says so once.

This file exists twice on purpose, as ``deploy/device.py`` and
``xr0/mibot/utils/device.py``, because the inference scripts and the post-training
package are installed separately. ``tests/test_device.py`` checks they stay equal.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Union

import torch

logger = logging.getLogger(__name__)

DeviceLike = Union[str, torch.device, None]
_ACCELERATORS = ("xpu", "cuda")
_warned: set[str] = set()


def is_available(device_type: str) -> bool:
    """Whether ``device_type`` is usable in this PyTorch build and on this host."""
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "cuda":
        return torch.cuda.is_available()
    return device_type == "cpu"


def default_device_type() -> str:
    """Best available backend: ``xpu``, then ``cuda``, then ``cpu``."""
    for device_type in _ACCELERATORS:
        if is_available(device_type):
            return device_type
    return "cpu"


def resolve_device(device: DeviceLike = None) -> torch.device:
    """Turn ``None``, a string or a ``torch.device`` into a concrete device.

    ``None`` auto-detects. An explicit value is returned as given, so an unsupported
    device fails at the first kernel instead of silently moving to another backend.
    """
    if isinstance(device, torch.device):
        return device
    if device is None:
        return torch.device(default_device_type())
    return torch.device(device)


def flash_attn_available() -> bool:
    """Whether the ``flash_attn`` package is installed (it is CUDA only)."""
    return importlib.util.find_spec("flash_attn") is not None


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message)


def resolve_attn_implementation(device: DeviceLike = None, requested: str | None = None) -> str:
    """Attention implementation to pass to ``from_pretrained``.

    An explicit ``requested`` value wins. Otherwise Flash Attention 2 is used on CUDA
    when ``flash_attn`` is installed, and PyTorch SDPA everywhere else.
    """
    if requested:
        return requested
    resolved = resolve_device(device)
    if resolved.type == "cuda" and flash_attn_available():
        return "flash_attention_2"
    _warn_once(
        resolved.type,
        f"Flash Attention 2 is not used on {resolved.type} (it needs CUDA and the flash_attn package), using sdpa.",
    )
    return "sdpa"


def checkpoint_attn_argument(attn_implementation: str) -> str | dict:
    """Value for ``attn_implementation=`` when loading a released Xiaomi-Robotics-0 checkpoint.

    The checkpoint's top level class (``MiBoTForActionGeneration``) does not declare SDPA
    support, so ``attn_implementation="sdpa"`` is rejected, even though only the Qwen3-VL
    backbone reads that setting (the action head always calls ``F.scaled_dot_product_attention``).
    SDPA is therefore requested for the backbone only. Other values pass through unchanged.
    """
    if attn_implementation == "sdpa":
        return {"": "eager", "vlm_config": "sdpa"}
    return attn_implementation
