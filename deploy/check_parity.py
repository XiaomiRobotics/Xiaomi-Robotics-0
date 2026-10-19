# Copyright (C) 2026 Xiaomi Corporation.
"""Compare the action chunk of a released checkpoint on a reference device and a target device.

The same fixed observation (two camera views, proprioceptive state and action mask, as in
the README example) goes through ``model(**inputs)`` on both devices. The model starts the
flow from noise drawn with ``torch.randn_like``, and that generator differs between
backends, so the noise is drawn on the CPU and moved, which gives both devices the same
starting point. The check passes when every element of the decoded action is inside
``atol + rtol * abs(reference)``.

Usage:
    python deploy/check_parity.py --model XiaomiRobotics/Xiaomi-Robotics-0-LIBERO --device xpu
"""

import argparse
import contextlib
import sys

import numpy as np
import torch
from device import checkpoint_attn_argument, resolve_attn_implementation, resolve_device
from PIL import Image
from transformers import AutoModel, AutoProcessor

ROBOT_TYPE = "libero_all"
INSTRUCTION = (
    "<|im_start|>user\nThe following observations are captured from multiple views.\n"
    "# Base View\n<|vision_start|><|image_pad|><|vision_end|>\n"
    "# Left-Wrist View\n<|vision_start|><|image_pad|><|vision_end|>\n"
    "Generate robot actions for the task:\n{task} /no_cot<|im_end|>\n"
    "<|im_start|>assistant\n<cot></cot><|im_end|>\n"
)


def make_observation(seed: int = 0, size: int = 224, state_dim: int = 32):
    """Two smooth synthetic camera views and a proprioceptive state, fixed by ``seed``."""
    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    images = []
    for view in range(2):
        phase = rng.uniform(0.0, 1.0, size=3).astype(np.float32)
        grid = np.stack([np.add.outer(axis * (c + 1 + view), axis * phase[c]) % 1.0 for c in range(3)], axis=-1)
        grid = 0.8 * grid + 0.2 * rng.uniform(size=grid.shape).astype(np.float32)
        images.append(Image.fromarray((grid * 255).astype(np.uint8)))
    state = rng.uniform(-1.0, 1.0, size=(state_dim,)).astype(np.float32)
    return images, state


@contextlib.contextmanager
def cpu_noise():
    """Draw ``torch.randn_like`` noise on the CPU, then move it to the requested device."""
    original = torch.randn_like

    def randn_like(tensor, *args, **kwargs):
        noise = original(tensor.detach().to("cpu", torch.float32), *args, **kwargs)
        return noise.to(tensor.device, tensor.dtype)

    torch.randn_like = randn_like
    try:
        yield
    finally:
        torch.randn_like = original


@torch.no_grad()
def run_once(model, processor, images, state, seed: int) -> torch.Tensor:
    device = model.device
    inputs = processor(
        text=[INSTRUCTION.format(task="Pick up the red block.")],
        images=images,
        videos=None,
        padding=True,
        return_tensors="pt",
    ).to(device)
    inputs["seed"] = seed
    inputs["state"] = torch.from_numpy(state).to(device, model.dtype).view(1, 1, -1)
    inputs["action_mask"] = processor.get_action_mask(ROBOT_TYPE).to(device, model.dtype)
    with cpu_noise():
        outputs = model(**inputs)
    return processor.decode_action(outputs.actions, robot_type=ROBOT_TYPE).float().cpu()


def compare(reference: torch.Tensor, actual: torch.Tensor, atol: float, rtol: float) -> dict:
    diff = (actual - reference).abs()
    inside = diff <= atol + rtol * reference.abs()
    mse = float(((actual - reference) ** 2).mean())
    return {
        "elements": int(reference.numel()),
        "reference_abs_max": float(reference.abs().max()),
        "mse": mse,
        "relative_mse": mse / max(float((reference**2).mean()), 1e-12),
        "max_abs_diff": float(diff.max()),
        "inside_tolerance": float(inside.float().mean()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="Path or Hugging Face id of the checkpoint.")
    parser.add_argument("--device", default=None, help="Device under test. Defaults to xpu, then cuda, then cpu.")
    parser.add_argument("--reference-device", default="cpu", help="Reference device. Default: cpu.")
    parser.add_argument("--attn-implementation", default=None, help="Defaults to sdpa unless CUDA with flash_attn.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    reference_device = resolve_device(args.reference_device)
    # Both runs use the same attention implementation so only the backend differs.
    attn = args.attn_implementation or ("sdpa" if device.type != reference_device.type else resolve_attn_implementation(device))

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, attn_implementation=checkpoint_attn_argument(attn), dtype=torch.bfloat16).eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    images, state = make_observation(args.seed)

    model.to(reference_device)
    reference = run_once(model, processor, images, state, args.seed)
    model.to(device)
    actual = run_once(model, processor, images, state, args.seed)

    stats = compare(reference, actual, args.atol, args.rtol)
    print(f"reference={reference_device} device={device} attn={attn} torch={torch.__version__} shape={tuple(reference.shape)}")
    for key, value in stats.items():
        print(f"{key}: {value}")
    passed = stats["inside_tolerance"] == 1.0
    print("PASS" if passed else "FAIL", f"(atol={args.atol}, rtol={args.rtol})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
