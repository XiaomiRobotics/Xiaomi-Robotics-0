"""A tiny random-weight XR0 for tests, built from config instead of the released checkpoint."""

import importlib.util
from unittest import mock

import torch
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

IMAGE_TOKEN = 100
VISION_START = 101
VISION_END = 102
VOCAB = 128


def mibot_importable() -> bool:
    """The post-training package needs mmengine, lightning and decord, which CI may not install."""
    return all(importlib.util.find_spec(name) is not None for name in ("mmengine", "lightning", "decord"))


def tiny_text_config() -> Qwen3VLTextConfig:
    # DiT hard-codes 8 key/value heads of size 128, so the backbone must match that.
    return Qwen3VLTextConfig(
        vocab_size=VOCAB,
        hidden_size=256,
        intermediate_size=384,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=512,
        rope_scaling={"rope_type": "default", "mrope_section": [24, 20, 20], "mrope_interleaved": True},
    )


def tiny_vlm_config() -> Qwen3VLConfig:
    vision = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=96,
        num_heads=4,
        out_hidden_size=256,
        num_position_embeddings=64,
        deepstack_visual_indexes=[0, 1],
    )
    return Qwen3VLConfig(
        text_config=tiny_text_config().to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=IMAGE_TOKEN,
        vision_start_token_id=VISION_START,
        vision_end_token_id=VISION_END,
    )


def build_tiny_xr0(seed: int = 0, **xr0_kwargs):
    """Build ``XR0`` with a tiny Qwen3-VL backbone. Returns ``(model, seen)``.

    ``seen`` records the keyword arguments the backbone was requested with, so tests can
    check which attention implementation XR0 asked for. The tiny backbone itself always
    runs sdpa, so a flash_attention_2 request can be checked without flash_attn installed.
    """
    from mibot.models.VLA.XR0 import XR0, Qwen3VLForConditionalGeneration
    from mibot.models.VLA.XR0 import Qwen3VLTextConfig as XR0TextConfig

    seen = {}

    def fake_vlm(name, **kwargs):
        seen.update(kwargs)
        config = tiny_vlm_config()
        torch.manual_seed(seed)
        return Qwen3VLForConditionalGeneration._from_config(
            config, attn_implementation="eager" if kwargs["attn_implementation"] == "eager" else "sdpa", dtype=kwargs["dtype"]
        )

    with mock.patch.object(Qwen3VLForConditionalGeneration, "from_pretrained", side_effect=fake_vlm), mock.patch.object(
        XR0TextConfig, "from_pretrained", return_value=tiny_text_config()
    ):
        torch.manual_seed(seed)
        model = XR0(dit_num_layers=2, num_steps=3, **xr0_kwargs)
    return model.eval(), seen


def tiny_batch(seed: int = 0) -> dict:
    """One image of 4 visual tokens plus text, a zero action chunk, a random state, all fixed by ``seed``."""
    generator = torch.Generator().manual_seed(seed)
    text = torch.randint(0, 90, (6,), generator=generator)
    input_ids = torch.cat([text[:3], torch.tensor([VISION_START] + [IMAGE_TOKEN] * 4 + [VISION_END]), text[3:]])[None]
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "pixel_values": torch.randn(16, 3 * 2 * 16 * 16, generator=generator),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
        "action": torch.zeros(1, 30, 32),
        "action_mask": torch.ones(1, 30, 32),
        "state": torch.randn(1, 1, 32, generator=generator),
    }
