"""Device and attention backend selection for the XPU, CUDA and CPU paths.

Most tests run anywhere: detection order, which attention implementation is requested,
how the servers and the parity tool use them. They patch availability, so no accelerator
and no checkpoint is needed. The last test class builds a tiny random-weight XR0 from
config and compares the CPU result with the XPU result, and is skipped without an XPU.

Run from the repository root with ``python -m pytest tests``.
"""

import logging
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "deploy", ROOT / "xr0", ROOT / "tests"):
    sys.path.insert(0, str(path))

import check_parity
import device as dev
import server as deploy_server
import tiny_xr0


def _patch_availability(*, xpu: bool, cuda: bool, flash_attn: bool = False, module=dev):
    """Make a copy of the device module see exactly the given backends and flash_attn install state."""

    def fake_is_available(device_type: str) -> bool:
        return {"xpu": xpu, "cuda": cuda}.get(device_type, device_type == "cpu")

    stack = mock.patch.object(module, "is_available", side_effect=fake_is_available)
    flash = mock.patch.object(module, "flash_attn_available", return_value=flash_attn)
    return _Both(stack, flash)


class _Both:
    def __init__(self, *managers):
        self.managers = managers

    def __enter__(self):
        for manager in self.managers:
            manager.__enter__()

    def __exit__(self, *exc):
        for manager in reversed(self.managers):
            manager.__exit__(*exc)


class DeviceDetectionTest(unittest.TestCase):
    def test_xpu_preferred_over_cuda(self):
        with _patch_availability(xpu=True, cuda=True):
            self.assertEqual(dev.default_device_type(), "xpu")
            self.assertEqual(dev.resolve_device(None), torch.device("xpu"))

    def test_cuda_when_no_xpu(self):
        with _patch_availability(xpu=False, cuda=True):
            self.assertEqual(dev.default_device_type(), "cuda")

    def test_cpu_when_no_accelerator(self):
        with _patch_availability(xpu=False, cuda=False):
            self.assertEqual(dev.resolve_device(None), torch.device("cpu"))

    def test_explicit_device_wins_over_detection(self):
        with _patch_availability(xpu=True, cuda=True):
            self.assertEqual(dev.resolve_device("cpu"), torch.device("cpu"))
            self.assertEqual(dev.resolve_device("cuda:1"), torch.device("cuda:1"))
            self.assertEqual(dev.resolve_device(torch.device("xpu:1")), torch.device("xpu:1"))


class AttnImplementationTest(unittest.TestCase):
    def setUp(self):
        dev._warned.clear()

    def test_cuda_with_flash_attn_keeps_flash_attention_2(self):
        with _patch_availability(xpu=False, cuda=True, flash_attn=True):
            self.assertEqual(dev.resolve_attn_implementation("cuda"), "flash_attention_2")
            self.assertEqual(dev.resolve_attn_implementation(None), "flash_attention_2")

    def test_cuda_without_flash_attn_uses_sdpa(self):
        with _patch_availability(xpu=False, cuda=True, flash_attn=False):
            self.assertEqual(dev.resolve_attn_implementation("cuda"), "sdpa")

    def test_xpu_never_uses_flash_attention_even_if_installed(self):
        with _patch_availability(xpu=True, cuda=False, flash_attn=True):
            self.assertEqual(dev.resolve_attn_implementation("xpu"), "sdpa")
            self.assertEqual(dev.resolve_attn_implementation(None), "sdpa")

    def test_requested_value_wins(self):
        with _patch_availability(xpu=True, cuda=False):
            self.assertEqual(dev.resolve_attn_implementation("xpu", "eager"), "eager")

    def test_fallback_is_logged_once(self):
        with _patch_availability(xpu=True, cuda=False):
            with self.assertLogs(dev.logger, level=logging.WARNING) as logs:
                dev.resolve_attn_implementation("xpu")
                dev.resolve_attn_implementation("xpu")
            self.assertEqual(len(logs.records), 1)

    def test_checkpoint_argument_asks_sdpa_for_the_backbone_only(self):
        # The released top level class does not declare SDPA support, so a plain "sdpa" is rejected.
        self.assertEqual(dev.checkpoint_attn_argument("sdpa"), {"": "eager", "vlm_config": "sdpa"})
        self.assertEqual(dev.checkpoint_attn_argument("flash_attention_2"), "flash_attention_2")
        self.assertEqual(dev.checkpoint_attn_argument("eager"), "eager")

    def test_both_copies_of_the_module_are_identical(self):
        # deploy/ and xr0/ are installed separately, so the module exists twice.
        self.assertEqual((ROOT / "deploy" / "device.py").read_text(), (ROOT / "xr0" / "mibot" / "utils" / "device.py").read_text())


class DeployServerTest(unittest.TestCase):
    def setUp(self):
        dev._warned.clear()

    def _build(self, **kwargs):
        model = mock.MagicMock()
        model.to.return_value = model
        with mock.patch.object(deploy_server.AutoModel, "from_pretrained", return_value=model) as load, mock.patch.object(
            deploy_server.AutoProcessor, "from_pretrained"
        ):
            server = deploy_server.Server("ckpt", "localhost", 1, **kwargs)
            self.assertFalse(load.called, "the model must be built in the spawned process, not in __init__")
            server._build_model()
        return load, model

    def test_cli_defaults_detect_the_device(self):
        with mock.patch.object(sys, "argv", ["server.py", "--model", "m"]):
            args = deploy_server.parse_args()
        self.assertIsNone(args.device)
        self.assertIsNone(args.attn_implementation)

    def test_xpu_host_loads_sdpa_and_moves_to_xpu(self):
        with _patch_availability(xpu=True, cuda=False, flash_attn=True):
            load, model = self._build()
        self.assertEqual(load.call_args.kwargs["attn_implementation"], {"": "eager", "vlm_config": "sdpa"})
        model.to.assert_any_call(torch.device("xpu"))

    def test_cuda_host_keeps_flash_attention_2_and_cuda(self):
        with _patch_availability(xpu=False, cuda=True, flash_attn=True):
            load, model = self._build()
        self.assertEqual(load.call_args.kwargs["attn_implementation"], "flash_attention_2")
        model.to.assert_any_call(torch.device("cuda"))

    def test_explicit_device_and_attention_are_honoured(self):
        with _patch_availability(xpu=True, cuda=True, flash_attn=True):
            load, model = self._build(device="cuda:1", attn_implementation="eager")
        self.assertEqual(load.call_args.kwargs["attn_implementation"], "eager")
        model.to.assert_any_call(torch.device("cuda:1"))


class CheckParityToolTest(unittest.TestCase):
    def test_compare_reports_error_statistics(self):
        reference = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        actual = reference + torch.tensor([[0.0, 0.0, 0.0, 0.5]])
        stats = check_parity.compare(reference, actual, atol=0.1, rtol=0.0)
        self.assertAlmostEqual(stats["mse"], 0.0625)
        self.assertAlmostEqual(stats["max_abs_diff"], 0.5)
        self.assertAlmostEqual(stats["inside_tolerance"], 0.75)

    def test_observation_is_fixed_by_the_seed(self):
        first_images, first_state = check_parity.make_observation(3)
        second_images, second_state = check_parity.make_observation(3)
        self.assertEqual(first_images[0].tobytes(), second_images[0].tobytes())
        self.assertTrue((first_state == second_state).all())
        self.assertNotEqual(first_images[0].tobytes(), first_images[1].tobytes())

    def test_noise_context_draws_on_cpu_and_restores_randn_like(self):
        original = torch.randn_like
        template = torch.zeros(2, 3, dtype=torch.bfloat16)
        torch.manual_seed(5)
        expected = original(template.float()).to(torch.bfloat16)
        torch.manual_seed(5)
        with check_parity.cpu_noise():
            noise = torch.randn_like(template)
        self.assertTrue(torch.equal(noise, expected))
        self.assertIs(torch.randn_like, original)


@unittest.skipUnless(tiny_xr0.mibot_importable(), "mmengine, lightning and decord are needed to import mibot")
class XR0AttentionSelectionTest(unittest.TestCase):
    def setUp(self):
        from mibot.utils import device as mibot_dev

        self.mibot_dev = mibot_dev
        mibot_dev._warned.clear()

    def _patch(self, **kwargs):
        return _patch_availability(module=self.mibot_dev, **kwargs)

    def test_xpu_host_asks_the_backbone_for_sdpa(self):
        with self._patch(xpu=True, cuda=False, flash_attn=True):
            model, seen = tiny_xr0.build_tiny_xr0()
        self.assertEqual(seen["attn_implementation"], "sdpa")
        self.assertEqual(model.vlm.config._attn_implementation, "sdpa")

    def test_cuda_host_keeps_flash_attention_2(self):
        with self._patch(xpu=False, cuda=True, flash_attn=True):
            _, seen = tiny_xr0.build_tiny_xr0()
        self.assertEqual(seen["attn_implementation"], "flash_attention_2")

    def test_configured_value_wins(self):
        with self._patch(xpu=True, cuda=False):
            _, seen = tiny_xr0.build_tiny_xr0(attn_implementation="eager")
        self.assertEqual(seen["attn_implementation"], "eager")


@unittest.skipUnless(tiny_xr0.mibot_importable() and hasattr(torch, "xpu") and torch.xpu.is_available(), "needs an XPU and the mibot dependencies")
class TinyXR0XpuParityTest(unittest.TestCase):
    """Random-weight XR0 from config: the same batch and the same noise on CPU and XPU."""

    @staticmethod
    def _run(model, batch, device):
        import copy

        model = copy.deepcopy(model).to(device)
        batch = {key: value.to(device) for key, value in batch.items()}
        with check_parity.cpu_noise():
            torch.manual_seed(1)
            return model.generate(batch).float().cpu()

    def test_cpu_and_xpu_actions_agree_in_bf16(self):
        model, seen = tiny_xr0.build_tiny_xr0()
        self.assertEqual(seen["attn_implementation"], "sdpa")
        batch = tiny_xr0.tiny_batch()
        reference = self._run(model, batch, "cpu")
        actual = self._run(model, batch, "xpu")
        self.assertEqual(tuple(reference.shape), (1, 30, 32))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreater(float(reference.abs().max()), 0.1, "the comparison would be trivial")
        stats = check_parity.compare(reference, actual, atol=5e-2, rtol=2e-2)
        self.assertEqual(stats["inside_tolerance"], 1.0, stats)
        self.assertLess(stats["mse"], 1e-4, stats)


if __name__ == "__main__":
    unittest.main()
