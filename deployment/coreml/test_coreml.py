"""Run without weights or Core ML runtime: python -m unittest deployment.coreml.test_coreml."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from .attention import RotaryEmbedding3D
from .export import compare_outputs, load_transformer, load_vae, main, make_inputs, parse_args
from .transformer import MobileI2VTransformer
from .vae import TurboVAEDecoder


class ExportContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.config = {
            "in_channels": 4, "out_channels": 4, "num_attention_heads": 2,
            "attention_head_dim": 12, "caption_channels": 5, "mlp_ratio": 1.5,
            "flow": True,
        }

    def test_optimized_hybrid_matches_reference_across_shapes_and_conditions(self):
        for shape, scale in [((2, 2, 3), (1, 1.4375, 2.5)), ((3, 1, 2), (1, 1, 1))]:
            with self.subTest(shape=shape):
                optimized = MobileI2VTransformer(self.config, shape, scale).eval()
                reference = MobileI2VTransformer(self.config, shape, scale, optimized=False).eval()
                reference.load_state_dict(optimized.state_dict(), strict=True)
                inputs = make_inputs("transformer", 4, shape, torch.Generator().manual_seed(9))
                with torch.no_grad():
                    for timestep, flow in [(0, 0), (731, 1.2)]:
                        inputs["timestep"].fill_(timestep)
                        inputs["flow_score"].fill_(flow)
                        result = optimized(*inputs.values())
                        torch.testing.assert_close(result, reference(*inputs.values()), atol=2e-5, rtol=2e-5)
                        self.assertEqual(tuple(result.shape), (1, 4, *shape))

    def test_rotary_uses_distinct_temporal_height_width_coordinates(self):
        rotary = RotaryEmbedding3D(6, (2, 2, 3), (1, 2, 4))
        result = rotary(torch.ones(1, 1, 12, 6))[0, 0]
        # Last token is (t=1, h=1, w=2), not a flattened position index.
        phase = torch.tensor([1.0, 1.0, 0.5, 0.5, 0.5, 0.5])
        rotation = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        torch.testing.assert_close(result[-1], phase.cos() + rotation * phase.sin())
        torch.testing.assert_close(result[0], torch.ones(6))

    def test_checkpoint_loaders_reject_missing_inference_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(self.config))
            model = MobileI2VTransformer(self.config, (2, 1, 2), (1, 1, 1))
            weights = model.state_dict()
            save_file(weights, path / "diffusion_pytorch_model.safetensors")
            loaded = load_transformer(path, (2, 1, 2), (1, 1, 1))
            torch.testing.assert_close(loaded.proj_out.weight, model.proj_out.weight)
            del weights["proj_out.weight"]
            save_file(weights, path / "diffusion_pytorch_model.safetensors")
            with self.assertRaisesRegex(RuntimeError, "proj_out.weight"):
                load_transformer(path, (2, 1, 2), (1, 1, 1))

            config = {"decoder_block_out_channels": [32] * 4, "decoder_layers_per_block": [1] * 5,
                      "latent_channels": 4}
            (path / "vae.json").write_text(json.dumps(config))
            decoder = TurboVAEDecoder(config).eval()
            weights = decoder.state_dict()
            weights["latents_mean"] = torch.zeros(4)
            weights["aligned_feature_projection_heads.0.weight"] = torch.ones(1)
            torch.save({"Turbo-VAED-LTX": weights}, path / "vae.pth")
            loaded = load_vae(path / "vae.json", path / "vae.pth")
            with torch.no_grad():
                video = loaded(torch.randn(1, 4, 2, 1, 2))
            self.assertEqual(tuple(video.shape), (1, 3, 9, 32, 64))
            self.assertTrue(torch.isfinite(video).all())
            del weights["conv_out.conv.bias"]
            torch.save({"Turbo-VAED-LTX": weights}, path / "vae.pth")
            with self.assertRaisesRegex(RuntimeError, "conv_out.conv.bias"):
                load_vae(path / "vae.json", path / "vae.pth")

    def test_cli_rejects_invalid_shapes_missing_weights_and_overwrites(self):
        base = ["transformer", "--transformer", "weights", "--rope-interpolation", "1", "1", "1",
                "--output", "new.mlpackage"]
        for extra in [["--height", "720"], ["--frames", "16"], ["--width", "0"],
                      ["--rope-interpolation", "1", "nan", "1"], ["--check-samples", "0"]]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(base + extra)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["vae", "--output", "new.mlpackage"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.mlpackage"
            path.mkdir()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(base + ["--output", str(path)])

    def test_metrics_reject_nan_shape_mismatch_and_measure_error(self):
        reference = np.array([-2, 0, 1, 4])
        metrics = compare_outputs(reference, reference + np.array([1, 0, 0, 0]))
        self.assertEqual(metrics["max_abs_error"], 1)
        self.assertAlmostEqual(metrics["psnr_db"], 10 * np.log10(64))
        self.assertIsNone(compare_outputs(reference, reference)["psnr_db"])
        for actual in [np.array([np.nan, 0, 1, 4]), np.array([0])]:
            with self.assertRaises(ValueError):
                compare_outputs(reference, actual)

    @unittest.skipUnless(os.environ.get("MOBILEI2V_RUN_COREML_TESTS") == "1", "optional macOS Core ML runtime test")
    def test_export_reports_passed_failed_and_unvalidated_truthfully(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(self.config))
            model = MobileI2VTransformer(self.config, (2, 1, 2), (1, 1, 1))
            save_file(model.state_dict(), path / "diffusion_pytorch_model.safetensors")
            base = ["transformer", "--transformer", str(path), "--height", "32", "--width", "64",
                    "--frames", "9", "--rope-interpolation", "1", "1", "1", "--check-samples", "1"]
            for status, extra in [("passed", []), ("unvalidated", ["--skip-check"]),
                                  ("failed", ["--min-psnr", "300"])]:
                with self.subTest(status=status):
                    args = base + ["--output", str(path / f"{status}.mlpackage")] + extra
                    if status == "failed":
                        with self.assertRaisesRegex(ValueError, "PSNR threshold"):
                            main(args)
                    else:
                        main(args)
                    report = json.loads((path / f"{status}.validation.json").read_text())
                    self.assertEqual(report["status"], status)
                    self.assertEqual(report["output_shape"], [1, 4, 2, 1, 2])
                    self.assertEqual(len(report["samples"]), 0 if status == "unvalidated" else 1)


if __name__ == "__main__":
    unittest.main()
