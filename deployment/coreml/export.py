# Copyright 2026 MobileI2V contributors.
# SPDX-License-Identifier: Apache-2.0
"""Export trained MobileI2V components, with mandatory-by-default validation."""

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .transformer import MobileI2VTransformer
from .vae import TurboVAEDecoder


def read_json(path):
    with Path(path).open() as file:
        return json.load(file)


def load_transformer(path, latent_shape, interpolation, optimized=True):
    path = Path(path)
    config = read_json(path / "config.json")
    model = MobileI2VTransformer(config, latent_shape, interpolation, optimized)
    weights = load_file(str(path / "diffusion_pytorch_model.safetensors"))
    model.load_state_dict(weights, strict=True)
    return model.eval()


def load_vae(config_path, checkpoint_path):
    model = TurboVAEDecoder(read_json(config_path))
    path = Path(checkpoint_path)
    if path.suffix == ".safetensors":
        weights = load_file(str(path))
    else:
        weights = torch.load(path, map_location="cpu", weights_only=True)
        weights = weights.get("Turbo-VAED-LTX", weights)
    # Explicitly omit only the training-only feature alignment heads and
    # latent statistics, which the original decoder.forward does not consume.
    weights = {
        key: value for key, value in weights.items()
        if key not in {"latents_mean", "latents_std"}
        and not key.startswith("aligned_feature_projection_heads.")
    }
    model.load_state_dict(weights, strict=True)
    return model.eval()


def compare_outputs(reference, actual):
    reference, actual = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape:
        raise ValueError(f"Output shapes differ: {reference.shape} != {actual.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("Non-finite model output")
    error = reference - actual
    mse = float(np.mean(error ** 2))
    peak = float(np.max(np.abs(reference)))
    # JSON null represents exact equality, rather than non-standard Infinity.
    psnr = None if mse == 0 else 10 * math.log10(max(peak ** 2, 1e-30) / mse)
    return {"max_abs_error": float(np.max(np.abs(error))), "psnr_db": psnr}


def make_inputs(component, channels, latent_shape, generator):
    frames, height, width = latent_shape
    latent = torch.randn((1, channels, frames, height, width), generator=generator)
    if component == "vae":
        return {"latents": latent}
    mask = torch.zeros(1, frames * height * width)
    mask[:, :height * width] = 1
    return {
        "hidden_states": latent, "timestep": torch.tensor([731.0]),
        "cond_mask": mask, "flow_score": torch.tensor([0.6]),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=["transformer", "vae"])
    parser.add_argument("--transformer", type=Path, help="Diffusers-format hybrid checkpoint directory")
    parser.add_argument("--vae-config", type=Path, default=Path(__file__).with_name("turbo_vae.json"))
    parser.add_argument("--vae-checkpoint", type=Path, help="Turbo-VAED-LTX checkpoint, not the teacher VAE")
    parser.add_argument("--output", type=Path, required=True, help="New .mlpackage path; never overwritten")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--rope-interpolation", type=float, nargs=3, metavar=("T", "H", "W"),
                        help="Required for transformer; use 1 1.4375 2.5 for the original 736x1280 export")
    parser.add_argument("--precision", choices=["float32", "float16"], default="float32",
                        help="float16 keeps normalization and attention accumulation in float32")
    parser.add_argument("--target", choices=["iOS18"], default="iOS18")
    parser.add_argument("--compute-unit", choices=["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"], default="CPU_ONLY")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-samples", type=int, default=3)
    parser.add_argument("--min-psnr", type=float, default=30)
    parser.add_argument("--skip-check", action="store_true", help="Export only; report explicitly remains unvalidated")
    args = parser.parse_args(argv)
    if args.height <= 0 or args.width <= 0 or args.height % 32 or args.width % 32:
        parser.error("height and width must be positive multiples of 32 (use 736, not 720, for padded 720p)")
    if args.frames < 1 or (args.frames - 1) % 8:
        parser.error("frames must be 1 + 8k, e.g. 1, 9, 17 or 33")
    if args.component == "transformer" and (not args.transformer or not args.rope_interpolation):
        parser.error("transformer requires --transformer and --rope-interpolation")
    if args.rope_interpolation and any(not math.isfinite(n) or n <= 0 for n in args.rope_interpolation):
        parser.error("RoPE interpolation factors must be finite and positive")
    if args.component == "vae" and not args.vae_checkpoint:
        parser.error("vae requires --vae-checkpoint; random-weight exports are not supported")
    if args.check_samples < 1 or not math.isfinite(args.min_psnr) or args.min_psnr <= 0:
        parser.error("check-samples and min-psnr must be positive")
    if args.output.suffix != ".mlpackage":
        parser.error("output must end in .mlpackage")
    if args.output.exists() or args.output.with_suffix(".validation.json").exists():
        parser.error("output or validation report already exists; choose a new output path")
    return args


def main(argv=None):
    args = parse_args(argv)
    import coremltools as ct
    import diffusers

    torch.manual_seed(args.seed)
    latent_shape = ((args.frames - 1) // 8 + 1, args.height // 32, args.width // 32)
    if args.component == "transformer":
        model = load_transformer(args.transformer, latent_shape, args.rope_interpolation)
        channels = model.in_channels
    else:
        model = load_vae(args.vae_config, args.vae_checkpoint)
        channels = model.latent_channels
    generator = torch.Generator().manual_seed(args.seed)
    inputs = make_inputs(args.component, channels, latent_shape, generator)
    with torch.no_grad():
        expected = model(*inputs.values())
        if not torch.isfinite(expected).all():
            raise ValueError("Non-finite PyTorch output before conversion")
        output_grid = latent_shape if args.component == "transformer" else (args.frames, args.height, args.width)
        expected_shape = (1, model.out_channels, *output_grid)
        if tuple(expected.shape) != expected_shape:
            raise ValueError(f"Unexpected PyTorch output: {tuple(expected.shape)} != {expected_shape}")
        traced = torch.jit.trace(model, tuple(inputs.values()))
    mlmodel = ct.convert(
        traced, convert_to="mlprogram", minimum_deployment_target=getattr(ct.target, args.target),
        inputs=[ct.TensorType(name=name, shape=tensor.shape, dtype=np.float32) for name, tensor in inputs.items()],
        outputs=[ct.TensorType(name="sample", dtype=np.float32)],
        compute_precision=(
            ct.precision.FLOAT32 if args.precision == "float32" else
            ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type not in {
                "reduce_mean", "reduce_sum", "pow", "square", "rsqrt", "add", "sub",
                "mul", "real_div", "matmul", "layer_norm",
            })
        ),
        compute_units=getattr(ct.ComputeUnit, args.compute_unit),
    )
    report = {
        "component": args.component, "status": "unvalidated", "seed": args.seed,
        "precision": args.precision, "target": args.target, "compute_unit": args.compute_unit,
        "precision_policy": "full_fp32" if args.precision == "float32" else "fp16_with_fp32_norm_and_accumulation",
        "latent_shape": [1, channels, *latent_shape], "output_shape": list(expected.shape),
        "rope_interpolation": args.rope_interpolation,
        "checkpoint": str(args.transformer if args.component == "transformer" else args.vae_checkpoint),
        "versions": {"torch": torch.__version__, "diffusers": diffusers.__version__,
                     "coremltools": ct.__version__, "macos": platform.mac_ver()[0]},
        "min_psnr_db": args.min_psnr, "samples": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.output))
    report_path = args.output.with_suffix(".validation.json")
    try:
        if not args.skip_check:
            for index in range(args.check_samples):
                sample = make_inputs(args.component, channels, latent_shape, generator)
                if args.component == "transformer":
                    # Include the final denoising boundary and a different flow score.
                    sample["timestep"].fill_(0 if index == 0 else 1000 / (index + 1))
                    sample["flow_score"].fill_(0 if index == 0 else 1.2)
                with torch.no_grad():
                    reference = model(*sample.values()).numpy()
                actual = mlmodel.predict({name: tensor.numpy() for name, tensor in sample.items()})["sample"]
                metrics = compare_outputs(reference, actual)
                report["samples"].append(metrics)
                if metrics["psnr_db"] is not None and metrics["psnr_db"] < args.min_psnr:
                    raise ValueError(f"Sample {index} failed PSNR threshold: {metrics}")
            report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}; validation={report['status']}; report={report_path}")


if __name__ == "__main__":
    main()
