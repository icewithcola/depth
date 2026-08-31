#!/usr/bin/env python3
"""Run the official MoGe-2 PyTorch post-processing as a JSON reference.

Development-only utility for comparing the browser implementation with
``moge/model/v2.py::MoGeModel.infer``.  It intentionally imports the official
package and model checkpoint rather than duplicating model code.

Requirements (outside this web project):
  pip install git+https://github.com/microsoft/MoGe.git
  pip install torch torchvision pillow opencv-python numpy

The checkpoint is downloaded by Hugging Face on first use.  The utility emits
JSON to stdout unless ``--output-json`` is provided.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


MODEL_ID = "Ruicheng/moge-2-vits-normal"


def _json_number(value: Any) -> Any:
    """Convert NumPy/Torch scalars and non-finite values to JSON-safe values."""

    try:
        value = value.item()
    except AttributeError:
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_tree(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _json_tree(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_tree(item) for item in value]
    return _json_number(value)


def _finite_depth_stats(depth, mask) -> dict[str, Any]:
    import numpy as np

    depth_np = depth.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy().astype(bool)
    finite = depth_np[np.isfinite(depth_np) & mask_np]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p05": float(np.percentile(finite, 5)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
    }


def _parse_samples(values: list[str], width: int, height: int) -> list[tuple[int, int]]:
    if not values:
        return [(0, 0), (width // 2, height // 2), (width - 1, height - 1)]
    samples: list[tuple[int, int]] = []
    for value in values:
        try:
            x_text, y_text = value.split(",", 1)
            x, y = int(x_text), int(y_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"sample must be written x,y; received {value!r}"
            ) from exc
        if not (0 <= x < width and 0 <= y < height):
            raise argparse.ArgumentTypeError(
                f"sample {value!r} is outside image dimensions {width}x{height}"
            )
        samples.append((x, y))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only official MoGe-2 reference. Reads one image, "
            "runs Ruicheng/moge-2-vits-normal through moge/model/v2.py, and "
            "writes dimensions, intrinsics, mask/depth statistics, samples, "
            "and metric-scale geometry evidence as JSON."
        ),
        epilog=(
            "Requires the official MoGe Python package, PyTorch, OpenCV, "
            "NumPy, and a Hugging Face network/cache for the checkpoint."
        ),
    )
    parser.add_argument("image", type=Path, help="input image (read as RGB)")
    parser.add_argument(
        "-o",
        "--output-json",
        "--output",
        dest="output_json",
        type=Path,
        help="write JSON to this path (default: stdout)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device (default: cuda when available, otherwise cpu)",
    )
    parser.add_argument(
        "--sample",
        dest="samples",
        action="append",
        default=[],
        metavar="X,Y",
        help="pixel sample; repeatable (default: top-left, centre, bottom-right)",
    )
    parser.add_argument(
        "--fov-x",
        type=float,
        default=None,
        help="optional known horizontal FOV in degrees passed to model.infer",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=1800,
        help="base ViT token count; must match the browser (default: 1800)",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=800,
        help="aspect-preserving image long-side cap matching the browser (default: 800)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="opt into mixed precision (default is FP32 to match the browser ONNX graph)",
    )
    args = parser.parse_args()

    # Keep heavyweight imports after argparse so `--help` works in an
    # environment that has not installed the research dependencies.
    import numpy as np
    import torch
    import cv2
    from PIL import Image, ImageOps
    from moge.model.v2 import MoGeModel

    if not args.image.is_file():
        parser.error(f"image does not exist or is not a file: {args.image}")
    try:
        # Browser createImageBitmap(..., imageOrientation='from-image') applies
        # the same EXIF transform before resizing.
        with Image.open(args.image) as image:
            image_rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    except Exception as exc:
        parser.error(f"Pillow could not decode image: {args.image} ({exc})")
    if args.num_tokens <= 0:
        parser.error("--num-tokens must be positive")
    if args.max_side <= 0:
        parser.error("--max-side must be positive")
    original_height, original_width = image_rgb.shape[:2]
    scale = min(1.0, args.max_side / max(original_width, original_height))
    # Match JavaScript Math.round used by browser preprocessing (rather than
    # Python's ties-to-even round).
    target_width = max(1, math.floor(original_width * scale + 0.5))
    target_height = max(1, math.floor(original_height * scale + 0.5))
    if (target_width, target_height) != (original_width, original_height):
        image_rgb = cv2.resize(
            image_rgb,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
    height, width = image_rgb.shape[:2]
    samples = _parse_samples(args.samples, width, height)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.fp16 and device.type != "cuda":
        parser.error("--fp16 requires a CUDA device")
    model = MoGeModel.from_pretrained(MODEL_ID).to(device).eval()
    image_tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)

    infer_kwargs = {
        "use_fp16": args.fp16,
        "num_tokens": args.num_tokens,
    }
    if args.fov_x is not None:
        infer_kwargs["fov_x"] = args.fov_x
    with torch.inference_mode():
        result = model.infer(image_tensor, **infer_kwargs)

        # Re-run only the raw forward at the same default resolution level so
        # the JSON contains evidence for the metric-scale multiplication that
        # infer() applies after affine focal/shift recovery.
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.fp16):
            raw = model.forward(image_tensor.unsqueeze(0), num_tokens=args.num_tokens)

    output_mask = result["mask"].detach().cpu()
    output_depth = result["depth"].detach().cpu()
    output_points = result["points"].detach().cpu()
    output_normal = result.get("normal")
    if output_normal is not None:
        output_normal = output_normal.detach().cpu()
    raw_points = raw["points"][0].detach().cpu()
    raw_scale = raw.get("metric_scale")
    metric_scale = None if raw_scale is None else float(raw_scale.reshape(-1)[0].item())
    selected: dict[str, Any] = {}
    for x, y in samples:
        key = f"{x},{y}"
        item: dict[str, Any] = {
            "depth": float(output_depth[y, x].item()),
            "xyz": _json_tree(output_points[y, x]),
        }
        if output_normal is not None:
            item["normal"] = _json_tree(output_normal[y, x])
        item["mask"] = bool(output_mask[y, x].item())
        item["affine_xyz"] = _json_tree(raw_points[y, x])
        selected[key] = item

    mask_count = int(output_mask.sum().item())
    payload: dict[str, Any] = {
        "model": MODEL_ID,
        "num_tokens": args.num_tokens,
        "max_side": args.max_side,
        "fp16": args.fp16,
        "width": width,
        "height": height,
        "intrinsics": _json_tree(result["intrinsics"]),
        "mask_count": mask_count,
        "mask_ratio": mask_count / (width * height),
        "finite_depth": _finite_depth_stats(output_depth, output_mask),
        "samples": selected,
        "metric_scale_geometry": {
            "metric_scale": _json_number(metric_scale),
            "affine_points_finite_count": int(torch.isfinite(raw_points).all(dim=-1).sum().item()),
            "affine_z_min": _json_number(raw_points[..., 2].min().item()),
            "affine_z_max": _json_number(raw_points[..., 2].max().item()),
            "metric_depth_finite_count": int(torch.isfinite(output_depth).sum().item()),
        },
    }
    serialized = json.dumps(_json_tree(payload), indent=2, sort_keys=True) + "\n"
    if args.output_json is None:
        print(serialized, end="")
    else:
        args.output_json.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
