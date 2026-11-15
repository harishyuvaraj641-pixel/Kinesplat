"""Command-line interface for kinesplat.

Examples
--------
    python -m splatlab.cli demo --preset draft
    python -m splatlab.cli images .\\photos --name "Back garden"
    python -m splatlab.cli serve --port 8000 --open
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import DEMO_DIR, ROOT, get_preset


def _demo_pipeline(preset: str, out_dir: Path, threads: int | None,
                   device: str = "auto") -> dict:
    from .pipeline import PipelineConfig, run_reconstruction

    if not (DEMO_DIR / "source").exists():
        raise SystemExit(f"Demo images are missing from {DEMO_DIR / 'source'}")
    return run_reconstruction(
        out_dir,
        [DEMO_DIR / "source"],
        PipelineConfig(name="Toy Courtyard", preset=preset, threads=threads, device=device),
        log=lambda m: print(m, flush=True),
        progress=lambda stage, pct, msg: print(
            f"[{int(pct * 100):3d}%] {msg}", flush=True
        ),
    )


def cmd_demo(args) -> int:
    out = Path(args.out or (ROOT / "outputs" / "demo"))
    started = time.time()
    result = _demo_pipeline(args.preset, out, args.threads, args.device)
    print("\nDone in", round(time.time() - started, 1), "s")
    print("Scene :", out / "output" / "scene.json")
    print("Splat :", out / "output" / "model.splat")
    print("Splats:", f"{result['gaussians']:,}")
    print("PSNR  :", f"{result['metrics']['psnr']:.2f} dB")
    return 0


def cmd_images(args) -> int:
    from .pipeline import PipelineConfig, run_reconstruction

    inputs = [Path(p) for p in args.inputs]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise SystemExit("Missing input path(s): " + ", ".join(missing))
    out = Path(args.out or (ROOT / "outputs" / (args.name or "scene").replace(" ", "_")))
    cfg = PipelineConfig(
        name=args.name or out.name.replace("_", " ").title(),
        preset=args.preset,
        backend=args.backend,
        pose_backend=args.poses,
        use_depth=not args.no_depth,
        threads=args.threads,
        device=args.device,
    )
    result = run_reconstruction(
        out, inputs, cfg,
        log=lambda m: print(m, flush=True),
        progress=lambda stage, pct, msg: print(f"[{int(pct * 100):3d}%] {msg}", flush=True),
    )
    print("\nFinished in", result["seconds"], "s")
    print(json.dumps(result["metrics"], indent=2))
    print("Scene:", out / "output" / "scene.json")
    if result.get("mesh"):
        print("Mesh :", out / result["mesh"])
    return 0


def cmd_mesh(args) -> int:
    """Single photo -> textured mesh (TripoSR)."""
    from .device import resolve_device
    from .mesh import reconstruct_mesh

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"Missing input file: {src}")
    out = Path(args.out or (ROOT / "outputs" / (src.stem + "_mesh")))
    result = reconstruct_mesh(
        src, out / "output",
        name=args.name or src.stem.replace("_", " ").title(),
        device=resolve_device(args.device).type,
        resolution=args.resolution,
        threshold=args.threshold,
        chunk_size=args.chunk_size,
        remove_background=not args.no_remove_bg,
        foreground_ratio=args.foreground_ratio,
        log=lambda m: print(m, flush=True),
        progress=lambda stage, pct, msg: print(f"[{int(pct * 100):3d}%] {msg}", flush=True),
    )
    print("\nFinished in", result["seconds"], "s")
    print(json.dumps(result["metrics"], indent=2))
    print("Scene:", out / "output" / "scene.json")
    print("Mesh :", out / "output" / "mesh.glb")
    return 0


def cmd_serve(args) -> int:
    from .server import serve

    serve(args.host, args.port, args.open, args.device)
    return 0


def cmd_model(args) -> int:
    from .depth import DepthEstimator, download_depth_model

    if args.triposr:
        from .mesh import download_weights, is_downloaded, weights_path

        if args.check:
            if is_downloaded():
                path = weights_path()
                print(f"TripoSR weights ready: {path} ({path.stat().st_size / 1e6:.0f} MB)")
                return 0
            print("TripoSR weights not downloaded.")
            return 1
        download_weights(
            progress=lambda p, done, total, speed: print(
                f"\r  {done / 1e6:7.1f} / {total / 1e6:.0f} MB  ({speed / 1e6:.1f} MB/s)",
                end="", flush=True,
            )
        )
        print("\nTripoSR weights downloaded.")
        return 0

    if args.check:
        estimator = DepthEstimator()
        if estimator.available():
            path = estimator.model_path
            print(f"Depth model ready: {path} ({path.stat().st_size / 1e6:.0f} MB)")
            return 0
        print("Depth model not downloaded.")
        return 1
    download_depth_model(
        progress=lambda p, done, total, speed: print(
            f"\r  {done / 1e6:6.1f} / {total / 1e6:.0f} MB  ({speed / 1e6:.1f} MB/s)",
            end="", flush=True,
        )
    )
    print("\nDepth model downloaded.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="splatlab",
        description="CPU-first 3D Gaussian Splatting reconstruction studio.",
    )
    parser.add_argument("--version", action="version", version="kinesplat 1.0.0")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="reconstruct the bundled Toy Courtyard scene")
    demo.add_argument("--preset", default="draft", choices=["draft", "balanced", "quality"])
    demo.add_argument("--out", default=None, help="output workspace directory")
    demo.add_argument("--threads", type=int, default=None)
    demo.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    demo.set_defaults(func=cmd_demo)

    images = sub.add_parser("images", help="reconstruct a scene from photos / videos")
    images.add_argument("inputs", nargs="+", help="image files, video files or folders")
    images.add_argument("--name", default="", help="scene name")
    images.add_argument("--preset", default="balanced", choices=["draft", "balanced", "quality"])
    images.add_argument("--out", default=None)
    images.add_argument("--backend", default="auto", choices=["auto", "splat", "mesh"],
                        help="auto picks the mesh path for a single photo")
    images.add_argument("--poses", default="auto", choices=["auto", "sfm", "colmap"])
    images.add_argument("--no-depth", action="store_true", help="skip the depth model")
    images.add_argument("--threads", type=int, default=None)
    images.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    images.set_defaults(func=cmd_images)

    mesh = sub.add_parser("mesh", help="reconstruct a mesh from a single photo (TripoSR)")
    mesh.add_argument("input", help="one image file")
    mesh.add_argument("--name", default="", help="scene name")
    mesh.add_argument("--out", default=None)
    mesh.add_argument("--resolution", type=int, default=None,
                      help="marching-cubes grid size (official default: 256)")
    mesh.add_argument("--threshold", type=float, default=25.0,
                      help="density iso-level; lower keeps more of the surface")
    mesh.add_argument("--no-remove-bg", action="store_true",
                      help="use the official pre-composed RGB input mode")
    mesh.add_argument("--foreground-ratio", type=float, default=0.85,
                      help="foreground coverage after rembg (official default: 0.85)")
    mesh.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    mesh.add_argument("--chunk-size", type=int, default=8192,
                      help="TripoSR renderer query chunk (official default: 8192)")
    mesh.set_defaults(func=cmd_mesh)

    serve = sub.add_parser("serve", help="start the local web studio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--open", action="store_true")
    serve.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    serve.set_defaults(func=cmd_serve)

    model = sub.add_parser("model", help="download or check the pretrained models")
    model.add_argument("--check", action="store_true")
    model.add_argument("--triposr", action="store_true",
                       help="use the TripoSR mesh weights instead of the depth model")
    model.set_defaults(func=cmd_model)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preset = get_preset(getattr(args, "preset", None))
    if getattr(args, "preset", None):
        print(f"Preset: {preset.label} ({preset.max_gaussians:,} splats, {preset.iterations} steps)")
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
