import argparse
import json
import os
import re
import time
from pathlib import Path

import torch


def find_latest_iteration(model_path: Path) -> int:
    pc_root = model_path / "point_cloud"
    iters = []
    if pc_root.exists():
        for d in pc_root.iterdir():
            m = re.match(r"iteration_(\d+)$", d.name)
            if m and d.is_dir():
                iters.append(int(m.group(1)))
    if not iters:
        raise FileNotFoundError(f"No iteration_xxxx found under: {pc_root}")
    return max(iters)


def load_cfg_args(model_path: Path):
    """
    EndoGaussian 会在 output/<exp>/cfg_args 里保存一个 argparse.Namespace(...) 字符串。
    用受限 eval 解析。
    """
    cfg_file = model_path / "cfg_args"
    if not cfg_file.exists():
        return None

    from argparse import Namespace

    txt = cfg_file.read_text(encoding="utf-8", errors="ignore").strip()
    try:
        cfg = eval(txt, {"Namespace": Namespace})
        return cfg
    except Exception as e:
        raise RuntimeError(f"Failed to parse cfg_args: {cfg_file}\nContent head: {txt[:200]}\nErr: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_path", required=True, help="e.g. output/endonerf/cutting")
    ap.add_argument("--iteration", type=int, default=-1, help="-1 means latest")
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--views", type=int, default=1, help="cycle over N views (1=repeat same view)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", action="store_true", help="print JSON only")
    args = ap.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if args.iteration == -1:
        iteration = find_latest_iteration(model_path)
    else:
        iteration = args.iteration

    # ---- Import EndoGaussian modules (assumes running from repo root) ----
    try:
        from gaussian_renderer import render
        from scene import Scene
        # Gaussian model class name might differ across forks, so try a few
        try:
            from scene import GaussianModel
        except Exception:
            from scene.gaussian_model import GaussianModel  # fallback (3DGS style)
        from arguments import ModelParams, PipelineParams
    except Exception as e:
        raise RuntimeError(
            "Import failed. Please run this script from EndoGaussian-master root.\n"
            "Example: cd EndoGaussian-master && python bench_fps.py -m output/...\n"
            f"Original error: {e}"
        )

    # Load cfg_args (for source_path, sh_degree, white_background, etc.)
    cfg = load_cfg_args(model_path)
    if cfg is None:
        raise RuntimeError(f"Missing cfg_args under {model_path}. Run training/render once to generate it.")

    # Build minimal arg objects similarly to render.py/train.py
    # (Most repos use ModelParams/PipelineParams.extract(Namespace))
    model_params = ModelParams()
    pipe_params = PipelineParams()

    dataset = model_params.extract(cfg) if hasattr(model_params, "extract") else cfg
    pipe = pipe_params.extract(cfg) if hasattr(pipe_params, "extract") else cfg

    # Instantiate gaussians (needs sh_degree usually)
    sh_degree = getattr(cfg, "sh_degree", 3)
    gaussians = GaussianModel(sh_degree)

    # Load scene & trained model at iteration
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    # Cameras
    cams = scene.getTestCameras() if args.split == "test" else scene.getTrainCameras()
    if not cams:
        raise RuntimeError(f"No cameras found for split={args.split}")

    # Background
    white_bg = getattr(cfg, "white_background", False)
    background = torch.tensor([1.0, 1.0, 1.0] if white_bg else [0.0, 0.0, 0.0], device=args.device)

    # FPS benchmark (pure render, no saving)
    gaussians.eval() if hasattr(gaussians, "eval") else None
    torch.set_grad_enabled(False)

    # choose views
    vcnt = max(1, min(args.views, len(cams)))
    use_cams = cams[:vcnt]

    # warmup
    for i in range(args.warmup):
        cam = use_cams[i % vcnt]
        _ = render(cam, gaussians, pipe, background)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    for i in range(args.iters):
        cam = use_cams[i % vcnt]
        _ = render(cam, gaussians, pipe, background)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    total = t1 - t0
    fps = args.iters / total if total > 0 else 0.0
    ms = 1000.0 / fps if fps > 0 else float("inf")

    out = {
        "model_path": str(model_path).replace("\\", "/"),
        "iteration": iteration,
        "split": args.split,
        "views_used": vcnt,
        "warmup": args.warmup,
        "iters": args.iters,
        "fps": fps,
        "ms_per_frame": ms,
    }

    if args.json:
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(f"[BENCH] {model_path} iter={iteration} split={args.split} views={vcnt} FPS={fps:.2f} ({ms:.2f} ms/frame)")
        print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
