import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


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


def prepare_windows_dll_search():
    if os.name != "nt":
        return
    try:
        env_root = os.environ.get("CONDA_PREFIX")
        if not env_root:
            env_root = str(Path(sys.executable).resolve().parent)
        dll_dirs = [
            Path(env_root) / "Library" / "bin",
            Path(env_root) / "Library" / "usr" / "bin",
            Path(env_root) / "Library" / "mingw-w64" / "bin",
            Path(env_root) / "DLLs",
            Path(env_root) / "Scripts",
            Path(env_root),
        ]
        for p in dll_dirs:
            if p.exists():
                os.add_dll_directory(str(p))
        path_sep = os.pathsep
        cur_path = os.environ.get("PATH", "")
        prepend = path_sep.join([str(p) for p in dll_dirs if p.exists()])
        if prepend:
            os.environ["PATH"] = prepend + path_sep + cur_path
    except Exception:
        pass


def main():
    prepare_windows_dll_search()
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_path", required=True, help="e.g. output/endonerf/cutting")
    ap.add_argument("--iteration", type=int, default=-1, help="-1 means latest")
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--views", type=int, default=1, help="cycle over N views (1=repeat same view)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dct_expand_codebook", action="store_true", default=False)
    ap.add_argument("--fp16_static", action="store_true", default=False)
    ap.add_argument("--dct_masked", action="store_true", default=False)
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
        from gaussian_renderer import render, GaussianModel
        from scene import Scene
        from arguments import ModelParams, PipelineParams, ModelHiddenParams
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
    # Always use the passed model_path even if cfg_args points elsewhere.
    setattr(cfg, "model_path", str(model_path))

    # Build minimal arg objects similarly to render.py/train.py
    # (Most repos use ModelParams/PipelineParams.extract(Namespace))
    parser = argparse.ArgumentParser(add_help=False)
    model_params = ModelParams(parser, sentinel=True)
    pipe_params = PipelineParams(parser)
    hyper_params = ModelHiddenParams(parser)

    if args.dct_expand_codebook:
        setattr(cfg, "dct_expand_codebook", True)
    if args.fp16_static:
        setattr(cfg, "fp16_static", True)
    if args.dct_masked:
        setattr(cfg, "dct_masked", True)
    dataset = model_params.extract(cfg) if hasattr(model_params, "extract") else cfg
    pipe = pipe_params.extract(cfg) if hasattr(pipe_params, "extract") else cfg
    hyper = hyper_params.extract(cfg) if hasattr(hyper_params, "extract") else cfg

    # Instantiate gaussians (needs sh_degree usually)
    sh_degree = getattr(cfg, "sh_degree", 3)
    gaussians = GaussianModel(sh_degree, hyper)

    # Load scene & trained model at iteration
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, load_coarse=getattr(dataset, "no_fine", False))

    def move_cam_to_device(cam, device):
        for attr in ("world_view_transform", "full_proj_transform", "camera_center"):
            t = getattr(cam, attr, None)
            if torch.is_tensor(t):
                setattr(cam, attr, t.to(device))
        if hasattr(cam, "data_device"):
            cam.data_device = device
        return cam

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
    use_cams = [move_cam_to_device(c, args.device) for c in cams[:vcnt]]

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()

    # warmup
    with torch.inference_mode():
        for i in range(args.warmup):
            cam = use_cams[i % vcnt]
            _ = render(cam, gaussians, pipe, background)

        if use_cuda:
            torch.cuda.synchronize()

        t0 = time.time()
        if use_cuda:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()

        for i in range(args.iters):
            cam = use_cams[i % vcnt]
            _ = render(cam, gaussians, pipe, background)

        if use_cuda:
            ender.record()
            torch.cuda.synchronize()
        t1 = time.time()

    total = t1 - t0
    fps = args.iters / total if total > 0 else 0.0
    ms = 1000.0 / fps if fps > 0 else float("inf")

    fps_gpu = None
    ms_gpu = None
    if use_cuda:
        gpu_ms = float(starter.elapsed_time(ender))
        if gpu_ms > 0:
            fps_gpu = args.iters / (gpu_ms / 1000.0)
            ms_gpu = gpu_ms / args.iters

    out = {
        "model_path": str(model_path).replace("\\", "/"),
        "iteration": iteration,
        "split": args.split,
        "views_used": vcnt,
        "warmup": args.warmup,
        "iters": args.iters,
        "fps": fps,
        "ms_per_frame": ms,
        "fps_gpu": fps_gpu,
        "ms_per_frame_gpu": ms_gpu,
    }

    if args.json:
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(f"[BENCH] {model_path} iter={iteration} split={args.split} views={vcnt} FPS={fps:.2f} ({ms:.2f} ms/frame)")
        print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
