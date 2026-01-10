import argparse
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict

import pandas as pd


def count_images(scene_dir: Path) -> Optional[int]:
    candidates = [
        scene_dir / "images",
        scene_dir / "image",
        scene_dir / "imgs",
        scene_dir / "data" / "rgb_data",
        scene_dir / "data" / "left",
        scene_dir / "data" / "right",
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            files = [f for f in p.iterdir() if f.is_file()]
            if files:
                return len(files)
    return None


def ply_vertex_count(ply_path: Path) -> Optional[int]:
    if not ply_path.exists():
        return None
    n = None
    with ply_path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            s = line.decode("ascii", "ignore").strip()
            if s.startswith("element vertex"):
                parts = s.split()
                if len(parts) == 3:
                    try:
                        n = int(parts[2])
                    except ValueError:
                        pass
            if s == "end_header":
                break
    return n


def mb(p: Path) -> float:
    return p.stat().st_size / (1024 * 1024) if p.exists() else 0.0


def overall_mb(iter_dir: Path) -> float:
    ply = iter_dir / "point_cloud.ply"
    dct = iter_dir / "dct_coeffs.pth"
    if dct.exists():
        return mb(ply) + mb(dct)
    deform = iter_dir / "deformation.pth"
    accum = iter_dir / "deformation_accum.pth"
    table = iter_dir / "deformation_table.pth"
    return mb(ply) + mb(deform) + mb(accum) + mb(table)


def parse_metrics(results_json: Path, method: str) -> Dict[str, Optional[float]]:
    if not results_json.exists():
        return {}
    data = json.loads(results_json.read_text(encoding="utf-8", errors="ignore"))
    if method not in data:
        return {}
    return data[method]


def run_cmd(cmd, cwd: Path):
    print("[RUN]", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_bench_fps(repo_root: Path, exp: Path, iteration: int, split="test", views=1, warmup=10, iters=200, python_exe: Optional[str] = None):
    python_cmd = python_exe or "python"
    cmd = [
        python_cmd, str(repo_root / "bench_fps.py"),
        "-m", str(exp),
        "--iteration", str(iteration),
        "--split", split,
        "--views", str(views),
        "--warmup", str(warmup),
        "--iters", str(iters),
        "--json",
    ]
    out = subprocess.check_output(cmd, cwd=str(repo_root), text=True, encoding="utf-8", errors="ignore")
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            j = json.loads(line)
            return float(j.get("fps", 0.0))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--output_root", default="output1")
    ap.add_argument("--distill_iterations", type=int, default=6000)
    ap.add_argument("--dct_k", type=int, default=16)
    ap.add_argument("--dct_lr_mult", type=float, default=140)
    ap.add_argument("--dct_xyz_lr_mult", type=float, default=0.01)
    ap.add_argument("--distill_ssim_weight", type=float, default=0.1)
    ap.add_argument("--distill_unfreeze_lr_mult", type=float, default=0.01)
    ap.add_argument("--out_xlsx", default="endo_all_metrics_dct_sr2.xlsx")
    ap.add_argument("--python", dest="python_exe", default="python", help="Python executable for running train/render/metrics")
    ap.add_argument("--fps_only", action="store_true", default=False, help="Only recompute FPS and update Excel")
    ap.add_argument("--dct_use_scale", action="store_true", default=False, help="Enable DCT scaling deformation")
    ap.add_argument("--dct_use_rot", action="store_true", default=False, help="Enable DCT rotation deformation")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    scenes = [
        {
            "dataset": "EndoNeRF",
            "scene": "pulling_soft_tissues",
            "data_path": repo_root / "assets" / "data" / "EndoNeRF" / "pulling_soft_tissues",
            "teacher": repo_root / "output" / "endonerf" / "pulling",
            "config": repo_root / "arguments" / "endonerf" / "pulling.py",
        },
        {
            "dataset": "EndoNeRF",
            "scene": "cutting_tissues_twice",
            "data_path": repo_root / "assets" / "data" / "EndoNeRF" / "cutting_tissues_twice",
            "teacher": repo_root / "output" / "endonerf" / "cutting",
            "config": repo_root / "arguments" / "endonerf" / "cutting.py",
        },
        {
            "dataset": "SCARED",
            "scene": "dataset_1/keyframe_1",
            "data_path": repo_root / "assets" / "data" / "SCARED" / "dataset_1" / "keyframe_1",
            "teacher": repo_root / "output" / "scared" / "dataset_1" / "keyframe_1",
            "config": repo_root / "arguments" / "scared" / "d1k1.py",
        },
        {
            "dataset": "SCARED",
            "scene": "dataset_1/keyframe_2",
            "data_path": repo_root / "assets" / "data" / "SCARED" / "dataset_1" / "keyframe_2",
            "teacher": repo_root / "output" / "scared" / "dataset_1" / "keyframe_2",
            "config": repo_root / "arguments" / "scared" / "d2k1.py",
        },
        {
            "dataset": "SCARED",
            "scene": "dataset_1/keyframe_3",
            "data_path": repo_root / "assets" / "data" / "SCARED" / "dataset_1" / "keyframe_3",
            "teacher": repo_root / "output" / "scared" / "dataset_1" / "keyframe_3",
            "config": repo_root / "arguments" / "scared" / "d3k1.py",
        },
    ]

    rows = []
    for s in scenes:
        t = count_images(s["data_path"])
        if t is None:
            print(f"[SKIP] no images found: {s['data_path']}")
            continue
        if not s["teacher"].exists():
            print(f"[SKIP] missing teacher: {s['teacher']}")
            continue

        out_dir = output_root / s["dataset"].lower() / s["scene"]
        out_dir.parent.mkdir(parents=True, exist_ok=True)

        if not args.fps_only:
            train_cmd = [
                args.python_exe, str(repo_root / "train.py"),
                "--configs", str(s["config"]),
                "--model_path", str(out_dir),
                "--distill_dct",
                "--teacher_model_path", str(s["teacher"]),
                "--distill_iteration", "-1",
                "--distill_iterations", str(args.distill_iterations),
                "--use_dct_deform",
            ]
            if args.dct_use_scale:
                train_cmd.append("--dct_use_scale")
            if args.dct_use_rot:
                train_cmd.append("--dct_use_rot")
            train_cmd += [
                "--dct_k", str(args.dct_k),
                "--dct_T", str(t),
                "--dct_lr_mult", str(args.dct_lr_mult),
                "--dct_xyz_lr_mult", str(args.dct_xyz_lr_mult),
                "--distill_ssim_weight", str(args.distill_ssim_weight),
                "--distill_unfreeze_all",
                "--distill_unfreeze_lr_mult", str(args.distill_unfreeze_lr_mult),
            ]
            run_cmd(train_cmd, repo_root)

            render_cmd = [
                args.python_exe, str(repo_root / "render.py"),
                "--model_path", str(out_dir),
                "--iteration", str(args.distill_iterations),
                "--skip_train",
                "--skip_video",
                "--configs", str(s["config"]),
                "--use_dct_deform",
            ]
            if args.dct_use_scale:
                render_cmd.append("--dct_use_scale")
            if args.dct_use_rot:
                render_cmd.append("--dct_use_rot")
            run_cmd(render_cmd, repo_root)

            metrics_cmd = [
                args.python_exe, str(repo_root / "metrics.py"),
                "-m", str(out_dir),
            ]
            run_cmd(metrics_cmd, repo_root)

        results_json = out_dir / "results.json"
        method = f"ours_{args.distill_iterations}"
        metrics = parse_metrics(results_json, method)

        iter_dir = out_dir / "point_cloud" / f"iteration_{args.distill_iterations}"
        ply = iter_dir / "point_cloud.ply"
        gauss = ply_vertex_count(ply)
        overall = overall_mb(iter_dir)
        fps = run_bench_fps(repo_root, out_dir, args.distill_iterations, python_exe=args.python_exe)

        rows.append({
            "Dataset": s["dataset"],
            "Scene": s["scene"],
            "Iteration": args.distill_iterations,
            "PSNR": metrics.get("PSNR"),
            "SSIM": metrics.get("SSIM"),
            "LPIPS": metrics.get("LPIPS"),
            "PSNR*": metrics.get("PSNR*"),
            "FLIP": metrics.get("FLIP"),
            "RMSE": metrics.get("RMSE"),
            "FPS": fps,
            "Gaussians": gauss,
            "Overall_MB": round(overall, 3),
        })

    if not rows:
        print("No results collected.")
        return

    df = pd.DataFrame(rows)
    numeric_cols = df.select_dtypes(include="number").columns
    summary = df.groupby("Dataset", as_index=False)[numeric_cols].mean(numeric_only=True)

    out_xlsx = output_root / args.out_xlsx
    with pd.ExcelWriter(out_xlsx) as writer:
        df.to_excel(writer, index=False, sheet_name="All")
        summary.to_excel(writer, index=False, sheet_name="DatasetSummary")
    print(f"\nSaved: {out_xlsx}")


if __name__ == "__main__":
    main()
