import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def find_experiments(output_root: Path):
    """Find exp dirs that have cfg_args and point_cloud/iteration_xxxx."""
    exps = []
    for cfg in output_root.glob("**/cfg_args"):
        exp = cfg.parent
        pc = exp / "point_cloud"
        if pc.exists():
            exps.append(exp)
    return sorted(set(exps))


def pick_latest_iter(exp: Path) -> Optional[int]:
    pc = exp / "point_cloud"
    iters = []
    if not pc.exists():
        return None
    for d in pc.iterdir():
        m = re.match(r"iteration_(\d+)$", d.name)
        if m and d.is_dir():
            iters.append(int(m.group(1)))
    return max(iters) if iters else None


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
            m = re.match(r"^element\s+vertex\s+(\d+)$", s)
            if m:
                n = int(m.group(1))
            if s == "end_header":
                break
    return n


def mb(p: Path) -> float:
    return p.stat().st_size / (1024 * 1024) if p.exists() else 0.0


def overall_mb(iter_dir: Path) -> float:
    ply = iter_dir / "point_cloud.ply"
    deform = iter_dir / "deformation.pth"
    accum = iter_dir / "deformation_accum.pth"
    table = iter_dir / "deformation_table.pth"
    return mb(ply) + mb(deform) + mb(accum) + mb(table)


def parse_metrics_from_log(log_path: Path):
    """Parse PSNR/SSIM/LPIPS (and optional PSNR*, FLIP, RMSE) from batch_baseline.log."""
    if not log_path.exists():
        return {}

    text = log_path.read_text(encoding="utf-8", errors="ignore")

    def last_float(pattern):
        ms = re.findall(pattern, text)
        return float(ms[-1]) if ms else None

    metrics = {
        "SSIM": last_float(r"SSIM\s*:\s*([0-9.]+)"),
        "PSNR": last_float(r"\bPSNR\s*:\s*([0-9.]+)"),
        "PSNR_star": last_float(r"PSNR\*\s*:\s*([0-9.]+)"),
        "LPIPS": last_float(r"LPIPS\s*:\s*([0-9.]+)"),
        "FLIP": last_float(r"FLIP\s*:\s*([0-9.]+)"),
        "RMSE": last_float(r"RMSE\s*:\s*([0-9.]+)"),
    }
    return metrics


def parse_json_line(text: str) -> Optional[dict]:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except Exception:
                continue
    return None


def run_bench_fps(
    repo_root: Path,
    exp: Path,
    iteration: int,
    split="test",
    views=1,
    warmup=10,
    iters=200,
    bench_python: Optional[str] = None,
    bench_conda_env: Optional[str] = None,
):
    bench = repo_root / "bench_fps.py"
    if not bench.exists():
        return None

    if bench_conda_env:
        cmd = [
            "conda", "run", "-n", bench_conda_env, "python", str(bench),
            "-m", str(exp),
            "--iteration", str(iteration),
            "--split", split,
            "--views", str(views),
            "--warmup", str(warmup),
            "--iters", str(iters),
            "--json",
        ]
    else:
        python_exe = bench_python or sys.executable
        cmd = [
            python_exe, str(bench),
            "-m", str(exp),
            "--iteration", str(iteration),
            "--split", split,
            "--views", str(views),
            "--warmup", str(warmup),
            "--iters", str(iters),
            "--json",
        ]
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    out = proc.stdout or ""

    j = parse_json_line(out)
    if j and "fps" in j:
        try:
            return float(j.get("fps", 0.0))
        except Exception:
            pass

    log_name = "bench_fps_error.log" if proc.returncode != 0 else "bench_fps_output.log"
    (exp / log_name).write_text(out, encoding="utf-8", errors="ignore")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".", help="EndoGaussian-master root")
    ap.add_argument("--output_root", default="output")
    ap.add_argument("--out_xlsx", default="endo_all_metrics.xlsx")
    ap.add_argument("--baseline_xlsx", default="EndoGaussian_baseline_metrics.xlsx")
    ap.add_argument("--bench_split", default="test", choices=["test", "train"])
    ap.add_argument("--bench_views", type=int, default=1)
    ap.add_argument("--bench_warmup", type=int, default=10)
    ap.add_argument("--bench_iters", type=int, default=200)
    ap.add_argument("--bench_python", default=None, help="Python executable with torch installed")
    ap.add_argument("--bench_conda_env", default=None, help="Conda env name for running bench_fps.py")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_root = (repo_root / args.output_root).resolve()

    exps = find_experiments(output_root)
    if not exps:
        print(f"No experiments found under {output_root}")
        return

    rows = []
    for exp in exps:
        it = pick_latest_iter(exp)
        if it is None:
            continue
        it_dir = exp / "point_cloud" / f"iteration_{it}"

        # metrics log
        log_path = exp / "batch_baseline.log"
        metrics = parse_metrics_from_log(log_path)

        # gaussians & size
        ply = it_dir / "point_cloud.ply"
        gauss = ply_vertex_count(ply)
        overall = overall_mb(it_dir)

        # fps
        fps = run_bench_fps(
            repo_root, exp, it,
            split=args.bench_split,
            views=args.bench_views,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
            bench_python=args.bench_python,
            bench_conda_env=args.bench_conda_env,
        )

        rel = str(exp.relative_to(output_root)).replace("\\", "/")

        row = {
            "Scene": rel,
            "Iteration": it,
            "FPS": fps,
            "Gaussians": gauss,
            "SSIM": metrics.get("SSIM"),
            "PSNR": metrics.get("PSNR"),
            "LPIPS": metrics.get("LPIPS"),
            "PSNR*": metrics.get("PSNR_star"),
            "FLIP": metrics.get("FLIP"),
            "RMSE": metrics.get("RMSE"),
            "Overall_MB": round(overall, 3),
        }
        rows.append(row)

        print(f"[OK] {rel} | it={it} | gauss={gauss} | overall={overall:.2f}MB | fps={fps}")

    df = pd.DataFrame(rows)

    # dataset column (from output path)
    def scene_to_dataset(scene: str) -> str:
        parts = scene.split("/")
        if not parts:
            return ""
        head = parts[0].lower()
        if head == "endonerf":
            return "EndoNeRF"
        if head == "scared":
            return "SCARED"
        return parts[0]

    df.insert(0, "Dataset", df["Scene"].map(scene_to_dataset))

    # merge baseline metrics if present
    baseline_path = (repo_root / args.baseline_xlsx).resolve()
    if baseline_path.exists():
        base = pd.read_excel(baseline_path)
        base = base.dropna(subset=["Dataset", "Scene"])
        base["Dataset_norm"] = base["Dataset"].astype(str).str.lower()
        base["Scene_norm"] = base["Scene"].astype(str).str.lower()

        def match_baseline(row):
            dataset = str(row["Dataset"]).lower()
            scene = str(row["Scene"]).lower()
            scene_tail = scene.split("/")[-1]

            # exact match on full scene
            exact = base[(base["Dataset_norm"] == dataset) & (base["Scene_norm"] == scene)]
            if not exact.empty:
                return exact.iloc[0]

            # match by suffix (e.g., scared/dataset_1/keyframe_1)
            suffix = base[(base["Dataset_norm"] == dataset) & (base["Scene_norm"].apply(lambda s: scene.endswith(s)))]
            if not suffix.empty:
                return suffix.iloc[0]

            # match by basename containment (e.g., endonerf/pulling -> pulling_soft_tissues)
            contain = base[(base["Dataset_norm"] == dataset) & (base["Scene_norm"].str.contains(scene_tail, regex=False))]
            if not contain.empty:
                return contain.iloc[0]

            return None

        baseline_rows = df.apply(match_baseline, axis=1)
        df["Baseline_PSNR"] = baseline_rows.apply(lambda r: getattr(r, "PSNR", None) if r is not None else None)
        df["Baseline_SSIM"] = baseline_rows.apply(lambda r: getattr(r, "SSIM", None) if r is not None else None)
        df["Baseline_LPIPS"] = baseline_rows.apply(lambda r: getattr(r, "LPIPS", None) if r is not None else None)

    # basic formatting: put common cols first
    col_order = [
        "Scene", "Iteration",
        "PSNR", "SSIM", "LPIPS",
        "FPS", "Gaussians",
        "Overall_MB",
        "PSNR*", "FLIP", "RMSE",
    ]
    df = df[[c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]]

    out_xlsx = (repo_root / args.out_xlsx).resolve()

    # dataset summary (mean of numeric columns)
    numeric_cols = df.select_dtypes(include="number").columns
    summary = df.groupby("Dataset", as_index=False)[numeric_cols].mean(numeric_only=True)

    with pd.ExcelWriter(out_xlsx) as writer:
        df.to_excel(writer, index=False, sheet_name="All")
        summary.to_excel(writer, index=False, sheet_name="DatasetSummary")
    print(f"\nSaved: {out_xlsx}")


if __name__ == "__main__":
    main()
