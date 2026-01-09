import os
import sys
import re
import glob
import subprocess
from datetime import datetime

# =========================
# 配置：你要批量跑的场景
# =========================
SCENES = [
    # SCARED
    ("assets/data/SCARED/dataset_1/keyframe_1", "scared/dataset_1/keyframe_1"),
    ("assets/data/SCARED/dataset_1/keyframe_2", "scared/dataset_1/keyframe_2"),
    ("assets/data/SCARED/dataset_1/keyframe_3", "scared/dataset_1/keyframe_3"),
    # EndoNeRF
    ("assets/data/EndoNeRF/pulling_soft_tissues", "endonerf/pulling"),
    ("assets/data/EndoNeRF/cutting_tissues_twice", "endonerf/cutting"),
]

DEFAULT_PORT = "6017"


def repo_root() -> str:
    # 假设脚本放在 repo 根目录
    return os.path.dirname(os.path.abspath(__file__))


def norm(p: str) -> str:
    return os.path.normpath(os.path.join(repo_root(), p))


def find_config(scene_path: str) -> str:
    """
    尽量自动找 config：
    - EndoNeRF pulling：arguments/endonerf/pulling.py（官方README示例）:contentReference[oaicite:1]{index=1}
    - EndoNeRF cutting：优先 arguments/endonerf/cutting*.py，不行就 default.py
    - SCARED：优先 arguments/scared/ 里包含 keyframe_x 的文件，否则 default.py，否则取第一个 scared 的 .py
    """
    sp = scene_path.replace("\\", "/").lower()

    # EndoNeRF
    if "endonerf" in sp:
        if "pulling" in sp:
            cand = norm("arguments/endonerf/pulling.py")
            if os.path.exists(cand):
                return cand
        if "cutting" in sp:
            # 常见命名：cutting.py / cutting*.py
            cands = glob.glob(norm("arguments/endonerf/cutting*.py"))
            if cands:
                return sorted(cands)[0]
        # fallback
        for fn in ["default.py", "endo.py"]:
            cand = norm(f"arguments/endonerf/{fn}")
            if os.path.exists(cand):
                return cand

    # SCARED
    if "scared" in sp:
        # 取 keyframe_x
        m = re.search(r"keyframe[_\- ]?(\d+)", sp)
        kf = m.group(1) if m else None

        scared_dir = norm("arguments/scared")
        if os.path.isdir(scared_dir):
            pyfiles = glob.glob(os.path.join(scared_dir, "**", "*.py"), recursive=True)
            if kf:
                # 优先包含 keyframe_x 的
                kf_hits = [p for p in pyfiles if re.search(rf"keyframe[_\- ]?{kf}\b", p.lower())]
                if kf_hits:
                    return sorted(kf_hits)[0]
            # 其次 default.py
            cand = norm("arguments/scared/default.py")
            if os.path.exists(cand):
                return cand
            # 再其次：随便取一个 scared 相关的 config
            if pyfiles:
                return sorted(pyfiles)[0]

    # 最后的兜底：arguments 下随便找一个 .py（不建议）
    any_py = glob.glob(norm("arguments/**/*.py"), recursive=True)
    if any_py:
        return sorted(any_py)[0]

    raise FileNotFoundError("找不到任何 config.py，请手动指定 config 路径。")


def run_cmd(cmd, log_file, cwd=None, allow_fail=False):
    """
    边跑边输出，并写入日志（Windows 兼容：不因编码崩溃）
    """
    import os, subprocess
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    env = os.environ.copy()
    # 让子进程尽量用 UTF-8 输出（Windows 上很关键）
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    with open(log_file, "a", encoding="utf-8", errors="ignore") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("CMD: " + " ".join(cmd) + "\n")
        f.write("=" * 80 + "\n")

        p = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=1,
        )

        # bytes -> str（容错：先 utf-8，不行再 gbk，再不行就替换）
        while True:
            line = p.stdout.readline()
            if not line:
                break
            try:
                s = line.decode("utf-8", errors="replace")
            except Exception:
                s = line.decode("gbk", errors="replace")

            print(s, end="")
            f.write(s)

        ret = p.wait()
        if ret != 0 and not allow_fail:
            raise RuntimeError(f"命令失败 (exit={ret}): {' '.join(cmd)}")
        return ret

def main():
    port = DEFAULT_PORT
    py = sys.executable  # 默认用当前 python（你也可以改成 conda env 的 python.exe）

    root = repo_root()
    print(f"[INFO] Repo root: {root}")
    print(f"[INFO] Python: {py}")
    print(f"[INFO] Port: {port}")
    print(f"[INFO] Total scenes: {len(SCENES)}")

    for i, (scene_rel, expname) in enumerate(SCENES, 1):
        scene_abs = norm(scene_rel)
        out_dir = norm(os.path.join("output", expname))
        log_file = os.path.join(out_dir, "batch_baseline.log")

        print("\n" + "#" * 80)
        print(f"[{i}/{len(SCENES)}] Scene: {scene_rel}")
        print(f"    Expname: {expname}")
        print(f"    Output : {out_dir}")

        if not os.path.exists(scene_abs):
            print(f"[SKIP] 场景路径不存在：{scene_abs}")
            continue

        cfg = find_config(scene_rel)
        print(f"    Config : {cfg}")

        # 1) Train
        print("\n[STEP 1/3] Training ...")
        train_cmd = [
            py, "train.py",
            "-s", scene_abs,
            "--port", port,
            "--expname", expname,
            "--configs", cfg
        ]
        run_cmd(train_cmd, log_file, cwd=root)

        # 2) Render
        # 说明：你之前在 Windows 上 render 有时会在写 mp4 时报错（imageio ffmpeg问题），
        #      但往往图片已经写出来了，所以这里 allow_fail=True，继续跑 metrics。
        print("\n[STEP 2/3] Rendering (skip_train, skip_video) ...")
        render_cmd = [
            py, "render.py",
            "--model_path", os.path.join("output", expname),
            "--skip_train",
            "--skip_video",
            "--configs", cfg
        ]
        run_cmd(render_cmd, log_file, cwd=root, allow_fail=True)

        # 3) Metrics
        print("\n[STEP 3/3] Metrics ...")
        metrics_cmd = [
            py, "metrics.py",
            "--model_path", os.path.join("output", expname),
        ]
        run_cmd(metrics_cmd, log_file, cwd=root)

        print(f"\n[DONE] {expname} finished. Log: {log_file}")

    print("\n[ALL DONE] 批量 baseline 完成。")


if __name__ == "__main__":
    main()
