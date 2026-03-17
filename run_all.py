\
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

from src.common import load_yaml

def run(cmd, cwd=None):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg_path = Path(args.config)

    # 1) generate data -> capture RUN_DIR line
    import subprocess, re
    p = subprocess.run([sys.executable, "scripts/01_generate_data.py", "--config", str(cfg_path)], capture_output=True, text=True)
    print(p.stdout)
    if p.returncode != 0:
        print(p.stderr)
        raise SystemExit(p.returncode)

    m = re.search(r"RUN_DIR=(.*)", p.stdout)
    if not m:
        raise RuntimeError("Could not parse RUN_DIR from generator output.")
    run_dir = m.group(1).strip()
    print("Parsed run_dir:", run_dir)

    # 2) synergy
    run([sys.executable, "scripts/02_fit_synergy.py", "--run_dir", run_dir])

    # 3) phase models (fusion + ablation)
    for mod in ["fusion","eeg","emg"]:
        run([sys.executable, "scripts/03_train_phase.py", "--run_dir", run_dir, "--modality", mod])

    # 4) torque model (fusion)
    run([sys.executable, "scripts/04_train_torque.py", "--run_dir", run_dir, "--modality", "fusion"])

    # 5) figures
    run([sys.executable, "scripts/05_make_figures.py", "--run_dir", run_dir])

    # 6) skeleton grid
    run([sys.executable, "scripts/06_render_skeleton.py", "--run_dir", run_dir])

    # 7) validation report
    run([sys.executable, "scripts/09_make_validation_report.py", "--run_dir", run_dir])

    print("\nDONE. Outputs in:", run_dir)

if __name__ == "__main__":
    main()
