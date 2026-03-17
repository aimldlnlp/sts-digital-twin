\
from __future__ import annotations
import argparse
from pathlib import Path

from src.common import load_yaml, set_seed, now_run_id, ensure_dir, save_json
from src.data_gen.generate import generate_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out_root", type=str, default="outputs")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    set_seed(int(cfg["seed"]))

    run_id = now_run_id("sts")
    run_dir = ensure_dir(Path(args.out_root)/run_id)
    # persist config for reproducibility
    (run_dir/"config.yaml").write_text(Path(args.config).read_text())

    meta = generate_dataset(cfg, run_dir)
    save_json(run_dir/"run_meta.json", meta)
    print(f"RUN_DIR={run_dir}")
    print(meta)

if __name__ == "__main__":
    main()
