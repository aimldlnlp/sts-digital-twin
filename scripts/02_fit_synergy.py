\
from __future__ import annotations
import argparse
from pathlib import Path

from src.common import load_yaml, set_seed
from src.features.synergy import fit_nmf_synergy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_yaml(Path(args.config)) if args.config else load_yaml(run_dir/"config.yaml")
    set_seed(int(cfg["seed"]))

    res = fit_nmf_synergy(run_dir, cfg)
    print(res)

if __name__ == "__main__":
    main()
