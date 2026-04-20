from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from src.common import ensure_dir, load_yaml, now_run_id


@dataclass(frozen=True)
class PipelineStep:
    key: str
    label: str
    progress_end: int
    cmd: list[str]


class TeeLogger:
    def __init__(self, stream: TextIO, log_path: Path):
        self.stream = stream
        self.log_path = log_path
        self.handle = open(log_path, "a", encoding="utf-8")

    def write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()
        self.handle.write(text)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class ProgressTracker:
    def __init__(self, run_id: str, run_dir: Path, progress_path: Path, logger: TeeLogger):
        self.run_id = run_id
        self.run_dir = run_dir
        self.progress_path = progress_path
        self.logger = logger

    def emit(self, *, step: str, label: str, status: str, progress_pct: int, detail: str | None = None) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "step": step,
            "label": label,
            "status": status,
            "progress_pct": int(progress_pct),
        }
        if detail:
            event["detail"] = detail
        with open(self.progress_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        line = f"[progress] progress={int(progress_pct):3d}% step={step} status={status}"
        if detail:
            line += f" detail={detail}"
        self.logger.write(line + "\n")


def build_steps(python_bin: str, cfg_path: Path, out_root: Path, run_id: str, trial_index: int) -> list[PipelineStep]:
    return [
        PipelineStep(
            key="generate_data",
            label="Generate multimodal trials",
            progress_end=16,
            cmd=[python_bin, "scripts/01_generate_data.py", "--config", str(cfg_path), "--out_root", str(out_root), "--run_id", run_id],
        ),
        PipelineStep(
            key="fit_synergy",
            label="Fit EMG synergies",
            progress_end=23,
            cmd=[python_bin, "scripts/02_fit_synergy.py", "--run_dir", str(out_root / run_id)],
        ),
        PipelineStep(
            key="train_phase_fusion",
            label="Train fusion phase model",
            progress_end=31,
            cmd=[python_bin, "scripts/03_train_phase.py", "--run_dir", str(out_root / run_id), "--modality", "fusion"],
        ),
        PipelineStep(
            key="train_phase_eeg",
            label="Train EEG phase model",
            progress_end=39,
            cmd=[python_bin, "scripts/03_train_phase.py", "--run_dir", str(out_root / run_id), "--modality", "eeg"],
        ),
        PipelineStep(
            key="train_phase_emg",
            label="Train EMG phase model",
            progress_end=47,
            cmd=[python_bin, "scripts/03_train_phase.py", "--run_dir", str(out_root / run_id), "--modality", "emg"],
        ),
        PipelineStep(
            key="train_torque_emg",
            label="Train EMG torque regressor",
            progress_end=55,
            cmd=[python_bin, "scripts/04_train_torque.py", "--run_dir", str(out_root / run_id), "--modality", "emg"],
        ),
        PipelineStep(
            key="train_torque",
            label="Train fusion torque regressor",
            progress_end=63,
            cmd=[python_bin, "scripts/04_train_torque.py", "--run_dir", str(out_root / run_id), "--modality", "fusion"],
        ),
        PipelineStep(
            key="make_figures",
            label="Render figures",
            progress_end=71,
            cmd=[python_bin, "scripts/05_make_figures.py", "--run_dir", str(out_root / run_id)],
        ),
        PipelineStep(
            key="render_skeleton",
            label="Render skeleton grid",
            progress_end=76,
            cmd=[python_bin, "scripts/06_render_skeleton.py", "--run_dir", str(out_root / run_id), "--trial_index", str(trial_index)],
        ),
        PipelineStep(
            key="make_video_multimodal",
            label="Render multimodal showcase video",
            progress_end=82,
            cmd=[python_bin, "scripts/07_make_video.py", "--run_dir", str(out_root / run_id), "--trial_index", str(trial_index)],
        ),
        PipelineStep(
            key="make_video_ablation",
            label="Render ablation video",
            progress_end=87,
            cmd=[python_bin, "scripts/08_make_video_ablation.py", "--run_dir", str(out_root / run_id), "--trial_index", str(trial_index)],
        ),
        PipelineStep(
            key="make_video_stress_phase",
            label="Render stress phase robustness video",
            progress_end=91,
            cmd=[python_bin, "scripts/11_make_video_stress_phase.py", "--run_dir", str(out_root / run_id), "--trial_index", str(trial_index)],
        ),
        PipelineStep(
            key="make_video_torque_range",
            label="Render torque error range video",
            progress_end=95,
            cmd=[python_bin, "scripts/12_make_video_torque_error_range.py", "--run_dir", str(out_root / run_id), "--trial_index", str(trial_index)],
        ),
        PipelineStep(
            key="make_video_fusion_montage",
            label="Render fusion benchmark montage video",
            progress_end=98,
            cmd=[python_bin, "scripts/13_make_video_fusion_benchmark_montage.py", "--run_dir", str(out_root / run_id), "--trial_index", str(trial_index)],
        ),
        PipelineStep(
            key="validation_report",
            label="Build validation report",
            progress_end=100,
            cmd=[python_bin, "scripts/09_make_validation_report.py", "--run_dir", str(out_root / run_id)],
        ),
    ]


def filter_steps(steps: list[PipelineStep], start_from: str | None) -> tuple[list[PipelineStep], int]:
    if start_from is None:
        return steps, 0
    keys = [step.key for step in steps]
    if start_from not in keys:
        raise ValueError(f"Unknown step: {start_from}")
    index = keys.index(start_from)
    previous_progress = 0 if index == 0 else steps[index - 1].progress_end
    return steps[index:], previous_progress


def stream_command(cmd: list[str], logger: TeeLogger, env: dict[str, str], cwd: Path) -> None:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=cwd,
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.write(line)
    code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out-root", type=str, default="outputs")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--trial-index", type=int, default=0)
    parser.add_argument("--start-from", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    out_root = Path(args.out_root).resolve()
    repo_root = Path(__file__).resolve().parent
    cfg = load_yaml(cfg_path)
    _ = cfg

    run_id = args.run_id or now_run_id("sts")
    run_dir = out_root / run_id
    log_dir = ensure_dir(run_dir / "logs")
    pipeline_log_path = log_dir / "pipeline.log"
    progress_path = log_dir / "progress.jsonl"
    meta_path = log_dir / "run_context.json"

    logger = TeeLogger(sys.stdout, pipeline_log_path)
    tracker = ProgressTracker(run_id, run_dir, progress_path, logger)
    steps_all = build_steps(sys.executable, cfg_path, out_root, run_id, args.trial_index)
    steps, previous_progress = filter_steps(steps_all, args.start_from)

    meta = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "config_path": str(cfg_path),
        "trial_index": int(args.trial_index),
        "start_from": args.start_from,
        "pipeline_log": str(pipeline_log_path),
        "progress_log": str(progress_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": [{"step": step.key, "label": step.label, "progress_end": step.progress_end, "cmd": step.cmd} for step in steps_all],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.write(f"RUN_ID={run_id}\n")
    logger.write(f"RUN_DIR={run_dir}\n")
    logger.write(f"PIPELINE_LOG={pipeline_log_path}\n")
    logger.write(f"PROGRESS_LOG={progress_path}\n")

    if args.dry_run:
        tracker.emit(step="pipeline", label="Dry run", status="ready", progress_pct=0, detail="No commands executed")
        for step in steps:
            logger.write(f"DRY_RUN step={step.key} cmd={' '.join(step.cmd)}\n")
        logger.close()
        return

    base_env = os.environ.copy()
    base_env["PYTHONUNBUFFERED"] = "1"
    base_env.setdefault("TQDM_DISABLE", "1")
    existing_pythonpath = base_env.get("PYTHONPATH", "")
    base_env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}:{existing_pythonpath}"

    tracker.emit(
        step="pipeline",
        label="Pipeline boot",
        status="running",
        progress_pct=previous_progress,
        detail="Starting end-to-end run" if args.start_from is None else f"Resuming from {args.start_from}",
    )
    start_time = time.time()

    try:
        for step in steps:
            tracker.emit(step=step.key, label=step.label, status="running", progress_pct=previous_progress)
            logger.write(f"[step] {step.label}\n")
            logger.write(f"[cmd] {' '.join(step.cmd)}\n")
            stream_command(step.cmd, logger, base_env, repo_root)
            previous_progress = step.progress_end
            tracker.emit(step=step.key, label=step.label, status="completed", progress_pct=previous_progress)
        elapsed = time.time() - start_time
        tracker.emit(step="pipeline", label="Pipeline complete", status="completed", progress_pct=100, detail=f"elapsed_s={elapsed:.1f}")
        logger.write(f"\nDONE. Outputs in: {run_dir}\n")
    except subprocess.CalledProcessError as exc:
        tracker.emit(step=step.key, label=step.label, status="failed", progress_pct=previous_progress, detail=f"returncode={exc.returncode}")
        logger.write(f"\nFAILED at step={step.key} returncode={exc.returncode}\n")
        raise SystemExit(exc.returncode) from exc
    finally:
        logger.close()


if __name__ == "__main__":
    main()
