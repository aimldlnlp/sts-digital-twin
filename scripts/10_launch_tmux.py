from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common import ensure_dir, now_run_id


def session_exists(name: str) -> bool:
    probe = subprocess.run(["tmux", "has-session", "-t", name], capture_output=True, text=True)
    return probe.returncode == 0


def allocate_session_name(requested: str, run_id: str) -> str:
    if not session_exists(requested):
        return requested
    candidate = f"{requested}-{run_id.split('_')[-1]}"
    if not session_exists(candidate):
        return candidate
    counter = 2
    while session_exists(f"{candidate}-{counter}"):
        counter += 1
    return f"{candidate}-{counter}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out-root", type=str, default="outputs")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--session", type=str, default="sts-e2e")
    parser.add_argument("--trial-index", type=int, default=0)
    args = parser.parse_args()

    repo_root = REPO_ROOT
    config_path = Path(args.config).resolve()
    out_root = Path(args.out_root).resolve()
    run_id = args.run_id or now_run_id("sts")
    run_dir = out_root / run_id
    log_dir = ensure_dir(run_dir / "logs")
    session_name = allocate_session_name(args.session, run_id)

    command = [
        sys.executable,
        "run_all.py",
        "--config",
        str(config_path),
        "--out-root",
        str(out_root),
        "--run-id",
        run_id,
        "--trial-index",
        str(args.trial_index),
    ]
    shell_command = "cd {cwd} && {cmd}".format(
        cwd=shlex.quote(str(repo_root)),
        cmd=shlex.join(command),
    )

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}:{existing_pythonpath}"

    subprocess.check_call(["tmux", "new-session", "-d", "-s", session_name, shell_command], env=env)
    subprocess.check_call(["tmux", "set-option", "-t", session_name, "remain-on-exit", "on"])

    metadata = {
        "session_name": session_name,
        "requested_session": args.session,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "trial_index": int(args.trial_index),
        "launch_command": command,
        "pipeline_log": str(log_dir / "pipeline.log"),
        "progress_log": str(log_dir / "progress.jsonl"),
        "launched_at": datetime.now(timezone.utc).isoformat(),
    }
    (log_dir / "tmux_session.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"TMUX_SESSION={session_name}")
    print(f"RUN_ID={run_id}")
    print(f"RUN_DIR={run_dir}")
    print(f"PIPELINE_LOG={log_dir / 'pipeline.log'}")
    print(f"PROGRESS_LOG={log_dir / 'progress.jsonl'}")
    print(f"ATTACH=tmux attach -t {session_name}")
    print(f"TAIL=tail -f {log_dir / 'pipeline.log'}")


if __name__ == "__main__":
    main()
