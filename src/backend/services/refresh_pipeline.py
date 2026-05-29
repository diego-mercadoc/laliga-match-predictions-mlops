from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from services.multi_season_experiments import (
    CURRENT_SEASON,
    DEFAULT_SEASONS,
    download_football_data,
    run_multi_season_experiments,
)
from services.multi_target_experiments import run_multi_target_experiments


STATUS_FILENAME = "refresh_status.json"


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def status_path(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "multi_season" / STATUS_FILENAME


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(repo_root: Path, payload: Dict[str, object]) -> None:
    path = status_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_refresh_status(repo_root: Optional[Path] = None) -> Dict[str, object]:
    root = repo_root or repo_root_from_here()
    path = status_path(root)
    if not path.exists():
        return {"state": "never_run", "status_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def run_refresh_pipeline(
    repo_root: Optional[Path] = None,
    force_download: bool = False,
    skip_experiments: bool = False,
) -> Dict[str, object]:
    root = repo_root or repo_root_from_here()
    output_dir = root / "artifacts" / "multi_season"
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now()
    _write_status(
        root,
        {
            "state": "running",
            "started_at": started_at,
            "force_download": force_download,
            "skip_experiments": skip_experiments,
        },
    )

    try:
        raw_dir = root / "data" / "raw" / "football_data"
        raw_paths = download_football_data(raw_dir, DEFAULT_SEASONS, force=force_download)
        raw_files = [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            }
            for path in raw_paths
        ]

        run_result = None
        target_result = None
        if not skip_experiments:
            run_result = run_multi_season_experiments(
                root,
                seasons=DEFAULT_SEASONS,
                current_season=CURRENT_SEASON,
                force_download=False,
            )
            target_result = run_multi_target_experiments(root, current_season=CURRENT_SEASON)

        completed_at = _utc_now()
        payload: Dict[str, object] = {
            "state": "succeeded",
            "started_at": started_at,
            "completed_at": completed_at,
            "force_download": force_download,
            "skip_experiments": skip_experiments,
            "raw_files": raw_files,
            "output_dir": str(output_dir),
        }
        if run_result:
            payload["summary"] = {
                "split_date": run_result.split_date,
                "data_rows": run_result.data_rows,
                "train_current_rows": run_result.train_current_rows,
                "test_current_rows": run_result.test_current_rows,
                "best_experiment": asdict(run_result.best_experiment),
            }
        if target_result:
            payload["multi_target_summary"] = {
                "target_count": target_result["target_count"],
                "best_accuracy_target": target_result["best_accuracy_target"],
                "best_balanced_target": target_result["best_balanced_target"],
                "best_regression_target": target_result["best_regression_target"],
            }
        _write_status(root, payload)
        return payload
    except Exception as exc:
        failed = {
            "state": "failed",
            "started_at": started_at,
            "failed_at": _utc_now(),
            "force_download": force_download,
            "skip_experiments": skip_experiments,
            "error": repr(exc),
        }
        _write_status(root, failed)
        raise


def start_refresh_subprocess(
    repo_root: Optional[Path] = None,
    force_download: bool = False,
    skip_experiments: bool = False,
) -> Dict[str, object]:
    root = repo_root or repo_root_from_here()
    output_dir = root / "artifacts" / "multi_season"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "refresh_pipeline.log"

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
    ]
    if force_download:
        command.append("--force-download")
    if skip_experiments:
        command.append("--skip-experiments")

    env = os.environ.copy()
    backend_path = str(root / "src" / "backend")
    env["PYTHONPATH"] = backend_path + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYSPARK_PYTHON", sys.executable)
    env.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(root),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

    payload = {
        "state": "starting",
        "started_at": _utc_now(),
        "pid": process.pid,
        "force_download": force_download,
        "skip_experiments": skip_experiments,
        "log_path": str(log_path),
    }
    _write_status(root, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh La Liga data and multi-season experiments.")
    parser.add_argument("--force-download", action="store_true", help="Re-download raw CSVs even if cached files exist.")
    parser.add_argument("--skip-experiments", action="store_true", help="Only refresh raw CSV cache.")
    args = parser.parse_args()
    payload = run_refresh_pipeline(
        force_download=args.force_download,
        skip_experiments=args.skip_experiments,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
