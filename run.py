"""Production entry point: run the calibration agent across all Drive accounts.

Usage:
    python run.py --config config.yaml

Requires a populated config.yaml, service_account.json, and ANTHROPIC_API_KEY.
The core measurement and FFT detection are validated and run without these; this
script wires them to live Google Drive access and the orchestrator agent.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import yaml

from src.agent import DEFAULT_BATCH_SIZE, run_agent
from src.drive_client import DriveClient
from src.state import StateStore
from src.tools import ToolBox


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the calibration measurement agent")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    base_dir = args.config.resolve().parent

    drive = DriveClient(str(base_dir / config["credentials"]["service_account_json"]))
    store = StateStore(str(base_dir / config["state"]["db_path"]))
    temp_dir = base_dir / config["state"]["temp_dir"]
    toolbox = ToolBox(drive, store, config, temp_dir)

    folder_paths = _discover_unprocessed(drive, store, config)
    if not folder_paths:
        print("No unprocessed folders. Nothing to do.")
        return

    client = _anthropic_client()
    summary = run_agent(
        client=client,
        toolbox=toolbox,
        folder_paths=folder_paths,
        model=config["agent"]["model"],
        max_tokens=config["agent"]["max_tokens"],
        batch_size=int(config["agent"].get("batch_size", DEFAULT_BATCH_SIZE)),
    )

    print(summary)
    _report(store)
    _write_outputs(store, base_dir, config)
    store.close()


def _discover_unprocessed(drive: DriveClient, store: StateStore, config: dict) -> list[str]:
    paths: list[str] = []
    for account in config["accounts"]:
        if account["root_folder_id"] == "FILL_IN":
            continue
        for folder in drive.list_specimen_folders(account["name"], account["root_folder_id"]):
            case_id = f"{account['name']}:{folder.folder_id}"
            store.register_pending(case_id, folder.subfolder_path, account["name"])
            if not store.is_processed(case_id):
                paths.append(folder.subfolder_path)
    return paths


def _anthropic_client():
    import anthropic

    if "ANTHROPIC_API_KEY" not in os.environ:
        raise RuntimeError("set ANTHROPIC_API_KEY to run the agent")
    return anthropic.Anthropic()


def _report(store: StateStore) -> None:
    results = store.all_results()
    reviews = store.review_items()
    successes = [row for row in results if row["status"] == "success"]
    print(f"\nProcessed {len(results)} cases: {len(successes)} measured, {len(reviews)} flagged")
    if successes:
        values = [row["nm_per_pixel"] for row in successes if row["nm_per_pixel"]]
        if values:
            print(f"nm/pixel range: {min(values):.4f} - {max(values):.4f}")


def _write_outputs(store: StateStore, base_dir: Path, config: dict) -> None:
    """Write the results and review CSVs named in the config."""
    output = config.get("output", {})
    results_path = base_dir / output.get("results_filename", "calibration_results.csv")
    review_path = base_dir / output.get("review_filename", "review_queue.csv")

    _write_rows(results_path, store.all_results())
    _write_rows(review_path, store.review_items())
    print(f"\nWrote {results_path.name} and {review_path.name}")


def _write_rows(path: Path, rows: list) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(dict(row) for row in rows)


if __name__ == "__main__":
    main()
