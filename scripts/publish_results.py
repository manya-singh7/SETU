#!/usr/bin/env python3
"""Publish completed SETU scans and validation comparison JSON for the demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterable, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_KEYS = (
    "spectrum_plot", "displacement_plot", "magnified_video", "vibration_map", "sound",
)
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validated_name(value: str) -> str:
    name = value[:-5] if value.lower().endswith(".json") else value
    if not SAFE_NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError("NAME must contain only letters, numbers, dots, underscores, or hyphens")
    return name


def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON file: {path}") from exc


def _scan_output_names(result: dict[str, Any]) -> List[str]:
    outputs = result.get("outputs") or {}
    names = [outputs[key] for key in OUTPUT_KEYS if isinstance(outputs.get(key), str) and outputs[key]]
    traffic = result.get("traffic") or {}
    response_plot = traffic.get("response_plot") if isinstance(traffic, dict) else None
    if isinstance(response_plot, str) and response_plot:
        names.append(response_plot)
    return list(dict.fromkeys(names))


def publish_scan(scan_folder: Path | str, name: str, root: Path = ROOT) -> List[Path]:
    """Copy a completed non-demo scan result and its declared output files."""
    name = _validated_name(name)
    source = Path(scan_folder).expanduser().resolve()
    result_path = source / "result.json"
    if not source.is_dir() or not result_path.is_file():
        raise ValueError("--scan must point to a scan folder containing result.json")
    result = _read_json(result_path)
    if not isinstance(result, dict):
        raise ValueError("result.json must contain a JSON object")
    if result.get("is_demo") is True:
        raise ValueError("Demo results cannot be published as showcase scans")
    if result.get("status") != "complete":
        raise ValueError("Only completed scans can be published")

    excluded = {"device_motion.json"}
    video_name = (result.get("video") or {}).get("filename")
    if isinstance(video_name, str) and video_name:
        excluded.add(Path(video_name).name.lower())

    files: List[tuple[Path, Path]] = [(result_path, Path("result.json"))]
    for output_name in _scan_output_names(result):
        relative = Path(output_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe output path in result.json: {output_name}")
        if relative.name.lower() in excluded:
            continue
        candidate = (source / relative).resolve()
        try:
            candidate.relative_to(source)
        except ValueError as exc:
            raise ValueError(f"Output file points outside the scan folder: {output_name}") from exc
        if not candidate.is_file():
            raise ValueError(f"Declared output file is missing: {output_name}")
        files.append((candidate, relative))

    destination_root = Path(root).resolve() / "demo_scans"
    destination = destination_root / name
    if destination.exists():
        raise ValueError(f"Showcase scan already exists: demo_scans/{name}")
    destination.mkdir(parents=True, exist_ok=False)
    copied: List[Path] = []
    try:
        for src, relative in files:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            copied.append(target.relative_to(Path(root).resolve()))
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return copied


def _validation_index_entries(index: Any) -> List[str]:
    if not isinstance(index, list):
        raise ValueError("validation/index.json must contain a JSON array")
    names: List[str] = []
    for entry in index:
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, dict):
            file = entry.get("file") or entry.get("path")
            if isinstance(file, str):
                names.append(file)
    return names


def publish_validation(validation_json: Path | str, name: str, root: Path = ROOT) -> List[Path]:
    """Copy validation output JSON and register it once in the frontend index."""
    name = _validated_name(name)
    source = Path(validation_json).expanduser().resolve()
    if not source.is_file():
        raise ValueError("--validation must point to an existing JSON file")
    payload = _read_json(source)
    if not isinstance(payload, dict):
        raise ValueError("Validation output must contain a JSON object")

    validation_dir = Path(root).resolve() / "frontend" / "public" / "validation"
    index_path = validation_dir / "index.json"
    index = _read_json(index_path) if index_path.exists() else []
    entries = _validation_index_entries(index)
    filename = f"{name}.json"
    entries = [entry for entry in entries if Path(entry).name != filename]
    entries.append(filename)
    entries = sorted(set(entries))

    validation_dir.mkdir(parents=True, exist_ok=True)
    destination = validation_dir / filename
    shutil.copy2(source, destination)
    index_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return [destination.relative_to(Path(root).resolve()), index_path.relative_to(Path(root).resolve())]


def list_published(root: Path = ROOT) -> tuple[List[str], List[str]]:
    """Return published showcase and validation names."""
    root = Path(root).resolve()
    scans_dir = root / "demo_scans"
    scans = sorted(
        folder.name for folder in scans_dir.iterdir()
        if folder.is_dir() and (folder / "result.json").is_file()
    ) if scans_dir.is_dir() else []

    validation_dir = root / "frontend" / "public" / "validation"
    index_path = validation_dir / "index.json"
    entries = _validation_index_entries(_read_json(index_path)) if index_path.exists() else []
    validations = sorted({
        Path(entry).stem for entry in entries
        if Path(entry).name == entry and (validation_dir / entry).is_file()
    })
    return scans, validations


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scan", metavar="FOLDER", help="completed scan folder to publish")
    group.add_argument("--validation", metavar="JSON", help="validation comparison JSON to publish")
    group.add_argument("--list", action="store_true", help="list published scans and validation results")
    parser.add_argument("--name", help="published name (required with --scan or --validation)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.list and not args.name:
        parser.error("--name is required with --scan or --validation")
    try:
        if args.scan:
            copied = publish_scan(args.scan, args.name)
            for path in copied:
                print(path.as_posix())
        elif args.validation:
            copied = publish_validation(args.validation, args.name)
            for path in copied:
                print(path.as_posix())
        else:
            scans, validations = list_published()
            print("Showcase scans:")
            for name in scans:
                print(f"  {name}")
            if not scans:
                print("  (none)")
            print("Validation results:")
            for name in validations:
                print(f"  {name}")
            if not validations:
                print("  (none)")
        return 0
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
