"""Build bridge engineer-report data and CSV representations."""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote


REPORT_COLUMNS = [
    "date", "scan_id", "verdict", "frequency_hz", "ci_low_hz", "ci_high_hz",
    "target_prompt", "target_source", "background_source", "cable_tension_change_pct",
    "cable_tension_change_low_pct", "cable_tension_change_high_pct",
    "field_note_tags", "scan_page_link",
]


def bridge_details(scan: Dict[str, Any]) -> Dict[str, Any]:
    """Extract bridge name and known OSM identity from a scan result."""
    interpretation = scan.get("interpretation") or {}
    bridge = interpretation.get("bridge") or scan.get("bridge_name")
    if isinstance(bridge, dict):
        return {
            "name": bridge.get("name") or scan.get("bridge_name"),
            "osm_id": bridge.get("osm_id") or bridge.get("osmId") or bridge.get("id"),
            "bridge_type": bridge.get("bridge_type") or bridge.get("type"),
            "material": bridge.get("material"),
        }
    return {"name": str(bridge) if bridge else None, "osm_id": None,
            "bridge_type": None, "material": None}


def bridge_key_for_scan(scan: Dict[str, Any]) -> str:
    """Return stable OSM identity, or the scan location rounded to 3 decimals."""
    details = bridge_details(scan)
    if details.get("osm_id") is not None:
        return str(details["osm_id"])
    location = scan.get("location") or {}
    try:
        lat, lon = float(location.get("lat", 0.0)), float(location.get("lon", 0.0))
    except (TypeError, ValueError):
        lat, lon = 0.0, 0.0
    return f"{round(lat, 3):.3f},{round(lon, 3):.3f}"


def _verdict(scan: Dict[str, Any]) -> Optional[str]:
    verdict = scan.get("verdict") or {}
    return verdict.get("level") if isinstance(verdict, dict) else str(verdict)


def _field_note_tags(scan: Dict[str, Any]) -> List[Dict[str, str]]:
    tags: List[Dict[str, str]] = []
    for note in scan.get("field_notes") or []:
        for item in note.get("tags") or []:
            tag, evidence = item.get("tag"), item.get("evidence")
            if tag is not None and evidence is not None:
                tags.append({"tag": str(tag), "evidence": str(evidence)})
    return tags


def scan_report_row(scan: Dict[str, Any]) -> Dict[str, Any]:
    """Create one flat report row from a canonical scan result."""
    measurement = scan.get("measurement") or {}
    ci = measurement.get("frequency_ci_hz")
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        ci = [None, None]
    segmentation = scan.get("segmentation") or {}
    cable = scan.get("cable") or {}
    change_range = cable.get("tension_change_range_pct")
    if not isinstance(change_range, (list, tuple)) or len(change_range) != 2:
        change_range = [None, None]
    scan_id = str(scan.get("scan_id") or "")
    return {
        "date": scan.get("created_at"),
        "scan_id": scan_id,
        "verdict": _verdict(scan),
        "frequency_hz": measurement.get("dominant_frequency_hz"),
        "ci_low_hz": ci[0],
        "ci_high_hz": ci[1],
        "target_prompt": segmentation.get("target_prompt") or scan.get("target_prompt"),
        "target_source": segmentation.get("target_source"),
        "background_source": segmentation.get("background_source"),
        "cable_tension_change_pct": cable.get("tension_change_pct"),
        "cable_tension_change_low_pct": change_range[0],
        "cable_tension_change_high_pct": change_range[1],
        "field_note_tags": _field_note_tags(scan),
        "scan_page_link": f"/?scan={quote(scan_id, safe='')}",
    }


def build_bridge_report(bridge_key: str, scans: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Return report JSON data for one bridge; demo scans are always excluded."""
    matching = [scan for scan in scans if isinstance(scan, dict)
                and scan.get("is_demo") is not True and bridge_key_for_scan(scan) == str(bridge_key)]
    matching.sort(key=lambda scan: str(scan.get("created_at") or ""))
    latest = matching[-1] if matching else {}
    location = latest.get("location") or {}
    details = bridge_details(latest)
    bridge = {
        "bridge_key": str(bridge_key),
        "name": details.get("name"),
        "osm_id": details.get("osm_id"),
        "bridge_type": details.get("bridge_type"),
        "material": details.get("material"),
        "location": location,
    }
    return {
        "bridge": bridge,
        "scans": [scan_report_row(scan) for scan in matching],
    }


def report_to_csv(report: Dict[str, Any]) -> str:
    """Serialize report rows as RFC-compatible CSV text."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=REPORT_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in report.get("scans") or []:
        csv_row = dict(row)
        csv_row["field_note_tags"] = json.dumps(row.get("field_note_tags", []), ensure_ascii=False)
        writer.writerow(csv_row)
    return output.getvalue()
