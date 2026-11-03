"""Tests for bridge engineer report CSV and JSON construction."""

import csv
import io

from setu.report import bridge_key_for_scan, build_bridge_report, report_to_csv


def _scan(scan_id, *, demo=False, osm_id=None):
    bridge = {"name": "Harbor Bridge", "osm_id": osm_id} if osm_id else "Harbor Bridge"
    return {
        "scan_id": scan_id,
        "created_at": "2026-11-03T12:30:00+05:30",
        "is_demo": demo,
        "location": {"lat": 13.0827, "lon": 80.2707},
        "verdict": {"level": "usable", "reasons": []},
        "segmentation": {"target_prompt": "stay cable", "target_source": "sam", "background_source": "sam"},
        "measurement": {"dominant_frequency_hz": 3.7, "frequency_ci_hz": [3.6, 3.8]},
        "interpretation": {"bridge": bridge},
        "cable": {"is_cable": True, "tension_change_pct": -9.75,
                  "tension_change_range_pct": [-11.2, -8.3]},
        "field_notes": [{"english": "A crack is visible.", "tags": [
            {"tag": "crack", "evidence": "crack is visible"},
        ]}],
    }


def test_bridge_key_prefers_osm_id_then_rounded_coordinates():
    assert bridge_key_for_scan(_scan("osm", osm_id="way/987")) == "way/987"
    assert bridge_key_for_scan(_scan("gps")) == "13.083,80.271"


def test_json_and_csv_reports_include_fields_and_exclude_demo_scans():
    production = _scan("scan_prod")
    demo = _scan("scan_demo", demo=True)
    key = bridge_key_for_scan(production)
    report = build_bridge_report(key, [production, demo])

    assert report["bridge"]["bridge_key"] == key
    assert report["bridge"]["name"] == "Harbor Bridge"
    assert len(report["scans"]) == 1
    row = report["scans"][0]
    assert row["scan_id"] == "scan_prod"
    assert row["ci_low_hz"] == 3.6 and row["ci_high_hz"] == 3.8
    assert row["cable_tension_change_pct"] == -9.75
    assert row["cable_tension_change_low_pct"] == -11.2
    assert row["field_note_tags"] == [{"tag": "crack", "evidence": "crack is visible"}]
    assert row["scan_page_link"] == "/?scan=scan_prod"

    csv_rows = list(csv.DictReader(io.StringIO(report_to_csv(report))))
    assert len(csv_rows) == 1
    assert csv_rows[0]["scan_id"] == "scan_prod"
    assert "crack is visible" in csv_rows[0]["field_note_tags"]
    assert csv_rows[0]["cable_tension_change_pct"] == "-9.75"


def test_report_excludes_demos_even_when_only_matching_scan():
    demo = _scan("scan_demo", demo=True)
    report = build_bridge_report(bridge_key_for_scan(demo), [demo])
    assert report["scans"] == []
