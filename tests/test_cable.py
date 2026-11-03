"""Tests for cable identification and relative tension estimates."""

from setu.cable import apply_cable_comparison, is_cable_scan, tension_change


def _scan(scan_id, created_at, frequency, ci, verdict="usable", prompt="stay cable", lat=1.0):
    return {
        "scan_id": scan_id,
        "created_at": created_at,
        "is_demo": False,
        "verdict": {"level": verdict, "reasons": []},
        "location": {"lat": lat, "lon": 2.0},
        "segmentation": {"target_prompt": prompt},
        "measurement": {"dominant_frequency_hz": frequency, "frequency_ci_hz": ci},
    }


def test_tension_change_for_five_percent_frequency_drop():
    change, interval = tension_change(10.0, 9.5, [9.9, 10.1], [9.4, 9.6])
    assert change == -9.75
    assert interval[0] < change < interval[1]


def test_uncertainty_uses_both_frequency_confidence_intervals():
    _, narrow = tension_change(10.0, 10.0, [9.99, 10.01], [9.99, 10.01])
    _, wide = tension_change(10.0, 10.0, [9.8, 10.2], [9.8, 10.2])
    assert narrow[1] - narrow[0] > 0
    assert wide[1] - wide[0] > narrow[1] - narrow[0]


def test_cable_detection_from_prompt_and_explicit_upload_flag():
    for prompt in ("cable", "stay cable", "hanger", "suspender"):
        assert is_cable_scan(prompt)
    assert not is_cable_scan("concrete pier")
    assert is_cable_scan("concrete pier", explicit=True)


def test_comparison_uses_prior_usable_scan_and_same_bridge_prompt():
    prior = _scan("prior", "2026-11-01T12:00:00+00:00", 10.0, [9.9, 10.1])
    weak = _scan("weak", "2026-11-02T12:00:00+00:00", 9.8, [9.7, 9.9], "weak")
    current = _scan("current", "2026-11-03T12:00:00+00:00", 9.5, [9.4, 9.6])
    comparison = apply_cable_comparison(current, [prior, weak])
    assert comparison["compared_to_scan_id"] == "prior"
    assert comparison["tension_change_pct"] == -9.75

    only_weak = _scan("only_weak", "2026-11-02T12:00:00+00:00", 9.8, [9.7, 9.9], "weak")
    no_comparison = apply_cable_comparison(current, [only_weak])
    assert no_comparison["is_cable"] is True
    assert no_comparison["tension_change_pct"] is None
    assert no_comparison["compared_to_scan_id"] is None

    current_weak = _scan("current_weak", "2026-11-03T12:00:00+00:00", 9.5, [9.4, 9.6], "weak")
    weak_current_result = apply_cable_comparison(current_weak, [prior])
    assert weak_current_result["tension_change_pct"] is None


def test_comparison_requires_same_cable_prompt_and_bridge():
    prior = _scan("prior", "2026-11-01T12:00:00+00:00", 10.0, [9.9, 10.1])
    current = _scan("current", "2026-11-03T12:00:00+00:00", 9.5, [9.4, 9.6], prompt="hanger")
    assert apply_cable_comparison(current, [prior])["tension_change_pct"] is None
    current["segmentation"]["target_prompt"] = "stay cable"
    prior["location"]["lat"] = 4.0
    assert apply_cable_comparison(current, [prior])["tension_change_pct"] is None
