import csv
from pathlib import Path

import numpy as np

from setu.device_motion import analyze_device_motion


FIXTURE = Path(__file__).parent / "fixtures" / "device_motion_12hz.csv"


def _fixture_samples():
    with FIXTURE.open(encoding="utf-8", newline="") as stream:
        return [
            {"t": float(row["t"]), "ax": float(row["ax"]),
             "ay": float(row["ay"]), "az": float(row["az"])}
            for row in csv.DictReader(stream)
        ]


def test_recovers_12_hz_and_reports_steady_sampling_and_agreement():
    result = analyze_device_motion(_fixture_samples(), camera_frequency_hz=12.0,
                                   camera_ci_hz=[11.95, 12.05])
    assert result["sampling_rate_steady"] is True
    assert np.isclose(result["sampling_rate_hz"], 100.0)
    assert abs(result["dominant_frequency_hz"] - 12.0) < 0.1
    assert result["confidence_interval_hz"][0] - 0.02 <= 12.0 <= result["confidence_interval_hz"][1] + 0.02
    assert result["comparison"]["verdict"] == "PASS"


def test_disagreement_verdict_when_combined_uncertainty_does_not_cover_difference():
    result = analyze_device_motion(_fixture_samples(), camera_frequency_hz=14.0,
                                   camera_ci_hz=[13.9, 14.1])
    assert result["comparison"]["verdict"] == "FAIL"
    assert result["comparison"]["agreement"] is False


def test_handheld_low_frequency_motion_invalidates_comparison():
    fs = 100.0
    times = np.arange(1000) / fs
    samples = [
        {"t": float(t), "ax": float(2.0 * np.sin(2 * np.pi * 0.2 * t) + 0.03 * np.sin(2 * np.pi * 12 * t)),
         "ay": float(0.01 * np.sin(2 * np.pi * 12 * t)),
         "az": float(9.81 + 0.01 * np.sin(2 * np.pi * 12 * t))}
        for t in times
    ]
    result = analyze_device_motion(samples, camera_frequency_hz=12.0, camera_ci_hz=[11.9, 12.1])
    assert result["resting_on_structure"] is False
    assert result["comparison"] is None
    assert result["message"] == "not resting on the structure; comparison not valid"
