"""Tests of spatial frequency measurements and the frame-0 vibration map."""

import cv2
import numpy as np
import pytest

from scripts.vibration_map_demo import FREQUENCIES_HZ, generate_synthetic_video
from setu.vibration_map import MAX_INSTANCES, build_vibration_map, mode_shape_from_instances


def test_four_post_frequencies_recovered_and_map_written(tmp_path):
    video = tmp_path / "four_posts.mp4"
    output = tmp_path / "vibration_map.png"
    masks, background = generate_synthetic_video(video)

    records = build_vibration_map(video, masks, background, output)

    assert len(records) == 4
    assert output.is_file()
    png = cv2.imread(str(output))
    assert png is not None and png.ndim == 3
    assert [record["label"] for record in records] == [label for label, _ in masks]
    for index, record in enumerate(records):
        assert record["verdict"] == "usable", record
        assert record["frequency_hz"] is not None
        assert abs(record["frequency_hz"] - FREQUENCIES_HZ[index]) <= 0.1, record
        assert record["amplitude_px"] is not None and record["amplitude_px"] > 0
        assert record["bbox"] == [55 + 135 * index, 82, 56, 210]


def test_instance_cap_is_sixteen(tmp_path, monkeypatch):
    video = tmp_path / "one_frame.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (100, 80))
    writer.write(np.full((80, 100, 3), 120, np.uint8))
    writer.release()
    background = np.ones((80, 100), dtype=bool)
    masks = []
    for i in range(20):
        mask = np.zeros((80, 100), dtype=bool)
        mask[10:30, 2 + i * 4 : 5 + i * 4] = True
        masks.append((f"target {i}", mask))

    monkeypatch.setattr("setu.vibration_map.track_motion", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("not used")))
    output = tmp_path / "capped.png"
    records = build_vibration_map(video, masks, background, output)
    assert len(records) == MAX_INSTANCES
    assert records[-1]["label"] == "target 15"
    assert output.is_file()


def test_mode_shape_projects_collinear_instances_and_uses_shared_frequency():
    records = [
        {"label": "A", "bbox": [0, 10, 10, 10], "frequency_hz": 3.7,
         "amplitude_px": 0.02, "verdict": "usable"},
        {"label": "B", "bbox": [20, 11, 10, 10], "frequency_hz": 3.72,
         "amplitude_px": 0.05, "verdict": "usable"},
        {"label": "C", "bbox": [40, 9, 10, 10], "frequency_hz": 3.69,
         "amplitude_px": 0.03, "verdict": "usable"},
    ]
    shape = mode_shape_from_instances(records, dominant_frequency_hz=3.7)
    assert [point["label"] for point in shape] == ["A", "B", "C"]
    assert [point["position_norm"] for point in shape] == pytest.approx([0.0, 0.5, 1.0], abs=0.002)
    assert [point["amplitude_px"] for point in shape] == [0.02, 0.05, 0.03]


def test_mode_shape_skips_non_collinear_or_too_few_instances():
    records = [
        {"label": "A", "bbox": [0, 0, 10, 10], "frequency_hz": 3.7,
         "amplitude_px": 0.02, "verdict": "usable"},
        {"label": "B", "bbox": [30, 0, 10, 10], "frequency_hz": 3.7,
         "amplitude_px": 0.02, "verdict": "usable"},
        {"label": "C", "bbox": [15, 30, 10, 10], "frequency_hz": 3.7,
         "amplitude_px": 0.02, "verdict": "usable"},
    ]
    assert mode_shape_from_instances(records, 3.7) == []
    assert mode_shape_from_instances(records[:2], 3.7) == []
