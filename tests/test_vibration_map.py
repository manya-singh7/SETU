"""Tests of spatial frequency measurements and the frame-0 vibration map."""

import cv2
import numpy as np

from scripts.vibration_map_demo import FREQUENCIES_HZ, generate_synthetic_video
from setu.vibration_map import MAX_INSTANCES, build_vibration_map


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
