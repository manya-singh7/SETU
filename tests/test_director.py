"""Tests for video-directed scan setup."""

import json

import cv2
import numpy as np

from setu.director import direct_scan
from setu.meta_client import MetaClient


def make_video(path, frames=40, width=640, height=480, fps=20):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    for i in range(frames):
        frame = np.full((height, width, 3), (i * 5) % 255, dtype=np.uint8)
        cv2.rectangle(frame, (width // 4, height // 4), (width // 2, height * 3 // 4), (40, 120, 180), -1)
        writer.write(frame)
    writer.release()


def test_mock_video_is_prepared_small_and_prompts_are_truncated(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    make_video(source, width=1280, height=720)
    client = MetaClient(mode="mock", fixtures_dir="tests/fixtures")
    captured = {}
    original = client.video_completion

    def inspect_video(path, *args, **kwargs):
        captured["size"] = path.stat().st_size
        cap = cv2.VideoCapture(str(path))
        captured["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        captured["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        captured["fps"] = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(client, "video_completion", inspect_video)
    result = direct_scan(source, "en", meta_client=client)

    assert result is not None
    assert result["target_prompt"] == "weathered concrete bridge support pillar"
    assert len(result["target_prompt"].split()) <= 5
    assert len(result["background_prompt"].split()) <= 5
    assert captured["size"] < source.stat().st_size
    assert captured["width"] <= 854 and captured["height"] <= 480
    assert captured["fps"] <= 5.1


def test_invalid_json_retried_once_then_returns_none(tmp_path):
    source = tmp_path / "source.mp4"
    make_video(source)

    class InvalidMockClient:
        mode = "mock"

        def __init__(self):
            self.calls = 0

        def video_completion(self, *args, **kwargs):
            self.calls += 1
            return {"choices": [{"message": {"content": "not valid JSON"}}]}

    client = InvalidMockClient()
    assert direct_scan(source, "en", meta_client=client) is None
    assert client.calls == 2


def test_invalid_schema_is_retried_then_none(tmp_path):
    source = tmp_path / "source.mp4"
    make_video(source)
    invalid = {"structure_visible": "yes"}

    class InvalidSchemaClient:
        def __init__(self):
            self.calls = 0

        def video_completion(self, *args, **kwargs):
            self.calls += 1
            return {"choices": [{"message": {"content": json.dumps(invalid)}}]}

    client = InvalidSchemaClient()
    assert direct_scan(source, "en", meta_client=client) is None
    assert client.calls == 2
