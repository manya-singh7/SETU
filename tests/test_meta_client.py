"""Unit tests for Meta Model API client wrapper and quality gate."""

import json
import os
from unittest.mock import MagicMock
import numpy as np
import pytest
import requests

from setu.meta_client import (
    BudgetExceededError,
    MetaAPIError,
    MetaClient,
    get_running_total_cost,
)
from setu.quality_gate import QualityGate, check_frame_quality, resize_frame_to_768


def test_meta_client_mock_mode_zero_network(tmp_path):
    """Verify mock mode returns fixture with zero network requests."""
    usage_file = tmp_path / "usage.jsonl"
    client = MetaClient(
        mode="mock",
        usage_log_path=str(usage_file),
    )

    resp = client.chat_completion(
        messages=[{"role": "user", "content": "Analyze bridge"}],
        fixture_name="muse_spark_assess.json",
    )

    assert resp is not None
    assert "choices" in resp
    content = resp["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    assert parsed["severity"] in ("none", "low", "medium", "high")
    assert parsed["assessor"] == "muse_spark"
    # Ensure usage.jsonl was NOT modified in mock mode
    assert not usage_file.exists()


def test_meta_client_cached_mode(tmp_path, monkeypatch):
    """Verify cached mode writes to cache on miss and reads on hit."""
    cache_dir = tmp_path / "cache_meta"
    usage_file = tmp_path / "usage.jsonl"
    client = MetaClient(
        api_key="meta_test_secret_12345",
        mode="cached",
        cache_dir=str(cache_dir),
        usage_log_path=str(usage_file),
    )

    network_calls = 0

    def mock_post(*args, **kwargs):
        nonlocal network_calls
        network_calls += 1
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps({"result": "cached_ok"})}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        return resp

    monkeypatch.setattr(requests, "post", mock_post)

    messages = [{"role": "user", "content": "Query 1"}]

    # Call 1: Cache Miss -> Hits network & writes to cache
    res1 = client.chat_completion(messages=messages)
    assert network_calls == 1
    assert "result" in res1["choices"][0]["message"]["content"]

    # Call 2: Cache Hit -> Reads from disk, zero network calls
    res2 = client.chat_completion(messages=messages)
    assert network_calls == 1
    assert res1 == res2


def test_meta_client_budget_cap_refuses_calls(tmp_path):
    """Verify that reaching SETU_BUDGET_CAP refuses further live API calls."""
    usage_file = tmp_path / "usage.jsonl"
    # Seed usage log with $10.50 of spend
    with open(usage_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"estimated_cost": 10.50}) + "\n")

    client = MetaClient(
        api_key="meta_key_fake",
        mode="live",
        budget_cap=10.0,  # $10 cap is exceeded by $10.50
        usage_log_path=str(usage_file),
    )

    with pytest.raises(BudgetExceededError) as exc_info:
        client.chat_completion(messages=[{"role": "user", "content": "Should fail"}])

    assert "budget cap" in str(exc_info.value).lower()


def test_never_log_api_key(tmp_path, monkeypatch, capsys):
    """Verify that META_API_KEY is never logged to usage.jsonl or terminal output."""
    secret_key = "SECRET_SUPER_SENSITIVE_KEY_XYZ999"
    usage_file = tmp_path / "usage.jsonl"

    client = MetaClient(
        api_key=secret_key,
        mode="live",
        usage_log_path=str(usage_file),
    )

    def mock_post(*args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }
        return resp

    monkeypatch.setattr(requests, "post", mock_post)

    client.chat_completion(messages=[{"role": "user", "content": "Hello"}])

    # Check terminal stdout/stderr
    captured = capsys.readouterr()
    assert secret_key not in captured.out
    assert secret_key not in captured.err

    # Check usage file
    with open(usage_file, "r", encoding="utf-8") as f:
        file_content = f.read()
    assert secret_key not in file_content


def test_quality_gate_mock_mode():
    """Verify quality gate returns expected visual metrics in mock mode."""
    # Use a dedicated mock client so the fixture and mode are independent of
    # the process-wide MetaClient singleton mutated by other application tests.
    gate = QualityGate(client=MetaClient(mode="mock", fixtures_dir=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "tests", "fixtures"
    )))
    # Dummy frame
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = gate.check_frame(dummy_frame)

    assert isinstance(result, dict)
    assert result["is_structural_element"] is True
    assert result["has_static_background"] is True
    assert result["appears_stable"] is True
    assert isinstance(result["issues"], list)


def test_resize_frame_to_768():
    """Verify frame resizing to 768px maximum dimension."""
    # 1920x1080 frame
    large_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    resized = resize_frame_to_768(large_frame)
    h, w = resized.shape[:2]

    assert max(h, w) == 768
    assert w == 768
    assert h == 432  # 1080 * (768 / 1920) = 432


def test_sam_video_mock_mode_loads_fixture_without_network(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"mock video bytes")
    client = MetaClient(mode="mock", usage_log_path=str(tmp_path / "usage.jsonl"))
    fixture = {"status": "completed", "output_text": "<0f>0<|box;x1=1;y1=2;x2=3;y2=4;w=10;h=10|>"}
    monkeypatch.setattr(client, "_load_fixture", lambda _name: fixture)
    monkeypatch.setattr(requests, "post", lambda *_a, **_kw: pytest.fail("mock mode must not call network"))
    assert client.track_video(video, "truck", frame_count=20) == fixture


def test_sam_video_cached_mode_caches_and_logs_frame_cost(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"mock video bytes")
    usage = tmp_path / "usage.jsonl"
    cache = tmp_path / "cache"
    client = MetaClient(api_key="test-key", mode="cached", cache_dir=str(cache), usage_log_path=str(usage))
    response = {"status": "completed", "output_text": "<0f>"}
    calls = []

    def mock_post(url, **kwargs):
        calls.append((url, kwargs))
        result = MagicMock(status_code=200)
        result.json.return_value = response
        return result

    monkeypatch.setattr(requests, "post", mock_post)
    assert client.track_video(video, "truck", frame_count=900) == response
    assert client.track_video(video, "truck", frame_count=900) == response
    assert len(calls) == 1
    assert calls[0][0].endswith("/responses")
    assert calls[0][1]["json"]["input"][0]["content"][0]["text"] == "truck"
    assert calls[0][1]["json"]["input"][0]["content"][1]["type"] == "input_video"
    entry = json.loads(usage.read_text(encoding="utf-8").strip())
    assert entry["frames"] == 900
    assert entry["estimated_cost"] == 0.18


def test_sam_video_missing_key_returns_none_without_network(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"mock video bytes")
    client = MetaClient(api_key="", mode="live", usage_log_path=str(tmp_path / "usage.jsonl"))
    client.api_key = ""
    monkeypatch.setattr(requests, "post", lambda *_a, **_kw: pytest.fail("missing key must not call network"))
    assert client.track_video(video, "truck", frame_count=900) is None


def test_sam_video_budget_cap_checks_estimated_video_cost(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"mock video bytes")
    usage = tmp_path / "usage.jsonl"
    usage.write_text(json.dumps({"estimated_cost": 1.0}) + "\n", encoding="utf-8")
    client = MetaClient(api_key="test-key", mode="live", budget_cap=1.1, usage_log_path=str(usage))
    monkeypatch.setattr(requests, "post", lambda *_a, **_kw: pytest.fail("over budget must not call network"))
    with pytest.raises(BudgetExceededError):
        client.track_video(video, "car", frame_count=1000)


def test_meta_api_calls_retry_timeout_and_connection_errors(tmp_path, monkeypatch, caplog):
    """Muse, SAM image, and SAM video requests retry transient network failures three times."""
    import time

    monkeypatch.setattr(time, "sleep", lambda _delay: None)
    client = MetaClient(
        api_key="test-key", mode="live", budget_cap=5.0,
        usage_log_path=str(tmp_path / "usage.jsonl"),
    )
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    responses = {"calls": 0}
    successful = {
        "status_code": 200,
        "json": {"choices": [{"message": {"content": "ok"}}], "usage": {}},
    }

    def flaky_post(*_args, **_kwargs):
        responses["calls"] += 1
        if responses["calls"] % 4 != 0:
            if responses["calls"] % 2:
                raise requests.exceptions.Timeout("read timed out")
            raise requests.exceptions.ConnectionError("connection reset")
        result = MagicMock(status_code=successful["status_code"])
        result.json.return_value = successful["json"]
        return result

    monkeypatch.setattr(requests, "post", flaky_post)
    client.chat_completion([{"role": "user", "content": "hello"}])
    client.segment_image("aW1hZ2U=", "pillar")
    client.track_video(video, "car", frame_count=1)
    assert responses["calls"] == 12
    assert sum("retrying attempt" in rec.message for rec in caplog.records) == 9
