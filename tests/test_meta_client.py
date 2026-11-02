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
    gate = QualityGate()
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
