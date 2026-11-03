import json
from datetime import datetime, timezone
from pathlib import Path

from setu import weather


FIXTURES = Path(__file__).parent / "fixtures"


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_historical_weather_parsing_and_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(weather, "CACHE_DIR", tmp_path / "weather")
    payload = json.loads((FIXTURES / "weather_response.json").read_text(encoding="utf-8"))
    calls = []
    monkeypatch.setattr(weather.requests, "get", lambda url, **kwargs: (calls.append((url, kwargs)) or FakeResponse(payload)))

    when = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
    expected = {"temperature_c": 28.0, "humidity_pct": 68.0, "wind_speed_ms": 3.61}
    assert weather.get_weather(13.08, 80.27, when) == expected
    assert "archive-api.open-meteo.com" in calls[0][0]
    assert calls[0][1]["params"]["wind_speed_unit"] == "ms"
    assert weather.get_weather(13.08, 80.27, when) == expected
    assert len(calls) == 1


def test_current_weather_parsing(tmp_path, monkeypatch):
    monkeypatch.setattr(weather, "CACHE_DIR", tmp_path)
    payload = {"current": {"temperature_2m": 26.0, "relative_humidity_2m": 75, "wind_speed_10m": 2.36}}
    calls = []
    monkeypatch.setattr(weather.requests, "get", lambda url, **kwargs: (calls.append((url, kwargs)) or FakeResponse(payload)))
    now = datetime.now(timezone.utc)
    assert weather.get_weather(1, 2, now) == {
        "temperature_c": 26.0, "humidity_pct": 75.0, "wind_speed_ms": 2.36,
    }
    assert "api.open-meteo.com/v1/forecast" in calls[0][0]
    assert calls[0][1]["params"]["wind_speed_unit"] == "ms"


def test_weather_failure_returns_none_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(weather, "CACHE_DIR", tmp_path)

    def fail(*args, **kwargs):
        raise TimeoutError("offline")

    monkeypatch.setattr(weather.requests, "get", fail)
    result = weather.get_weather(1, 2, datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert result == {"temperature_c": None, "humidity_pct": None, "wind_speed_ms": None}
