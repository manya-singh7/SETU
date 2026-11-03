"""Small Open-Meteo client for timestamped environmental context."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests


CACHE_DIR = Path("cache/weather")
TIMEOUT_SECONDS = 8
_EMPTY = {"temperature_c": None, "humidity_pct": None, "wind_speed_ms": None}


def _when_datetime(when: Union[datetime, str]) -> datetime:
    if isinstance(when, datetime):
        result = when
    else:
        result = datetime.fromisoformat(str(when).replace("Z", "+00:00"))
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _cache_key(lat: float, lon: float, when: datetime) -> str:
    stamp = when.strftime("%Y%m%dT%H")
    return f"{lat:.4f}_{lon:.4f}_{stamp}.json"


def _none_values() -> Dict[str, Optional[float]]:
    return dict(_EMPTY)


def _parse_payload(payload: Dict[str, Any], when: datetime, historical: bool) -> Dict[str, Optional[float]]:
    if historical:
        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        target = when.strftime("%Y-%m-%dT%H:00")
        try:
            index = times.index(target)
        except (ValueError, AttributeError):
            return _none_values()
        source = hourly
    else:
        source = payload.get("current") or {}
        index = None

    def read(name: str) -> Optional[float]:
        values = source.get(name)
        try:
            value = values[index] if index is not None else values
            number = float(value)
            return number if number == number and abs(number) != float("inf") else None
        except (TypeError, ValueError, IndexError, KeyError):
            return None

    return {
        "temperature_c": read("temperature_2m"),
        "humidity_pct": read("relative_humidity_2m"),
        "wind_speed_ms": read("wind_speed_10m"),
    }


def get_weather(lat: float, lon: float, when: Union[datetime, str]) -> Dict[str, Optional[float]]:
    """Fetch Open-Meteo weather fields, caching successful parsed responses.

    Past timestamps use the archive endpoint; current or future timestamps use
    the forecast endpoint's current conditions. Failures return None per field.
    """
    try:
        lat, lon = float(lat), float(lon)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return _none_values()
        moment = _when_datetime(when)
        now = datetime.now(timezone.utc)
        # Treat the current UTC hour as current conditions even if the caller's
        # timestamp was captured a few milliseconds before this request began.
        historical = moment.replace(minute=0, second=0, microsecond=0) < now.replace(minute=0, second=0, microsecond=0)
        cache_path = CACHE_DIR / _cache_key(lat, lon, moment)
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached, dict) and set(_EMPTY).issubset(cached):
                    return {key: cached.get(key) for key in _EMPTY}
            except (OSError, json.JSONDecodeError):
                pass

        if historical:
            url = "https://archive-api.open-meteo.com/v1/archive"
            day = moment.strftime("%Y-%m-%d")
            params = {
                "latitude": lat, "longitude": lon, "start_date": day, "end_date": day,
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
                "wind_speed_unit": "ms",
                "timezone": "UTC",
            }
        else:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
                "wind_speed_unit": "ms",
                "timezone": "UTC",
            }
        response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        result = _parse_payload(response.json(), moment, historical)
        if all(value is None for value in result.values()):
            return result
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
        return result
    except Exception:
        return _none_values()
