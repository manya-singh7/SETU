"""Unit tests for OpenStreetMap Overpass bridge query client and parser."""

import json
from unittest.mock import MagicMock
import pytest
import requests

from setu.osm import (
    OSMBridgeInfo,
    OverpassBridgeClient,
    compute_geometry_length,
    haversine_distance,
    parse_tagged_length,
)

# Overpass JSON fixture representing two bridges near Napier Bridge, Chennai
NAPIER_BRIDGE_FIXTURE = {
    "version": 0.6,
    "generator": "Overpass API",
    "elements": [
        {
            "type": "way",
            "id": 24891152,
            "tags": {
                "bridge": "yes",
                "bridge:structure": "arch",
                "highway": "primary",
                "layer": "1",
                "material": "concrete",
                "name": "Napier Bridge",
            },
            "geometry": [
                {"lat": 13.067120, "lon": 80.284550},
                {"lat": 13.067420, "lon": 80.284850},
                {"lat": 13.067720, "lon": 80.285150},
                {"lat": 13.068020, "lon": 80.285450},
            ],
        },
        {
            "type": "way",
            "id": 99999999,
            "tags": {
                "bridge": "yes",
                "bridge:structure": "beam",
                "highway": "secondary",
                "material": "steel",
                "name": "Farther Bypass Span",
                "length": "85.5 m",
            },
            "geometry": [
                {"lat": 13.069500, "lon": 80.287000},
                {"lat": 13.069900, "lon": 80.287500},
            ],
        },
    ],
}


def test_haversine_distance():
    """Verify haversine distance calculation matches expected spherical distance."""
    # ~111 km per degree of latitude
    dist = haversine_distance(13.0, 80.0, 13.001, 80.0)
    assert 110.0 < dist < 112.0


def test_compute_geometry_length():
    """Verify polyline length calculation from coordinate points."""
    coords = [
        (13.067120, 80.284550),
        (13.067420, 80.284850),
        (13.067720, 80.285150),
        (13.068020, 80.285450),
    ]
    length = compute_geometry_length(coords)
    # Total distance is approx 140 meters
    assert 130.0 < length < 160.0


def test_parse_tagged_length():
    """Verify parsing of tagged bridge length with various units."""
    assert parse_tagged_length("120") == 120.0
    assert parse_tagged_length("120 m") == 120.0
    assert parse_tagged_length("85.5 meters") == 85.5
    assert parse_tagged_length("invalid") is None
    assert parse_tagged_length(None) is None


def test_parse_fixture_nearest_bridge_geometry_length():
    """Verify parsing fixture correctly computes geometry length and extracts tags."""
    client = OverpassBridgeClient()
    # Query location close to Napier Bridge (approx 20m from start node)
    lat, lon = 13.067200, 80.284600
    bridge = client.parse_overpass_response(NAPIER_BRIDGE_FIXTURE, lat, lon, radius_meters=150.0)

    assert bridge is not None
    assert bridge.osm_id == "way/24891152"
    assert bridge.name == "Napier Bridge"
    assert bridge.bridge_type == "arch"
    assert bridge.structure == "arch"
    assert bridge.material == "concrete"
    assert bridge.highway == "primary"
    assert bridge.layer == "1"
    # Tagged length was absent, so length_meters was computed from geometry
    assert bridge.length_meters is not None
    assert 130.0 < bridge.length_meters < 160.0
    assert bridge.distance_meters < 50.0


def test_tagged_length_takes_precedence():
    """Verify that explicit length tag overrides geometry calculation."""
    client = OverpassBridgeClient()
    # Query location close to the second bridge
    lat, lon = 13.069510, 80.287010
    bridge = client.parse_overpass_response(NAPIER_BRIDGE_FIXTURE, lat, lon, radius_meters=150.0)

    assert bridge is not None
    assert bridge.osm_id == "way/99999999"
    assert bridge.name == "Farther Bypass Span"
    assert bridge.bridge_type == "beam"
    assert bridge.length_meters == 85.5


def test_nearest_bridge_selection():
    """Verify that when multiple bridges are in range, the nearest one is selected."""
    client = OverpassBridgeClient()
    # Point midway but closer to Napier Bridge
    bridge = client.parse_overpass_response(
        NAPIER_BRIDGE_FIXTURE, 13.067500, 80.284900, radius_meters=500.0
    )
    assert bridge is not None
    assert bridge.name == "Napier Bridge"


def test_no_bridge_found_empty():
    """Verify empty Overpass response returns None gracefully."""
    client = OverpassBridgeClient()
    empty_fixture = {"version": 0.6, "elements": []}
    result = client.parse_overpass_response(empty_fixture, 13.0, 80.0, radius_meters=150.0)
    assert result is None


def test_no_bridge_within_radius():
    """Verify bridges beyond the 150m radius threshold are not returned."""
    client = OverpassBridgeClient()
    # Far coordinates (several kilometers away from the fixture bridges)
    result = client.parse_overpass_response(
        NAPIER_BRIDGE_FIXTURE, 13.150000, 80.350000, radius_meters=150.0
    )
    assert result is None


def test_overpass_timeout_handled_gracefully(monkeypatch):
    """Verify network timeout returns None without raising an uncaught exception."""
    client = OverpassBridgeClient(cache_dir="tmp_cache_test")

    def mock_post(*args, **kwargs):
        raise requests.exceptions.Timeout("Overpass connection timed out")

    monkeypatch.setattr(requests, "post", mock_post)

    # Coordinates not in cache
    result = client.find_nearest_bridge(13.99999, 80.99999, radius_meters=150.0)
    assert result is None
    assert client.timeout == 10
    assert "timed out" in client.last_error.lower()


def test_disk_caching(tmp_path, monkeypatch):
    """Verify that results are cached to disk in cache/osm keyed by rounded lat/lon."""
    cache_dir = tmp_path / "cache" / "osm"
    client = OverpassBridgeClient(cache_dir=str(cache_dir))

    call_count = 0

    def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = NAPIER_BRIDGE_FIXTURE
        return mock_resp

    monkeypatch.setattr(requests, "post", mock_post)

    lat, lon = 13.067123, 80.284567
    # First call: hits network and writes to cache
    bridge1 = client.find_nearest_bridge(lat, lon, radius_meters=150.0)
    assert bridge1 is not None
    assert call_count == 1

    # Check cache file was created with rounded key
    rounded_key = f"{round(lat, 3):.3f}_{round(lon, 3):.3f}.json"
    cache_file = cache_dir / rounded_key
    assert cache_file.exists()

    # Second call with slightly different lat/lon that rounds to same key:
    # lat 13.067123 rounds to 13.067, lon 80.284567 rounds to 80.285
    bridge2 = client.find_nearest_bridge(13.067200, 80.284800, radius_meters=150.0)
    assert bridge2 is not None
    assert bridge2.osm_id == bridge1.osm_id
    # Ensure network was NOT called a second time
    assert call_count == 1
