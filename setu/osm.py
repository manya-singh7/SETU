"""
OpenStreetMap (OSM) Overpass API bridge metadata client for Project SETU.

Step 6 (Part A) in SETU Pipeline:
Queries the Overpass API for the nearest bridge infrastructure relative to user's GPS coordinates,
extracting bridge type (truss, beam, suspension), construction material, span length, and tags.
Includes local disk caching and graceful error/timeout handling.
"""

from dataclasses import dataclass, field
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple
import requests

from setu.config import settings

logger = logging.getLogger(__name__)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great-circle distance between two GPS points in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return R * c


def compute_geometry_length(coords: List[Tuple[float, float]]) -> float:
    """Calculate the cumulative polyline length in meters from a list of (lat, lon) coordinates."""
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for i in range(len(coords) - 1):
        total += haversine_distance(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
    return round(total, 2)


def parse_tagged_length(val: Optional[str]) -> Optional[float]:
    """Parse numeric meter length from OSM tag value (e.g. '120', '120 m', '120.5 meters')."""
    if not val:
        return None
    cleaned = (
        str(val)
        .strip()
        .lower()
        .replace("meters", "")
        .replace("meter", "")
        .replace("m", "")
        .strip()
    )
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


def extract_tagged_length(tags: Dict[str, str]) -> Optional[float]:
    """Check bridge tags for explicit length attributes."""
    for key in ("length", "bridge:length", "est_length"):
        if key in tags:
            parsed = parse_tagged_length(tags[key])
            if parsed is not None:
                return parsed
    return None


@dataclass
class OSMBridgeInfo:
    """Bridge metadata retrieved from OpenStreetMap.

    Attributes:
        osm_id: OpenStreetMap identifier (e.g. 'way/12345').
        name: Name of the bridge.
        bridge_type: Type classification (e.g. 'truss', 'beam', 'suspension', 'arch').
        material: Primary material ('concrete', 'steel', 'masonry').
        length_meters: Bridge span or total length in meters.
        latitude: Center latitude coordinate.
        longitude: Center longitude coordinate.
        distance_meters: Distance from queried GPS point to bridge.
        structure: Specific bridge:structure tag if available.
        highway: Road classification if highway bridge.
        railway: Track classification if railway bridge.
        layer: Layer tag if specified.
        tags: Full raw OSM tags dictionary.
    """

    osm_id: str
    name: str = "Unnamed Bridge"
    bridge_type: Optional[str] = None
    material: Optional[str] = None
    length_meters: Optional[float] = None
    latitude: float = 0.0
    longitude: float = 0.0
    distance_meters: float = 0.0
    structure: Optional[str] = None
    highway: Optional[str] = None
    railway: Optional[str] = None
    layer: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize bridge metadata to dictionary for disk caching."""
        return {
            "osm_id": self.osm_id,
            "name": self.name,
            "bridge_type": self.bridge_type,
            "material": self.material,
            "length_meters": self.length_meters,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "distance_meters": self.distance_meters,
            "structure": self.structure,
            "highway": self.highway,
            "railway": self.railway,
            "layer": self.layer,
            "tags": self.tags,
        }

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def keys(self) -> Any:
        return self.to_dict().keys()

    def __iter__(self) -> Any:
        return iter(self.to_dict())

    def items(self) -> Any:
        return self.to_dict().items()

    def values(self) -> Any:
        return self.to_dict().values()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OSMBridgeInfo":
        """Reconstruct bridge info from dictionary."""
        return cls(
            osm_id=data.get("osm_id", ""),
            name=data.get("name", "Unnamed Bridge"),
            bridge_type=data.get("bridge_type"),
            material=data.get("material"),
            length_meters=data.get("length_meters"),
            latitude=float(data.get("latitude", 0.0)),
            longitude=float(data.get("longitude", 0.0)),
            distance_meters=float(data.get("distance_meters", 0.0)),
            structure=data.get("structure"),
            highway=data.get("highway"),
            railway=data.get("railway"),
            layer=data.get("layer"),
            tags=data.get("tags") or {},
        )


class OverpassBridgeClient:
    """Client for querying bridge structural records from OpenStreetMap Overpass API."""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        timeout: int = 20,
        cache_dir: str = "cache/osm",
        mode: Optional[str] = None,
        fixture_path: Optional[str] = None,
    ) -> None:
        self.endpoint_url = endpoint_url or settings.OVERPASS_API_URL
        self.timeout = timeout
        self.cache_dir = cache_dir
        self.mode = mode.lower() if mode else None
        self.fixture_path = fixture_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "tests", "fixtures", "osm_napier_bridge.json"
        )

    def _get_cache_path(self, latitude: float, longitude: float, precision: int = 3) -> str:
        """Compute disk cache file path keyed by rounded lat/lon."""
        rounded_lat = round(latitude, precision)
        rounded_lon = round(longitude, precision)
        filename = f"{rounded_lat:.{precision}f}_{rounded_lon:.{precision}f}.json"
        return os.path.join(self.cache_dir, filename)

    def _read_cache(self, cache_path: str) -> Optional[Tuple[bool, Optional[OSMBridgeInfo]]]:
        """Read cached query result.

        Returns:
            Tuple of (hit, bridge_info_or_None) if cached, otherwise None.
        """
        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            if cached_data is None:
                return (True, None)
            if isinstance(cached_data, dict):
                if cached_data.get("no_bridge"):
                    return (True, None)
                if "elements" in cached_data:
                    # Raw Overpass cache
                    parsed = self.parse_overpass_response(cached_data, 0.0, 0.0, float("inf"))
                    return (True, parsed)
                if "osm_id" in cached_data and cached_data["osm_id"]:
                    return (True, OSMBridgeInfo.from_dict(cached_data))
                if cached_data.get("found") is False:
                    return (True, None)
            return (True, None)
        except Exception as e:
            logger.warning(f"Error reading OSM cache file {cache_path}: {e}")
            return None

    def _write_cache(self, cache_path: str, bridge: Optional[OSMBridgeInfo]) -> None:
        """Write query result to disk cache."""
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                if bridge is not None:
                    json.dump(bridge.to_dict(), f, indent=2)
                else:
                    json.dump({"no_bridge": True}, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write OSM cache to {cache_path}: {e}")

    def build_query(self, latitude: float, longitude: float, radius_meters: float = 150.0) -> str:
        """Build Overpass QL query string for bridge=yes ways within radius."""
        return f"""[out:json][timeout:{self.timeout}];
(
  way["bridge"="yes"](around:{radius_meters},{latitude},{longitude});
);
out body geom;"""

    def parse_overpass_response(
        self,
        data: Dict[str, Any],
        latitude: float,
        longitude: float,
        radius_meters: float = 150.0,
    ) -> Optional[OSMBridgeInfo]:
        """Parse Overpass API JSON response to find and return the nearest bridge within radius."""
        elements = data.get("elements", [])
        if not elements:
            return None

        # Build node coordinate lookup in case nodes are provided separately
        node_coords: Dict[int, Tuple[float, float]] = {
            elem["id"]: (elem["lat"], elem["lon"])
            for elem in elements
            if elem.get("type") == "node" and "lat" in elem and "lon" in elem
        }

        candidate_bridges: List[OSMBridgeInfo] = []

        for elem in elements:
            if elem.get("type") != "way":
                continue

            tags = elem.get("tags", {})
            # Verify it has bridge=yes (or bridge tag)
            bridge_tag = tags.get("bridge")
            if not bridge_tag:
                continue

            # Extract polyline geometry coordinates
            coords: List[Tuple[float, float]] = []
            if "geometry" in elem and isinstance(elem["geometry"], list):
                for pt in elem["geometry"]:
                    if isinstance(pt, dict) and "lat" in pt and "lon" in pt:
                        coords.append((pt["lat"], pt["lon"]))
                    elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        coords.append((pt[0], pt[1]))
            elif "nodes" in elem:
                for nid in elem["nodes"]:
                    if nid in node_coords:
                        coords.append(node_coords[nid])

            # Calculate distance from query coordinate to way geometry
            if coords:
                min_dist = min(haversine_distance(latitude, longitude, c[0], c[1]) for c in coords)
                center_lat = sum(c[0] for c in coords) / len(coords)
                center_lon = sum(c[1] for c in coords) / len(coords)
            elif "center" in elem:
                c = elem["center"]
                center_lat = c.get("lat", latitude)
                center_lon = c.get("lon", longitude)
                min_dist = haversine_distance(latitude, longitude, center_lat, center_lon)
            else:
                # Without geometry or center, cannot compute spatial distance reliably
                continue

            # If beyond radius, skip
            if min_dist > radius_meters:
                continue

            # Length calculation: tagged length takes priority, else geometry polyline
            tagged_len = extract_tagged_length(tags)
            computed_len = compute_geometry_length(coords) if coords else None
            length_meters = tagged_len if tagged_len is not None else computed_len

            # Extract tags
            name = tags.get("name") or tags.get("bridge:name") or "Unnamed Bridge"
            structure = tags.get("bridge:structure") or tags.get("structure")
            material = tags.get("material") or tags.get("bridge:material") or tags.get("building:material")
            highway = tags.get("highway")
            railway = tags.get("railway")
            layer = tags.get("layer")

            # Determine classification bridge_type
            bridge_type = structure
            if not bridge_type and bridge_tag != "yes":
                bridge_type = bridge_tag

            raw_id = str(elem.get("id", ""))
            osm_id = f"way/{raw_id}" if not raw_id.startswith(("way/", "node/", "relation/")) else raw_id

            candidate_bridges.append(
                OSMBridgeInfo(
                    osm_id=osm_id,
                    name=name,
                    bridge_type=bridge_type,
                    material=material,
                    length_meters=length_meters,
                    latitude=round(center_lat, 6),
                    longitude=round(center_lon, 6),
                    distance_meters=round(min_dist, 2),
                    structure=structure,
                    highway=highway,
                    railway=railway,
                    layer=layer,
                    tags=tags,
                )
            )

        if not candidate_bridges:
            return None

        # Sort by distance and return the nearest bridge
        candidate_bridges.sort(key=lambda b: b.distance_meters)
        return candidate_bridges[0]

    def find_nearest_bridge(
        self,
        latitude: float,
        longitude: float,
        radius_meters: float = 150.0,
    ) -> Optional[OSMBridgeInfo]:
        """Query Overpass API for bridge ways around given coordinates within radius_meters.

        Checks disk cache in cache/osm/ keyed by rounded lat/lon first.
        Handles no-bridge-found and Overpass timeouts gracefully.

        Args:
            latitude: Query latitude coordinate.
            longitude: Query longitude coordinate.
            radius_meters: Search radius in meters (default 150 m).

        Returns:
            OSMBridgeInfo for nearest bridge if found, otherwise None.
        """
        # In mock mode, evaluate directly against local offline fixture
        if self.mode == "mock":
            if os.path.exists(self.fixture_path):
                try:
                    with open(self.fixture_path, "r", encoding="utf-8") as f:
                        fixture_data = json.load(f)
                    return self.parse_overpass_response(fixture_data, latitude, longitude, radius_meters)
                except Exception as e:
                    logger.warning(f"Error reading mock OSM fixture {self.fixture_path}: {e}")
            return None

        cache_path = self._get_cache_path(latitude, longitude)
        cached_result = self._read_cache(cache_path)
        if cached_result is not None:
            return cached_result[1]

        query = self.build_query(latitude, longitude, radius_meters)

        try:
            response = requests.post(
                self.endpoint_url,
                data={"data": query},
                timeout=self.timeout,
                headers={"User-Agent": "SETU-Bridge-Client/0.1.0"},
            )
            if response.status_code != 200:
                logger.warning(
                    f"Overpass API returned status {response.status_code}: {response.text[:200]}"
                )
                return None

            data = response.json()
            bridge = self.parse_overpass_response(data, latitude, longitude, radius_meters)

            # Cache the successful result or negative result
            self._write_cache(cache_path, bridge)
            return bridge

        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as exc:
            logger.warning(f"Overpass API request failed or timed out: {exc}")
            return None
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(f"Failed to parse Overpass response: {exc}")
            return None


def get_nearest_bridge(
    latitude: float,
    longitude: float,
    radius_meters: float = 150.0,
    client: Optional[OverpassBridgeClient] = None,
) -> Optional[OSMBridgeInfo]:
    """Convenience function querying nearest bridge within radius_meters."""
    c = client or OverpassBridgeClient()
    return c.find_nearest_bridge(latitude=latitude, longitude=longitude, radius_meters=radius_meters)

