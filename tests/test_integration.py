"""End-to-end integration tests for Project SETU vibration analysis API.

Tests full pipeline execution with SETU_DEMO_MODE=false and SETU_META_MODE=mock (no network):
1. Synthetic video generation and upload via POST /api/scans.
2. Background vibration tracking, modal PSD computation, and result retrieval via GET /api/scans/{id}.
3. Plot download verification via GET /api/scans/{id}/files/{name}.
4. Corrupt video upload handling (verdict: failed, status: complete, no server crash).
5. Bridge aggregation verification in GET /api/bridges.
"""

import io
import os
import tempfile
import pytest
from fastapi.testclient import TestClient

from setu.api import app
from setu.osm import OSMBridgeInfo
from tests.synth import generate_synthetic_video

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_integration_env(monkeypatch):
    """Enforce offline execution mode with no network access."""
    monkeypatch.setenv("SETU_DEMO_MODE", "false")
    monkeypatch.setenv("SETU_META_MODE", "mock")

    # Guard against missing attribute and eliminate external Overpass network latency
    OSMBridgeInfo.span_count = None
    monkeypatch.setattr(
        "setu.osm.OverpassBridgeClient.find_nearest_bridge",
        lambda *args, **kwargs: None,
    )


@pytest.mark.slow
def test_end_to_end_synthetic_scan():
    """Upload short synthetic video (10s, 640x480, 60fps, 3.7Hz) and verify full analysis pipeline."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_video_path = tmp_file.name

    try:
        # 1. Generate 10s synthetic video at 3.7 Hz resonant frequency
        generate_synthetic_video(
            output_path=tmp_video_path,
            fps=60.0,
            duration_sec=10.0,
            width=640,
            height=480,
            freq_hz=3.7,
        )

        # 2. Upload through POST /api/scans
        with open(tmp_video_path, "rb") as video_file:
            upload_resp = client.post(
                "/api/scans",
                files={"file": ("synth_pillar_10s.mp4", video_file, "video/mp4")},
                data={
                    "lat": "13.0827",
                    "lon": "80.2707",
                    "target_prompt": "concrete pillar",
                    "background_prompt": "building",
                },
            )
        assert upload_resp.status_code == 200
        upload_data = upload_resp.json()
        scan_id = upload_data["scan_id"]
        assert scan_id.startswith("scan_")

        # 3. Retrieve scan result via GET /api/scans/{id}
        # In TestClient, BackgroundTasks are processed before response completion
        get_resp = client.get(f"/api/scans/{scan_id}")
        assert get_resp.status_code == 200
        scan_result = get_resp.json()

        # Assert status "complete", is_demo False, verdict "usable"
        assert scan_result["status"] == "complete"
        assert scan_result["is_demo"] is False
        assert scan_result["verdict"]["level"] == "usable"

        # Assert dominant frequency within 0.1 Hz of 3.7 Hz
        meas = scan_result["measurement"]
        assert meas["dominant_frequency_hz"] is not None
        assert abs(meas["dominant_frequency_hz"] - 3.7) <= 0.1

        # Assert measurement.psd is present
        assert meas["psd"] is not None
        assert "freq_hz" in meas["psd"]
        assert "power" in meas["psd"]
        assert len(meas["psd"]["freq_hz"]) > 0

        # Assert spectrum plot is downloadable via /api/scans/{id}/files/
        spec_plot_file = scan_result["outputs"]["spectrum_plot"] or "spectrum.png"
        plot_resp = client.get(f"/api/scans/{scan_id}/files/{spec_plot_file}")
        assert plot_resp.status_code == 200
        assert plot_resp.headers["content-type"] == "image/png"
        assert len(plot_resp.content) > 0

        # 4. Confirm scan appears in GET /api/bridges
        bridges_resp = client.get("/api/bridges")
        assert bridges_resp.status_code == 200
        bridges = bridges_resp.json()
        assert isinstance(bridges, list)

        all_scans_in_bridges = []
        for b in bridges:
            all_scans_in_bridges.extend(b.get("scans", []))

        assert scan_id in all_scans_in_bridges

    finally:
        if os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)


def test_corrupt_video_upload_fails_gracefully():
    """Verify that uploading a corrupt .mp4 file ends with status 'complete' and verdict 'failed' with reason."""
    corrupt_bytes = b"CORRUPTED_VIDEO_STREAM_DATA_HEADER_INVALID_12345"
    response = client.post(
        "/api/scans",
        files={"file": ("corrupt_recording.mp4", io.BytesIO(corrupt_bytes), "video/mp4")},
        data={"lat": "13.0827", "lon": "80.2707"},
    )
    assert response.status_code == 200
    scan_id = response.json()["scan_id"]

    get_resp = client.get(f"/api/scans/{scan_id}")
    assert get_resp.status_code == 200
    res = get_resp.json()

    # The scan must end with status "complete" and verdict "failed" with reasons (not a 500 error)
    assert res["status"] == "complete"
    assert res["is_demo"] is False
    assert res["verdict"]["level"] == "failed"
    assert len(res["verdict"]["reasons"]) > 0

    # Ensure measurements are nullified per specification when verdict is failed
    assert res["measurement"]["dominant_frequency_hz"] is None
    assert res["measurement"]["frequency_ci_hz"] is None
    assert res["measurement"]["peaks"] == []

    # Confirm corrupt scan also registers in /api/bridges under its location
    bridges_resp = client.get("/api/bridges")
    assert bridges_resp.status_code == 200
    all_scan_ids = []
    for b in bridges_resp.json():
        all_scan_ids.extend(b.get("scans", []))
    assert scan_id in all_scan_ids
