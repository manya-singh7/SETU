"""Tests for SETU API endpoints in Demo Mode."""

import io
import os
import pytest
from fastapi.testclient import TestClient

from setu.api import app
import setu.storage as storage

# Ensure Demo Mode is active for these tests
os.environ["SETU_DEMO_MODE"] = "true"

client = TestClient(app)


@pytest.fixture(autouse=True)
def enable_demo_mode(monkeypatch):
    """Ensure SETU_DEMO_MODE is set for all tests in this module."""
    monkeypatch.setenv("SETU_DEMO_MODE", "true")


def test_health_endpoint():
    """Verify GET /health returns HTTP 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data


def test_scan_upload_demo_mode():
    """Verify POST /api/scans in demo mode skips analysis and returns fixture with new scan_id."""
    video_bytes = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41demo_content"
    files = {"file": ("test_clip.mp4", io.BytesIO(video_bytes), "video/mp4")}
    data = {
        "lat": "13.0827",
        "lon": "80.2707",
        "target_prompt": "concrete pillar",
        "background_prompt": "building",
        "bridge_name": "Napier Bridge",
    }

    response = client.post("/api/scans", files=files, data=data)
    assert response.status_code == 200
    result = response.json()

    # Verify shared result format
    assert result["schema_version"] == "1.0"
    assert result["scan_id"].startswith("scan_")
    assert result["status"] == "complete"
    assert result["is_demo"] is True
    assert result["verdict"]["level"] == "usable"
    assert result["location"]["lat"] == 13.0827
    assert result["location"]["lon"] == 80.2707
    assert result["measurement"]["dominant_frequency_hz"] == 3.70
    assert len(result["measurement"]["peaks"]) >= 1
    assert result["measurement"]["peaks"][0]["frequency_hz"] == 3.70
    assert result["interpretation"]["bridge"] == "Napier Bridge"
    assert "outputs" in result
    assert result["outputs"]["magnified_video"] is None
    assert result["outputs"]["spectrum_plot"] is None
    assert result["outputs"]["displacement_plot"] is None


def test_demo_result_has_is_demo_true_and_null_outputs():
    """Verify demo mode results explicitly set is_demo: true and null outputs without placeholder files."""
    video_bytes = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41demo_content"
    files = {"file": ("demo_check.mp4", io.BytesIO(video_bytes), "video/mp4")}
    resp = client.post("/api/scans", files=files)
    assert resp.status_code == 200
    data = resp.json()

    # Assert is_demo is True
    assert data["is_demo"] is True

    # Assert all output file fields are null
    outputs = data["outputs"]
    assert outputs["magnified_video"] is None
    assert outputs["spectrum_plot"] is None
    assert outputs["displacement_plot"] is None

    # Assert no placeholder output files were created on disk
    scan_dir = storage.get_scan_dir(data["scan_id"])
    assert not (scan_dir / "magnified.mp4").exists()
    assert not (scan_dir / "spectrum.png").exists()
    assert not (scan_dir / "displacement.png").exists()


def test_get_scan_result():
    """Verify GET /api/scans/{id} retrieves the complete scan result JSON."""
    video_bytes = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41demo_content"
    files = {"file": ("bridge_scan.mp4", io.BytesIO(video_bytes), "video/mp4")}
    post_resp = client.post("/api/scans", files=files, data={"lat": "13.0827", "lon": "80.2707"})
    assert post_resp.status_code == 200
    scan_id = post_resp.json()["scan_id"]

    # Fetch scan by ID
    get_resp = client.get(f"/api/scans/{scan_id}")
    assert get_resp.status_code == 200
    scan_data = get_resp.json()
    assert scan_data["scan_id"] == scan_id
    assert scan_data["status"] == "complete"
    assert scan_data["is_demo"] is True
    assert scan_data["outputs"]["magnified_video"] is None
    assert scan_data["outputs"]["spectrum_plot"] is None
    assert scan_data["outputs"]["displacement_plot"] is None
    assert scan_data["measurement"]["dominant_frequency_hz"] == 3.70


def test_get_scan_files():
    """Verify GET /api/scans/{id}/files/{name} serves files with path-traversal protection."""
    video_bytes = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41demo_content"
    files = {"file": ("telemetry_clip.mp4", io.BytesIO(video_bytes), "video/mp4")}
    post_resp = client.post("/api/scans", files=files)
    assert post_resp.status_code == 200
    scan_id = post_resp.json()["scan_id"]

    # 1. Serve uploaded video file
    resp_video = client.get(f"/api/scans/{scan_id}/files/telemetry_clip.mp4")
    assert resp_video.status_code == 200
    assert "video/mp4" in resp_video.headers["content-type"]
    assert len(resp_video.content) > 0

    # 2. Serve output plot once placed in scan directory
    scan_dir = storage.get_scan_dir(scan_id)
    (scan_dir / "spectrum.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    resp_plot = client.get(f"/api/scans/{scan_id}/files/spectrum.png")
    assert resp_plot.status_code == 200
    assert resp_plot.headers["content-type"] == "image/png"
    assert len(resp_plot.content) > 0


def test_bad_extension_rejected():
    """Verify invalid file extensions are rejected with HTTP 400."""
    # Test .txt
    bad_file = {"file": ("malicious.txt", io.BytesIO(b"not a video"), "text/plain")}
    resp = client.post("/api/scans", files=bad_file)
    assert resp.status_code == 400
    assert "extension" in resp.json()["detail"].lower()

    # Test .avi
    bad_avi = {"file": ("unsupported.avi", io.BytesIO(b"RIFF....AVI"), "video/x-msvideo")}
    resp_avi = client.post("/api/scans", files=bad_avi)
    assert resp_avi.status_code == 400
    assert "extension" in resp_avi.json()["detail"].lower()

    # Test .exe
    bad_exe = {"file": ("payload.exe", io.BytesIO(b"MZ..."), "application/x-msdownload")}
    resp_exe = client.post("/api/scans", files=bad_exe)
    assert resp_exe.status_code == 400


def test_oversized_rejected(monkeypatch):
    """Verify files exceeding 200 MB are rejected with HTTP 413."""
    # Temporarily set max size to 1024 bytes to test threshold without large memory allocation
    monkeypatch.setattr(storage, "MAX_UPLOAD_SIZE_BYTES", 1024)

    oversized_data = b"0" * 2048
    files = {"file": ("large_video.mp4", io.BytesIO(oversized_data), "video/mp4")}
    response = client.post("/api/scans", files=files)
    assert response.status_code in (413, 400)
    assert "200 mb" in response.json()["detail"].lower() or "size" in response.json()["detail"].lower()


def test_path_traversal_rejected():
    """Verify directory traversal attempts in file downloads are rejected with HTTP 400."""
    video_bytes = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41demo_content"
    files = {"file": ("safe_video.mp4", io.BytesIO(video_bytes), "video/mp4")}
    post_resp = client.post("/api/scans", files=files)
    assert post_resp.status_code == 200
    scan_id = post_resp.json()["scan_id"]

    # Traversal in filename
    resp_traversal = client.get(f"/api/scans/{scan_id}/files/../../etc/passwd")
    assert resp_traversal.status_code in (400, 404)

    # Encoded traversal
    resp_encoded = client.get(f"/api/scans/{scan_id}/files/%2e%2e%2f%2e%2e%2fsecret.txt")
    assert resp_encoded.status_code in (400, 404)

    # Relative dot-dot
    resp_dotdot = client.get(f"/api/scans/{scan_id}/files/../conftest.py")
    assert resp_dotdot.status_code in (400, 404)


def test_get_nonexistent_scan():
    """Verify querying an unknown scan_id returns HTTP 404."""
    response = client.get("/api/scans/scan_unknown_999999")
    assert response.status_code == 404


def test_get_nonexistent_scan_file():
    """Verify querying an unknown file for a valid scan returns HTTP 404."""
    video_bytes = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41demo_content"
    files = {"file": ("safe_video.mp4", io.BytesIO(video_bytes), "video/mp4")}
    post_resp = client.post("/api/scans", files=files)
    assert post_resp.status_code == 200
    scan_id = post_resp.json()["scan_id"]

    resp = client.get(f"/api/scans/{scan_id}/files/nonexistent_file.png")
    assert resp.status_code == 404


def test_bridges_excludes_demo_scans():
    """Verify GET /api/bridges excludes scans where is_demo is true."""
    video_bytes = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41demo_content"
    files = {"file": ("demo_bridge.mp4", io.BytesIO(video_bytes), "video/mp4")}
    post_resp = client.post(
        "/api/scans",
        files=files,
        data={"lat": "13.0827", "lon": "80.2707", "bridge_name": "Demo Exclusive Bridge"},
    )
    assert post_resp.status_code == 200
    demo_scan_id = post_resp.json()["scan_id"]
    assert post_resp.json()["is_demo"] is True

    # GET /api/bridges must exclude all demo scans
    resp = client.get("/api/bridges")
    assert resp.status_code == 200
    bridges = resp.json()
    for b in bridges:
        assert b["bridge"] != "Demo Exclusive Bridge"
        assert demo_scan_id not in b["scans"]


def test_bridges_grouped_endpoint():
    """Verify GET /api/bridges returns non-demo scans grouped with verdict, timestamp, and frequency history."""
    from setu.api import _sync_scan_record_to_db

    # Non-demo scan 1 (Named bridge)
    scan1_data = {
        "schema_version": "1.0",
        "is_demo": False,
        "scan_id": "scan_prod_01",
        "created_at": "2026-11-03T19:00:00+05:30",
        "status": "complete",
        "verdict": {"level": "usable", "reasons": []},
        "location": {"lat": 13.0827, "lon": 80.2707},
        "measurement": {"dominant_frequency_hz": 3.70, "peaks": []},
        "interpretation": {"bridge": "Pamban Bridge", "assessment": {"severity": "none"}},
    }
    storage.save_scan_result("scan_prod_01", scan1_data)
    _sync_scan_record_to_db("scan_prod_01", scan1_data)

    # Non-demo scan 2 (Unnamed bridge, grouped by rounded location)
    scan2_data = {
        "schema_version": "1.0",
        "is_demo": False,
        "scan_id": "scan_prod_02",
        "created_at": "2026-11-03T19:05:00+05:30",
        "status": "complete",
        "verdict": {"level": "usable", "reasons": []},
        "location": {"lat": 18.9221, "lon": 72.8346},
        "measurement": {"dominant_frequency_hz": 2.45, "peaks": []},
        "interpretation": {"bridge": None, "assessment": {"severity": "low"}},
    }
    storage.save_scan_result("scan_prod_02", scan2_data)
    _sync_scan_record_to_db("scan_prod_02", scan2_data)

    resp = client.get("/api/bridges")
    assert resp.status_code == 200
    bridges = resp.json()
    assert isinstance(bridges, list)
    assert len(bridges) >= 2

    # Verify bridge summary structure and that non-demo scans are present
    pamban_found = False
    for b in bridges:
        assert "bridge" in b
        assert "location" in b
        assert "latest_verdict" in b
        assert "last_scanned_time" in b
        assert "frequency_history" in b
        assert isinstance(b["frequency_history"], list)
        assert len(b["frequency_history"]) >= 1
        assert "frequency_hz" in b["frequency_history"][0]
        if b["bridge"] == "Pamban Bridge":
            pamban_found = True
            assert "scan_prod_01" in b["scans"]
    assert pamban_found


def test_triage_queue_endpoint():
    """Verify GET /api/triage returns a list."""
    response = client.get("/api/triage")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_scan_pipeline_non_demo_mode(monkeypatch):
    """Verify non-demo mode queues scan and handles background task execution gracefully."""
    monkeypatch.setenv("SETU_DEMO_MODE", "false")
    video_bytes = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41demo_content"
    files = {"file": ("nondemo_clip.mp4", io.BytesIO(video_bytes), "video/mp4")}
    data = {"lat": "13.0827", "lon": "80.2707"}

    response = client.post("/api/scans", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["schema_version"] == "1.0"
    scan_id = res["scan_id"]

    # In TestClient, background tasks execute before returning
    get_resp = client.get(f"/api/scans/{scan_id}")
    assert get_resp.status_code == 200
    updated_res = get_resp.json()
    assert updated_res["status"] in ("queued", "running", "complete", "error")
    assert updated_res.get("is_demo") is False
    if updated_res["verdict"]["level"] == "failed":
        assert updated_res["measurement"]["dominant_frequency_hz"] is None
        assert updated_res["measurement"]["frequency_ci_hz"] is None
        assert updated_res["measurement"]["peaks"] == []

