"""Tests for SETU API endpoints in Demo Mode."""

import io
import os
import wave
import json
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
    assert result["schema_version"] == "1.2"
    assert result["scan_id"].startswith("scan_")
    assert result["status"] == "complete"
    assert result["is_demo"] is True
    assert result["verdict"]["level"] == "usable"
    assert result["cable"]["is_cable"] is False
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
    assert result["outputs"]["vibration_map"] is None


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
    assert outputs["vibration_map"] is None

    # Assert no placeholder output files were created on disk
    scan_dir = storage.get_scan_dir(data["scan_id"])
    assert not (scan_dir / "magnified.mp4").exists()
    assert not (scan_dir / "spectrum.png").exists()
    assert not (scan_dir / "displacement.png").exists()


def _wav_bytes(seconds=1):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00" * 24000 * seconds)
    return stream.getvalue()


def _demo_scan_id():
    video = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41demo_content"
    response = client.post("/api/scans", files={"file": ("note_scan.mp4", io.BytesIO(video), "video/mp4")})
    assert response.status_code == 200
    return response.json()["scan_id"]


def test_field_note_endpoint_transcribes_tags_and_deletes_audio():
    scan_id = _demo_scan_id()
    response = client.post(f"/api/scans/{scan_id}/field-notes", files={"audio": ("note.wav", _wav_bytes(), "audio/wav")}, data={"language": "en"})
    assert response.status_code == 201, response.text
    data = response.json()
    assert data["transcript"] == "The concrete has a crack and rust near the water."
    assert data["english"] is None
    assert {tag["tag"] for tag in data["tags"]} == {"crack", "rust"}
    assert all(tag["evidence"] in data["transcript"] for tag in data["tags"])
    scan_dir = storage.get_scan_dir(scan_id)
    assert not list(scan_dir.glob("field_note_*"))
    saved = client.get(f"/api/scans/{scan_id}").json()
    assert len(saved["field_notes"]) == 1


def test_field_note_enforces_size_and_duration_limits():
    scan_id = _demo_scan_id()
    too_large = client.post(f"/api/scans/{scan_id}/field-notes", files={"audio": ("large.wav", io.BytesIO(b"x" * (10 * 1024 * 1024 + 1)), "audio/wav")}, data={"language": "en"})
    assert too_large.status_code == 413
    too_long = client.post(f"/api/scans/{scan_id}/field-notes", files={"audio": ("long.wav", _wav_bytes(121), "audio/wav")}, data={"language": "en"})
    assert too_long.status_code == 413


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
    pamban_scan_group = None
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
            pamban_scan_group = b
    if pamban_scan_group is None:
        pamban_scan_group = next(b for b in bridges if "scan_prod_01" in b["scans"])
    assert "scan_prod_01" in pamban_scan_group["scans"]
    assert pamban_scan_group["bridge_key"] == "13.083,80.271"


def test_triage_queue_endpoint():
    """Verify GET /api/triage returns a list."""
    response = client.get("/api/triage")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_bridges_include_stable_key_and_reports_exclude_demo_rows():
    from setu.api import _sync_scan_record_to_db

    scan = {
        "schema_version": "1.2", "is_demo": False,
        "scan_id": "scan_report_api_prod", "created_at": "2026-11-03T20:00:00+05:30",
        "status": "complete", "verdict": {"level": "usable", "reasons": []},
        "location": {"lat": 27.12345, "lon": 78.54321},
        "segmentation": {"target_prompt": "stay cable", "target_source": "sam", "background_source": "sam"},
        "measurement": {"dominant_frequency_hz": 3.7, "frequency_ci_hz": [3.6, 3.8]},
        "interpretation": {"bridge": {"name": "Report Cable Bridge", "osm_id": "way/765"}},
        "cable": {"is_cable": True, "tension_change_pct": -9.75,
                  "tension_change_range_pct": [-11.0, -8.5]},
        "field_notes": [{"tags": [{"tag": "rust", "evidence": "rust by the joint"}]}],
    }
    demo = dict(scan, scan_id="scan_report_api_demo", is_demo=True)
    for item in (scan, demo):
        storage.save_scan_result(item["scan_id"], item)
        _sync_scan_record_to_db(item["scan_id"], item)

    bridges = client.get("/api/bridges").json()
    group = next(item for item in bridges if item.get("bridge_key") == "way/765")
    assert "bridge_key" in group

    json_response = client.get("/api/bridges/way%2F765/report.json")
    assert json_response.status_code == 200, json_response.text
    report = json_response.json()
    assert report["bridge"]["osm_id"] == "way/765"
    assert [row["scan_id"] for row in report["scans"]] == ["scan_report_api_prod"]
    assert report["scans"][0]["field_note_tags"][0]["evidence"] == "rust by the joint"

    csv_response = client.get("/api/bridges/way%2F765/report.csv")
    assert csv_response.status_code == 200
    assert "scan_report_api_prod" in csv_response.text
    assert "scan_report_api_demo" not in csv_response.text
    assert "cable_tension_change_low_pct" in csv_response.text


def test_upload_can_explicitly_mark_cable_scan():
    video_bytes = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41demo_content"
    response = client.post(
        "/api/scans",
        files={"file": ("explicit_cable.mp4", io.BytesIO(video_bytes), "video/mp4")},
        data={"target_prompt": "concrete structure", "is_cable": "true"},
    )
    assert response.status_code == 200
    assert response.json()["schema_version"] == "1.2"
    assert response.json()["cable"]["is_cable"] is True


def test_showcase_loading_and_endpoint(tmp_path, monkeypatch):
    from setu import api

    showcase_dir = tmp_path / "showcase"
    scan_dir = showcase_dir / "real-bridge"
    scan_dir.mkdir(parents=True)
    (scan_dir / "result.json").write_text(json.dumps({
        "scan_id": "showcase_real_1", "schema_version": "1.2",
        "is_demo": True, "verdict": {"level": "usable"},
        "outputs": {"spectrum_plot": "spectrum.png"},
    }), encoding="utf-8")
    (scan_dir / "spectrum.png").write_bytes(b"png")
    monkeypatch.setattr(api, "SHOWCASE_DIR", showcase_dir)
    monkeypatch.setattr(api.app.state, "showcase_scans", None, raising=False)
    loaded = api.load_showcase_scans()
    assert loaded["showcase_real_1"]["is_showcase"] is True
    assert loaded["showcase_real_1"]["is_demo"] is False
    response = client.get("/api/showcase")
    assert response.status_code == 200
    assert response.json()[0]["scan_id"] == "showcase_real_1"
    assert client.get("/api/scans/showcase_real_1").json()["is_showcase"] is True
    image = client.get("/api/scans/showcase_real_1/files/spectrum.png")
    assert image.status_code == 200 and image.content == b"png"


def _public_video(monkeypatch, name="public.mp4"):
    from setu import api
    monkeypatch.setenv("SETU_PUBLIC_DEMO", "true")
    monkeypatch.setenv("SETU_DEMO_MODE", "false")
    monkeypatch.setattr(api, "_public_video_duration", lambda _path: 10.0)
    monkeypatch.setattr(api, "_execute_analysis_task", lambda **_kwargs: None)
    return {"file": (name, io.BytesIO(b"video"), "video/mp4")}


def test_public_demo_rate_limit_and_real_pipeline(monkeypatch):
    from setu import api
    api._upload_times.clear()
    responses = [client.post("/api/scans", files=_public_video(monkeypatch)) for _ in range(3)]
    assert all(response.status_code == 200 for response in responses)
    assert all(response.json()["is_demo"] is False for response in responses)
    assert all(response.json()["status"] == "queued" for response in responses)
    assert api.settings.SETU_META_MODE == "cached"
    limited = client.post("/api/scans", files=_public_video(monkeypatch))
    assert limited.status_code == 429


def test_public_demo_rejects_over_60_seconds_and_deletes_video(monkeypatch):
    from setu import api
    monkeypatch.setenv("SETU_PUBLIC_DEMO", "true")
    monkeypatch.setenv("SETU_DEMO_MODE", "false")
    monkeypatch.setattr(api, "_public_video_duration", lambda _path: 60.01)
    api._upload_times.clear()
    response = client.post("/api/scans", files={"file": ("long.mp4", io.BytesIO(b"video"), "video/mp4")})
    assert response.status_code == 413
    assert "60 seconds" in response.json()["detail"]
    assert not list(storage.SCANS_DIR.glob("scan_*/long.mp4"))


def test_public_upload_cleanup_removes_only_old_original(tmp_path, monkeypatch):
    from setu import api
    import time
    scan_dir = tmp_path / "scan"; scan_dir.mkdir()
    (scan_dir / "result.json").write_text(json.dumps({"video": {"filename": "source.mp4"}}), encoding="utf-8")
    upload = scan_dir / "source.mp4"; upload.write_bytes(b"video")
    output = scan_dir / "magnified.mp4"; output.write_bytes(b"output")
    old = time.time() - 90000
    os.utime(upload, (old, old))
    monkeypatch.setattr(api, "SCANS_DIR", tmp_path)
    assert api.cleanup_expired_uploads() == 1
    assert not upload.exists()
    assert output.exists()


def test_scan_pipeline_non_demo_mode(monkeypatch):
    """Verify non-demo mode queues scan and handles background task execution gracefully."""
    monkeypatch.setenv("SETU_DEMO_MODE", "false")
    video_bytes = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41demo_content"
    files = {"file": ("nondemo_clip.mp4", io.BytesIO(video_bytes), "video/mp4")}
    data = {"lat": "13.0827", "lon": "80.2707"}

    response = client.post("/api/scans", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["schema_version"] == "1.2"
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

