"""Tests for SETU API endpoints in Demo Mode."""

import io
import os
import wave
import json
from pathlib import Path
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
    assert result["schema_version"] == "1.4"
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
    assert result["outputs"]["sound"] is None
    assert result["outputs"]["sound"] is None


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
    assert outputs["sound"] is None

    # Assert no placeholder output files were created on disk
    scan_dir = storage.get_scan_dir(data["scan_id"])
    assert not (scan_dir / "magnified.mp4").exists()
    assert not (scan_dir / "spectrum.png").exists()
    assert not (scan_dir / "displacement.png").exists()


def test_device_motion_is_validated_and_saved(tmp_path):
    samples = [
        {"t": 0.0, "ax": 0.1, "ay": 0.2, "az": 9.8},
        {"t": 0.02, "ax": 0.11, "ay": 0.19, "az": 9.81},
    ]
    response = client.post(
        "/api/scans",
        files={"file": ("device.mp4", io.BytesIO(b"video"), "video/mp4")},
        data={"target_prompt": "pier", "device_motion": json.dumps(samples)},
    )
    assert response.status_code == 200
    saved = storage.get_scan_dir(response.json()["scan_id"]) / "device_motion.json"
    assert json.loads(saved.read_text(encoding="utf-8")) == samples


def test_device_motion_limits_and_malformed_data(monkeypatch):
    video = {"file": ("device_limit.mp4", io.BytesIO(b"video"), "video/mp4")}
    too_many = [{"t": i * 0.001, "ax": 0, "ay": 0, "az": 0} for i in range(20_001)]
    response = client.post("/api/scans", files=video, data={"target_prompt": "pier", "device_motion": json.dumps(too_many)})
    assert response.status_code == 413
    assert "20,000" in response.json()["detail"]

    malformed = client.post(
        "/api/scans",
        files={"file": ("device_bad.mp4", io.BytesIO(b"video"), "video/mp4")},
        data={"target_prompt": "pier", "device_motion": "not-json"},
    )
    assert malformed.status_code == 400
    assert "JSON list" in malformed.json()["detail"]

    long_samples = [
        {"t": 0.0, "ax": 0, "ay": 0, "az": 0},
        {"t": 61.0, "ax": 0, "ay": 0, "az": 0},
    ]
    over_duration = client.post(
        "/api/scans",
        files={"file": ("device_long.mp4", io.BytesIO(b"video"), "video/mp4")},
        data={"target_prompt": "pier", "device_motion": json.dumps(long_samples)},
    )
    assert over_duration.status_code == 413
    assert "60 seconds" in over_duration.json()["detail"]


def test_scan_sound_is_served_only_when_pipeline_output_exists():
    scan_id = "scan_sound_api_test"
    result = {
        "schema_version": "1.4", "scan_id": scan_id, "is_demo": False,
        "status": "complete", "video": {"fps": 100.0},
        "measurement": {"dominant_frequency_hz": 4.0,
                         "peaks": [{"frequency_hz": 4.0, "relative_power": 1.0, "snr_db": 20.0}]},
        "outputs": {"sound": "bridge_sound.wav"},
        "setup": {"source": "user", "structure_type": None, "recording_issues": [], "advice": None},
    }
    storage.save_scan_result(scan_id, result)
    (storage.get_scan_dir(scan_id) / "bridge_sound.wav").write_bytes(_wav_bytes())

    loaded = client.get(f"/api/scans/{scan_id}")
    assert loaded.json()["sound_available"] is True
    assert loaded.json()["sound_url"] == f"/api/scans/{scan_id}/sound.wav"
    assert loaded.json()["sonify_url"] is None
    assert loaded.json()["sound_summary"]["peak_count"] == 1

    download = client.get(f"/api/scans/{scan_id}/sound.wav")
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("audio/wav")
    file_route = client.get(f"/api/scans/{scan_id}/files/bridge_sound.wav")
    assert file_route.status_code == 200
    assert file_route.headers["content-type"].startswith("audio/wav")

    empty_id = "scan_sound_missing"
    storage.save_scan_result(empty_id, {"scan_id": empty_id, "outputs": {"sound": None}})
    absent = client.get(f"/api/scans/{empty_id}/sound.wav")
    assert absent.status_code == 404
    assert client.post(f"/api/scans/{scan_id}/sound").status_code == 405
    assert client.get(f"/api/scans/{scan_id}/sonify").status_code == 404


def test_analysis_completes_before_magnification_and_tracks_status(monkeypatch):
    import setu.scan as scan_module
    from setu import api

    scan_id = "scan_magnification_flow_test"
    scan_dir = storage.get_scan_dir(scan_id)
    (scan_dir / "clip.mp4").write_bytes(b"source video")
    storage.save_scan_result(scan_id, {
        "schema_version": "1.4", "scan_id": scan_id, "status": "queued",
        "is_demo": False, "video": {"filename": "clip.mp4", "fps": 30.0},
        "setup": {"source": "user", "structure_type": None, "recording_issues": [], "advice": None},
        "outputs": {"magnified_video": None},
    })
    pipeline_calls = []
    def fake_analyze_video(**kwargs):
        pipeline_calls.append(kwargs)
        return {
            "schema_version": "1.4", "status": "complete", "is_demo": False,
            "verdict": {"level": "usable", "reasons": []},
            "video": {"filename": "clip.mp4", "fps": 30.0},
            "segmentation": {"target_bbox": [10, 20, 30, 40]},
            "measurement": {"dominant_frequency_hz": 3.2, "peaks": [], "instances": []},
            "outputs": {"magnified_video": None},
        }
    monkeypatch.setattr(scan_module, "analyze_video", fake_analyze_video)
    queued_jobs = []
    monkeypatch.setattr(api, "_start_background_task", lambda target, *args: queued_jobs.append((target, args)))

    api._execute_analysis_task(
        scan_id=scan_id, video_path=str(scan_dir / "clip.mp4"), lat=0.0, lon=0.0,
        target_prompt="pier", background_prompt="building", out_dir=str(scan_dir),
    )

    complete_result = storage.load_scan_result(scan_id)
    assert pipeline_calls[0]["magnify"] is False
    assert complete_result["status"] == "complete"
    assert complete_result["magnification_status"] == "pending"
    assert complete_result["outputs"]["magnified_video"] is None
    assert queued_jobs == [(api._execute_magnification_task, (scan_id,))]

    import setu.evm as evm
    def fake_magnify(**kwargs):
        assert storage.load_scan_result(scan_id)["magnification_status"] == "running"
        Path(kwargs["output_video_path"]).write_bytes(b"magnified")
        return kwargs["output_video_path"]
    monkeypatch.setattr(evm, "magnify_video", fake_magnify)
    queued_jobs[0][0](*queued_jobs[0][1])
    done_result = storage.load_scan_result(scan_id)
    assert done_result["status"] == "complete"
    assert done_result["magnification_status"] == "done"
    assert done_result["outputs"]["magnified_video"] == "magnified.mp4"

    missing_bbox_id = "scan_magnification_missing_bbox_test"
    missing_bbox_dir = storage.get_scan_dir(missing_bbox_id)
    (missing_bbox_dir / "clip.mp4").write_bytes(b"source video")
    storage.save_scan_result(missing_bbox_id, {
        "schema_version": "1.4", "scan_id": missing_bbox_id, "status": "complete",
        "verdict": {"level": "usable", "reasons": []},
        "video": {"filename": "clip.mp4", "fps": 30.0},
        "measurement": {"dominant_frequency_hz": 3.0},
        "segmentation": {"target_bbox": None},
        "outputs": {"magnified_video": None},
    })
    magnify_calls = []
    monkeypatch.setattr(evm, "magnify_video", lambda **kwargs: magnify_calls.append(kwargs))
    api._execute_magnification_task(missing_bbox_id)
    no_box_result = storage.load_scan_result(missing_bbox_id)
    assert no_box_result["magnification_status"] == "skipped"
    assert no_box_result["magnification_reason"] == "No target region recorded"
    assert magnify_calls == []


def test_weak_scan_skips_magnification_and_magnification_errors_fail_status(monkeypatch):
    import setu.scan as scan_module
    import setu.evm as evm
    from setu import api

    scan_id = "scan_magnification_weak_test"
    scan_dir = storage.get_scan_dir(scan_id)
    (scan_dir / "clip.mp4").write_bytes(b"source video")
    storage.save_scan_result(scan_id, {
        "schema_version": "1.4", "scan_id": scan_id, "status": "queued",
        "is_demo": False, "video": {"filename": "clip.mp4", "fps": 30.0},
        "outputs": {"magnified_video": None},
    })
    monkeypatch.setattr(scan_module, "analyze_video", lambda **_kwargs: {
        "schema_version": "1.4", "status": "complete", "is_demo": False,
        "verdict": {"level": "weak", "reasons": ["low quality"]},
        "video": {"filename": "clip.mp4", "fps": 30.0},
        "measurement": {"dominant_frequency_hz": 3.0, "peaks": [], "instances": []},
        "outputs": {"magnified_video": None},
    })
    jobs = []
    monkeypatch.setattr(api, "_start_background_task", lambda *args: jobs.append(args))
    api._execute_analysis_task(
        scan_id=scan_id, video_path=str(scan_dir / "clip.mp4"), lat=0.0, lon=0.0,
        target_prompt="pier", background_prompt="building", out_dir=str(scan_dir),
    )
    weak_result = storage.load_scan_result(scan_id)
    assert weak_result["status"] == "complete"
    assert weak_result["magnification_status"] == "skipped"
    assert jobs == []

    failed_id = "scan_magnification_error_test"
    failed_dir = storage.get_scan_dir(failed_id)
    (failed_dir / "clip.mp4").write_bytes(b"source video")
    storage.save_scan_result(failed_id, {
        "schema_version": "1.4", "scan_id": failed_id, "status": "complete",
        "verdict": {"level": "usable", "reasons": []},
        "video": {"filename": "clip.mp4", "fps": 30.0},
        "segmentation": {"target_bbox": [10, 10, 40, 40]},
        "measurement": {"dominant_frequency_hz": 3.0},
        "outputs": {"magnified_video": None},
    })
    monkeypatch.setattr(evm, "magnify_video", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("render failed")))
    api._execute_magnification_task(failed_id)
    failed_result = storage.load_scan_result(failed_id)
    assert failed_result["status"] == "complete"
    assert failed_result["magnification_status"] == "failed"
    assert failed_result["outputs"]["magnified_video"] is None


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
    assert response.json()["schema_version"] == "1.4"
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


def _mock_director(monkeypatch, result):
    import sys
    import types
    module = types.ModuleType("setu.director")
    calls = []
    def direct_scan(video_path, language):
        calls.append((video_path, language))
        return result
    module.direct_scan = direct_scan
    monkeypatch.setitem(sys.modules, "setu.director", module)
    return calls


def test_empty_prompt_calls_director_and_uses_its_setup(monkeypatch):
    from setu import api
    monkeypatch.setenv("SETU_DEMO_MODE", "true")
    monkeypatch.setattr(api.settings, "META_API_KEY", "test-key")
    calls = _mock_director(monkeypatch, {
        "structure_visible": True, "structure_type": "bridge pier",
        "target_prompt": "concrete pier", "background_prompt": "distant building",
        "is_cable": True, "recording_issues": [], "advice": "Hold the phone steady.",
    })
    response = client.post(
        "/api/scans",
        files={"file": ("director.mp4", io.BytesIO(b"video"), "video/mp4")},
        data={"target_prompt": "", "background_prompt": "", "language": "hi", "is_cable": "false"},
    )
    assert response.status_code == 200
    result = response.json()
    assert len(calls) == 1 and calls[0][1] == "hi"
    assert result["segmentation"]["target_prompt"] == "concrete pier"
    assert result["segmentation"]["background_prompt"] == "distant building"
    assert result["setup"]["source"] == "muse_spark"
    assert result["setup"]["structure_type"] == "bridge pier"
    assert result["cable"]["is_cable"] is False


def test_user_prompt_skips_director(monkeypatch):
    import sys
    import types
    from setu import api
    monkeypatch.setenv("SETU_DEMO_MODE", "true")
    module = types.ModuleType("setu.director")
    module.direct_scan = lambda *_args: pytest.fail("director should not run for user prompts")
    monkeypatch.setitem(sys.modules, "setu.director", module)
    response = client.post(
        "/api/scans",
        files={"file": ("user.mp4", io.BytesIO(b"video"), "video/mp4")},
        data={"target_prompt": "steel cable", "background_prompt": "sky"},
    )
    assert response.status_code == 200
    assert response.json()["setup"]["source"] == "user"
    assert response.json()["segmentation"]["target_prompt"] == "steel cable"


def test_director_rejects_invisible_structure_without_pipeline(monkeypatch):
    from setu import api
    monkeypatch.setenv("SETU_DEMO_MODE", "false")
    monkeypatch.setattr(api.settings, "META_API_KEY", "test-key")
    _mock_director(monkeypatch, {
        "structure_visible": False, "structure_type": None,
        "target_prompt": "", "background_prompt": "", "is_cable": False,
        "recording_issues": ["structure not visible"], "advice": "Move closer.",
    })
    pipeline_calls = []
    monkeypatch.setattr(api, "_execute_analysis_task", lambda **kwargs: pipeline_calls.append(kwargs))
    response = client.post(
        "/api/scans",
        files={"file": ("invisible.mp4", io.BytesIO(b"video"), "video/mp4")},
        data={"target_prompt": "", "background_prompt": ""},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["verdict"]["level"] == "failed"
    assert result["verdict"]["reasons"] == ["structure not visible", "Move closer."]
    assert pipeline_calls == []


def test_director_none_uses_defaults_and_adds_reason(monkeypatch):
    from setu import api
    monkeypatch.setenv("SETU_DEMO_MODE", "true")
    monkeypatch.setattr(api.settings, "META_API_KEY", "test-key")
    _mock_director(monkeypatch, None)
    response = client.post(
        "/api/scans",
        files={"file": ("defaults.mp4", io.BytesIO(b"video"), "video/mp4")},
        data={"target_prompt": "", "background_prompt": ""},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["segmentation"]["target_prompt"] == "concrete pillar"
    assert result["segmentation"]["background_prompt"] == "building"
    assert result["setup"]["source"] == "default"
    assert result["setup"]["advice"] == "Prompts chosen automatically by default; AI setup unavailable."
    assert "Prompts chosen automatically by default; AI setup unavailable." in result["verdict"]["reasons"]


def test_missing_meta_key_skips_director_and_uses_defaults(monkeypatch):
    import sys
    import types
    from setu import api
    monkeypatch.setenv("SETU_DEMO_MODE", "true")
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setattr(api.settings, "META_API_KEY", None)
    module = types.ModuleType("setu.director")
    module.direct_scan = lambda *_args: pytest.fail("director must not call Meta without a key")
    monkeypatch.setitem(sys.modules, "setu.director", module)
    response = client.post(
        "/api/scans",
        files={"file": ("no_key.mp4", io.BytesIO(b"video"), "video/mp4")},
        data={"target_prompt": "", "background_prompt": ""},
    )
    assert response.status_code == 200
    assert response.json()["setup"]["source"] == "default"
    assert response.json()["setup"]["advice"] == "Prompts chosen automatically by default; AI setup unavailable."


def test_public_mode_skips_director_even_when_key_is_configured(monkeypatch):
    import sys
    import types
    from setu import api
    api._upload_times.clear()
    monkeypatch.setattr(api.settings, "META_API_KEY", "test-key")
    module = types.ModuleType("setu.director")
    module.direct_scan = lambda *_args: pytest.fail("public mode must not invoke live setup")
    monkeypatch.setitem(sys.modules, "setu.director", module)
    response = client.post(
        "/api/scans",
        files=_public_video(monkeypatch, name="public_setup.mp4"),
        data={"target_prompt": "", "background_prompt": ""},
    )
    assert response.status_code == 200
    assert response.json()["setup"]["source"] == "default"


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


def test_public_cache_miss_never_calls_network_and_labels_fallback(monkeypatch, tmp_path):
    import numpy as np
    import requests
    from setu import api
    from setu.assess import get_assessor
    from setu.meta_client import get_meta_client
    from setu.segment import MetaSAM31Segmenter

    monkeypatch.setenv("SETU_PUBLIC_DEMO", "true")
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setattr(api.settings, "META_API_KEY", None)
    client = get_meta_client()
    monkeypatch.setattr(client, "cache_dir", str(tmp_path / "empty-cache"))
    monkeypatch.setattr(client, "api_key", None)
    monkeypatch.setattr(client, "mode", "cached")
    network_calls = []
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: network_calls.append(args))

    api.configure_public_meta_mode()
    segmented = MetaSAM31Segmenter(meta_client=client).segment(
        np.zeros((96, 128, 3), dtype=np.uint8),
        target_prompt="concrete pillar",
        background_prompt="building",
    )
    assessment = get_assessor().assess()

    assert network_calls == []
    assert segmented.method == "fallback"
    assert segmented.target_source == "fallback"
    assert segmented.background_source == "fallback"
    assert assessment.assessor == "local_rules"


def test_public_upload_cleanup_removes_only_old_original(tmp_path, monkeypatch):
    from setu import api
    import time
    scan_dir = tmp_path / "scan"; scan_dir.mkdir()
    (scan_dir / "result.json").write_text(json.dumps({"video": {"filename": "source.mp4"}}), encoding="utf-8")
    upload = scan_dir / "source.mp4"; upload.write_bytes(b"video")
    output = scan_dir / "magnified.mp4"; output.write_bytes(b"output")
    old = time.time() - 90000
    os.utime(upload, (old, old))
    monkeypatch.setattr(api, "SCANS_DIR", scan_dir)
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
    assert res["schema_version"] == "1.4"
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

