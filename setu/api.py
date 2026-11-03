"""
FastAPI application and REST endpoints for Project SETU.

Serves:
- POST /api/scans: Video upload, validation, background vibration analysis
- GET /api/scans/{id}: Shared result format telemetry
- GET /api/scans/{id}/files/{name}: Secure file server (plots, videos) with path-traversal protection
- GET /api/bridges: Scans grouped by bridge (or rounded GPS coordinates) with verdict and frequency history
- GET /health: Service status
- Demo Mode: Controlled by SETU_DEMO_MODE=true
"""

from contextlib import asynccontextmanager
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import mimetypes
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import uuid

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi import Request
from sqlmodel import Session, select

from setu.config import settings
from setu.database import create_db_and_tables, engine, get_session
from setu.models import (
    AnalysisResult,
    Bridge,
    ScanRecord,
    TriageItem,
    VideoUpload,
)
from setu.storage import (
    SCANS_DIR,
    get_scan_dir,
    get_scan_file_path,
    load_scan_result,
    save_scan_result,
    save_uploaded_video,
)
from setu.voice import transcribe_audio
from setu.field_notes import extract_field_note
from setu.cable import apply_cable_comparison, is_cable_scan
from setu.report import bridge_details, bridge_key_for_scan, build_bridge_report, report_to_csv

logger = logging.getLogger("setu.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize database tables and storage folders upon startup."""
    os.makedirs(settings.STORAGE_DIR, exist_ok=True)
    SCANS_DIR.mkdir(parents=True, exist_ok=True)
    create_db_and_tables()
    app.state.showcase_scans = load_showcase_scans()
    if is_public_demo_mode():
        configure_public_meta_mode()
        cleanup_expired_uploads()
        cleanup_task = asyncio.create_task(_public_cleanup_loop())
    else:
        cleanup_task = None
    try:
        yield
    finally:
        if cleanup_task is not None:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass


async def _public_cleanup_loop() -> None:
    while True:
        await asyncio.sleep(3600)
        cleanup_expired_uploads()


app = FastAPI(
    title="Project SETU API",
    description="Bridge Vibration Telemetry & Engineer Triage Screening API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS setup: allow localhost:5173 and configured origins
origins = set(settings.cors_origin_list)
origins.add("http://localhost:5173")
origins.add("http://127.0.0.1:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FIELD_NOTE_BYTES = 10 * 1024 * 1024
MAX_FIELD_NOTE_SECONDS = 120
PUBLIC_MAX_VIDEO_SECONDS = 60
PUBLIC_UPLOADS_PER_HOUR = 3
PUBLIC_UPLOAD_WINDOW_SECONDS = 3600
SHOWCASE_DIR = Path("demo_scans")
_upload_lock = threading.Lock()
_upload_times: Dict[str, List[float]] = {}


def is_public_demo_mode() -> bool:
    return os.environ.get("SETU_PUBLIC_DEMO", "").strip().lower() in ("true", "1", "yes")


def load_showcase_scans(root: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Load bundled read-only showcase results from demo_scans/<folder>/result.json."""
    scans: Dict[str, Dict[str, Any]] = {}
    root = root or SHOWCASE_DIR
    if not root.is_dir():
        return scans
    for folder in sorted(root.iterdir()):
        result_path = folder / "result.json"
        if not folder.is_dir() or not result_path.is_file():
            continue
        try:
            with result_path.open(encoding="utf-8") as source:
                result = json.load(source)
            scan_id = str(result.get("scan_id") or folder.name)
            result["scan_id"] = scan_id
            result["is_showcase"] = True
            result["is_demo"] = False
            result.setdefault("field_notes", [])
            scans[scan_id] = result
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Could not load showcase scan %s: %s", folder, exc)
    return scans


def _reserve_public_upload(ip: str) -> None:
    now = time.monotonic()
    with _upload_lock:
        recent = [t for t in _upload_times.get(ip, []) if now - t < PUBLIC_UPLOAD_WINDOW_SECONDS]
        if len(recent) >= PUBLIC_UPLOADS_PER_HOUR:
            _upload_times[ip] = recent
            raise HTTPException(status_code=429, detail="Upload limit reached (3 per hour).")
        recent.append(now)
        _upload_times[ip] = recent


def _public_video_duration(path: Path) -> float:
    import av
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise ValueError("No video stream")
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return float(container.duration / av.time_base)
    raise ValueError("Video duration unavailable")


def cleanup_expired_uploads(now: Optional[float] = None) -> int:
    """Remove original upload files older than 24 hours while retaining analysis outputs."""
    cutoff = (time.time() if now is None else now) - 86400
    deleted = 0
    if not SCANS_DIR.is_dir():
        return deleted
    for folder in SCANS_DIR.iterdir():
        result = load_scan_result(folder.name) if folder.is_dir() else None
        filename = (result or {}).get("video", {}).get("filename")
        if not filename or Path(filename).name != filename:
            continue
        upload = folder / filename
        try:
            if upload.is_file() and upload.stat().st_mtime <= cutoff:
                upload.unlink()
                deleted += 1
        except OSError as exc:
            logger.warning("Could not remove expired upload %s: %s", upload, exc)
    return deleted


def configure_public_meta_mode() -> None:
    """Set cached mode and prevent cache misses from reaching Meta over the network."""
    settings.SETU_META_MODE = "cached"
    os.environ["SETU_META_MODE"] = "cached"
    import requests
    if getattr(requests.post, "_setu_public_guard", False):
        return
    original_post = requests.post
    def guarded_post(url: Any, *args: Any, **kwargs: Any) -> Any:
        if is_public_demo_mode() and str(url).startswith(settings.META_API_BASE_URL):
            raise requests.ConnectionError("Meta live calls are disabled in public demo mode")
        return original_post(url, *args, **kwargs)
    guarded_post._setu_public_guard = True  # type: ignore[attr-defined]
    requests.post = guarded_post


def generate_scan_id() -> str:
    """Generate canonical scan ID matching schema format: scan_YYYYMMDD_hex."""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    unique_suffix = uuid.uuid4().hex[:6]
    return f"scan_{date_str}_{unique_suffix}"


def is_demo_mode() -> bool:
    """Check whether demo mode is enabled via environment variable."""
    return os.environ.get("SETU_DEMO_MODE", "").strip().lower() in ("true", "1", "yes")


def _sync_scan_record_to_db(scan_id: str, scan_data: Dict[str, Any]) -> None:
    """Persist or update scan metadata in SQLite database."""
    try:
        with Session(engine) as session:
            record = session.get(ScanRecord, scan_id)
            loc = scan_data.get("location") or {}
            lat = float(loc.get("lat", 0.0))
            lon = float(loc.get("lon", 0.0))
            raw_bridge = (
                scan_data.get("interpretation", {}).get("bridge")
                or scan_data.get("bridge_name")
            )
            if isinstance(raw_bridge, dict):
                bridge_str = raw_bridge.get("name")
            elif raw_bridge:
                bridge_str = str(raw_bridge)
            else:
                bridge_str = None

            verdict_obj = scan_data.get("verdict") or {}
            verdict = verdict_obj.get("level")
            meas_obj = scan_data.get("measurement") or {}
            dominant_hz = meas_obj.get("dominant_frequency_hz")
            status_val = scan_data.get("status", "complete")

            if not record:
                record = ScanRecord(
                    id=scan_id,
                    status=status_val,
                    latitude=lat,
                    longitude=lon,
                    bridge_name=bridge_str,
                    verdict_level=verdict,
                    dominant_frequency_hz=dominant_hz,
                    result_json=json.dumps(scan_data),
                    created_at=datetime.now(timezone.utc),
                )
                session.add(record)
            else:
                record.status = status_val
                record.latitude = lat
                record.longitude = lon
                record.bridge_name = bridge_str
                record.verdict_level = verdict
                record.dominant_frequency_hz = dominant_hz
                record.result_json = json.dumps(scan_data)
                session.add(record)
            session.commit()
    except Exception as exc:
        logger.warning("Could not sync scan %s to database: %s", scan_id, exc)


def _collect_all_scan_data(session: Optional[Session] = None) -> List[Dict[str, Any]]:
    """Load scan results from the database and scan directories, deduplicated by ID."""
    owns_session = session is None
    db_session = Session(engine) if owns_session else session
    scans_by_id: Dict[str, Dict[str, Any]] = {}
    try:
        records = db_session.exec(select(ScanRecord)).all() if db_session is not None else []
        for record in records:
            if record.result_json:
                try:
                    scans_by_id[record.id] = json.loads(record.result_json)
                    continue
                except (TypeError, json.JSONDecodeError):
                    pass
            disk = load_scan_result(record.id)
            if disk:
                scans_by_id[record.id] = disk
    finally:
        if owns_session and db_session is not None:
            db_session.close()
    if SCANS_DIR.is_dir():
        for item in SCANS_DIR.iterdir():
            if item.is_dir() and item.name not in scans_by_id:
                disk = load_scan_result(item.name)
                if disk:
                    scans_by_id[item.name] = disk
    return list(scans_by_id.values())


def _load_demo_fixture() -> Dict[str, Any]:
    """Load the demo result fixture JSON from tests/fixtures/scan_result_example.json."""
    fixture_path = Path("tests") / "fixtures" / "scan_result_example.json"
    if fixture_path.is_file():
        try:
            with open(fixture_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to parse fixture file %s: %s", fixture_path, exc)

    # Exact fallback example from specification if file cannot be read
    return {
        "schema_version": "1.2",
        "scan_id": "scan_20261103_a1b2c3",
        "created_at": "2026-11-03T19:30:00+05:30",
        "status": "complete",
        "verdict": {"level": "usable", "reasons": []},
        "location": {"lat": 13.0827, "lon": 80.2707},
        "video": {
            "filename": "clip.mp4",
            "fps": 60.0,
            "duration_s": 30.0,
            "width": 1920,
            "height": 1080,
            "frame_count": 1800,
        },
        "segmentation": {
            "target_prompt": "concrete pillar",
            "background_prompt": "building",
            "target_source": "sam",
            "background_source": "sam",
            "target_persistence": 0.98,
            "background_persistence": 0.99,
            "warnings": [],
        },
        "measurement": {
            "units": "px",
            "scale": None,
            "dominant_frequency_hz": 3.70,
            "frequency_ci_hz": [3.68, 3.72],
            "peaks": [
                {
                    "frequency_hz": 3.70,
                    "relative_power": 0.91,
                    "snr_db": 38.5,
                }
            ],
            "instances": [],
            "background_residual_px": 0.016,
        },
        "interpretation": {
            "bridge": None,
            "bridge_warning": "No mapped bridge within 150 m",
            "quality_gate": {
                "is_structural_element": True,
                "has_static_background": True,
                "appears_stable": True,
                "issues": [],
            },
            "assessment": {
                "severity": "none",
                "reasons": [],
                "expected_range_hz": None,
                "confidence": "low",
                "recommended_action": "",
                "assessor": "local_rules",
            },
        },
        "outputs": {
            "magnified_video": "magnified.mp4",
            "spectrum_plot": "spectrum.png",
            "displacement_plot": "displacement.png",
            "vibration_map": None,
        },
        "field_notes": [],
        "costs_usd": {"sam": 0.0025, "muse_spark": 0.0013, "total": 0.0038},
        "disclaimer": (
            "SETU does not assess safety. Flagged readings are for engineer review."
        ),
    }


def _execute_analysis_task(
    scan_id: str,
    video_path: str,
    lat: float,
    lon: float,
    target_prompt: str,
    background_prompt: str,
    out_dir: str,
    is_cable_override: Optional[bool] = None,
) -> None:
    """Background task to run video vibration analysis pipeline."""
    # 1. Update status to 'running'
    scan_data = load_scan_result(scan_id)
    if scan_data:
        scan_data["status"] = "running"
        scan_data["verdict"] = {"level": "running", "reasons": []}
        save_scan_result(scan_id, scan_data)
        _sync_scan_record_to_db(scan_id, scan_data)

    try:
        # Lazy import so API runs even before setu/scan.py exists
        from setu.scan import analyze_video  # type: ignore

        result_obj = analyze_video(
            path=video_path,
            lat=lat,
            lon=lon,
            target_prompt=target_prompt,
            background_prompt=background_prompt,
            magnify=True,
            out_dir=out_dir,
        )

        if hasattr(result_obj, "to_json"):
            raw = result_obj.to_json()
            completed_result = json.loads(raw) if isinstance(raw, str) else raw
        elif isinstance(result_obj, dict):
            completed_result = result_obj
        else:
            completed_result = dict(result_obj)

        completed_result["scan_id"] = scan_id
        completed_result["status"] = "complete"
        completed_result["is_demo"] = False
        completed_result["schema_version"] = "1.2"
        completed_result.setdefault("field_notes", [])
        completed_result.setdefault("measurement", {}).setdefault("instances", [])
        completed_result.setdefault("outputs", {}).setdefault("vibration_map", None)
        apply_cable_comparison(completed_result, _collect_all_scan_data(), is_cable_override)
        save_scan_result(scan_id, completed_result)
        _sync_scan_record_to_db(scan_id, completed_result)

    except Exception as exc:
        logger.exception("Analysis pipeline failed for scan %s: %s", scan_id, exc)
        curr = load_scan_result(scan_id) or {}
        curr["scan_id"] = scan_id
        curr["status"] = "error"
        curr["is_demo"] = False
        curr["verdict"] = {"level": "failed", "reasons": [str(exc)]}
        if "measurement" not in curr or not isinstance(curr["measurement"], dict):
            curr["measurement"] = {}
        curr["measurement"]["dominant_frequency_hz"] = None
        curr["measurement"]["frequency_ci_hz"] = None
        curr["measurement"]["peaks"] = []
        curr["schema_version"] = "1.2"
        curr.setdefault("cable", {
            "is_cable": is_cable_scan(curr, is_cable_override),
            "tension_change_pct": None,
            "tension_change_range_pct": None,
            "compared_to_scan_id": None,
            "compared_to_date": None,
        })
        save_scan_result(scan_id, curr)
        _sync_scan_record_to_db(scan_id, curr)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def get_health() -> Dict[str, Any]:
    """Health check endpoint confirming API service and configuration status."""
    return {
        "status": "ok",
        "service": "SETU Civil Telemetry API",
        "fallbacks_enabled": settings.USE_LOCAL_FALLBACKS,
        "database": settings.DATABASE_URL,
    }


@app.post("/api/scans/{scan_id}/field-notes", status_code=status.HTTP_201_CREATED)
async def add_field_note(
    scan_id: str,
    audio: UploadFile = File(...),
    language: str = Form(...),
) -> Dict[str, Any]:
    """Transcribe a short driver note, retain only its text, and append it to the scan."""
    if not language.strip():
        raise HTTPException(status_code=400, detail="language is required")
    filename = (audio.filename or "").lower()
    if not filename.endswith((".webm", ".m4a", ".wav", ".mp4")):
        raise HTTPException(status_code=400, detail="Audio must be webm, m4a, or wav.")
    scan = load_scan_result(scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    scan_dir = get_scan_dir(scan_id)
    scan_dir.mkdir(parents=True, exist_ok=True)
    temp_audio = scan_dir / f"field_note_{uuid.uuid4().hex}{Path(filename).suffix}"
    size = 0
    try:
        with open(temp_audio, "wb") as target:
            while True:
                chunk = await audio.read(64 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_FIELD_NOTE_BYTES:
                    raise HTTPException(status_code=413, detail="Audio exceeds 10 MB.")
                target.write(chunk)
        try:
            transcript = transcribe_audio(str(temp_audio), language.strip(), max_duration_s=MAX_FIELD_NOTE_SECONDS)
        except Exception as exc:
            logger.exception("Field note transcription failed for scan %s", scan_id)
            if "exceeds 2 minutes" in str(exc):
                raise HTTPException(status_code=413, detail=str(exc)) from exc
            raise HTTPException(status_code=422, detail=f"Could not transcribe audio: {exc}") from exc
        note_data = extract_field_note(transcript)
        note = {
            "note_id": uuid.uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "language": language.strip(), "transcript": transcript,
            "english": note_data["english"], "tags": note_data["tags"],
        }
        scan.setdefault("field_notes", []).append(note)
        scan["schema_version"] = "1.2"
        scan.setdefault("measurement", {}).setdefault("instances", [])
        scan.setdefault("outputs", {}).setdefault("vibration_map", None)
        save_scan_result(scan_id, scan)
        _sync_scan_record_to_db(scan_id, scan)
        return note
    finally:
        temp_audio.unlink(missing_ok=True)
        await audio.close()


@app.post("/api/scans", status_code=status.HTTP_200_OK)
@app.post("/scans", status_code=status.HTTP_200_OK)
async def create_scan(
    background_tasks: BackgroundTasks,
    request: Request,
    file: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    lat: Optional[float] = Form(None),
    latitude: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    target_prompt: str = Form("concrete pillar"),
    background_prompt: str = Form("building"),
    bridge_name: Optional[str] = Form(None),
    is_cable: Optional[bool] = Form(None),
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """Create a new vibration scan from a video upload."""
    upload_file = file or video
    if not upload_file or not upload_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A video file (.mp4, .mov, .webm) is required.",
        )

    public_mode = is_public_demo_mode()
    if public_mode:
        configure_public_meta_mode()
        _reserve_public_upload(request.client.host if request.client else "unknown")
        cleanup_expired_uploads()

    scan_id = generate_scan_id()

    # Validate extension, size <= 200 MB, sanitize filename, save to data/scans/<scan_id>/
    dest_path, clean_filename = await save_uploaded_video(upload_file, scan_id)

    if public_mode:
        try:
            duration = _public_video_duration(dest_path)
        except Exception as exc:
            dest_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Could not determine video duration.") from exc
        if duration > PUBLIC_MAX_VIDEO_SECONDS:
            dest_path.unlink(missing_ok=True)
            raise HTTPException(status_code=413, detail="Public demo videos must be 60 seconds or shorter.")

    actual_lat = lat if lat is not None else (latitude if latitude is not None else 13.0827)
    actual_lon = lon if lon is not None else (longitude if longitude is not None else 80.2707)

    # 5. Demo Mode Check
    if is_demo_mode() and not public_mode:
        demo_data = _load_demo_fixture()
        demo_data["is_demo"] = True
        demo_data["scan_id"] = scan_id
        demo_data["created_at"] = datetime.now(timezone.utc).astimezone().isoformat()
        if "video" in demo_data:
            demo_data["video"]["filename"] = clean_filename
        if "location" in demo_data:
            demo_data["location"] = {"lat": float(actual_lat), "lon": float(actual_lon)}
        if "segmentation" in demo_data:
            demo_data["segmentation"]["target_prompt"] = target_prompt
            demo_data["segmentation"]["background_prompt"] = background_prompt
        if bridge_name:
            if "interpretation" in demo_data:
                demo_data["interpretation"]["bridge"] = bridge_name

        # In demo mode, outputs must be null and no placeholder output files created
        demo_data["outputs"] = {
            "magnified_video": None,
            "spectrum_plot": None,
            "displacement_plot": None,
            "vibration_map": None,
        }
        demo_data["schema_version"] = "1.2"
        demo_data.setdefault("field_notes", [])
        demo_data.setdefault("measurement", {}).setdefault("instances", [])
        demo_data["cable"] = {
            "is_cable": is_cable_scan({"segmentation": {"target_prompt": target_prompt}}, is_cable),
            "tension_change_pct": None,
            "tension_change_range_pct": None,
            "compared_to_scan_id": None,
            "compared_to_date": None,
        }

        # Save result JSON and persist in DB
        save_scan_result(scan_id, demo_data)
        _sync_scan_record_to_db(scan_id, demo_data)

        return demo_data

    # Production / Full Pipeline Mode
    initial_result = {
        "schema_version": "1.2",
        "is_demo": False,
        "scan_id": scan_id,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "queued",
        "verdict": {"level": "queued", "reasons": []},
        "location": {"lat": float(actual_lat), "lon": float(actual_lon)},
        "video": {
            "filename": clean_filename,
            "fps": 60.0,
            "duration_s": 30.0,
            "width": 1920,
            "height": 1080,
            "frame_count": 1800,
        },
        "segmentation": {
            "target_prompt": target_prompt,
            "background_prompt": background_prompt,
            "target_source": "sam",
            "background_source": "sam",
            "target_persistence": None,
            "background_persistence": None,
            "warnings": [],
        },
        "measurement": {
            "units": "px",
            "scale": None,
            "dominant_frequency_hz": None,
            "frequency_ci_hz": None,
            "peaks": [],
            "instances": [],
            "background_residual_px": None,
        },
        "interpretation": {
            "bridge": bridge_name,
            "bridge_warning": None if bridge_name else "No mapped bridge within 150 m",
            "quality_gate": {
                "is_structural_element": True,
                "has_static_background": True,
                "appears_stable": True,
                "issues": [],
            },
            "assessment": {
                "severity": "none",
                "reasons": [],
                "expected_range_hz": None,
                "confidence": "low",
                "recommended_action": "",
                "assessor": "local_rules",
            },
        },
        "outputs": {
            "magnified_video": "magnified.mp4",
            "spectrum_plot": "spectrum.png",
            "displacement_plot": "displacement.png",
            "vibration_map": None,
        },
        "field_notes": [],
        "cable": {
            "is_cable": is_cable_scan({"segmentation": {"target_prompt": target_prompt}}, is_cable),
            "tension_change_pct": None,
            "tension_change_range_pct": None,
            "compared_to_scan_id": None,
            "compared_to_date": None,
        },
        "costs_usd": {"sam": 0.0025, "muse_spark": 0.0013, "total": 0.0038},
        "disclaimer": (
            "SETU does not assess safety. Flagged readings are for engineer review."
        ),
    }

    save_scan_result(scan_id, initial_result)
    _sync_scan_record_to_db(scan_id, initial_result)

    scan_dir = get_scan_dir(scan_id)
    background_tasks.add_task(
        _execute_analysis_task,
        scan_id=scan_id,
        video_path=str(dest_path),
        lat=float(actual_lat),
        lon=float(actual_lon),
        target_prompt=target_prompt,
        background_prompt=background_prompt,
        out_dir=str(scan_dir),
        is_cable_override=is_cable,
    )

    return initial_result


@app.get("/api/scans/{id}")
@app.get("/scans/{id}")
def get_scan(
    id: str,
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """Retrieve scan result JSON (status queued, running, complete, or error)."""
    # 1. Attempt to load directly from data/scans/<id>/result.json
    result = load_scan_result(id)
    if result:
        return result

    showcase = getattr(app.state, "showcase_scans", None) or load_showcase_scans()
    if id in showcase:
        return showcase[id]

    # 2. Check SQLModel scan table
    record = session.get(ScanRecord, id)
    if record and record.result_json:
        try:
            return json.loads(record.result_json)
        except Exception:
            pass

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Scan '{id}' not found.",
    )


@app.get("/api/scans/{id}/files/{name}")
@app.get("/scans/{id}/files/{name}")
def get_scan_file(id: str, name: str) -> FileResponse:
    """Serve output files (magnified video, plots) with strict path-traversal protection."""
    showcase = getattr(app.state, "showcase_scans", None) or load_showcase_scans()
    if id in showcase:
        if ".." in name or "/" in name or "\\" in name:
            raise HTTPException(status_code=400, detail="Path traversal detected in file name")
        folder = None
        for candidate in SHOWCASE_DIR.iterdir() if SHOWCASE_DIR.is_dir() else []:
            try:
                data = json.loads((candidate / "result.json").read_text(encoding="utf-8"))
                if str(data.get("scan_id") or candidate.name) == id:
                    folder = candidate.resolve()
                    break
            except (OSError, ValueError):
                continue
        if folder is None:
            raise HTTPException(status_code=404, detail="Showcase file not found")
        file_path = (folder / name).resolve()
        try:
            file_path.relative_to(folder)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Path traversal detected") from exc
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail=f"File '{name}' not found")
    else:
        file_path = get_scan_file_path(scan_id=id, filename=name)

    # Determine media type
    media_type, _ = mimetypes.guess_type(str(file_path))
    if not media_type:
        if name.endswith(".mp4"):
            media_type = "video/mp4"
        elif name.endswith(".png"):
            media_type = "image/png"
        elif name.endswith(".json"):
            media_type = "application/json"
        else:
            media_type = "application/octet-stream"

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=file_path.name,
    )


@app.get("/api/bridges")
@app.get("/bridges")
def get_bridges(
    session: Session = Depends(get_session),
) -> List[Dict[str, Any]]:
    """Retrieve scans grouped by bridge or rounded location, with latest verdict and history."""
    all_scans = _collect_all_scan_data(session)

    # Group by stable OSM identity, falling back to rounded coordinates.
    groups: Dict[str, List[Dict[str, Any]]] = {}
    group_labels: Dict[str, str] = {}
    group_coords: Dict[str, Dict[str, float]] = {}

    for scan in all_scans:
        # Exclude demo scans from bridge aggregation
        if scan.get("is_demo") is True:
            continue

        raw_bridge = (
            (scan.get("interpretation") or {}).get("bridge")
            or scan.get("bridge_name")
        )
        loc = scan.get("location") or {}
        lat = float(loc.get("lat", 0.0))
        lon = float(loc.get("lon", 0.0))
        identity = bridge_key_for_scan(scan)
        key = identity
        details = bridge_details(scan)
        rounded_lat, rounded_lon = round(lat, 3), round(lon, 3)
        label = str(details.get("name") or f"Bridge @ {rounded_lat:.3f}, {rounded_lon:.3f}")

        if key not in groups:
            groups[key] = []
            group_labels[key] = label
            group_coords[key] = {"lat": lat, "lon": lon}

        groups[key].append(scan)

    result_summaries: List[Dict[str, Any]] = []

    for key, scan_list in groups.items():
        # Sort scans chronologically
        scan_list.sort(key=lambda s: s.get("created_at", ""))
        latest_scan = scan_list[-1]

        # Extract frequency history
        frequency_history = []
        for s in scan_list:
            freq_hz = s.get("measurement", {}).get("dominant_frequency_hz")
            frequency_history.append({
                "scan_id": s.get("scan_id"),
                "timestamp": s.get("created_at"),
                "frequency_hz": freq_hz,
                "freq_hz": freq_hz,
                "verdict": s.get("verdict"),
            })

        latest_verdict = latest_scan.get(
            "verdict", {"level": "unknown", "reasons": []}
        )
        last_time = latest_scan.get("created_at")
        coords = latest_scan.get("location") or group_coords[key]
        latest_meas = latest_scan.get("measurement") or {}
        dom_freq = latest_meas.get("dominant_frequency_hz")
        assessment = (latest_scan.get("interpretation") or {}).get("assessment") or {}

        bridge_summary = {
            "id": f"bridge-{hashlib.md5(key.encode('utf-8')).hexdigest()[:8]}",
            "bridge_key": key,
            "bridge": group_labels[key],
            "bridge_name": group_labels[key],
            "location": coords,
            "lat": coords.get("lat"),
            "lon": coords.get("lon"),
            "latest_verdict": latest_verdict,
            "verdict": latest_verdict,
            "last_scanned_time": last_time,
            "last_scanned_at": last_time,
            "timestamp": last_time,
            "frequency_history": frequency_history,
            "history": [
                {"timestamp": h["timestamp"], "freq_hz": h["freq_hz"]}
                for h in frequency_history
                if h["freq_hz"] is not None
            ],
            "scans": [s.get("scan_id") for s in scan_list],
            "scan_count": len(scan_list),
            "dominant_frequency_hz": dom_freq,
            "fundamental_hz": dom_freq,
            "severity": assessment.get("severity", "none"),
            "reasons": assessment.get("reasons", []),
            "recommended_action": assessment.get("recommended_action", ""),
        }
        result_summaries.append(bridge_summary)

    return result_summaries


@app.get("/api/showcase")
def get_showcase() -> List[Dict[str, Any]]:
    """List real, bundled showcase scans (read-only and never marked as demos)."""
    scans = getattr(app.state, "showcase_scans", None)
    if scans is None:
        scans = load_showcase_scans()
        app.state.showcase_scans = scans
    return list(scans.values())


@app.get("/api/bridges/{bridge_key:path}/report.json")
def get_bridge_report_json(bridge_key: str, session: Session = Depends(get_session)) -> Dict[str, Any]:
    """Return an engineer report for a bridge, excluding demo scans."""
    report = build_bridge_report(bridge_key, _collect_all_scan_data(session))
    if not report["scans"]:
        raise HTTPException(status_code=404, detail="Bridge report not found")
    return report


@app.get("/api/bridges/{bridge_key:path}/report.csv")
def get_bridge_report_csv(bridge_key: str, session: Session = Depends(get_session)) -> Response:
    """Download CSV report rows for a bridge, excluding demo scans."""
    report = build_bridge_report(bridge_key, _collect_all_scan_data(session))
    if not report["scans"]:
        raise HTTPException(status_code=404, detail="Bridge report not found")
    return Response(
        content=report_to_csv(report),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="bridge-report.csv"'},
    )


# ---------------------------------------------------------------------------
# Legacy Endpoints (Preserved for compatibility)
# ---------------------------------------------------------------------------

@app.post("/api/upload", status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    target_prompt: str = Form("corner of concrete pillar"),
    background_prompt: str = Form("stationary ground reference"),
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """Legacy Step 1 upload endpoint."""
    os.makedirs(settings.STORAGE_DIR, exist_ok=True)
    file_path = os.path.join(settings.STORAGE_DIR, f"{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    upload_record = VideoUpload(
        filename=file.filename or "recording.mp4",
        filepath=file_path,
        latitude=latitude,
        longitude=longitude,
        target_prompt=target_prompt,
        background_prompt=background_prompt,
    )
    session.add(upload_record)
    session.commit()
    session.refresh(upload_record)

    return {
        "message": "Video uploaded successfully. Ready for pipeline analysis.",
        "upload_id": upload_record.id,
        "filename": upload_record.filename,
        "location": {"latitude": latitude, "longitude": longitude},
    }


@app.post("/api/analyze/{upload_id}", status_code=status.HTTP_200_OK)
def trigger_analysis(
    upload_id: int,
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """Legacy Step 2 analyze endpoint."""
    upload = session.get(VideoUpload, upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload record not found.")

    return {
        "upload_id": upload_id,
        "status": "pipeline_stub_ready",
        "message": "Analysis pipeline scaffolded. Ready for full signal processing implementation.",
    }


@app.get("/api/triage", response_model=List[TriageItem])
def get_triage_queue(
    session: Session = Depends(get_session),
) -> List[TriageItem]:
    """Legacy Step 7 triage endpoint."""
    statement = select(TriageItem).order_by(TriageItem.created_at.desc())
    return session.exec(statement).all()


@app.get("/{frontend_path:path}", include_in_schema=False)
def serve_frontend(frontend_path: str) -> FileResponse:
    """Serve the built single-page frontend from the same origin as the API."""
    if frontend_path == "api" or frontend_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if not dist.is_dir():
        raise HTTPException(status_code=404, detail="Frontend build not found")
    candidate = (dist / frontend_path).resolve() if frontend_path else dist / "index.html"
    try:
        candidate.relative_to(dist.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Not found") from exc
    if not candidate.is_file():
        candidate = dist / "index.html"
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Frontend build not found")
    return FileResponse(candidate)
