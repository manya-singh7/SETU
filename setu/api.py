"""
FastAPI application and REST endpoints for Project SETU.

Step 7 in SETU Pipeline:
Serves endpoints for video + GPS upload, pipeline execution trigger,
retrieval of bridge map coordinates, and the civil engineer triage queue.
"""

from contextlib import asynccontextmanager
import os
import shutil
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select

from setu.config import settings
from setu.database import create_db_and_tables, get_session
from setu.models import (
    AnalysisResult,
    Bridge,
    TriageItem,
    TriageSeverity,
    TriageStatus,
    VideoUpload,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize SQLite database tables and storage folders upon startup."""
    os.makedirs(settings.STORAGE_DIR, exist_ok=True)
    create_db_and_tables()
    yield


app = FastAPI(
    title="Project SETU API",
    description="Bridge Vibration Telemetry & Engineer Triage Screening API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS setup for Vite frontend (localhost:5173) and local clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def get_health() -> Dict[str, Any]:
    """Health check endpoint confirming API service and configuration status."""
    return {
        "status": "ok",
        "service": "SETU Civil Telemetry API",
        "fallbacks_enabled": settings.USE_LOCAL_FALLBACKS,
        "database": settings.DATABASE_URL,
    }


@app.post("/api/upload", status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    target_prompt: str = Form("corner of concrete pillar"),
    background_prompt: str = Form("stationary ground reference"),
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """Step 1: Upload a 30s video + GPS location with segmentation prompts."""
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
    """Trigger the 7-stage vibration analysis and triage generation for an upload."""
    upload = session.get(VideoUpload, upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload record not found.")

    # STUB: Pipeline integration will be wired here in implementation phase
    return {
        "upload_id": upload_id,
        "status": "pipeline_stub_ready",
        "message": "Analysis pipeline scaffolded. Ready for full signal processing implementation.",
    }


@app.get("/api/triage", response_model=List[TriageItem])
def get_triage_queue(
    session: Session = Depends(get_session),
) -> List[TriageItem]:
    """Step 7: Retrieve all triage items for human engineer inspection."""
    statement = select(TriageItem).order_by(TriageItem.created_at.desc())
    return session.exec(statement).all()


@app.get("/api/bridges", response_model=List[Bridge])
def get_bridges(
    session: Session = Depends(get_session),
) -> List[Bridge]:
    """Retrieve bridge geographic coordinates for the Leaflet frontend map."""
    statement = select(Bridge)
    return session.exec(statement).all()
