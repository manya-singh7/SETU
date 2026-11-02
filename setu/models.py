"""
SQLModel database models for Project SETU.

Includes entities for bridges, video uploads, spectral analysis results,
and the engineer triage review queue.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from sqlmodel import Field, SQLModel


class TriageSeverity(str, Enum):
    """Categorization of triage urgency.

    NOTE: In accordance with SETU's safety philosophy, the system NEVER
    declares a bridge safe. It only flags anomalies for human engineer inspection.
    """

    UNUSUAL = "UNUSUAL"
    INCONCLUSIVE = "INCONCLUSIVE"
    REQUIRES_INSPECTION = "REQUIRES_INSPECTION"


class TriageStatus(str, Enum):
    """Review workflow status for civil engineers."""

    PENDING = "PENDING"
    UNDER_REVIEW = "UNDER_REVIEW"
    INSPECTED = "INSPECTED"
    DISMISSED = "DISMISSED"


class Bridge(SQLModel, table=True):
    """Physical bridge infrastructure entity, correlated with OSM."""

    id: Optional[int] = Field(default=None, primary_key=True)
    osm_id: Optional[str] = Field(default=None, index=True)
    name: str = Field(default="Unnamed Bridge", index=True)
    latitude: float = Field(index=True)
    longitude: float = Field(index=True)
    bridge_type: Optional[str] = None  # e.g., truss, cable_stayed, arch, beam
    material: Optional[str] = None     # e.g., concrete, steel
    length_meters: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class VideoUpload(SQLModel, table=True):
    """Uploaded 30-second mobile phone recording and prompt parameters."""

    id: Optional[int] = Field(default=None, primary_key=True)
    bridge_id: Optional[int] = Field(default=None, foreign_key="bridge.id")
    filename: str
    filepath: str
    latitude: float
    longitude: float
    target_prompt: str = Field(default="corner of concrete pillar")
    background_prompt: str = Field(default="stationary ground reference")
    fps: float = Field(default=30.0)
    duration_seconds: float = Field(default=30.0)
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)


class AnalysisResult(SQLModel, table=True):
    """Quantitative vibration analysis telemetry extracted from video."""

    id: Optional[int] = Field(default=None, primary_key=True)
    upload_id: int = Field(foreign_key="videoupload.id")
    fundamental_frequency_hz: Optional[float] = None
    harmonic_frequencies_json: Optional[str] = None  # JSON list of floats
    peak_displacement_px: Optional[float] = None
    estimated_damping_ratio: Optional[float] = None
    magnified_video_path: Optional[str] = None
    osm_matched: bool = Field(default=False)
    processed_at: datetime = Field(default_factory=datetime.utcnow)


class TriageItem(SQLModel, table=True):
    """Engineer triage item for reviewing unusual or inconclusive readings."""

    id: Optional[int] = Field(default=None, primary_key=True)
    analysis_id: int = Field(foreign_key="analysisresult.id")
    bridge_id: Optional[int] = Field(default=None, foreign_key="bridge.id")
    severity: TriageSeverity = Field(default=TriageSeverity.REQUIRES_INSPECTION)
    anomaly_score: float = Field(default=0.0)
    assessment_summary: str
    recommended_action: str = Field(
        default="Schedule physical civil engineer on-site inspection. SETU does not declare structures safe."
    )
    status: TriageStatus = Field(default=TriageStatus.PENDING, index=True)
    engineer_notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
