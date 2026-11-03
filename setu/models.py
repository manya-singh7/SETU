"""
SQLModel database models and Pydantic schema representations for Project SETU.

Includes entities for bridges, video uploads, spectral analysis results,
the engineer triage review queue, and scan records aligned with the SETU
vibration telemetry shared result format.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field as PydanticField
from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


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


class ScanStatus(str, Enum):
    """Lifecycle status for a video vibration scan."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"


class VerdictLevel(str, Enum):
    """Reliability verdict for vibration analysis results."""

    USABLE = "usable"
    WEAK = "weak"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Database Tables (SQLModel)
# ---------------------------------------------------------------------------

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
    created_at: datetime = Field(default_factory=utc_now)


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
    uploaded_at: datetime = Field(default_factory=utc_now)


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
    processed_at: datetime = Field(default_factory=utc_now)


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
    created_at: datetime = Field(default_factory=utc_now)


class ScanRecord(SQLModel, table=True):
    """Persistent database entity for video vibration scans."""

    id: str = Field(primary_key=True, index=True)
    created_at: datetime = Field(default_factory=utc_now, index=True)
    status: str = Field(default="queued", index=True)  # queued | running | complete | error
    latitude: float = Field(index=True)
    longitude: float = Field(index=True)
    bridge_name: Optional[str] = Field(default=None, index=True)
    target_prompt: str = Field(default="concrete pillar")
    background_prompt: str = Field(default="building")
    verdict_level: Optional[str] = Field(default=None)
    dominant_frequency_hz: Optional[float] = Field(default=None)
    result_json: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Shared Result Format Schemas (Pydantic)
# ---------------------------------------------------------------------------

class VerdictModel(BaseModel):
    level: str = "usable"  # usable | weak | failed | queued | running
    reasons: List[str] = []


class LocationModel(BaseModel):
    lat: float
    lon: float


class VideoMetaModel(BaseModel):
    filename: str
    fps: Optional[float] = None
    duration_s: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    frame_count: Optional[int] = None


class SegmentationMetaModel(BaseModel):
    target_prompt: str
    background_prompt: str
    target_source: str = "sam"
    background_source: str = "sam"
    target_persistence: Optional[float] = None
    background_persistence: Optional[float] = None
    warnings: List[str] = []


class PeakModel(BaseModel):
    frequency_hz: float
    relative_power: float
    snr_db: float


class MeasurementMetaModel(BaseModel):
    units: str = "px"
    scale: Optional[float] = None
    dominant_frequency_hz: Optional[float] = None
    frequency_ci_hz: Optional[List[float]] = None
    peaks: List[PeakModel] = []
    background_residual_px: Optional[float] = None


class QualityGateModel(BaseModel):
    is_structural_element: bool = True
    has_static_background: bool = True
    appears_stable: bool = True
    issues: List[str] = []


class AssessmentModel(BaseModel):
    severity: str = "none"
    reasons: List[str] = []
    expected_range_hz: Optional[List[float]] = None
    confidence: str = "low"
    recommended_action: str = ""
    assessor: str = "local_rules"


class InterpretationMetaModel(BaseModel):
    bridge: Optional[str] = None
    bridge_warning: Optional[str] = None
    quality_gate: QualityGateModel = PydanticField(default_factory=QualityGateModel)
    assessment: AssessmentModel = PydanticField(default_factory=AssessmentModel)


class OutputsMetaModel(BaseModel):
    magnified_video: Optional[str] = None
    spectrum_plot: Optional[str] = None
    displacement_plot: Optional[str] = None


class CostsUsdModel(BaseModel):
    sam: float = 0.0
    muse_spark: float = 0.0
    total: float = 0.0


class ScanResultModel(BaseModel):
    """Complete SETU scan result object schema."""

    schema_version: str = "1.0"
    scan_id: str
    created_at: str
    status: str
    verdict: VerdictModel
    location: LocationModel
    video: VideoMetaModel
    segmentation: SegmentationMetaModel
    measurement: MeasurementMetaModel
    interpretation: InterpretationMetaModel
    outputs: OutputsMetaModel
    costs_usd: CostsUsdModel
    disclaimer: str = (
        "SETU does not assess safety. Flagged readings are for engineer review."
    )
