/**
 * TypeScript Type Definitions for SETU Telemetry Platform
 * Matching Shared Result Format (schema_version 1.0)
 */

export type ScanStatus = "queued" | "running" | "complete" | "error";

export type VerdictLevel = "usable" | "weak" | "failed";

export type SegmentationSource = "sam" | "fallback" | "none" | string;

export type TriageSeverity = "none" | "low" | "medium" | "high";

export interface Verdict {
  level: VerdictLevel;
  reasons: string[];
}

export interface LocationCoordinates {
  lat: number;
  lon: number;
}

export interface VideoMetadata {
  filename: string;
  fps: number;
  duration_s: number;
  width: number;
  height: number;
  frame_count: number;
}

export interface SegmentationDetails {
  target_prompt: string;
  background_prompt: string;
  target_source: SegmentationSource;
  background_source: SegmentationSource;
  target_persistence: number;
  background_persistence: number;
  warnings: string[];
}

export interface SpectralPeak {
  frequency_hz: number;
  relative_power: number;
  snr_db: number;
}

export interface PowerSpectralDensity {
  freq_hz: number[];
  power: number[];
}

export interface MeasurementDetails {
  units: string;
  scale: number | null;
  dominant_frequency_hz: number | null;
  frequency_ci_hz: [number, number] | null;
  peaks: SpectralPeak[];
  background_residual_px: number;
  psd?: PowerSpectralDensity | null;
}

export interface QualityGateDetails {
  is_structural_element: boolean;
  has_static_background: boolean;
  appears_stable: boolean;
  issues: string[];
}

export interface StructuralAssessment {
  severity: TriageSeverity;
  reasons: string[];
  expected_range_hz: [number, number] | null;
  confidence: "low" | "medium" | "high" | string;
  recommended_action: string;
  assessor: string;
}

export interface MappedBridge {
  id?: string | number;
  name?: string;
  osm_id?: string;
  latitude?: number;
  longitude?: number;
  bridge_type?: string | null;
  material?: string | null;
  length_meters?: number | null;
  [key: string]: unknown;
}

export interface InterpretationDetails {
  bridge: MappedBridge | null;
  bridge_warning: string | null;
  quality_gate: QualityGateDetails;
  assessment: StructuralAssessment;
}

export interface ScanOutputs {
  magnified_video?: string | null;
  spectrum_plot?: string | null;
  displacement_plot?: string | null;
  [key: string]: string | null | undefined;
}

export interface CostBreakdown {
  sam: number;
  muse_spark: number;
  total: number;
  [key: string]: number;
}

export interface ScanResult {
  schema_version: string;
  scan_id: string;
  is_demo?: boolean;
  created_at: string;
  status: ScanStatus;
  verdict: Verdict;
  location: LocationCoordinates;
  video: VideoMetadata;
  segmentation: SegmentationDetails;
  measurement: MeasurementDetails;
  interpretation: InterpretationDetails;
  outputs: ScanOutputs;
  costs_usd: CostBreakdown;
  disclaimer: string;
  error?: string | null;
}

// Backward-compatible alias
export type ScanRecord = ScanResult;

export interface HistoryPoint {
  timestamp: string;
  freq_hz: number;
}

export interface BridgeSummary {
  id: string;
  bridge_name: string;
  city?: string;
  lat: number;
  lon: number;
  timestamp: string;
  severity: TriageSeverity;
  verdict_level?: VerdictLevel;
  fundamental_hz: number | null;
  confidence: number | string;
  reasons: string[];
  recommended_action: string;
  history: HistoryPoint[];
  is_sample_data?: boolean;
  is_demo?: boolean;
}

export interface CreateScanPayload {
  file?: File | null;
  preview_url?: string;
  lat: number | string;
  lon: number | string;
  bridge_name?: string;
  target_prompt?: string;
  background_prompt?: string;
  city?: string;
}
