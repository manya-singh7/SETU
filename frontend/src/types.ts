/**
 * TypeScript Type Definitions for SETU Telemetry Platform
 */

export type TriageSeverity = "none" | "low" | "medium" | "high";

export interface Peak {
  freq_hz: number;
  rel_power: number;
  snr: number;
}

export interface QualityMetrics {
  background_residual_px: number;
  snr: number;
  warnings: string[];
}

export interface Assessment {
  severity: TriageSeverity;
  reasons: string[];
  confidence: number;
  data_quality_issues: string[];
  recommended_action: string;
}

export interface HistoryPoint {
  timestamp: string;
  freq_hz: number;
}

export interface ScanRecord {
  id: string;
  bridge_name: string;
  city?: string;
  lat: number;
  lon: number;
  timestamp: string;
  peaks: Peak[];
  quality: QualityMetrics;
  assessment: Assessment;
  video_url: string;
  magnified_url: string;
  history: HistoryPoint[];
  _sample_data_notice?: string;
}

export interface BridgeSummary {
  id: string;
  bridge_name: string;
  city?: string;
  lat: number;
  lon: number;
  timestamp: string;
  severity: TriageSeverity;
  fundamental_hz: number | null;
  confidence: number;
  reasons: string[];
  recommended_action: string;
  history: HistoryPoint[];
}

export interface CreateScanPayload {
  file?: File | null;
  preview_url?: string;
  lat: number;
  lon: number;
  bridge_name?: string;
  target_prompt?: string;
  city?: string;
}
