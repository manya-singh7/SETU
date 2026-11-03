import { ScanResult } from "../types";

export const mockUsableScan: ScanResult = {
  schema_version: "1.0",
  scan_id: "scan_usable_001",
  created_at: "2026-11-03T19:30:00+05:30",
  status: "complete",
  verdict: {
    level: "usable",
    reasons: ["Clear resonance peak detected above SNR threshold"],
  },
  location: { lat: 13.0827, lon: 80.2707 },
  video: {
    filename: "clip.mp4",
    fps: 60.0,
    duration_s: 30.0,
    width: 1920,
    height: 1080,
    frame_count: 1800,
  },
  segmentation: {
    target_prompt: "concrete pillar",
    background_prompt: "building",
    target_source: "sam",
    background_source: "sam",
    target_persistence: 0.98,
    background_persistence: 0.99,
    warnings: [],
  },
  measurement: {
    units: "px",
    scale: null,
    dominant_frequency_hz: 3.70,
    frequency_ci_hz: [3.68, 3.72],
    peaks: [
      { frequency_hz: 3.70, relative_power: 0.91, snr_db: 38.5 },
    ],
    psd: {
      freq_hz: [1.0, 2.0, 3.0, 3.7, 4.0, 5.0],
      power: [0.01, 0.05, 0.12, 0.91, 0.10, 0.02],
    },
    background_residual_px: 0.016,
  },
  interpretation: {
    bridge: {
      id: "napier-01",
      name: "Napier Bridge Pier #2",
    },
    bridge_warning: null,
    quality_gate: {
      is_structural_element: true,
      has_static_background: true,
      appears_stable: true,
      issues: [],
    },
    assessment: {
      severity: "none",
      reasons: ["Frequency within nominal envelope"],
      expected_range_hz: [3.5, 4.0],
      confidence: "high",
      recommended_action: "Routine annual inspection.",
      assessor: "local_rules",
    },
  },
  outputs: {
    magnified_video: "magnified.mp4",
    spectrum_plot: "spectrum.png",
    displacement_plot: "displacement.png",
  },
  costs_usd: {
    sam: 0.0025,
    muse_spark: 0.0013,
    total: 0.0038,
  },
  disclaimer: "SETU does not assess safety. Flagged readings are for engineer review.",
  is_demo: false,
};

export const mockWeakScan: ScanResult = {
  ...mockUsableScan,
  scan_id: "scan_weak_002",
  verdict: {
    level: "weak",
    reasons: ["Low vibration amplitude observed during recording"],
  },
  measurement: {
    ...mockUsableScan.measurement,
    dominant_frequency_hz: 2.10,
    frequency_ci_hz: [2.00, 2.20],
    peaks: [
      { frequency_hz: 2.10, relative_power: 0.45, snr_db: 12.0 },
    ],
  },
  interpretation: {
    ...mockUsableScan.interpretation,
    assessment: {
      severity: "low",
      reasons: ["Marginal vibration amplitude"],
      expected_range_hz: null,
      confidence: "low",
      recommended_action: "Film again under heavier traffic.",
      assessor: "local_rules",
    },
  },
  is_demo: false,
};

export const mockFailedScan: ScanResult = {
  ...mockUsableScan,
  scan_id: "scan_failed_003",
  verdict: {
    level: "failed",
    reasons: ["No clear vibration peak detected", "SNR below rejection threshold"],
  },
  measurement: {
    units: "px",
    scale: null,
    dominant_frequency_hz: null,
    frequency_ci_hz: null,
    peaks: [],
    psd: null,
    background_residual_px: 0.045,
  },
  interpretation: {
    ...mockUsableScan.interpretation,
    assessment: {
      severity: "none",
      reasons: ["Measurement rejected"],
      expected_range_hz: null,
      confidence: "low",
      recommended_action: "Re-film bridge with stabilized camera.",
      assessor: "local_rules",
    },
  },
  is_demo: false,
};

export const mockDemoScan: ScanResult = {
  ...mockUsableScan,
  scan_id: "scan_demo_004",
  is_demo: true,
};

export const mockQueuedScan: ScanResult = {
  ...mockUsableScan,
  scan_id: "scan_queued_005",
  status: "queued",
};

export const mockRunningScan: ScanResult = {
  ...mockUsableScan,
  scan_id: "scan_running_006",
  status: "running",
};

export const mockErrorScan: ScanResult = {
  ...mockUsableScan,
  scan_id: "scan_error_007",
  status: "error",
  error: "Pipeline error: tracking lost due to sudden illumination change.",
};
