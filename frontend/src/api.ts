/**
 * SETU Telemetry Data Access API
 *
 * Real backend connection + shared result format (schema_version 1.0).
 * Handles createScan (POST /api/scans), getScan (GET /api/scans/{id} with polling),
 * getBridges (GET /api/bridges with mock fallback), and getFileUrl (/api/scans/{id}/files/{name}).
 */

import mockScansData from "./mock/scans.json";
import {
  BridgeSummary,
  BridgeReport,
  CreateScanPayload,
  FieldNote,
  ScanResult,
  TriageSeverity,
  VerdictLevel,
} from "./types";

/**
 * Resolve file URLs via /api/scans/{id}/files/{name}.
 * If the filename is already a data/blob/http URL, returns as-is.
 */
export function getFileUrl(scanId: string, name: string): string {
  if (!name) return "";
  if (
    name.startsWith("http://") ||
    name.startsWith("https://") ||
    name.startsWith("blob:") ||
    name.startsWith("data:")
  ) {
    return name;
  }
  return `/api/scans/${encodeURIComponent(scanId)}/files/${encodeURIComponent(name)}`;
}

// Backward-compatible alias
export const getScanFileUrl = getFileUrl;

/**
 * Canonical sample scan conforming to Shared Result Format (schema 1.0)
 */
export const SAMPLE_BENCHMARK_SCAN: ScanResult = {
  schema_version: "1.2",
  scan_id: "scan_20261103_a1b2c3",
  is_demo: false,
  created_at: "2026-11-03T19:30:00+05:30",
  status: "complete",
  verdict: {
    level: "usable",
    reasons: [],
  },
  location: {
    lat: 13.0827,
    lon: 80.2707,
  },
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
    dominant_frequency_hz: 3.7,
    frequency_ci_hz: [3.68, 3.72],
    peaks: [
      {
        frequency_hz: 3.7,
        relative_power: 0.91,
        snr_db: 38.5,
      },
    ],
    psd: {
      freq_hz: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.7, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
      power: [0.001, 0.001, 0.002, 0.003, 0.005, 0.008, 0.035, 0.371, 0.91, 0.123, 0.015, 0.004, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001],
    },
    background_residual_px: 0.016,
    instances: [
      {
        label: "Pier #2 South Column",
        bbox: [450, 200, 180, 520],
        frequency_hz: 3.70,
        amplitude_px: 0.42,
        verdict: "usable",
      },
      {
        label: "Midspan Girders",
        bbox: [120, 150, 720, 180],
        frequency_hz: 3.68,
        amplitude_px: 0.85,
        verdict: "usable",
      },
    ],
  },
  interpretation: {
    bridge: null,
    bridge_warning: "No mapped bridge within 150 m",
    quality_gate: {
      is_structural_element: true,
      has_static_background: true,
      appears_stable: true,
      issues: [],
    },
    assessment: {
      severity: "none",
      reasons: [],
      expected_range_hz: null,
      confidence: "low",
      recommended_action: "",
      assessor: "local_rules",
    },
  },
  outputs: {
    magnified_video: "magnified.mp4",
    spectrum_plot: "spectrum.png",
    displacement_plot: "displacement.png",
    vibration_map: "vibration_map.png",
  },
  costs_usd: {
    sam: 0.0025,
    muse_spark: 0.0013,
    total: 0.0038,
  },
  disclaimer: "SETU does not assess safety. Flagged readings are for engineer review.",
  field_notes: [],
  cable: null,
};

function convertLegacyMockToScanResult(mock: Record<string, any>): ScanResult {
  const primaryPeak = mock.peaks?.[0];
  const rawSev = typeof mock.assessment?.severity === "string" ? mock.assessment.severity.toLowerCase() : "none";
  const severity: TriageSeverity = ["high", "medium", "low", "none"].includes(rawSev) ? rawSev : "none";
  const verdictLevel: VerdictLevel = severity === "high" ? "weak" : "usable";

  return {
    schema_version: "1.0",
    scan_id: mock.id || "scan_sample",
    is_demo: true, // Marked as demo
    created_at: mock.timestamp || new Date().toISOString(),
    status: "complete",
    verdict: {
      level: verdictLevel,
      reasons: mock.assessment?.reasons || [],
    },
    location: {
      lat: Number(mock.lat ?? 13.0827),
      lon: Number(mock.lon ?? 80.2707),
    },
    video: {
      filename: "benchmark_clip.mp4",
      fps: 60.0,
      duration_s: 30.0,
      width: 1920,
      height: 1080,
      frame_count: 1800,
    },
    segmentation: {
      target_prompt: "structural deck midspan",
      background_prompt: "building",
      target_source: "sam",
      background_source: "sam",
      target_persistence: 0.98,
      background_persistence: 0.99,
      warnings: mock.quality?.warnings || [],
    },
    measurement: {
      units: "px",
      scale: null,
      dominant_frequency_hz: primaryPeak?.freq_hz ?? 3.7,
      frequency_ci_hz: primaryPeak?.freq_hz
        ? [
            Number((primaryPeak.freq_hz - 0.02).toFixed(2)),
            Number((primaryPeak.freq_hz + 0.02).toFixed(2)),
          ]
        : null,
      peaks: (mock.peaks || []).map((p: any) => ({
        frequency_hz: p.freq_hz,
        relative_power: p.rel_power,
        snr_db: p.snr || 24.0,
      })),
      psd: null, // No synthetic PSD curve! If null, only backend plot image or nothing is displayed
      background_residual_px: mock.quality?.background_residual_px ?? 0.05,
    },
    interpretation: {
      bridge: {
        name: mock.bridge_name,
        latitude: mock.lat,
        longitude: mock.lon,
      },
      bridge_warning: null,
      quality_gate: {
        is_structural_element: true,
        has_static_background: true,
        appears_stable: true,
        issues: [],
      },
      assessment: {
        severity,
        reasons: mock.assessment?.reasons || [],
        expected_range_hz: [2.0, 5.0],
        confidence: mock.assessment?.confidence || "medium",
        recommended_action: mock.assessment?.recommended_action || "Schedule routine inspection.",
        assessor: "local_rules",
      },
    },
    outputs: {
      magnified_video: mock.magnified_url || "magnified.mp4",
      spectrum_plot: "spectrum.png",
      displacement_plot: "displacement.png",
    },
    costs_usd: {
      sam: 0.0025,
      muse_spark: 0.0013,
      total: 0.0038,
    },
    disclaimer: "SETU does not assess safety. Flagged readings are for engineer review.",
  };
}

function getMockScanById(id: string): ScanResult | null {
  if (id === SAMPLE_BENCHMARK_SCAN.scan_id) {
    return SAMPLE_BENCHMARK_SCAN;
  }
  const rawList = mockScansData as Array<Record<string, any>>;
  const found = rawList.find((item) => item.id === id);
  if (found) {
    return convertLegacyMockToScanResult(found);
  }
  return null;
}

/**
 * Submit a new scan via POST /api/scans (multipart FormData).
 */
export async function createScan(payload: CreateScanPayload): Promise<ScanResult> {
  const formData = new FormData();
  if (payload.file) {
    formData.append("file", payload.file);
  }
  formData.append("latitude", String(payload.lat));
  formData.append("longitude", String(payload.lon));
  formData.append("lat", String(payload.lat));
  formData.append("lon", String(payload.lon));
  formData.append("target_prompt", payload.target_prompt || "");
  formData.append("is_cable", String(Boolean(payload.is_cable)));
  formData.append("background_prompt", payload.background_prompt || "");
  if (payload.language) formData.append("language", payload.language);
  if (payload.device_motion) formData.append("device_motion", payload.device_motion);
  if (payload.bridge_name) {
    formData.append("bridge_name", payload.bridge_name);
  }

  let response: Response;
  try {
    response = await fetch("/api/scans", {
      method: "POST",
      body: formData,
    });
  } catch (err) {
    console.warn("Backend POST /api/scans unreachable:", err);
    throw new Error("Could not connect to backend server at /api/scans. Please ensure the backend is running.");
  }

  if (!response.ok) {
    const errorText = await response.text().catch(() => response.statusText);
    throw new Error(`Scan submission failed (${response.status}): ${errorText}`);
  }

  return (await response.json()) as ScanResult;
}

/**
 * Fetch a scan by ID via GET /api/scans/{id}.
 * Polls every 2 seconds until status is 'complete' or 'error'.
 * Calls onProgress callback with latest state during polling.
 */
export async function getScan(
  id: string,
  onProgress?: (scan: ScanResult) => void,
  waitForMagnification = false,
): Promise<ScanResult> {
  const POLL_INTERVAL_MS = 2000;

  while (true) {
    try {
      const response = await fetch(`/api/scans/${encodeURIComponent(id)}`);
      if (response.ok) {
        const scan = (await response.json()) as ScanResult;
        if (onProgress) {
          onProgress(scan);
        }
        const magnificationPending = scan.magnification_status === "pending" || scan.magnification_status === "running";
        if (scan.status === "error" || (scan.status === "complete" && !(waitForMagnification && magnificationPending))) {
          return scan;
        }
      } else if (response.status === 404) {
        const fallback = getMockScanById(id);
        if (fallback) {
          if (onProgress) onProgress(fallback);
          return fallback;
        }
        throw new Error(`Scan with id "${id}" was not found.`);
      } else {
        const errorText = await response.text().catch(() => response.statusText);
        throw new Error(`Failed to fetch scan (${response.status}): ${errorText}`);
      }
    } catch (err: any) {
      const fallback = getMockScanById(id);
      if (fallback) {
        if (onProgress) onProgress(fallback);
        return fallback;
      }
      throw err;
    }

    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }
}

/** Fetch a bridge report from GET /api/bridges/{bridge_key}/report.json. */
export async function getBridgeReport(bridgeKey: string): Promise<BridgeReport> {
  const response = await fetch(`/api/bridges/${encodeURIComponent(bridgeKey)}/report.json`);
  if (!response.ok) {
    throw new Error(`Could not load bridge report (${response.status}).`);
  }
  return (await response.json()) as BridgeReport;
}

/**
 * Fetch bridges from GET /api/bridges.
 * If empty or the backend is down, falls back to mock data with is_sample_data: true.
 */
export async function getBridges(): Promise<BridgeSummary[]> {
  try {
    const response = await fetch("/api/bridges");
    if (response.ok) {
      const data = await response.json();
      if (Array.isArray(data) && data.length > 0) {
        return data.map((b: any, index: number) => ({
          id: String(b.id ?? b.osm_id ?? b.bridge_key ?? `bridge-${index}`),
          bridge_key: b.bridge_key == null ? undefined : String(b.bridge_key),
          bridge_name: b.name ?? b.bridge_name ?? `Bridge ${index + 1}`,
          city: b.city ?? (b.latitude && b.latitude > 16 ? "Mumbai Region" : "Chennai Region"),
          lat: Number(b.latitude ?? b.lat ?? 13.0827),
          lon: Number(b.longitude ?? b.lon ?? 80.2707),
          timestamp: b.created_at ?? b.timestamp ?? new Date().toISOString(),
          severity: (b.severity?.toLowerCase() ?? "none") as TriageSeverity,
          verdict_level: (b.verdict?.level ?? b.verdict_level ?? (b.fundamental_hz ? "usable" : "failed")) as VerdictLevel,
          fundamental_hz: b.fundamental_hz ?? b.dominant_frequency_hz ?? null,
          confidence: b.confidence ?? "high",
          reasons: b.reasons ?? [],
          recommended_action: b.recommended_action ?? "Routine inspection schedule.",
          history: b.history ?? [],
          is_sample_data: Boolean(b.is_demo || b.is_sample_data),
          is_demo: Boolean(b.is_demo),
        }));
      }
    }
  } catch (err) {
    console.warn("Backend /api/bridges unavailable. Falling back to sample data.", err);
  }

  // Fallback to existing mock data with clear Sample Data indicators
  return getMockBridges();
}

/**
 * Generates sample data bridges from local mock dataset with is_sample_data: true and is_demo: true
 */
export function getMockBridges(): BridgeSummary[] {
  const scans = mockScansData as Array<Record<string, any>>;
  return scans.map((scan) => {
    const isWeak = scan.id === "scan-chn-005" || scan.verdict?.level === "weak";
    const isFailed = scan.id === "scan-mum-006" || scan.verdict?.level === "failed";
    const verdictLevel: VerdictLevel = isFailed ? "failed" : isWeak ? "weak" : "usable";

    return {
      id: scan.id,
      bridge_name: scan.bridge_name.includes("Sample") ? scan.bridge_name : `${scan.bridge_name} (Sample data)`,
      city: scan.city || "Unknown",
      lat: scan.lat,
      lon: scan.lon,
      timestamp: scan.timestamp,
      severity: (scan.assessment?.severity || "none") as TriageSeverity,
      verdict_level: verdictLevel,
      fundamental_hz: isFailed ? null : (scan.peaks?.[0]?.freq_hz ?? null),
      confidence: scan.assessment?.confidence ?? 0.9,
      reasons: scan.assessment?.reasons || [],
      recommended_action: scan.assessment?.recommended_action || "",
      history: scan.history || [],
      is_sample_data: true,
      is_demo: true,
    };
  });
}

/**
 * Upload a voice field note to POST /api/scans/{id}/field-notes as multipart.
 * Sends the audio file and current language code.
 */
export async function uploadFieldNote(
  scanId: string,
  audioBlob: Blob,
  language: string
): Promise<FieldNote> {
  const formData = new FormData();
  formData.append("audio", audioBlob, "field-note.webm");
  formData.append("language", language);

  let response: Response;
  try {
    response = await fetch(`/api/scans/${encodeURIComponent(scanId)}/field-notes`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error("Could not connect to field note service.");
  }

  if (!response.ok) {
    throw new Error(`Field note upload failed (${response.status}).`);
  }

  const result = await response.json();
  return result.field_note || result;
}

