/**
 * SETU Telemetry Data Access API
 *
 * Provides typed functions to read mock scans (Chennai and Mumbai benchmarks)
 * with localStorage persistence, easily switchable to production endpoints
 * (GET /bridges, GET /scans/{id}, POST /scans).
 */

import mockScansData from "./mock/scans.json";
import { BridgeSummary, CreateScanPayload, ScanRecord, TriageSeverity } from "./types";

const mockScans = mockScansData as ScanRecord[];

const API_BASE_URL = import.meta.env.VITE_API_URL || "";
const USE_REAL_BACKEND = Boolean(API_BASE_URL && import.meta.env.VITE_USE_REAL_BACKEND === "true");
const STORAGE_KEY = "setu_mock_scans_v1";

function getStoredScans(): ScanRecord[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length > 0) {
        return parsed as ScanRecord[];
      }
    }
  } catch (err) {
    console.warn("Could not read scans from localStorage:", err);
  }

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(mockScans));
  } catch (err) {
    console.warn("Could not seed mock scans to localStorage:", err);
  }
  return mockScans;
}

function saveStoredScans(scans: ScanRecord[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(scans));
  } catch (err) {
    console.warn("Could not save scans to localStorage:", err);
  }
}

/**
 * Fetch all bridges with latest scan telemetry.
 * In production: GET /bridges
 */
export async function getBridges(): Promise<BridgeSummary[]> {
  if (USE_REAL_BACKEND) {
    const response = await fetch(`${API_BASE_URL}/bridges`);
    if (!response.ok) {
      throw new Error(`Failed to fetch bridges: ${response.statusText}`);
    }
    return (await response.json()) as BridgeSummary[];
  }

  // Realistic mock latency
  await new Promise((resolve) => setTimeout(resolve, 80));

  const scans = getStoredScans();
  return scans.map((scan) => ({
    id: scan.id,
    bridge_name: scan.bridge_name,
    city: scan.city || "Unknown",
    lat: scan.lat,
    lon: scan.lon,
    timestamp: scan.timestamp,
    severity: (scan.assessment?.severity || "none") as TriageSeverity,
    fundamental_hz: scan.peaks?.[0]?.freq_hz ?? null,
    confidence: scan.assessment?.confidence ?? 0.9,
    reasons: scan.assessment?.reasons || [],
    recommended_action: scan.assessment?.recommended_action || "",
    history: scan.history || [],
  }));
}

/**
 * Fetch full scan details by ID.
 * In production: GET /scans/{id}
 */
export async function getScan(id: string): Promise<ScanRecord> {
  if (USE_REAL_BACKEND) {
    const response = await fetch(`${API_BASE_URL}/scans/${encodeURIComponent(id)}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch scan ${id}: ${response.statusText}`);
    }
    return (await response.json()) as ScanRecord;
  }

  await new Promise((resolve) => setTimeout(resolve, 60));

  const scans = getStoredScans();
  const found = scans.find((s) => s.id === id);
  if (!found) {
    throw new Error(`Scan with id "${id}" not found.`);
  }
  return found;
}

/**
 * Submit a new video vibration scan.
 * In production: POST /scans
 */
export async function createScan(payload: CreateScanPayload): Promise<ScanRecord> {
  if (USE_REAL_BACKEND) {
    const formData = new FormData();
    if (payload.file) formData.append("file", payload.file);
    formData.append("latitude", String(payload.lat));
    formData.append("longitude", String(payload.lon));
    if (payload.target_prompt) formData.append("target_prompt", payload.target_prompt);
    if (payload.bridge_name) formData.append("bridge_name", payload.bridge_name);

    const response = await fetch(`${API_BASE_URL}/scans`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(`Scan submission failed: ${response.statusText}`);
    }
    return (await response.json()) as ScanRecord;
  }

  await new Promise((resolve) => setTimeout(resolve, 300));

  const scans = getStoredScans();
  const newId = `scan-usr-${Date.now().toString(36)}`;
  const nowIso = new Date().toISOString();

  // Synthetic frequency based on latitude seed
  const seedFundHz = Number((1.5 + (Math.abs(payload.lat || 13) % 2.5)).toFixed(2));
  const harmonicHz = Number((seedFundHz * 2).toFixed(2));

  // Determine realistic simulated severity
  const severityRoll = Math.random();
  const severity: TriageSeverity = severityRoll > 0.7 ? "medium" : severityRoll > 0.4 ? "low" : "none";

  const newScan: ScanRecord = {
    id: newId,
    _sample_data_notice: "USER SUBMITTED SCAN (LOCAL DEMO SESSION)",
    bridge_name: payload.bridge_name || `Bridge @ ${Number(payload.lat || 0).toFixed(4)}°N`,
    city: payload.city || (payload.lat > 16 ? "Mumbai Region" : "Chennai Region"),
    lat: Number(payload.lat) || 13.0827,
    lon: Number(payload.lon) || 80.2707,
    timestamp: nowIso,
    peaks: [
      { freq_hz: seedFundHz, rel_power: 1.0, snr: 19.5 },
      { freq_hz: harmonicHz, rel_power: 0.38, snr: 10.2 },
    ],
    quality: {
      background_residual_px: 0.06,
      snr: 18.4,
      warnings: payload.file && payload.file.size > 50000000 ? ["Large video payload processed"] : [],
    },
    assessment: {
      severity,
      reasons: [
        `Fundamental frequency identified at ${seedFundHz} Hz`,
        `Target prompt applied: "${payload.target_prompt || "structural deck midspan"}"`,
      ],
      confidence: 0.89,
      data_quality_issues: [],
      recommended_action:
        severity === "none"
          ? "Routine baseline established. Re-scan per maintenance schedule."
          : "Flagged for human engineer review. SETU does not declare structures safe.",
    },
    video_url: payload.preview_url || "",
    magnified_url: payload.preview_url || "",
    history: [
      {
        timestamp: new Date(Date.now() - 60 * 24 * 3600 * 1000).toISOString(),
        freq_hz: Number((seedFundHz * 1.04).toFixed(2)),
      },
      { timestamp: nowIso, freq_hz: seedFundHz },
    ],
  };

  scans.unshift(newScan);
  saveStoredScans(scans);
  return newScan;
}

export function resetMockData(): ScanRecord[] {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(mockScans));
  return mockScans;
}
