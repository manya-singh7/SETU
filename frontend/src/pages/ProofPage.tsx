import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { formatLocaleNumber } from "../i18n";

interface ProofPoint {
  frequency_hz: number;
  normalized_power: number;
}

interface ProofComparison {
  title?: string;
  camera_frequency_hz: number;
  camera_frequency_ci_hz?: [number, number] | null;
  accelerometer_frequency_hz: number;
  accelerometer_frequency_ci_hz?: [number, number] | null;
  camera_spectrum?: { available?: boolean; points?: ProofPoint[] } | ProofPoint[];
  accelerometer_spectrum?: { available?: boolean; points?: ProofPoint[] } | ProofPoint[];
  difference_hz: number;
  difference_pct: number;
  agreement?: boolean;
  is_agreement?: boolean;
  verdict?: string;
}

interface ValidationIndexEntry {
  file?: string;
  path?: string;
  title?: string;
}

function comparisonPoints(value: ProofComparison["camera_spectrum"]): ProofPoint[] {
  const points = Array.isArray(value) ? value : value?.points || [];
  return points.filter((point) => Number.isFinite(point.frequency_hz) && Number.isFinite(point.normalized_power) && point.frequency_hz >= 0 && point.frequency_hz <= 30 && point.normalized_power >= 0).slice(0, 300);
}

function entriesFromIndex(index: unknown): ValidationIndexEntry[] {
  const raw = Array.isArray(index) ? index : (index as { results?: unknown; files?: unknown; comparisons?: unknown } | null)?.results ?? (index as { files?: unknown; comparisons?: unknown } | null)?.files ?? (index as { comparisons?: unknown } | null)?.comparisons;
  if (!Array.isArray(raw)) return [];
  return raw.flatMap((entry): ValidationIndexEntry[] => {
    if (typeof entry === "string") return [{ file: entry }];
    if (!entry || typeof entry !== "object") return [];
    const item = entry as ValidationIndexEntry;
    return item.file || item.path ? [item] : [];
  });
}

export default function ProofPage() {
  const { t } = useTranslation();
  const [results, setResults] = useState<Array<{ item: ProofComparison; title?: string }>>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const response = await fetch("/validation/index.json");
        if (!response.ok) throw new Error("index unavailable");
        const index = entriesFromIndex(await response.json());
        const loaded = await Promise.all(index.map(async (entry) => {
          const file = entry.file || entry.path || "";
          if (!file || file.split(/[\\/]/).includes("..")) return null;
          const url = file.startsWith("/validation/") ? file : `/validation/${file.replace(/^\/+/, "")}`;
          try {
            const resultResponse = await fetch(url);
            if (!resultResponse.ok) return null;
            const item = await resultResponse.json() as ProofComparison;
            if (!Number.isFinite(item.camera_frequency_hz) || !Number.isFinite(item.accelerometer_frequency_hz) || !Number.isFinite(item.difference_pct)) return null;
            return { item, title: entry.title };
          } catch {
            return null;
          }
        }));
        if (active) setResults(loaded.filter((result): result is { item: ProofComparison; title: string | undefined } => result !== null));
      } catch {
        if (active) setLoadError(true);
      } finally {
        if (active) setLoading(false);
      }
    };
    void load();
    return () => { active = false; };
  }, []);

  return <div className="max-w-5xl mx-auto space-y-6 py-3 font-sans text-asphalt">
    <header className="border-b border-concrete-rule pb-3">
      <h1 className="text-2xl sm:text-3xl font-heading">{t("proof.title", "Validation proof")}</h1>
    </header>
    {loading ? <p role="status">{t("proof.loading", "Loading validation results...")}</p>
      : results.length === 0 ? <p className="rounded border border-concrete-rule bg-white p-5 text-sm">{loadError ? t("proof.load_error", "Validation results are not available yet.") : t("proof.coming_soon", "Validation results coming soon")}</p>
      : <div className="space-y-6">{results.map(({ item, title }, index) => {
        const cameraPoints = comparisonPoints(item.camera_spectrum);
        const accelerometerPoints = comparisonPoints(item.accelerometer_spectrum);
        const agreement = item.agreement ?? item.is_agreement ?? item.verdict?.toUpperCase() === "PASS";
        const rangeText = (range?: [number, number] | null) => range ? ` (${formatLocaleNumber(range[0])}\u2013${formatLocaleNumber(range[1])} Hz)` : "";
        return <article key={`${title || "proof"}-${index}`} className="rounded border border-concrete-rule bg-white p-4 sm:p-5 space-y-4">
          <h2 className="text-lg font-bold">{title || item.title || `${t("proof.comparison", "Sensor comparison")} ${index + 1}`}</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
            <p><strong>{t("proof.camera", "Camera frequency")}:</strong> {formatLocaleNumber(item.camera_frequency_hz)} Hz{rangeText(item.camera_frequency_ci_hz)}</p>
            <p><strong>{t("proof.accelerometer", "Accelerometer frequency")}:</strong> {formatLocaleNumber(item.accelerometer_frequency_hz)} Hz{rangeText(item.accelerometer_frequency_ci_hz)}</p>
          </div>
          <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-2 border-y border-concrete-rule py-3">
            <div><div className="text-xs font-semibold uppercase text-asphalt-muted">{t("proof.difference", "Difference")}</div><div className="text-3xl font-bold tabular-nums">{formatLocaleNumber(item.difference_pct, { maximumFractionDigits: 2 })}%</div><div className="text-xs text-asphalt-muted">{formatLocaleNumber(item.difference_hz, { maximumFractionDigits: 3 })} Hz</div></div>
            <p className="font-semibold">{agreement ? t("proof.agreement", "The sensors agree within the comparison criteria") : t("proof.disagreement", "The readings differ beyond the comparison criteria")}</p>
          </div>
          {(cameraPoints.length > 0 || accelerometerPoints.length > 0) ? <section aria-label={t("proof.spectra", "Normalized spectra")}>
            <h3 className="mb-2 text-sm font-bold">{t("proof.spectra", "Normalized spectra")}</h3>
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 8, right: 18, bottom: 18, left: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="frequency_hz" name={t("proof.frequency_axis", "Frequency")} unit=" Hz" domain={[0, 30]} />
                  <YAxis type="number" dataKey="normalized_power" name={t("proof.normalized_power", "Normalized power")} domain={[0, 1]} />
                  <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                  <Legend />
                  {cameraPoints.length > 0 && <Scatter name={t("proof.camera", "Camera frequency")} data={cameraPoints} fill="#d95f02" line={{ stroke: "#d95f02", strokeWidth: 2 }} lineType="joint" />}
                  {accelerometerPoints.length > 0 && <Scatter name={t("proof.accelerometer", "Accelerometer frequency")} data={accelerometerPoints} fill="#1f77b4" line={{ stroke: "#1f77b4", strokeWidth: 2 }} lineType="joint" />}
                </ScatterChart>
              </ResponsiveContainer>
            </div>
            {(cameraPoints.length === 0 || accelerometerPoints.length === 0) && <p className="text-xs text-asphalt-muted">{t("proof.spectrum_unavailable", "A spectrum was not included for one of these sensors.")}</p>}
          </section> : <p className="text-xs text-asphalt-muted">{t("proof.spectra_unavailable", "Spectra were not included in this comparison.")}</p>}
        </article>;
      })}</div>}
  </div>;
}
