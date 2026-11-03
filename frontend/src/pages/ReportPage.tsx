import React, { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getBridgeReport } from "../api";
import { BridgeReport, BridgeReportScan } from "../types";
import { formatLocaleDate, formatLocaleNumber } from "../i18n";

interface ReportPageProps {
  bridgeKey: string;
  initialReport?: BridgeReport;
}

function verdictText(verdict: BridgeReportScan["verdict"]): string {
  if (typeof verdict === "string") return verdict;
  return verdict?.level || "";
}

function scanDate(scan: BridgeReportScan): string | null {
  return scan.date ?? scan.created_at ?? null;
}

function frequencyFor(scan: BridgeReportScan): number | null {
  return scan.frequency_hz ?? scan.measurement?.dominant_frequency_hz ?? null;
}

function rangeFor(scan: BridgeReportScan): [number, number] | null {
  if (scan.frequency_range_hz) return scan.frequency_range_hz;
  if (scan.ci_low_hz != null && scan.ci_high_hz != null) return [scan.ci_low_hz, scan.ci_high_hz];
  return scan.measurement?.frequency_ci_hz ?? null;
}

function cableChangeFor(scan: BridgeReportScan): number | null {
  return scan.cable_tension_change_pct ?? scan.cable?.tension_change_pct ?? null;
}

function cableRangeFor(scan: BridgeReportScan): [number, number] | null {
  if (scan.cable?.tension_change_range_pct) return scan.cable.tension_change_range_pct;
  if (scan.cable_tension_change_low_pct != null && scan.cable_tension_change_high_pct != null) return [scan.cable_tension_change_low_pct, scan.cable_tension_change_high_pct];
  return null;
}

function noteTags(scan: BridgeReportScan, translate: (key: string, fallback: string) => string): string[] {
  const tags = scan.field_note_tags || (scan.field_notes || []).flatMap((note) => note.tags);
  return tags.map((tag) => `${translate(`result.tag_${tag.tag}`, tag.tag)}: ${tag.evidence}`);
}

export default function ReportPage({ bridgeKey, initialReport }: ReportPageProps) {
  const { t } = useTranslation();
  const [report, setReport] = useState<BridgeReport | null>(initialReport || null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (initialReport) {
      setReport(initialReport);
      return;
    }
    let active = true;
    getBridgeReport(bridgeKey).then((data) => {
      if (active) setReport(data);
    }).catch((err: unknown) => {
      if (active) setError(err instanceof Error ? err.message : t("report.load_error", "Could not load this report."));
    });
    return () => { active = false; };
  }, [bridgeKey, initialReport, t]);

  const chartData = useMemo(() => (report?.scans || []).map((scan) => ({
    date: scanDate(scan) ? formatLocaleDate(scanDate(scan), { dateStyle: "short" }) : "",
    frequency_hz: frequencyFor(scan),
  })).filter((row) => row.frequency_hz != null), [report]);

  if (!report) return <main className="print-report max-w-5xl mx-auto p-6 bg-white text-black">{error || t("report.loading", "Loading report...")}</main>;

  const bridge = report.bridge || {};
  const bridgeKeyValue = report.bridge_key || bridge.bridge_key || bridgeKey;
  const name = bridge.name || bridge.bridge_name || report.name || report.bridge_name || bridgeKeyValue;
  const city = bridge.city ?? report.city;
  const bridgeType = bridge.bridge_type ?? report.bridge_type;
  const location = bridge.location || (bridge.latitude != null && bridge.longitude != null ? { lat: bridge.latitude, lon: bridge.longitude } : null);
  const material = bridge.material ?? report.material;

  return <main className="print-report max-w-5xl mx-auto p-6 bg-white text-black space-y-6">
    <div className="print-hidden flex justify-end">
      <button type="button" onClick={() => window.print()} className="rounded border border-black px-4 py-2 font-semibold">{t("report.print", "Print or save as PDF")}</button>
    </div>
    <header className="border-b border-black pb-4">
      <h1 className="text-2xl font-bold">{t("report.page_title", "Bridge report")}</h1>
      <h2 className="mt-2 text-xl font-semibold">{name}</h2>
      <dl className="mt-3 grid grid-cols-2 gap-2 text-sm">
        <div><dt className="font-semibold">{t("report.bridge_key", "Bridge key")}</dt><dd>{bridgeKeyValue}</dd></div>
        {city && <div><dt className="font-semibold">{t("report.city", "Area")}</dt><dd>{city}</dd></div>}
        {bridgeType && <div><dt className="font-semibold">{t("report.bridge_type", "Bridge type")}</dt><dd>{bridgeType}</dd></div>}
        {material && <div><dt className="font-semibold">{t("report.material", "Material")}</dt><dd>{material}</dd></div>}
        {location && location.lat != null && location.lon != null && <div><dt className="font-semibold">{t("report.location", "Location")}</dt><dd>{formatLocaleNumber(location.lat, { maximumFractionDigits: 5 })}, {formatLocaleNumber(location.lon, { maximumFractionDigits: 5 })}</dd></div>}
      </dl>
    </header>

    <section>
      <h3 className="mb-2 text-lg font-bold">{t("report.scan_history", "Scan history")}</h3>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-left text-sm">
          <thead><tr className="border-b-2 border-black">
            <th className="p-2">{t("report.date", "Date")}</th>
            <th className="p-2">{t("report.verdict", "Reading")}</th>
            <th className="p-2">{t("report.frequency", "Frequency")}</th>
            <th className="p-2">{t("report.cable_change", "Cable change")}</th>
            <th className="p-2">{t("report.field_notes", "Field-note tags and evidence")}</th>
          </tr></thead>
          <tbody>{report.scans.map((scan, index) => {
            const frequency = frequencyFor(scan);
            const range = rangeFor(scan);
            const tags = noteTags(scan, (key, fallback) => t(key, fallback));
            const cableChange = cableChangeFor(scan);
            const cableRange = cableRangeFor(scan);
            const date = scanDate(scan);
            return <tr key={scan.scan_id || `${scan.created_at}-${index}`} className="border-b border-gray-400 align-top">
              <td className="p-2">{date ? formatLocaleDate(date, { dateStyle: "medium" }) : "?"}</td>
              <td className="p-2">{verdictText(scan.verdict) || "?"}</td>
              <td className="p-2">{frequency == null ? "?" : `${formatLocaleNumber(frequency)} Hz${range ? ` (${formatLocaleNumber(range[0])}?${formatLocaleNumber(range[1])} Hz)` : ""}`}</td>
              <td className="p-2">{cableChange == null ? "?" : `${formatLocaleNumber(cableChange)}%${cableRange ? ` (${formatLocaleNumber(cableRange[0])}% to ${formatLocaleNumber(cableRange[1])}%)` : ""}`}</td>
              <td className="p-2">{tags.length ? tags.join("; ") : "?"}</td>
            </tr>;
          })}{report.scans.length === 0 && <tr><td colSpan={5} className="p-3">{t("report.no_scans", "No scans are available.")}</td></tr>}</tbody>
        </table>
      </div>
    </section>

    {chartData.length > 0 && <section className="break-inside-avoid">
      <h3 className="mb-2 text-lg font-bold">{t("report.frequency_chart", "Frequency over time")}</h3>
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 8, right: 20, bottom: 8, left: 8 }}>
            <XAxis dataKey="date" stroke="#000" />
            <YAxis unit=" Hz" stroke="#000" />
            <Tooltip />
            <Line type="monotone" dataKey="frequency_hz" name={t("report.frequency", "Frequency")} stroke="#000" connectNulls={false} dot />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>}

    <p className="border-t border-black pt-3 text-sm">{report.disclaimer || t("common.safety_principle", "SETU does not assess safety. Flagged readings are for engineer review.")}</p>
  </main>;
}
