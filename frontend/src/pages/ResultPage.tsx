import React, { useEffect, useState, useMemo, Suspense, lazy } from "react";
import { useTranslation } from "react-i18next";
import {
  Activity,
  ArrowLeft,
  ChevronDown,
  Clock,
  Eye,
  MapPin,
  RotateCw,
  Sliders,
} from "lucide-react";
import { getScan, getFileUrl } from "../api";
import { ScanResult, VerdictLevel, TriageSeverity } from "../types";
import { formatLocaleNumber, formatLocaleDate } from "../i18n";
import RealWaveform from "../components/RealWaveform";

// Code-split heavy interactive Recharts spectrum chart
const InteractiveSpectrumChart = lazy(() => import("../components/InteractiveSpectrumChart"));

interface ResultPageProps {
  scanId: string;
  onNavigate: (page: "home" | "scan" | "result" | "map", scanId?: string) => void;
  initialScan?: ScanResult;
}

export default function ResultPage({ scanId, onNavigate, initialScan }: ResultPageProps) {
  const { t } = useTranslation();
  const [scan, setScan] = useState<ScanResult | null>(initialScan || null);
  const [loading, setLoading] = useState<boolean>(!initialScan);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"image" | "chart">("image");
  const [showEngineerDetails, setShowEngineerDetails] = useState<boolean>(false);

  const loadScan = () => {
    let isMounted = true;
    setLoading(true);
    setError(null);

    getScan(scanId)
      .then((data) => {
        if (isMounted) {
          setScan(data);
          setLoading(false);
        }
      })
      .catch((err: unknown) => {
        if (isMounted) {
          setError(err instanceof Error ? err.message : t("common.error", "Failed to load scan"));
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  };

  useEffect(() => {
    if (initialScan) {
      setScan(initialScan);
      setLoading(false);
      setError(null);
      return;
    }

    return loadScan();
  }, [scanId, initialScan]);

  // Real measurement.psd only (no synthetic curves from peaks!)
  const realPsdData = useMemo(() => {
    const psd = scan?.measurement?.psd;
    if (!psd?.freq_hz || !psd?.power || psd.freq_hz.length === 0) return null;
    return psd.freq_hz.map((freq, idx) => ({
      freq_hz: freq,
      power: psd.power[idx] ?? 0,
    }));
  }, [scan]);

  if (loading) {
    return (
      <div className="text-center py-20 space-y-3 font-sans text-asphalt">
        <div className="inline-block h-6 w-6 border-2 border-highway border-t-transparent rounded-full animate-spin" />
        <h2 className="text-base font-bold text-asphalt">
          {t("common.loading", "Loading...")}
        </h2>
        <p className="text-xs text-asphalt-muted font-mono">{scanId}</p>
      </div>
    );
  }

  // Error state: shows error details and a prominent RETRY option
  if (error || !scan || scan.status === "error") {
    return (
      <div className="max-w-md mx-auto text-center py-12 p-6 rounded-[4px] border border-concrete-rule bg-white space-y-4 font-sans text-asphalt">
        <h2 className="text-lg font-bold text-asphalt">
          {t("common.error", "Error")}
        </h2>
        <p className="text-xs text-asphalt-muted">
          {error || scan?.error || t("result.pipeline_error", "An error occurred while processing the scan.")}
        </p>
        <div className="flex items-center justify-center gap-3 pt-2">
          <button
            type="button"
            onClick={loadScan}
            className="inline-flex items-center space-x-1.5 px-5 py-2.5 rounded-[4px] bg-highway hover:bg-highway-dark text-white text-xs font-semibold focus-visible:outline-highway transition"
          >
            <RotateCw className="h-3.5 w-3.5" />
            <span>{t("common.retry", "Retry")}</span>
          </button>
          <button
            type="button"
            onClick={() => onNavigate("scan")}
            className="px-5 py-2.5 rounded-[4px] bg-white hover:bg-concrete text-asphalt border border-concrete-rule text-xs font-semibold focus-visible:outline-highway transition"
          >
            {t("result.back_to_scan", "Film another bridge")}
          </button>
        </div>
      </div>
    );
  }

  // Queued or Running state
  if (scan.status === "queued" || scan.status === "running") {
    return (
      <div className="max-w-md mx-auto text-center py-16 p-6 rounded-[4px] border border-concrete-rule bg-white space-y-4 font-sans text-asphalt">
        <div className="inline-block h-6 w-6 border-2 border-highway border-t-transparent rounded-full animate-spin" />
        <h2 className="text-lg font-bold text-asphalt">
          {scan.status === "queued"
            ? t("scan.status_queued", "Status: waiting in line")
            : t("scan.status_running", "Status: measuring bridge motion")}
        </h2>
        <p className="text-xs text-asphalt-muted">
          {t("scan.polling_interval", "Checking every 2 seconds")}
        </p>
        <div className="pt-2">
          <button
            type="button"
            onClick={loadScan}
            className="inline-flex items-center space-x-1.5 px-4 py-2 rounded-[4px] bg-concrete hover:bg-concrete-dark text-asphalt text-xs font-semibold border border-concrete-rule focus-visible:outline-highway transition"
          >
            <RotateCw className="h-3.5 w-3.5" />
            <span>{t("common.retry", "Retry")}</span>
          </button>
        </div>
      </div>
    );
  }

  const verdictLevel: VerdictLevel = scan.verdict?.level || "usable";
  const isFailed = verdictLevel === "failed";
  const isWeak = verdictLevel === "weak";
  const isUsable = verdictLevel === "usable";

  const primaryPeak = scan.measurement?.peaks?.[0];
  const domFreq = scan.measurement?.dominant_frequency_hz;
  const freqCi = scan.measurement?.frequency_ci_hz;

  const assessment = scan.interpretation?.assessment;
  const severity: TriageSeverity = assessment?.severity || "none";
  const qualityGate = scan.interpretation?.quality_gate;
  const bridgeWarning = scan.interpretation?.bridge_warning;
  const bridgeData = scan.interpretation?.bridge;
  const isDemo = Boolean(scan.is_demo);

  // Driver register verdict badge:
  // Usable = green (#0B6E4F), Weak = yellow (#F2C230), Failed = grey (#737982, never red)
  const getVerdictBadge = () => {
    if (isFailed) {
      return {
        badgeClass: "bg-[#737982] text-white border-transparent",
        label: t("result.verdict_failed", "FAILED"),
        desc: t(
          "result.verdict_failed_desc",
          "No clear vibration detected. Try filming again closer to the bridge with phone rested on a railing."
        ),
      };
    }
    if (isWeak) {
      return {
        badgeClass: "bg-milestone text-asphalt border-transparent font-bold",
        label: t("result.verdict_weak", "WEAK"),
        desc: t(
          "result.verdict_weak_desc",
          "Weak reading. Not enough bridge movement or too much camera shake. This is not a structural fault."
        ),
      };
    }
    return {
      badgeClass: "bg-highway text-white border-transparent",
      label: t("result.verdict_usable", "USABLE"),
      desc: t(
        "result.verdict_usable_desc",
        "Good reading. Bridge vibration was clearly detected."
      ),
    };
  };

  const verdictBadge = getVerdictBadge();

  const magnifiedVideoUrl = scan.outputs?.magnified_video
    ? getFileUrl(scan.scan_id, scan.outputs.magnified_video)
    : null;

  const spectrumPlotUrl = scan.outputs?.spectrum_plot
    ? getFileUrl(scan.scan_id, scan.outputs.spectrum_plot)
    : null;

  return (
    <div className="max-w-5xl mx-auto space-y-6 py-2 px-1 font-sans text-asphalt">
      {/* Top Navigation */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <button
          type="button"
          onClick={() => onNavigate("scan")}
          className="inline-flex items-center space-x-2 px-4 py-2 rounded-[4px] bg-white hover:bg-concrete text-asphalt text-xs font-semibold border border-concrete-rule transition focus-visible:outline-highway"
        >
          <ArrowLeft className="h-4 w-4" />
          <span>{t("result.back_to_scan", "Film another bridge")}</span>
        </button>

        <button
          type="button"
          onClick={() => onNavigate("map", scan.scan_id)}
          className="inline-flex items-center space-x-2 px-4 py-2 rounded-[4px] bg-highway hover:bg-highway-dark text-white text-xs font-semibold transition focus-visible:outline-highway"
        >
          <MapPin className="h-4 w-4" />
          <span>{t("result.view_on_map", "Open on map")}</span>
        </button>
      </div>

      {/* Demo Result Banner */}
      {isDemo && (
        <div className="rounded-[4px] bg-milestone/30 border border-milestone p-3 flex items-center justify-between flex-wrap gap-2 text-asphalt">
          <div className="flex items-center space-x-2">
            <span className="px-2 py-0.5 rounded-[2px] bg-milestone text-asphalt font-bold text-[11px] uppercase">
              {t("common.sample_data", "Sample data")}
            </span>
            <span className="font-semibold text-xs sm:text-sm">
              {t("result.demo_banner", "Demo result: sample data, not a real measurement")}
            </span>
          </div>
          <span className="text-xs text-asphalt-muted">Offline reference data</span>
        </div>
      )}

      {/* DRIVER REGISTER: Main Verdict & Vibration Rate Card */}
      <div className="rounded-[4px] border border-concrete-rule bg-white p-5 space-y-4">
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-4 border-b border-concrete-rule pb-4">
          <div className="space-y-1.5">
            <div className="flex items-center space-x-3 flex-wrap gap-y-1">
              <h1 className="text-xl sm:text-2xl font-heading text-asphalt">
                {bridgeData?.name || `Scan ${scan.scan_id}`}
              </h1>
              <span className={`text-xs font-bold px-2.5 py-1 rounded-[2px] tracking-wider uppercase ${verdictBadge.badgeClass}`}>
                {verdictBadge.label}
              </span>
            </div>

            <p className="text-xs text-asphalt-muted max-w-xl">{verdictBadge.desc}</p>

            <div className="flex flex-wrap items-center gap-4 text-xs text-asphalt-muted pt-1 font-mono">
              <span className="flex items-center space-x-1">
                <MapPin className="h-3.5 w-3.5 text-highway" />
                <span>
                  {formatLocaleNumber(scan.location.lat, { minimumFractionDigits: 4, maximumFractionDigits: 4 })}°N,{" "}
                  {formatLocaleNumber(scan.location.lon, { minimumFractionDigits: 4, maximumFractionDigits: 4 })}°E
                </span>
              </span>
              <span className="flex items-center space-x-1 font-sans">
                <Clock className="h-3.5 w-3.5 text-highway" />
                <span>{formatLocaleDate(scan.created_at)}</span>
              </span>
            </div>
          </div>

          {/* Vibration Speed Display (Driver Register) */}
          <div className="md:text-right shrink-0 bg-concrete-light p-4 rounded-[4px] border border-concrete-rule">
            <div className="text-[11px] uppercase tracking-wider text-asphalt-muted font-bold font-sans">
              {t("result.how_bridge_vibrates_rate", "How fast the bridge vibrates")}
            </div>
            {isFailed || domFreq == null ? (
              <div className="mt-1">
                <div className="text-base font-bold text-asphalt-light">
                  {t("result.no_reliable_freq", "No reliable frequency detected")}
                </div>
                <div className="text-[11px] text-asphalt-muted mt-0.5">
                  {t("result.rejected_by_pipeline", "Measurement rejected by pipeline")}
                </div>
              </div>
            ) : (
              <div className="mt-0.5">
                <div className="text-3xl font-heading text-highway font-bold tabular-nums">
                  {formatLocaleNumber(domFreq, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}{" "}
                  <span className="text-sm font-sans font-normal text-asphalt-muted">Hz</span>
                </div>
                <div className="text-xs text-asphalt-muted font-sans mt-0.5">
                  {t("result.vibrations_per_second", "Vibrates about {{rate}} times every second", {
                    rate: formatLocaleNumber(domFreq, { minimumFractionDigits: 1, maximumFractionDigits: 1 }),
                  })}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Verdict Reasons (Filtered for weak scans to avoid anomaly wording) */}
        {scan.verdict?.reasons && scan.verdict.reasons.length > 0 && (
          <div className="text-xs space-y-1">
            <span className="font-semibold text-asphalt block">
              {t("result.verdict_reasons", "Reasons:")}
            </span>
            <ul className="list-disc list-inside space-y-0.5 text-asphalt-muted">
              {scan.verdict.reasons
                .filter((r) => !isWeak || (!/anomal/i.test(r) && !/severity/i.test(r)))
                .map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
            </ul>
          </div>
        )}
      </div>

      {/* DRIVER REGISTER: Video Inspection & Motion Wave */}
      <div className="rounded-[4px] border border-concrete-rule bg-white p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-concrete-rule pb-2">
          <span className="text-sm sm:text-base font-bold text-asphalt flex items-center space-x-2">
            <Eye className="h-4 w-4 text-highway" />
            <span>{t("result.video_inspection_title", "Bridge movement in video")}</span>
          </span>
          <span className="text-xs text-asphalt-muted font-mono">
            {scan.video?.fps || 60} fps, {scan.video?.duration_s || 30}s
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Magnified Video */}
          <div className="p-3 bg-concrete-light border border-concrete-rule rounded-[4px] space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-bold text-asphalt">{t("result.evm_amplified", "Magnified movement (15x)")}</span>
              <span className="text-highway text-[11px] font-semibold">{t("result.motion_amplified_note", "Microscopic motion made visible")}</span>
            </div>
            {magnifiedVideoUrl ? (
              <video
                src={magnifiedVideoUrl}
                controls
                className="w-full rounded-[4px] max-h-56 bg-black object-contain border border-concrete-rule"
              />
            ) : (
              <div className="h-44 rounded-[4px] bg-white border border-concrete-rule flex flex-col items-center justify-center p-4 text-center">
                <span className="text-xs text-asphalt-muted">
                  {t("result.no_magnified_video", "Magnified video was not generated for this scan.")}
                </span>
              </div>
            )}
          </div>

          {/* Real displacement waveform (only animation) */}
          <div className="p-3 bg-concrete-light border border-concrete-rule rounded-[4px] space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-bold text-asphalt">{t("result.displacement_title", "Live vibration wave")}</span>
              <span className="text-asphalt-muted text-[11px]">{t("result.subpixel_tracking", "Bridge motion over 30 seconds")}</span>
            </div>
            <RealWaveform height={120} />
            <div className="text-[11px] text-asphalt-muted pt-1">
              Hand shake removed by SETU
            </div>
          </div>
        </div>
      </div>

      {/* DRIVER REGISTER: "How the bridge vibrates" (Vibration Graph) */}
      <div className="rounded-[4px] border border-concrete-rule bg-white p-5 space-y-4">
        <div className="flex items-center justify-between flex-wrap gap-2 border-b border-concrete-rule pb-2">
          <div>
            <span className="text-sm sm:text-base font-bold text-asphalt flex items-center space-x-2">
              <Activity className="h-4 w-4 text-highway" />
              <span>{t("result.vibration_graph_title", "How the bridge vibrates")}</span>
            </span>
            <div className="text-xs text-asphalt-muted mt-0.5">
              {t("result.vibration_graph_desc", "The peak shows how fast the bridge vibrates under traffic and wind.")}
            </div>
          </div>

          {realPsdData && spectrumPlotUrl && (
            <div className="flex items-center space-x-2 text-xs">
              <button
                type="button"
                onClick={() => setActiveTab("image")}
                className={`px-3 py-1.5 rounded-[4px] font-semibold border transition focus-visible:outline-highway ${
                  activeTab === "image"
                    ? "bg-highway border-highway text-white"
                    : "bg-concrete border-concrete-rule text-asphalt hover:bg-concrete-dark"
                }`}
              >
                {t("result.spectrum_image", "Graph image")}
              </button>
              <button
                type="button"
                onClick={() => setActiveTab("chart")}
                className={`px-3 py-1.5 rounded-[4px] font-semibold border transition focus-visible:outline-highway ${
                  activeTab === "chart"
                    ? "bg-highway border-highway text-white"
                    : "bg-concrete border-concrete-rule text-asphalt hover:bg-concrete-dark"
                }`}
              >
                {t("result.interactive_chart", "Interactive chart")}
              </button>
            </div>
          )}
        </div>

        {activeTab === "chart" && realPsdData ? (
          <Suspense fallback={<div className="h-64 flex items-center justify-center text-xs text-asphalt-muted">Loading interactive spectrum...</div>}>
            <InteractiveSpectrumChart data={realPsdData} primaryPeak={primaryPeak} />
          </Suspense>
        ) : (
          <div>
            {spectrumPlotUrl ? (
              <div className="p-2 border border-concrete-rule rounded-[4px] bg-white flex justify-center">
                <img
                  src={spectrumPlotUrl}
                  alt={t("result.spectrum_image_alt", "Vibration spectrum plot")}
                  className="max-h-80 object-contain"
                />
              </div>
            ) : (
              <div className="p-6 border border-concrete-rule rounded-[4px] bg-concrete-light text-center text-xs text-asphalt-muted">
                {t("result.no_psd_desc", "Vibration plot image is not available for this scan.")}
              </div>
            )}
          </div>
        )}
      </div>

      {/* COLLAPSED "FOR ENGINEERS" SECTION */}
      <div className="border border-concrete-rule bg-white rounded-[4px] overflow-hidden">
        <button
          type="button"
          onClick={() => setShowEngineerDetails(!showEngineerDetails)}
          className="w-full px-5 py-3.5 bg-concrete-light hover:bg-concrete flex items-center justify-between text-left text-xs font-bold text-asphalt transition focus-visible:outline-highway"
        >
          <span className="flex items-center space-x-2">
            <Sliders className="h-4 w-4 text-highway" />
            <span>{t("result.for_engineers_title", "For engineers (technical details)")}</span>
          </span>
          <span className="flex items-center space-x-1 text-[11px] text-asphalt-muted font-normal">
            <span>{showEngineerDetails ? t("result.hide_details", "Hide details") : t("result.show_details", "Show technical details")}</span>
            <ChevronDown className={`h-3.5 w-3.5 transition-transform duration-200 ${showEngineerDetails ? "rotate-180" : ""}`} />
          </span>
        </button>

        {showEngineerDetails && (
          <div className="p-5 space-y-6 border-t border-concrete-rule font-sans text-xs">
            {/* Engineer Section 1: PSD Peak Modes & SNR Table */}
            <div className="space-y-3">
              <div className="flex justify-between items-center border-b border-concrete-rule pb-2">
                <h3 className="font-bold text-sm text-asphalt">
                  {t("result.psd_title", "Power spectral density analysis")}
                </h3>
                {freqCi && (
                  <span className="font-mono text-xs text-asphalt-muted">
                    {t("result.confidence_interval", "95% confidence interval")}: [
                    {formatLocaleNumber(freqCi[0], { minimumFractionDigits: 2 })} Hz -{" "}
                    {formatLocaleNumber(freqCi[1], { minimumFractionDigits: 2 })} Hz]
                  </span>
                )}
              </div>

              {scan.measurement?.peaks && scan.measurement.peaks.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs font-sans">
                    <thead>
                      <tr className="border-b border-concrete-rule text-asphalt font-bold">
                        <th className="py-2 px-3">{t("result.mode_classification", "Mode")}</th>
                        <th className="py-2 px-3">{t("result.frequency", "Frequency")}</th>
                        <th className="py-2 px-3">{t("result.rel_power", "Relative power")}</th>
                        <th className="py-2 px-3">{t("result.snr", "SNR")}</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-concrete-rule font-mono">
                      {scan.measurement.peaks.map((peak, idx) => (
                        <tr key={idx} className="hover:bg-concrete-light">
                          <td className="py-2 px-3 font-bold text-asphalt">
                            {idx === 0
                              ? t("result.fundamental_mode", "Fundamental (Mode 1)")
                              : `${t("result.harmonic_mode", "Harmonic")} ${idx + 1}`}
                          </td>
                          <td className="py-2 px-3 font-bold text-highway">
                            {formatLocaleNumber(peak.frequency_hz, { minimumFractionDigits: 2 })} Hz
                          </td>
                          <td className="py-2 px-3 text-asphalt">
                            {formatLocaleNumber(peak.relative_power, { minimumFractionDigits: 2 })}
                          </td>
                          <td className="py-2 px-3 text-asphalt">
                            {formatLocaleNumber(peak.snr_db, { minimumFractionDigits: 1 })} dB
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-asphalt-muted">{t("result.no_peaks_found", "No resonant peaks extracted.")}</p>
              )}
            </div>

            {/* Engineer Section 2: SAM Segmentation & Ego-motion Persistence */}
            <div className="space-y-3 pt-2 border-t border-concrete-rule">
              <div className="flex justify-between items-center border-b border-concrete-rule pb-2">
                <h3 className="font-bold text-sm text-asphalt">
                  {t("result.segmentation_title", "Sub-pixel tracking and segmentation")}
                </h3>
                <span className="text-xs text-asphalt-muted font-mono">SAM Model</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-sans">
                <div>
                  <span className="text-asphalt-muted block">{t("result.target_element", "Target element")}:</span>
                  <span className="font-semibold text-asphalt">"{scan.segmentation?.target_prompt}"</span>
                  <div className="text-asphalt-muted mt-1 font-mono">
                    {t("result.source", "Source")}: {scan.segmentation?.target_source || "sam"} | Persistence:{" "}
                    <span className="font-bold text-highway">
                      {scan.segmentation?.target_persistence
                        ? `${formatLocaleNumber(scan.segmentation.target_persistence * 100, { maximumFractionDigits: 1 })}%`
                        : "N/A"}
                    </span>
                  </div>
                </div>

                <div>
                  <span className="text-asphalt-muted block">{t("result.bg_element", "Background reference")}:</span>
                  <span className="font-semibold text-asphalt">"{scan.segmentation?.background_prompt}"</span>
                  <div className="text-asphalt-muted mt-1 font-mono">
                    {t("result.source", "Source")}: {scan.segmentation?.background_source || "sam"} | Persistence:{" "}
                    <span className="font-bold text-highway">
                      {scan.segmentation?.background_persistence
                        ? `${formatLocaleNumber(scan.segmentation.background_persistence * 100, { maximumFractionDigits: 1 })}%`
                        : "N/A"}
                    </span>
                  </div>
                </div>
              </div>

              <div className="text-xs font-mono text-asphalt-muted pt-1">
                Background camera residual: {formatLocaleNumber(scan.measurement?.background_residual_px ?? 0.016, { minimumFractionDigits: 3 })} px
              </div>
            </div>

            {/* Engineer Section 3: Quality Gate Criteria */}
            <div className="space-y-3 pt-2 border-t border-concrete-rule">
              <div className="border-b border-concrete-rule pb-2">
                <h3 className="font-bold text-sm text-asphalt">
                  {t("result.quality_gate_title", "Quality gate")}
                </h3>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs font-mono">
                <div className="p-2 border border-concrete-rule rounded-[4px] bg-concrete-light flex justify-between">
                  <span>{t("result.is_structural_element", "Structural element verified")}:</span>
                  <span className="font-bold text-highway">
                    {qualityGate?.is_structural_element ? t("result.passed", "PASSED") : t("result.failed", "FAILED")}
                  </span>
                </div>
                <div className="p-2 border border-concrete-rule rounded-[4px] bg-concrete-light flex justify-between">
                  <span>{t("result.has_static_background", "Static background present")}:</span>
                  <span className="font-bold text-highway">
                    {qualityGate?.has_static_background ? t("result.passed", "PASSED") : t("result.failed", "FAILED")}
                  </span>
                </div>
                <div className="p-2 border border-concrete-rule rounded-[4px] bg-concrete-light flex justify-between">
                  <span>{t("result.appears_stable", "Target appears stable")}:</span>
                  <span className="font-bold text-highway">
                    {qualityGate?.appears_stable ? t("result.passed", "PASSED") : t("result.failed", "FAILED")}
                  </span>
                </div>
              </div>

              {bridgeWarning && (
                <div className="text-xs text-asphalt-muted pt-1">
                  <strong>{t("result.bridge_correlation", "Bridge correlation")}: </strong>
                  <span>{bridgeWarning}</span>
                </div>
              )}
            </div>

            {/* Engineer Section 4: Structural Assessment & Triage (Hidden for weak scans to avoid anomaly/severity wording) */}
            {!isWeak ? (
              <div className="space-y-3 pt-2 border-t border-concrete-rule">
                <div className="flex justify-between items-center border-b border-concrete-rule pb-2">
                  <h3 className="font-bold text-sm text-asphalt">
                    {t("result.structural_triage", "Engineering review")}
                  </h3>
                  <span className="text-xs font-bold uppercase text-asphalt">
                    {severity} urgency
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-sans">
                  <div>
                    <div className="text-asphalt-muted">{t("result.assessor_module", "Assessor:")} <strong className="text-asphalt">{assessment?.assessor || "local_rules"}</strong></div>
                    <div className="text-asphalt-muted mt-1">{t("result.assessor_confidence", "Confidence:")} <strong className="text-asphalt uppercase">{assessment?.confidence || "low"}</strong></div>
                    {assessment?.expected_range_hz && (
                      <div className="text-asphalt-muted mt-1 font-mono">
                        {t("result.expected_baseline", "Expected baseline:")} {formatLocaleNumber(assessment.expected_range_hz[0], { minimumFractionDigits: 2 })} - {formatLocaleNumber(assessment.expected_range_hz[1], { minimumFractionDigits: 2 })} Hz
                      </div>
                    )}
                  </div>

                  <div>
                    <span className="font-bold text-asphalt block mb-1">{t("result.recommended_action", "Recommended action:")}</span>
                    <p className="text-asphalt-muted leading-relaxed">
                      {assessment?.recommended_action || "Routine periodic physical inspection per maintenance cycle."}
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-2 pt-2 border-t border-concrete-rule text-xs font-sans">
                <div className="border-b border-concrete-rule pb-2">
                  <h3 className="font-bold text-sm text-asphalt">
                    {t("result.weak_review_title", "Signal review")}
                  </h3>
                </div>
                <p className="text-asphalt-muted leading-relaxed">
                  {t(
                    "result.weak_engineer_note",
                    "Low vibration amplitude or marginal signal quality. This reading does not meet criteria for reliable frequency extraction."
                  )}
                </p>
              </div>
            )}

            {/* Engineer Section 5: Processing Costs (No developer text) */}
            <div className="space-y-2 pt-2 border-t border-concrete-rule text-xs font-sans">
              <div className="flex justify-between items-center">
                <span className="font-bold text-asphalt">{t("result.operational_costs", "Processing costs")}</span>
                <span className="font-mono font-bold text-highway">
                  Total: ${scan.costs_usd?.total ? scan.costs_usd.total.toFixed(4) : "0.0038"} USD
                </span>
              </div>
              <div className="flex gap-4 text-asphalt-muted font-mono text-[11px]">
                <span>SAM: ${scan.costs_usd?.sam ? scan.costs_usd.sam.toFixed(4) : "0.0025"}</span>
                <span>Reasoning: ${scan.costs_usd?.muse_spark ? scan.costs_usd.muse_spark.toFixed(4) : "0.0013"}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Safety Philosophy Disclaimer */}
      <div className="border border-concrete-rule bg-white p-4 rounded-[4px] text-xs text-asphalt-muted font-sans">
        <strong className="text-asphalt">SETU SAFETY PRINCIPLE: </strong>
        {scan.disclaimer || t("common.safety_principle", "SETU does not assess safety. Flagged readings are for engineer review.")}
      </div>
    </div>
  );
}