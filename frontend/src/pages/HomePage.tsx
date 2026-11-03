import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { getBridges, getShowcaseScans, getFileUrl } from "../api";
import { BridgeSummary, ScanResult } from "../types";
import KmStone from "../components/KmStone";

interface HomePageProps {
  onNavigate: (page: "home" | "scan" | "result" | "map" | "proof", scanId?: string) => void;
}

export default function HomePage({ onNavigate }: HomePageProps) {
  const { t } = useTranslation();
  const [realBridgeCount, setRealBridgeCount] = useState<number>(0);
  const [showcaseScan, setShowcaseScan] = useState<ScanResult | null>(null);
  const [showcaseLoaded, setShowcaseLoaded] = useState(false);

  useEffect(() => {
    // 1. Fetch real bridge count (hide if zero)
    getBridges()
      .then((data: BridgeSummary[]) => {
        const real = data.filter((b) => !b.is_demo && !b.is_sample_data);
        setRealBridgeCount(real.length);
      })
      .catch(() => {
        setRealBridgeCount(0);
      });

    getShowcaseScans()
      .then((scans) => {
        const scan = scans.find((item) => item.status === "complete" && (item.outputs?.magnified_video || item.outputs?.spectrum_plot));
        setShowcaseScan(scan || null);
      })
      .catch(() => setShowcaseScan(null))
      .finally(() => setShowcaseLoaded(true));
  }, []);

  const magnifiedUrl = showcaseScan?.outputs?.magnified_video
    ? getFileUrl(showcaseScan.scan_id, showcaseScan.outputs.magnified_video)
    : null;

  const spectrumUrl = showcaseScan?.outputs?.spectrum_plot
    ? getFileUrl(showcaseScan.scan_id, showcaseScan.outputs.spectrum_plot)
    : null;

  const verdictLabel = showcaseScan?.verdict?.level
    ? t(`result.verdict_${showcaseScan.verdict.level}`, showcaseScan.verdict.level.toUpperCase())
    : null;

  return (
    <div className="space-y-12 py-4 font-sans text-asphalt max-w-6xl mx-auto">
      {/* Left-aligned Asymmetric Hero */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start pt-2">
        {/* Left Column: Headline, Explanation, Action Buttons */}
        <div className="lg:col-span-7 space-y-5 text-left">
          <div className="space-y-3">
            <h1 className="text-3xl sm:text-4xl lg:text-5xl font-heading text-asphalt leading-[1.15]">
              {t("home.hero_headline", "Film a bridge for 30 seconds. SETU measures how it vibrates.")}
            </h1>
            <p className="text-base sm:text-lg text-asphalt-muted leading-relaxed font-sans max-w-xl">
              {t(
                "home.hero_explanation",
                "Mobile cameras track structural motion at sub-pixel scale. SETU separates bridge movement from camera shake to extract resonant frequencies."
              )}
            </p>
          </div>

          {/* Action Buttons: Highway Green primary & Border button */}
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 pt-2">
            <button
              type="button"
              onClick={() => onNavigate("scan")}
              className="inline-flex items-center justify-center px-6 py-3.5 bg-highway hover:bg-highway-dark text-white font-semibold text-sm rounded-[4px] transition focus-visible:outline-highway"
            >
              {t("home.cta_film", "Film a bridge")}
            </button>

          </div>

          <button
            type="button"
            onClick={() => onNavigate("proof")}
            className="inline-flex items-center justify-center px-6 py-3.5 bg-white hover:bg-concrete-light text-asphalt border border-concrete-rule font-medium text-sm rounded-[4px] transition focus-visible:outline-highway"
          >
            {t("home.proof_link", "See validation results")}
          </button>

        </div>

        {/* Right Column: Real Scan Display or Plain Text Explanation (Never a mockup) */}
        <div className="lg:col-span-5 flex flex-col items-start lg:items-end">
          {/* Milestone Motif (Shows real bridge count, hidden if zero) */}
          {realBridgeCount > 0 && (
            <div className="mb-4 self-start lg:self-end">
              <KmStone count={realBridgeCount} label="BRIDGES SCANNED" />
            </div>
          )}

          {showcaseScan ? (
            <div className="w-full bg-white border border-concrete-rule rounded-[4px] p-4 space-y-3">
              <div className="border-b border-concrete-rule pb-2">
                <div className="text-xs font-bold uppercase tracking-wide text-asphalt">
                  {t("home.real_scan_title", "Actual scan result")}
                </div>
                <div className="text-xs text-asphalt-muted">{showcaseScan.scan_id}</div>
              </div>
              {magnifiedUrl ? (
                <video src={magnifiedUrl} controls className="w-full h-44 bg-asphalt object-contain rounded-[4px] border border-concrete-rule" />
              ) : spectrumUrl ? (
                <img src={spectrumUrl} alt={t("result.spectrum_image_alt", "Vibration spectrum plot")} className="w-full h-36 bg-concrete-light object-contain rounded-[4px] border border-concrete-rule" />
              ) : null}
              <div className="flex items-center justify-between text-xs pt-1 border-t border-concrete-rule font-sans">
                <span className="text-asphalt-muted">{t("result.dominant_freq", "Dominant frequency")}</span>
                <span className="font-bold text-highway font-mono">
                  {showcaseScan.measurement?.dominant_frequency_hz != null
                    ? `${showcaseScan.measurement.dominant_frequency_hz.toFixed(2)} Hz`
                    : t("result.no_reliable_freq", "No reliable frequency detected")}
                </span>
              </div>
              {verdictLabel && <div className="flex items-center justify-between text-xs">
                <span className="text-asphalt-muted">{t("result.verdict", "Verdict")}</span>
                <span className="font-bold">{verdictLabel}</span>
              </div>}
              <button type="button" onClick={() => onNavigate("result", showcaseScan.scan_id)} className="text-xs font-semibold text-highway underline">
                {t("home.cta_real_scan", "See a real scan")}
              </button>
            </div>
          ) : showcaseLoaded ? (
            <p className="max-w-md text-sm text-asphalt-muted leading-relaxed">
              {t("home.real_scan_missing", "No completed scan is currently loaded. Film a bridge or choose an entry from the map to view real video and spectrum plots.")}
            </p>
          ) : (
            <p role="status" className="text-sm text-asphalt-muted">{t("common.loading", "Loading...")}</p>
          )}
        </div>
      </section>

      {/* Thin Horizontal Rule */}
      <hr className="border-t border-concrete-rule my-8" />

      {/* Three-step "How it works" in plain text separated by thin rules (NO card grids, NO cards in cards) */}
      <section className="space-y-6">
        <h2 className="text-2xl sm:text-3xl font-heading text-asphalt">
          {t("home.how_it_works_heading", "How it works")}
        </h2>

        <div className="divide-y divide-concrete-rule border-t border-b border-concrete-rule">
          {/* Step 1 */}
          <div className="py-4 sm:py-5 flex flex-col sm:flex-row sm:items-baseline gap-2 sm:gap-6">
            <span className="font-heading text-lg sm:text-xl text-asphalt font-bold sm:w-64 shrink-0">
              {t("home.step1_title", "1. Record 30 seconds of video")}
            </span>
            <p className="text-sm text-asphalt-muted leading-relaxed font-sans max-w-2xl">
              {t(
                "home.step1_desc",
                "Aim at a deck, pier, or stay cable. Rest the phone against a rigid railing or tripod to reduce hand shake."
              )}
            </p>
          </div>

          {/* Step 2 */}
          <div className="py-4 sm:py-5 flex flex-col sm:flex-row sm:items-baseline gap-2 sm:gap-6">
            <span className="font-heading text-lg sm:text-xl text-asphalt font-bold sm:w-64 shrink-0">
              {t("home.step2_title", "2. Sub-pixel motion tracking")}
            </span>
            <p className="text-sm text-asphalt-muted leading-relaxed font-sans max-w-2xl">
              {t(
                "home.step2_desc",
                "SAM tracks structural edges frame by frame against a stationary background to cancel camera movement."
              )}
            </p>
          </div>

          {/* Step 3 */}
          <div className="py-4 sm:py-5 flex flex-col sm:flex-row sm:items-baseline gap-2 sm:gap-6">
            <span className="font-heading text-lg sm:text-xl text-asphalt font-bold sm:w-64 shrink-0">
              {t("home.step3_title", "3. Spectral analysis")}
            </span>
            <p className="text-sm text-asphalt-muted leading-relaxed font-sans max-w-2xl">
              {t(
                "home.step3_desc",
                "Eulerian video magnification amplifies microscopic movement and reveals the dominant vibration frequency."
              )}
            </p>
          </div>
        </div>
      </section>

      {/* Engineering Principle Notice */}
      <section className="bg-white border-l-4 border-highway border-t border-r border-b border-concrete-rule p-4 text-xs text-asphalt space-y-1">
        <div className="font-bold text-highway uppercase tracking-wider text-[11px]">
          Engineering principle
        </div>
        <p className="text-asphalt-muted leading-relaxed">
          SETU measures physical vibration frequencies to assist structural engineers. It does not certify or declare any bridge safe.
        </p>
      </section>
    </div>
  );
}
