import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { getBridges, getScan, getFileUrl } from "../api";
import { BridgeSummary, ScanResult } from "../types";
import KmStone from "../components/KmStone";
import RealWaveform from "../components/RealWaveform";

interface HomePageProps {
  onNavigate: (page: "home" | "scan" | "result" | "map" | "proof", scanId?: string) => void;
}

export default function HomePage({ onNavigate }: HomePageProps) {
  const { t } = useTranslation();
  const [realBridgeCount, setRealBridgeCount] = useState<number>(0);
  const [sampleScan, setSampleScan] = useState<ScanResult | null>(null);
  const [scanLoadFailed, setScanLoadFailed] = useState<boolean>(false);

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

    // 2. Fetch a real scan result for the asymmetric hero
    getScan("scan_20261103_a1b2c3")
      .then((scanData) => {
        if (scanData && scanData.status === "complete") {
          setSampleScan(scanData);
        } else {
          setScanLoadFailed(true);
        }
      })
      .catch(() => {
        setScanLoadFailed(true);
      });
  }, []);

  const magnifiedUrl = sampleScan?.outputs?.magnified_video
    ? getFileUrl(sampleScan.scan_id, sampleScan.outputs.magnified_video)
    : null;

  const spectrumUrl = sampleScan?.outputs?.spectrum_plot
    ? getFileUrl(sampleScan.scan_id, sampleScan.outputs.spectrum_plot)
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

            <button
              type="button"
              onClick={() => onNavigate("result", sampleScan?.scan_id || "scan_20261103_a1b2c3")}
              className="inline-flex items-center justify-center px-6 py-3.5 bg-white hover:bg-concrete-light text-asphalt border border-concrete-rule font-medium text-sm rounded-[4px] transition focus-visible:outline-highway"
            >
              {t("home.cta_real_scan", "See a real scan")}
            </button>
          </div>

            <button
              type="button"
              onClick={() => onNavigate("proof")}
              className="inline-flex items-center justify-center px-6 py-3.5 bg-white hover:bg-concrete-light text-asphalt border border-concrete-rule font-medium text-sm rounded-[4px] transition focus-visible:outline-highway"
            >
              {t("home.proof_link", "See validation results")}
            </button>

          {/* Real displacement waveform playback (only animated element) */}
          <div className="pt-4 max-w-lg">
            <RealWaveform height={75} />
          </div>
        </div>

        {/* Right Column: Real Scan Display or Plain Text Explanation (Never a mockup) */}
        <div className="lg:col-span-5 flex flex-col items-start lg:items-end">
          {/* Milestone Motif (Shows real bridge count, hidden if zero) */}
          {realBridgeCount > 0 && (
            <div className="mb-4 self-start lg:self-end">
              <KmStone count={realBridgeCount} label="BRIDGES SCANNED" />
            </div>
          )}

          {/* Real Scan Box or Text Fallback */}
          <div className="w-full bg-white border border-concrete-rule rounded-[4px] p-4 space-y-3">
            <div className="border-b border-concrete-rule pb-2">
              <div className="text-xs font-bold uppercase tracking-wide text-asphalt">
                {t("home.real_scan_title", "Actual scan result")}
              </div>
              <div className="text-xs text-asphalt-muted">
                {sampleScan
                  ? `${sampleScan.scan_id} (${sampleScan.location.lat.toFixed(4)}°N, ${sampleScan.location.lon.toFixed(4)}°E)`
                  : "Napier Bridge Telemetry"}
              </div>
            </div>

            {sampleScan && (magnifiedUrl || spectrumUrl) ? (
              <div className="space-y-3">
                {magnifiedUrl ? (
                  <div>
                    <div className="text-[11px] font-semibold text-asphalt-muted mb-1">
                      Magnified video (15x deflection amplification)
                    </div>
                    <video
                      src={magnifiedUrl}
                      controls
                      className="w-full h-44 bg-asphalt object-contain rounded-[4px] border border-concrete-rule"
                    />
                  </div>
                ) : null}

                {spectrumUrl ? (
                  <div>
                    <div className="text-[11px] font-semibold text-asphalt-muted mb-1">
                      Power spectral density (Fourier transform)
                    </div>
                    <img
                      src={spectrumUrl}
                      alt="Spectrum decomposition plot from real scan"
                      className="w-full h-36 bg-concrete-light object-contain rounded-[4px] border border-concrete-rule"
                    />
                  </div>
                ) : null}

                {sampleScan.measurement.dominant_frequency_hz && (
                  <div className="flex items-center justify-between text-xs pt-1 border-t border-concrete-rule font-sans">
                    <span className="text-asphalt-muted">Dominant frequency:</span>
                    <span className="font-bold text-highway font-mono">
                      {sampleScan.measurement.dominant_frequency_hz.toFixed(2)} Hz
                    </span>
                  </div>
                )}
              </div>
            ) : scanLoadFailed ? (
              <p className="text-xs text-asphalt-muted py-6 leading-relaxed">
                {t(
                  "home.real_scan_missing",
                  "No completed scan is currently loaded. Film a bridge or choose an entry from the map to view real video and spectrum plots."
                )}
              </p>
            ) : (
              <p className="text-xs text-asphalt-muted py-6 leading-relaxed">
                Loading actual scan telemetry from backend...
              </p>
            )}
          </div>
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
