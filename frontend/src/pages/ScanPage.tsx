import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import {
  Camera,
  Upload,
  MapPin,
  Crosshair,
  RotateCw,
  Layers,
  AlertTriangle,
  Play,
} from "lucide-react";
import { createScan, getScan } from "../api";
import { ScanResult } from "../types";

interface ScanPageProps {
  onScanComplete: (scanId: string) => void;
}

export default function ScanPage({ onScanComplete }: ScanPageProps) {
  const { t } = useTranslation();

  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [lat, setLat] = useState("13.0827");
  const [lon, setLon] = useState("80.2707");
  const [bridgeName, setBridgeName] = useState("Napier Bridge Pier #2");
  const [targetPrompt, setTargetPrompt] = useState("concrete pillar");
  const [backgroundPrompt, setBackgroundPrompt] = useState("building"); // Default "building"
  const [isCable, setIsCable] = useState(false);
  const [gpsStatus, setGpsStatus] = useState<"idle" | "fetching" | "success" | "error">("idle");

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [pollStatus, setPollStatus] = useState<"queued" | "running" | "complete" | "error" | "submitting">("submitting");
  const [pollMessage, setPollMessage] = useState<string>("");
  const [pollCount, setPollCount] = useState<number>(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setVideoFile(file);
      setVideoPreview(URL.createObjectURL(file));
      setErrorMessage(null);
    }
  };

  const handleGetGps = () => {
    if (!navigator.geolocation) {
      alert("Geolocation is not supported by your browser.");
      return;
    }
    setGpsStatus("fetching");
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLat(pos.coords.latitude.toFixed(6));
        setLon(pos.coords.longitude.toFixed(6));
        setGpsStatus("success");
      },
      (err) => {
        console.warn("GPS lookup denied or unavailable:", err);
        setGpsStatus("error");
      },
      { timeout: 10000, enableHighAccuracy: true }
    );
  };

  const handleApplyPreset = (presetName: string, presetLat: string, presetLon: string) => {
    setBridgeName(presetName);
    setLat(presetLat);
    setLon(presetLon);
  };

  const handleStartAnalysis = async () => {
    setIsSubmitting(true);
    setErrorMessage(null);
    setPollStatus("submitting");
    setPollMessage(t("scan.status_queued", "Status: waiting in line"));
    setPollCount(0);

    try {
      const created = await createScan({
        file: videoFile,
        lat: parseFloat(lat),
        lon: parseFloat(lon),
        target_prompt: targetPrompt,
        background_prompt: backgroundPrompt,
        bridge_name: bridgeName,
        is_cable: isCable,
      });

      const scanId = created.scan_id || (created as any).id;
      if (!scanId) {
        throw new Error("Backend did not return a valid scan_id");
      }

      setPollStatus("queued");

      let count = 0;
      const finalResult = await getScan(scanId, (polled: ScanResult) => {
        count++;
        setPollCount(count);
        setPollStatus(polled.status);
        if (polled.status === "queued") {
          setPollMessage(`${t("scan.status_queued", "Status: waiting in line")} (#${count})`);
        } else if (polled.status === "running") {
          setPollMessage(`${t("scan.status_running", "Status: measuring bridge motion")} (#${count})`);
        } else if (polled.status === "complete") {
          setPollMessage(t("scan.status_complete", "Status: complete"));
        } else if (polled.status === "error") {
          setPollMessage(
            `${t("scan.status_error", "Status: could not process video")} (${polled.error || "Check verdict"})`
          );
        }
      });

      if (finalResult.status === "error") {
        setErrorMessage(
          finalResult.error ||
            (finalResult.verdict?.reasons?.length
              ? finalResult.verdict.reasons.join("; ")
              : "Measurement pipeline reported an error.")
        );
        setIsSubmitting(false);
      } else {
        onScanComplete(finalResult.scan_id);
      }
    } catch (err: unknown) {
      const errText =
        err instanceof Error ? err.message : "An unexpected error occurred during scan submission.";
      console.error("Scan submission/polling error:", err);
      setErrorMessage(errText);
      setIsSubmitting(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6 py-2 px-1 font-sans text-asphalt">
      {/* Header (Driver Register) */}
      <div className="border-b border-concrete-rule pb-3">
        <h1 className="text-2xl sm:text-3xl font-heading text-asphalt tracking-tight">
          {t("scan.title", "Film a bridge")}
        </h1>
        <p className="text-xs sm:text-sm text-asphalt-muted mt-1 font-sans">
          {t(
            "scan.subtitle",
            "Film the bridge for 30 seconds to measure how it vibrates."
          )}
        </p>
      </div>

      {/* Driver Safety Warning (Highway Warning Style) */}
      <div className="rounded-[4px] bg-white border-2 border-milestone p-4 flex items-start space-x-3 text-asphalt">
        <AlertTriangle className="h-5 w-5 text-asphalt shrink-0 mt-0.5" />
        <div className="space-y-0.5">
          <div className="font-bold text-sm text-asphalt">
            {t("scan.drive_warning", "Never record while driving. Stop safely first.")}
          </div>
          <p className="text-xs text-asphalt-muted">
            Pull over into a safe spot off the roadway before aiming the camera at the bridge.
          </p>
        </div>
      </div>

      {/* Polling / Progress State */}
      {isSubmitting && (
        <div className="rounded-[4px] border-2 border-highway bg-white p-5 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2.5">
              <RotateCw className="h-4 w-4 text-highway animate-spin" />
              <h2 className="text-sm font-bold text-asphalt font-sans">
                {t("scan.polling_title", "Measuring bridge vibration")}
              </h2>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-[11px] font-mono text-asphalt-muted">Check #{pollCount}</span>
              <span
                className={`text-[11px] font-bold px-2 py-0.5 rounded-[2px] uppercase ${
                  pollStatus === "complete"
                    ? "bg-highway text-white"
                    : pollStatus === "running"
                    ? "bg-milestone text-asphalt"
                    : pollStatus === "error"
                    ? "bg-asphalt-light text-white"
                    : "bg-concrete text-asphalt border border-concrete-rule"
                }`}
              >
                {pollStatus}
              </span>
            </div>
          </div>

          <div className="h-2 w-full bg-concrete rounded-[2px] overflow-hidden">
            <div
              className={`h-full transition-all duration-300 ${
                pollStatus === "complete"
                  ? "w-full bg-highway"
                  : pollStatus === "running"
                  ? "w-3/4 bg-highway"
                  : "w-1/3 bg-asphalt"
              }`}
            />
          </div>

          <div className="flex justify-between items-center text-xs">
            <span className="text-asphalt font-medium">{pollMessage}</span>
            <span className="text-asphalt-muted">{t("scan.polling_interval", "Checking every 2 seconds")}</span>
          </div>
        </div>
      )}

      {/* Error Banner */}
      {errorMessage && !isSubmitting && (
        <div className="rounded-[4px] border border-asphalt-light bg-white p-4 text-xs text-asphalt space-y-1">
          <strong className="font-bold text-asphalt">{t("common.error", "Error")}:</strong>
          <p className="text-asphalt-muted">{errorMessage}</p>
        </div>
      )}

      {/* Main Scan Form */}
      {!isSubmitting && (
        <div className="space-y-6">
          {/* 1. Video Capture / Upload (Mobile-first, large tap targets) */}
          <div className="rounded-[4px] border border-concrete-rule bg-white p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-concrete-rule pb-2">
              <span className="text-sm sm:text-base font-bold text-asphalt flex items-center space-x-2">
                <Camera className="h-4 w-4 text-highway" />
                <span>{t("scan.video_section_title", "1. Video")}</span>
              </span>
              <span className="text-xs text-asphalt-muted font-mono">
                {t("scan.video_recommended", "Hold still for 30 seconds")}
              </span>
            </div>

            <div className="p-4 bg-concrete-light border border-concrete-rule rounded-[4px] flex flex-col items-center space-y-4">
              {videoPreview ? (
                <div className="w-full max-w-md">
                  <video
                    src={videoPreview}
                    controls
                    className="w-full rounded-[4px] max-h-56 bg-black object-contain border border-concrete-rule"
                  />
                  <div className="mt-2 text-xs text-asphalt-muted">
                    {t("scan.selected_video", "Selected")}: {videoFile?.name || "recording.mp4"} (
                    {(videoFile?.size ? videoFile.size / (1024 * 1024) : 0).toFixed(1)} MB)
                  </div>
                </div>
              ) : (
                <div className="text-center py-4 space-y-1">
                  <div className="font-bold text-sm text-asphalt">
                    {t("scan.select_video_prompt", "Choose a video or film with camera")}
                  </div>
                  <p className="text-xs text-asphalt-muted">
                    {t("scan.formats_supported", "Standard MP4, MOV, or WebM")}
                  </p>
                </div>
              )}

              <div className="flex flex-col sm:flex-row gap-3 w-full sm:w-auto justify-center">
                <label className="cursor-pointer min-h-[48px] inline-flex items-center justify-center space-x-2 px-6 py-3 rounded-[4px] bg-highway hover:bg-highway-dark text-white text-xs font-semibold transition focus-visible:outline-highway">
                  <Upload className="h-4 w-4" />
                  <span>
                    {videoFile
                      ? t("scan.choose_diff_video", "Choose different video")
                      : t("scan.upload_file", "Upload video file")}
                  </span>
                  <input
                    type="file"
                    accept="video/*"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </label>

                <label className="cursor-pointer min-h-[48px] inline-flex items-center justify-center space-x-2 px-6 py-3 rounded-[4px] bg-white hover:bg-concrete text-asphalt text-xs font-semibold border border-concrete-rule transition focus-visible:outline-highway">
                  <Camera className="h-4 w-4 text-highway" />
                  <span>{t("scan.record_camera", "Film with camera")}</span>
                  <input
                    type="file"
                    accept="video/*"
                    capture="environment"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </label>
              </div>
            </div>
          </div>

          {/* 2. Bridge Details & Driver Prompts */}
          <div className="rounded-[4px] border border-concrete-rule bg-white p-5 space-y-4">
            <div className="border-b border-concrete-rule pb-2">
              <span className="text-sm sm:text-base font-bold text-asphalt flex items-center space-x-2">
                <MapPin className="h-4 w-4 text-highway" />
                <span>{t("scan.gps_section_title", "2. Bridge details")}</span>
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              <div className="md:col-span-2">
                <label className="block text-asphalt font-semibold mb-1">
                  {t("scan.bridge_name_label", "Bridge name (optional)")}
                </label>
                <input
                  type="text"
                  value={bridgeName}
                  onChange={(e) => setBridgeName(e.target.value)}
                  placeholder={t("scan.bridge_name_placeholder", "Napier Bridge Pier #2")}
                  className="w-full px-3 py-2.5 rounded-[4px] bg-concrete-light border border-concrete-rule text-asphalt focus-visible:outline-highway transition"
                />
              </div>

              <div>
                <label className="block text-asphalt font-semibold mb-1">
                  {t("scan.latitude_label", "Latitude")}
                </label>
                <input
                  type="text"
                  value={lat}
                  onChange={(e) => setLat(e.target.value)}
                  placeholder="13.0827"
                  className="w-full px-3 py-2.5 rounded-[4px] bg-concrete-light border border-concrete-rule text-asphalt font-mono focus-visible:outline-highway transition"
                />
              </div>

              <div>
                <label className="block text-asphalt font-semibold mb-1">
                  {t("scan.longitude_label", "Longitude")}
                </label>
                <input
                  type="text"
                  value={lon}
                  onChange={(e) => setLon(e.target.value)}
                  placeholder="80.2707"
                  className="w-full px-3 py-2.5 rounded-[4px] bg-concrete-light border border-concrete-rule text-asphalt font-mono focus-visible:outline-highway transition"
                />
              </div>

              <div className="md:col-span-2 flex flex-wrap items-center gap-2 pt-1">
                <button
                  type="button"
                  onClick={handleGetGps}
                  className="min-h-[40px] inline-flex items-center space-x-1.5 px-3 py-2 rounded-[4px] bg-concrete hover:bg-concrete-dark text-asphalt text-xs font-semibold border border-concrete-rule transition focus-visible:outline-highway"
                >
                  <MapPin className="h-3.5 w-3.5 text-highway" />
                  <span>
                    {gpsStatus === "fetching"
                      ? t("scan.acquiring_gps", "Acquiring GPS...")
                      : gpsStatus === "success"
                      ? t("scan.gps_acquired", "GPS acquired")
                      : t("scan.get_gps", "Get GPS location")}
                  </span>
                </button>

                <span className="text-[11px] text-asphalt-muted ml-1">{t("scan.presets_label", "Presets:")}</span>
                <button
                  type="button"
                  onClick={() => handleApplyPreset("Napier Bridge Pier #2", "13.0827", "80.2707")}
                  className="px-2.5 py-1.5 rounded-[4px] text-[11px] text-asphalt bg-concrete border border-concrete-rule hover:bg-concrete-dark"
                >
                  {t("scan.preset_chennai", "Napier Bridge (Chennai)")}
                </button>
                <button
                  type="button"
                  onClick={() => handleApplyPreset("Bandra-Worli Sea Link", "19.0368", "72.8172")}
                  className="px-2.5 py-1.5 rounded-[4px] text-[11px] text-asphalt bg-concrete border border-concrete-rule hover:bg-concrete-dark"
                >
                  {t("scan.preset_mumbai", "Bandra-Worli (Mumbai)")}
                </button>
              </div>

              {/* Target Prompt in Driver Words */}
              <div className="pt-2">
                <label className="flex items-center space-x-1.5 text-asphalt font-semibold mb-1">
                  <Crosshair className="h-3.5 w-3.5 text-highway" />
                  <span>{t("scan.target_prompt_label", "What part of the bridge are you filming?")}</span>
                </label>
                <input
                  type="text"
                  value={targetPrompt}
                  onChange={(e) => setTargetPrompt(e.target.value)}
                  placeholder="concrete pillar"
                  className="w-full px-3 py-2.5 rounded-[4px] bg-concrete-light border border-concrete-rule text-asphalt focus-visible:outline-highway transition"
                />
                <span className="text-[11px] text-asphalt-muted block mt-1">
                  {t("scan.target_prompt_desc", "For example: concrete pillar, road deck, or stay cable.")}
                </span>
              </div>

              <div className="pt-2 md:col-span-2">
                <label className="inline-flex items-center gap-2 text-sm font-semibold text-asphalt">
                  <input type="checkbox" checked={isCable} onChange={(event) => setIsCable(event.target.checked)} className="h-5 w-5 accent-highway" />
                  {t("scan.is_cable_label", "I'm filming a cable or hanger")}
                </label>
              </div>

              {/* Background Prompt in Driver Words */}
              <div className="pt-2">
                <label className="flex items-center space-x-1.5 text-asphalt font-semibold mb-1">
                  <Layers className="h-3.5 w-3.5 text-highway" />
                  <span>{t("scan.background_prompt_label", "Stationary background")}</span>
                </label>
                <input
                  type="text"
                  value={backgroundPrompt}
                  onChange={(e) => setBackgroundPrompt(e.target.value)}
                  placeholder="building"
                  className="w-full px-3 py-2.5 rounded-[4px] bg-concrete-light border border-concrete-rule text-asphalt focus-visible:outline-highway transition"
                />
                <span className="text-[11px] text-asphalt-muted block mt-1">
                  {t("scan.background_prompt_desc", "Keep a building or hill in view so SETU can cancel your hand shake.")}
                </span>
              </div>
            </div>
          </div>

          {/* 3. Recording Rules (Driver Register) */}
          <div className="rounded-[4px] border border-concrete-rule bg-white p-5 space-y-3">
            <div className="text-xs font-bold text-asphalt border-b border-concrete-rule pb-2">
              {t("scan.tips_title", "Rules for a good recording")}
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-asphalt-muted font-sans">
              <div>
                <strong className="text-asphalt font-semibold">{t("scan.tip1_title", "Brace your phone:")}</strong>{" "}
                {t(
                  "scan.tip1_desc",
                  "Rest the phone on a solid railing or wall. If you're in a vehicle, park and switch the engine off first; engine vibration spoils the reading."
                )}
              </div>
              <div>
                <strong className="text-asphalt font-semibold">{t("scan.tip2_title", "Turn off stabilization:")}</strong>{" "}
                {t(
                  "scan.tip2_desc",
                  "Turn off video stabilisation if your camera has the option."
                )}
              </div>
              <div>
                <strong className="text-asphalt font-semibold">{t("scan.tip3_title", "Get close enough:")}</strong>{" "}
                {t("scan.tip3_desc", "Fill the screen with the part of the bridge you are checking.")}
              </div>
              <div>
                <strong className="text-asphalt font-semibold">{t("scan.tip4_title", "Keep steady background:")}</strong>{" "}
                {t("scan.tip4_desc", "Keep a building or hill in view so SETU can cancel your hand shake.")}
              </div>
            </div>
          </div>

          {/* 4. Privacy Note */}
          <div className="border border-concrete-rule bg-concrete-light p-3 rounded-[4px] text-xs text-asphalt-muted">
            <strong>Note: </strong>
            {t("scan.privacy_note", "Video and location are stored for analysis. SETU does not assess safety.")}
          </div>

          {/* Submit Button */}
          <div className="flex justify-end pt-2">
            <button
              type="button"
              onClick={handleStartAnalysis}
              className="w-full sm:w-auto min-h-[48px] inline-flex items-center justify-center space-x-2 px-8 py-3.5 rounded-[4px] bg-highway hover:bg-highway-dark text-white font-semibold text-sm transition focus-visible:outline-highway"
            >
              <Play className="h-4 w-4 fill-current text-white" />
              <span>{t("scan.submit_btn", "Check this bridge")}</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}