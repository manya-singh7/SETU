import React, { useState } from "react";
import {
  Camera,
  Upload,
  MapPin,
  Crosshair,
  Info,
  CheckCircle2,
  Play,
  RotateCw,
  Zap,
} from "lucide-react";
import { createScan } from "../api";

interface ScanPageProps {
  onScanComplete: (scanId: string) => void;
}

export default function ScanPage({ onScanComplete }: ScanPageProps) {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [lat, setLat] = useState("13.0674");
  const [lon, setLon] = useState("80.2848");
  const [bridgeName, setBridgeName] = useState("Napier Bridge Approach");
  const [targetPrompt, setTargetPrompt] = useState("deck soffit at pier joint #2");
  const [gpsStatus, setGpsStatus] = useState<"idle" | "fetching" | "success" | "error">("idle");

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progressPercent, setProgressPercent] = useState(0);
  const [progressStage, setProgressStage] = useState("");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setVideoFile(file);
      setVideoPreview(URL.createObjectURL(file));
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
    setIsAnalyzing(true);
    setProgressPercent(15);
    setProgressStage("SAM 3.1 Video Segmentation: Masking structural target...");

    const steps = [
      { pct: 35, text: "Background Ego-Motion: Estimating camera jitter homography..." },
      { pct: 60, text: "Eulerian Video Magnification: Amplifying micro-deflections (15x)..." },
      { pct: 80, text: "FFT Spectral Decomposition: Extracting sub-pixel resonant peaks..." },
      { pct: 92, text: "OSM Overpass Correlation: Cross-referencing bridge geometry..." },
      { pct: 100, text: "Formulating civil engineer screening triage..." },
    ];

    let stepIdx = 0;
    const interval = setInterval(async () => {
      if (stepIdx < steps.length) {
        setProgressPercent(steps[stepIdx].pct);
        setProgressStage(steps[stepIdx].text);
        stepIdx++;
      } else {
        clearInterval(interval);
        try {
          const scan = await createScan({
            file: videoFile,
            preview_url: videoPreview || undefined,
            lat: parseFloat(lat),
            lon: parseFloat(lon),
            bridge_name: bridgeName,
            target_prompt: targetPrompt,
          });
          onScanComplete(scan.id);
        } catch (err: unknown) {
          const errMsg = err instanceof Error ? err.message : "An error occurred";
          console.error("Scan submission error:", err);
          setIsAnalyzing(false);
          alert("Error processing scan: " + errMsg);
        }
      }
    }, 600);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Intro Header */}
      <div>
        <div className="flex items-center space-x-2 mb-1">
          <Zap className="h-5 w-5 text-indigo-400" />
          <h1 className="text-xl font-bold text-white tracking-tight">
            New Vibration Telemetry Scan
          </h1>
        </div>
        <p className="text-sm text-slate-400">
          Record or upload a 30-second mobile video to extract micro-vibrational modal frequencies
          and screen for structural anomalies.
        </p>
      </div>

      {/* Progress State Overlay */}
      {isAnalyzing && (
        <div className="rounded-xl border border-indigo-500/80 bg-slate-900/90 p-6 shadow-2xl shadow-indigo-950/50 space-y-4">
          <div className="flex items-center space-x-3">
            <RotateCw className="h-5 w-5 text-indigo-400 animate-spin" />
            <h2 className="text-base font-semibold text-white">Analyzing Bridge Telemetry...</h2>
          </div>

          <div className="h-2.5 w-full bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-indigo-500 to-sky-400 transition-all duration-500 rounded-full"
              style={{ width: `${progressPercent}%` }}
            />
          </div>

          <div className="flex justify-between items-center text-xs">
            <span className="text-indigo-300 font-medium">{progressStage}</span>
            <span className="font-mono text-slate-400">{progressPercent}%</span>
          </div>
        </div>
      )}

      {/* Main Scan Form */}
      {!isAnalyzing && (
        <div className="space-y-6">
          {/* 1. Video Capture / Upload */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-white flex items-center space-x-2">
                <Camera className="h-4 w-4 text-indigo-400" />
                <span>1. Video Recording or File</span>
              </span>
              <span className="text-xs text-slate-400">30s @ 30/60fps ideal</span>
            </div>

            <div className="border-2 border-dashed border-slate-700/80 rounded-xl p-6 text-center bg-slate-950/40 flex flex-col items-center space-y-4">
              {videoPreview ? (
                <div className="w-full max-w-md">
                  <video
                    src={videoPreview}
                    controls
                    className="w-full rounded-lg max-h-56 bg-black object-contain border border-slate-800"
                  />
                  <div className="mt-2 text-xs text-slate-400">
                    Selected: {videoFile?.name || "recording.mp4"} (
                    {(videoFile?.size ? videoFile.size / (1024 * 1024) : 12).toFixed(1)} MB)
                  </div>
                </div>
              ) : (
                <>
                  <div className="h-12 w-12 rounded-full bg-indigo-950/80 border border-indigo-800 flex items-center justify-center text-indigo-400">
                    <Upload className="h-6 w-6" />
                  </div>
                  <div>
                    <div className="font-semibold text-sm text-slate-200">
                      Select Video or Record with Mobile Camera
                    </div>
                    <div className="text-xs text-slate-500 mt-0.5">
                      Standard MP4, MOV, or WebM formats supported
                    </div>
                  </div>
                </>
              )}

              <div className="flex flex-wrap gap-3 justify-center">
                <label className="cursor-pointer inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-md transition">
                  <Upload className="h-4 w-4" />
                  <span>{videoFile ? "Choose Different Video" : "Upload File"}</span>
                  <input
                    type="file"
                    accept="video/*"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </label>

                <label className="cursor-pointer inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition">
                  <Camera className="h-4 w-4 text-indigo-400" />
                  <span>Record Video (Camera)</span>
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

          {/* 2. GPS Location & Target Prompt */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-white flex items-center space-x-2">
                <MapPin className="h-4 w-4 text-indigo-400" />
                <span>2. Location & Structural Target</span>
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              <div className="md:col-span-2">
                <label className="block text-slate-400 font-medium mb-1">
                  Bridge or Structure Name (Optional)
                </label>
                <input
                  type="text"
                  value={bridgeName}
                  onChange={(e) => setBridgeName(e.target.value)}
                  placeholder="e.g. Napier Bridge, Cooum Span"
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-200 focus:outline-none focus:border-indigo-500"
                />
              </div>

              <div>
                <label className="block text-slate-400 font-medium mb-1">Latitude</label>
                <input
                  type="text"
                  value={lat}
                  onChange={(e) => setLat(e.target.value)}
                  placeholder="13.0674"
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-200 font-mono focus:outline-none focus:border-indigo-500"
                />
              </div>

              <div>
                <label className="block text-slate-400 font-medium mb-1">Longitude</label>
                <input
                  type="text"
                  value={lon}
                  onChange={(e) => setLon(e.target.value)}
                  placeholder="80.2848"
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-200 font-mono focus:outline-none focus:border-indigo-500"
                />
              </div>

              <div className="md:col-span-2 flex flex-wrap items-center gap-2 pt-1">
                <button
                  type="button"
                  onClick={handleGetGps}
                  className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700"
                >
                  <MapPin className="h-3.5 w-3.5 text-indigo-400" />
                  <span>
                    {gpsStatus === "fetching"
                      ? "Acquiring GPS..."
                      : gpsStatus === "success"
                      ? "GPS Acquired ✓"
                      : "Get Browser GPS"}
                  </span>
                </button>

                <span className="text-[11px] text-slate-500 ml-1">Presets:</span>
                <button
                  type="button"
                  onClick={() => handleApplyPreset("Napier Bridge", "13.0674", "80.2848")}
                  className="px-2 py-1 rounded text-[11px] text-slate-400 hover:text-white bg-slate-900 border border-slate-800"
                >
                  Napier Bridge (Chennai)
                </button>
                <button
                  type="button"
                  onClick={() => handleApplyPreset("Bandra-Worli Sea Link", "19.0368", "72.8172")}
                  className="px-2 py-1 rounded text-[11px] text-slate-400 hover:text-white bg-slate-900 border border-slate-800"
                >
                  Bandra-Worli (Mumbai)
                </button>
                <button
                  type="button"
                  onClick={() => handleApplyPreset("Vashi Bridge", "19.0628", "72.9739")}
                  className="px-2 py-1 rounded text-[11px] text-slate-400 hover:text-white bg-slate-900 border border-slate-800"
                >
                  Vashi Bridge (Mumbai)
                </button>
              </div>

              <div className="md:col-span-2 pt-2">
                <label className="flex items-center space-x-1.5 text-slate-400 font-medium mb-1">
                  <Crosshair className="h-3.5 w-3.5 text-indigo-400" />
                  <span>Target Prompt for SAM 3.1 Segmentation (Optional)</span>
                </label>
                <input
                  type="text"
                  value={targetPrompt}
                  onChange={(e) => setTargetPrompt(e.target.value)}
                  placeholder="e.g. corner of concrete pillar, rocker bearing, truss node"
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-200 focus:outline-none focus:border-indigo-500"
                />
                <span className="text-[11px] text-slate-500 block mt-1">
                  Directs promptable segmentation to the edge or feature undergoing structural deflection.
                </span>
              </div>
            </div>
          </div>

          {/* 3. Recording Best Practices & Tips Card */}
          <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-4 space-y-3">
            <div className="flex items-center space-x-2 text-xs font-semibold text-indigo-300">
              <Info className="h-4 w-4 text-indigo-400 shrink-0" />
              <span>Civil Engineer Field Recording Tips</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-slate-400">
              <div className="flex items-start space-x-2">
                <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                <span>
                  <strong className="text-slate-200">Brace the phone:</strong> Lean firmly against a solid
                  railing, tripod, or car hood to reduce hand shake.
                </span>
              </div>
              <div className="flex items-start space-x-2">
                <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                <span>
                  <strong className="text-slate-200">Turn off stabilisation:</strong> Disable digital/optical
                  stabilisation if accessible (EIS dampens subtle micro-vibrations).
                </span>
              </div>
              <div className="flex items-start space-x-2">
                <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                <span>
                  <strong className="text-slate-200">Fill the frame:</strong> Zoom or position close to the
                  target element (bearing, midspan deck, stay cable).
                </span>
              </div>
              <div className="flex items-start space-x-2">
                <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                <span>
                  <strong className="text-slate-200">Stationary background:</strong> Keep immovable ground,
                  skyline, or bedrock in the frame for ego-motion subtraction.
                </span>
              </div>
            </div>
          </div>

          {/* Action Trigger */}
          <div className="flex justify-end pt-2">
            <button
              type="button"
              onClick={handleStartAnalysis}
              className="inline-flex items-center space-x-2 px-6 py-3 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm shadow-lg shadow-indigo-600/30 transition"
            >
              <Play className="h-4 w-4 fill-current" />
              <span>Analyze Vibration Telemetry</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
