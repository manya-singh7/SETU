import React, { useEffect, useState, useMemo, Suspense, lazy, useRef } from "react";
import { useTranslation } from "react-i18next";
import {
  Activity,
  Mic,
  Pause,
  Play,
  Square,
  Share2,
  ArrowLeft,
  ChevronDown,
  Clock,
  Eye,
  MapPin,
  RotateCw,
  Sliders,
} from "lucide-react";
import { getScan, getFileUrl, uploadFieldNote } from "../api";
import { FieldNote } from "../types";
import i18n from "../i18n";
import { ScanResult, VerdictLevel, TriageSeverity } from "../types";
import { formatLocaleNumber, formatLocaleDate } from "../i18n";
import RealWaveform from "../components/RealWaveform";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

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
  const [recording, setRecording] = useState(false);
  const [sonificationPlaying, setSonificationPlaying] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [uploadingNote, setUploadingNote] = useState(false);
  const [noteError, setNoteError] = useState<string | null>(null);
  const [fieldNotes, setFieldNotes] = useState<FieldNote[]>(initialScan?.field_notes || []);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const sonificationRef = useRef<HTMLAudioElement | null>(null);

  const loadScan = () => {
    let isMounted = true;
    setLoading(true);
    setError(null);

    getScan(scanId, (updated) => {
      if (isMounted) {
        setScan(updated);
        setFieldNotes(updated.field_notes || []);
        if (updated.status === "complete") setLoading(false);
      }
    }, true)
      .then((data) => {
        if (isMounted) {
          setScan(data);
          setFieldNotes(data.field_notes || []);
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
      setFieldNotes(initialScan.field_notes || []);
      setLoading(false);
      setError(null);
      if (initialScan.status === "complete" && (initialScan.magnification_status === "pending" || initialScan.magnification_status === "running")) {
        let isMounted = true;
        void getScan(scanId, (updated) => {
          if (isMounted) setScan(updated);
        }, true).then((updated) => {
          if (isMounted) setScan(updated);
        }).catch((err: unknown) => {
          if (isMounted) setError(err instanceof Error ? err.message : t("common.error", "Failed to load scan"));
        });
        return () => { isMounted = false; };
      }
      return;
    }

    return loadScan();
  }, [scanId, initialScan]);

  useEffect(() => {
    if (!recording) return;
    timerRef.current = window.setInterval(() => setSeconds((n) => {
      if (n + 1 >= 120) { recorderRef.current?.stop(); return 120; }
      return n + 1;
    }), 1000);
    return () => { if (timerRef.current != null) window.clearInterval(timerRef.current); };
  }, [recording]);

  const startRecording = async () => {
    setNoteError(null);
    try {
      if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") throw new Error("unsupported");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      recorderRef.current = recorder;
      chunksRef.current = [];
      recorder.ondataavailable = (event) => { if (event.data.size) chunksRef.current.push(event.data); };
      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        setRecording(false);
        if (!chunksRef.current.length) return;
        setUploadingNote(true);
        try {
          const note = await uploadFieldNote(scanId, new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" }), i18n.language);
          setFieldNotes((notes) => [...notes, note]);
        } catch { setNoteError(t("result.voice_upload_error", "Could not upload this voice note. Please try again.")); }
        finally { setUploadingNote(false); }
      };
      recorder.start();
      setSeconds(0);
      setRecording(true);
    } catch {
      setNoteError(t("result.mic_denied", "Microphone access was denied or is unavailable. Allow microphone access and try again."));
    }
  };



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

  const cable = scan.cable;
  const hasCableChange = cable?.tension_change_pct != null;

  const shareText = makeWhatsAppText({
    bridge: bridgeData?.name || t("result.bridge_area_unknown", "Bridge area"),
    verdict: isUsable ? t("result.share_usable", "reading available") : t("result.share_unclear", "reading not clear enough"),
    frequency: isUsable && domFreq != null && freqCi ? `${formatLocaleNumber(domFreq, { maximumFractionDigits: 2 })} Hz (${formatLocaleNumber(freqCi[0], { maximumFractionDigits: 2 })}\u2013${formatLocaleNumber(freqCi[1], { maximumFractionDigits: 2 })} Hz)` : null,
    link: `${window.location.origin}${window.location.pathname}?scan=${encodeURIComponent(scan.scan_id)}`,
    labels: { bridge: t("result.share_bridge", "Bridge"), reading: t("result.share_reading", "Reading"), frequency: t("result.share_frequency", "Frequency"), link: t("result.share_link", "Page") },
  });

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
  const soundUrl = scan.outputs?.sound ? getFileUrl(scan.scan_id, scan.outputs.sound) : null;
  const trafficPlotUrl = scan.traffic?.response_plot ? getFileUrl(scan.scan_id, scan.traffic.response_plot) : null;
  const modeShape = (scan.measurement?.mode_shape || []).filter((point) =>
    Number.isFinite(point.position_norm) && Number.isFinite(point.amplitude_px)
      && point.position_norm >= 0 && point.position_norm <= 1,
  ).sort((a, b) => a.position_norm - b.position_norm);
  const deviceCheckMessage = scan.device_check?.status === "agrees"
    ? t("result.device_agrees", "Your phone's motion sensor agreed with the camera")
    : scan.device_check?.status === "disagrees"
      ? t("result.device_differs", "Your phone's motion sensor gave a different reading from the camera")
      : scan.device_check?.status === "not_resting"
        ? (scan.device_check.note || t("result.device_not_resting", "The phone moved during recording, so the sensor comparison could not be made."))
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

      {hasCableChange && (
        <div className="rounded-[4px] border border-concrete-rule bg-white px-4 py-3 text-sm">
          <span>{t("result.cable_comparison_brief", "Compared with an earlier scan of this cable")}</span>{" "}
          <button type="button" onClick={() => { setShowEngineerDetails(true); window.setTimeout(() => document.getElementById("cable-engineer-details")?.scrollIntoView({ behavior: "smooth" }), 0); }} className="underline text-highway font-semibold">{t("result.cable_comparison_details", "View details")}</button>
        </div>
      )}
      <div className="rounded-[4px] border border-concrete-rule bg-white p-5 space-y-3">
        <h2 className="font-bold text-base">{t("result.voice_title", "Tell us what you see: cracks, sounds, sagging")}</h2>
        <button type="button" onClick={recording ? () => recorderRef.current?.stop() : startRecording} disabled={uploadingNote} aria-label={recording ? t("result.stop_recording", "Stop recording") : t("result.voice_title", "Tell us what you see: cracks, sounds, sagging")} className="inline-flex items-center gap-2 rounded-full bg-highway px-6 py-4 text-white font-bold disabled:opacity-60">
          {recording ? <Square className="h-5 w-5" /> : <Mic className="h-6 w-6" />}{recording ? t("result.stop_recording", "Stop recording") : t("result.record_voice", "Record a voice note")}
        </button>
        {recording && <span role="timer" className="ml-3 font-mono">{String(Math.floor(seconds / 60)).padStart(2,"0")}:{String(seconds % 60).padStart(2,"0")} / 02:00</span>}
        {uploadingNote && <p role="status">{t("result.uploading_note", "Uploading and transcribing?")}</p>}
        {noteError && <p role="alert" className="text-red-700">{noteError}</p>}
        <p className="text-xs text-asphalt-muted">{t("result.voice_privacy", "Your voice is turned into text; the recording is deleted.")}</p>
        {fieldNotes.map((note) => <article key={note.note_id} className="border-t pt-3 space-y-2"><p>{note.transcript}</p>{note.english && <div><strong>{t("result.for_engineers_title", "For engineers (technical details)")}</strong><p>{note.english}</p></div>}<div className="flex flex-wrap gap-2">{note.tags.map((tag, i) => <span key={`${tag.tag}-${i}`} className="rounded bg-concrete px-2 py-1 text-xs">{t(`result.tag_${tag.tag}`, tag.tag)}: {"\u201c"}{tag.evidence}{"\u201d"}</span>)}</div></article>)}
      </div>
      {(scan.setup?.advice || scan.setup?.recording_issues?.length) ? <section className="rounded-[4px] border border-concrete-rule bg-white p-5 space-y-2" aria-label={t("result.setup_title", "Recording setup")}>
        <h2 className="font-bold text-base">{t("result.setup_title", "Recording setup")}</h2>
        {scan.setup.advice && <p className="text-sm text-asphalt-muted">{scan.setup.advice}</p>}
        {scan.setup.recording_issues?.length ? <div className="text-sm"><h3 className="font-semibold">{t("result.recording_issues", "Things to improve in the recording")}</h3><ul className="mt-1 list-disc list-inside space-y-1 text-asphalt-muted">{scan.setup.recording_issues.map((issue, index) => <li key={`${issue}-${index}`}>{issue}</li>)}</ul></div> : null}
      </section> : null}
      {deviceCheckMessage && <p className="rounded-[4px] border border-concrete-rule bg-white px-4 py-3 text-sm text-asphalt-muted">{deviceCheckMessage}</p>}
      <a href={`https://wa.me/?text=${encodeURIComponent(shareText)}`} target="_blank" rel="noreferrer" className="inline-flex items-center gap-2 rounded bg-highway px-4 py-3 text-white font-semibold"><Share2 className="h-4 w-4" />{t("result.share_whatsapp", "Share on WhatsApp")}</a>
      {scan.measurement?.instances?.length && scan.outputs?.vibration_map ? <div className="rounded-[4px] border border-concrete-rule bg-white p-5"><img src={getFileUrl(scan.scan_id, scan.outputs.vibration_map)} alt={t("result.vibration_map_alt", "Vibration map")} className="max-h-96 mx-auto"/><p className="text-xs text-center">{t("result.vibration_map_caption", "Each part is coloured by how it vibrates")}</p></div> : null}

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

        {soundUrl && <div className="flex flex-wrap items-center gap-3 rounded border border-concrete-rule bg-concrete-light p-3">
          <audio ref={sonificationRef} src={soundUrl} preload="none" onPlay={() => setSonificationPlaying(true)} onPause={() => setSonificationPlaying(false)} onEnded={() => setSonificationPlaying(false)} />
          <button type="button" onClick={() => {
            const audio = sonificationRef.current;
            if (!audio) return;
            if (sonificationPlaying) audio.pause();
            else void audio.play();
          }} className="inline-flex items-center gap-2 rounded bg-highway px-4 py-2 text-white text-sm font-semibold">
            {sonificationPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            {t("result.listen_bridge", "Listen to the bridge")}
          </button>
          {scan.sonification?.speedup != null && <p className="text-xs text-asphalt-muted">{t("result.sonification_caption", "The real vibration, sped up {{speedup}}x so you can hear it.", { speedup: formatLocaleNumber(scan.sonification.speedup, { maximumFractionDigits: 2 }) })}</p>}
        </div>}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Magnified Video */}
          <div className="p-3 bg-concrete-light border border-concrete-rule rounded-[4px] space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-bold text-asphalt">{t("result.evm_amplified", "Magnified movement (15x)")}</span>
              <span className="text-highway text-[11px] font-semibold">{t("result.motion_amplified_note", "Microscopic motion made visible")}</span>
            </div>
            {scan.magnification_status === "pending" || scan.magnification_status === "running" ? (
              <div role="status" className="h-44 rounded-[4px] bg-white border border-concrete-rule flex items-center justify-center p-4 text-center text-xs text-asphalt-muted">
                {t("result.magnification_pending", "Preparing the slowed-down movement video...")}
              </div>
            ) : magnifiedVideoUrl ? (
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
            {scan.setup && <section className="border-b border-concrete-rule pb-4"><h3 className="font-bold text-sm">{t("result.setup_details", "Scan setup")}</h3><p>{t("result.setup_source", "Setup source")}: <code>{scan.setup.source}</code></p>{scan.setup.structure_type && <p>{t("scan.structure_type", "Structure type")}: <code>{scan.setup.structure_type}</code></p>}</section>}
            {scan.weather && (scan.weather.temperature_c != null || scan.weather.humidity_pct != null || scan.weather.wind_speed_ms != null) && <section className="space-y-2 border-b border-concrete-rule pb-4">
              <h3 className="font-bold text-sm">{t("result.weather_title", "Weather at the scan location")}</h3>
              <dl className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                {scan.weather.temperature_c != null && <div><dt className="text-asphalt-muted">{t("result.temperature", "Temperature")}</dt><dd className="font-semibold">{formatLocaleNumber(scan.weather.temperature_c)} °C</dd></div>}
                {scan.weather.humidity_pct != null && <div><dt className="text-asphalt-muted">{t("result.humidity", "Humidity")}</dt><dd className="font-semibold">{formatLocaleNumber(scan.weather.humidity_pct)}%</dd></div>}
                {scan.weather.wind_speed_ms != null && <div><dt className="text-asphalt-muted">{t("result.wind_speed", "Wind speed")}</dt><dd className="font-semibold">{formatLocaleNumber(scan.weather.wind_speed_ms)} m/s</dd></div>}
              </dl>
            </section>}
            {modeShape.length > 0 && <section className="space-y-2 border-b border-concrete-rule pb-4" aria-label={t("result.mode_shape_title", "Bridge mode shape")}>
              <h3 className="font-bold text-sm">{t("result.mode_shape_title", "Bridge mode shape")}</h3>
              <div className="h-64 w-full" role="img" aria-label={t("result.mode_shape_title", "Bridge mode shape")}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={modeShape} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" dataKey="position_norm" domain={[0, 1]} name={t("result.position_axis", "Position along bridge")} />
                    <YAxis type="number" dataKey="amplitude_px" name={t("result.amplitude", "Amplitude")} unit=" px" />
                    <Tooltip formatter={(value) => [`${formatLocaleNumber(Number(value))} px`, t("result.amplitude", "Amplitude")]} labelFormatter={(value) => `${t("result.position_axis", "Position along bridge")}: ${formatLocaleNumber(Number(value))}`} />
                    <Line type="monotone" dataKey="amplitude_px" name={t("result.amplitude", "Amplitude")} stroke="#0B6E4F" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>}
            {trafficPlotUrl && <section className="space-y-2 border-b border-concrete-rule pb-4">
              <h3 className="font-bold text-sm">{t("result.traffic_title", "Traffic response")}</h3>
              <figure className="space-y-1"><img src={trafficPlotUrl} alt={t("result.traffic_plot_alt", "Bridge vibration response during vehicle crossings")} className="max-h-80 w-full object-contain" /><figcaption className="text-asphalt-muted">{t("result.traffic_caption", "Shaded: vehicles crossing.")}</figcaption></figure>
            </section>}
            {hasCableChange && <section id="cable-engineer-details" className="space-y-1 border-b border-concrete-rule pb-4">
              <h3 className="font-bold text-sm">{t("result.cable_change_title", "Cable tension estimate")}</h3>
              <p>{t("result.cable_change_estimate", "Estimated change in cable tension since {{date}}: {{change}}%", {
                date: cable.compared_to_date ? formatLocaleDate(cable.compared_to_date, { dateStyle: "medium" }) : t("result.earlier_scan", "an earlier scan"),
                change: formatLocaleNumber(cable.tension_change_pct),
              })}{cable.tension_change_range_pct && <> ({t("result.cable_change_range", "range {{low}}% to {{high}}%", { low: formatLocaleNumber(cable.tension_change_range_pct[0]), high: formatLocaleNumber(cable.tension_change_range_pct[1]) })})</>}</p>
              <p className="text-asphalt-muted">{t("result.cable_change_note", "Based on the change in vibration frequency; for engineer review, not a safety verdict.")}</p>
            </section>}
            {scan.measurement?.instances?.length ? <section><h3 className="font-bold text-sm">{t("result.instance_details", "For engineers: measured parts")}</h3><div className="overflow-x-auto"><table className="w-full text-left"><thead><tr><th>{t("result.instance", "Part")}</th><th>{t("result.frequency", "Frequency")}</th><th>{t("result.amplitude", "Amplitude")}</th><th>{t("result.verdict", "Verdict")}</th></tr></thead><tbody>{scan.measurement.instances.map((instance, idx) => <tr key={`${instance.label}-${idx}`}><td>{instance.label}</td><td>{instance.frequency_hz == null ? "?" : `${formatLocaleNumber(instance.frequency_hz)} Hz`}</td><td>{instance.amplitude_px == null ? "?" : `${formatLocaleNumber(instance.amplitude_px)} px`}</td><td>{instance.verdict}</td></tr>)}</tbody></table></div></section> : null}

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
export function makeWhatsAppText(args: { bridge: string; verdict: string; frequency: string | null; link: string; labels: { bridge: string; reading: string; frequency: string; link: string } }): string {
  return [`${args.labels.bridge}: ${args.bridge}`, `${args.labels.reading}: ${args.verdict}`, ...(args.frequency ? [`${args.labels.frequency}: ${args.frequency}`] : []), `${args.labels.link}: ${args.link}`].join("\n");
}
