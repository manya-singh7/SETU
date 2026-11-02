import React, { useState, useEffect } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Layers,
  MapPin,
  Radio,
  RefreshCw,
  ShieldAlert,
  UploadCloud,
  Video,
  Waves,
  Eye,
} from "lucide-react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import L from "leaflet";

// Fix Leaflet's default marker icon issue in bundlers like Vite
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

delete (L.Icon.Default.prototype as unknown as { _getIconUrl?: unknown })._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

interface TriageRecord {
  id: number;
  bridgeName: string;
  lat: number;
  lng: number;
  severity: "UNUSUAL" | "REQUIRES_INSPECTION" | "INCONCLUSIVE";
  fundamentalHz: number;
  expectedHz: string;
  status: "PENDING" | "UNDER_REVIEW" | "INSPECTED";
  summary: string;
  timestamp: string;
}

const INITIAL_TRIAGE_ITEMS: TriageRecord[] = [
  {
    id: 1,
    bridgeName: "Yamuna Railway Truss Span #4",
    lat: 28.665,
    lng: 77.251,
    severity: "UNUSUAL",
    fundamentalHz: 0.94,
    expectedHz: "1.40 - 1.80 Hz",
    status: "PENDING",
    summary:
      "Muse Spark 1.3: Fundamental resonance (0.94 Hz) is 38% lower than nominal 1.6 Hz expectation for a 90m steel through-truss. Possible stiffness reduction or joint slack.",
    timestamp: "10 mins ago",
  },
  {
    id: 2,
    bridgeName: "Hindon River Concrete Girder",
    lat: 28.641,
    lng: 77.378,
    severity: "REQUIRES_INSPECTION",
    fundamentalHz: 2.15,
    expectedHz: "2.00 - 2.60 Hz",
    status: "UNDER_REVIEW",
    summary:
      "Routine screening baseline recorded. High ambient vehicular noise; certified inspection required for baseline validation.",
    timestamp: "1 hour ago",
  },
  {
    id: 3,
    bridgeName: "Wazirabad Stay Cable Pier C",
    lat: 28.712,
    lng: 77.234,
    severity: "INCONCLUSIVE",
    fundamentalHz: 0.0,
    expectedHz: "0.80 - 1.20 Hz",
    status: "PENDING",
    summary:
      "Excessive camera micro-jitter or wind buffeting degraded subpixel phase correlation. High noise floor prevents distinct peak identification.",
    timestamp: "3 hours ago",
  },
];

export default function App() {
  const [triageItems, setTriageItems] = useState<TriageRecord[]>(INITIAL_TRIAGE_ITEMS);
  const [selectedItem, setSelectedItem] = useState<TriageRecord | null>(INITIAL_TRIAGE_ITEMS[0]);
  const [isUploading, setIsUploading] = useState(false);
  const [lat, setLat] = useState("28.6650");
  const [lng, setLng] = useState("77.2510");
  const [targetPrompt, setTargetPrompt] = useState("corner of concrete pillar");
  const [bgPrompt, setBgPrompt] = useState("stationary ground bedrock");
  const [backendHealth, setBackendHealth] = useState<string>("Checking...");

  useEffect(() => {
    fetch("http://127.0.0.1:8000/health")
      .then((res) => res.json())
      .then((data) => {
        setBackendHealth(data.status === "ok" ? "FastAPI Online (:8000)" : "Degraded");
      })
      .catch(() => {
        setBackendHealth("FastAPI Standby (Run backend locally)");
      });
  }, []);

  const handleStatusChange = (id: number, newStatus: "PENDING" | "UNDER_REVIEW" | "INSPECTED") => {
    setTriageItems((prev) =>
      prev.map((item) => (item.id === id ? { ...item, status: newStatus } : item))
    );
    if (selectedItem?.id === id) {
      setSelectedItem((prev) => (prev ? { ...prev, status: newStatus } : null));
    }
  };

  const handleSimulateUpload = (e: React.FormEvent) => {
    e.preventDefault();
    setIsUploading(true);

    setTimeout(() => {
      const newItem: TriageRecord = {
        id: Date.now(),
        bridgeName: "Upload Screening Target (GPS Match)",
        lat: parseFloat(lat) || 28.65,
        lng: parseFloat(lng) || 77.25,
        severity: "UNUSUAL",
        fundamentalHz: 1.18,
        expectedHz: "1.60 - 2.00 Hz",
        status: "PENDING",
        summary:
          "Meta SAM 3.1 isolated target joint. Welch PSD detected 1.18 Hz peak. Muse Spark 1.3 flag: Lower than expected nominal bounds. Scheduled for engineer inspection.",
        timestamp: "Just now",
      };
      setTriageItems([newItem, ...triageItems]);
      setSelectedItem(newItem);
      setIsUploading(false);
    }, 1200);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Strict Safety Philosophy Banner */}
      <div className="bg-amber-950/80 border-b border-amber-800/80 px-4 py-2 text-xs text-amber-200 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <ShieldAlert className="h-4 w-4 text-amber-400 shrink-0" />
          <span>
            <strong>SETU SAFETY PRINCIPLE:</strong> This system <em>never</em> declares a bridge safe.
            Phone-based frequency telemetry is an early screening tool to triage unusual vibration signatures for professional on-site civil engineer inspection.
          </span>
        </div>
        <div className="flex items-center space-x-2 text-[11px] font-mono text-amber-300">
          <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
          <span>{backendHealth}</span>
        </div>
      </div>

      {/* Main App Bar */}
      <header className="border-b border-slate-800 bg-slate-900/90 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="h-9 w-9 rounded-lg bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-600/30">
            <Activity className="h-5 w-5 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-extrabold text-xl tracking-wider text-white">SETU</span>
              <span className="text-[11px] font-semibold px-2 py-0.5 rounded bg-indigo-950 text-indigo-400 border border-indigo-800">
                Hackathon SHM Core
              </span>
            </div>
            <p className="text-xs text-slate-400">Structural Evaluation & Telemetry Utility</p>
          </div>
        </div>

        <div className="flex items-center space-x-4 text-xs">
          <span className="px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-slate-300">
            Triage Queue: <strong>{triageItems.filter((i) => i.status === "PENDING").length} Pending</strong>
          </span>
        </div>
      </header>

      {/* 2-Column Responsive Layout */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 overflow-hidden">
        {/* Left Column: Leaflet Map & Triage Queue (7 cols) */}
        <div className="lg:col-span-7 flex flex-col border-r border-slate-800 p-4 space-y-4 overflow-y-auto">
          {/* Leaflet Map Card */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3 flex flex-col space-y-2">
            <div className="flex items-center justify-between text-xs px-1">
              <div className="flex items-center space-x-2">
                <MapPin className="h-4 w-4 text-indigo-400" />
                <span className="font-semibold text-white">Regional Bridge Triage Map</span>
              </div>
              <span className="text-slate-400">Click marker to inspect structural assessment</span>
            </div>

            <div className="h-72 w-full rounded-lg overflow-hidden border border-slate-800">
              <MapContainer
                center={[28.665, 77.251]}
                zoom={11}
                scrollWheelZoom={false}
                style={{ height: "100%", width: "100%" }}
              >
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                {triageItems.map((item) => (
                  <Marker
                    key={item.id}
                    position={[item.lat, item.lng]}
                    eventHandlers={{
                      click: () => setSelectedItem(item),
                    }}
                  >
                    <Popup>
                      <div className="text-xs space-y-1">
                        <strong>{item.bridgeName}</strong>
                        <p>Frequency: {item.fundamentalHz > 0 ? `${item.fundamentalHz} Hz` : "Inconclusive"}</p>
                        <p className="text-amber-600 font-semibold">{item.severity}</p>
                      </div>
                    </Popup>
                  </Marker>
                ))}
              </MapContainer>
            </div>
          </div>

          {/* Triage Review Queue */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-bold text-white flex items-center space-x-2">
                <AlertTriangle className="h-4 w-4 text-amber-400" />
                <span>Civil Engineer Triage Queue</span>
              </h2>
              <span className="text-xs text-slate-400">{triageItems.length} records</span>
            </div>

            <div className="space-y-2">
              {triageItems.map((item) => (
                <div
                  key={item.id}
                  onClick={() => setSelectedItem(item)}
                  className={`p-4 rounded-xl border cursor-pointer transition ${
                    selectedItem?.id === item.id
                      ? "border-indigo-500 bg-indigo-950/30"
                      : "border-slate-800 bg-slate-900/60 hover:border-slate-700"
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="font-semibold text-sm text-white">{item.bridgeName}</span>
                        <span
                          className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                            item.severity === "UNUSUAL"
                              ? "bg-rose-950 text-rose-300 border border-rose-800"
                              : item.severity === "INCONCLUSIVE"
                              ? "bg-amber-950 text-amber-300 border border-amber-800"
                              : "bg-indigo-950 text-indigo-300 border border-indigo-800"
                          }`}
                        >
                          {item.severity}
                        </span>
                      </div>
                      <p className="text-xs text-slate-400 mt-1 line-clamp-2">{item.summary}</p>
                    </div>

                    <div className="text-right shrink-0 ml-4">
                      <div className="text-sm font-mono font-bold text-indigo-400">
                        {item.fundamentalHz > 0 ? `${item.fundamentalHz} Hz` : "N/A"}
                      </div>
                      <span className="text-[10px] text-slate-500">{item.timestamp}</span>
                    </div>
                  </div>

                  <div className="mt-3 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs">
                    <span className="text-slate-400">Status: <strong className="text-white">{item.status}</strong></span>
                    <div className="flex items-center space-x-2" onClick={(e) => e.stopPropagation()}>
                      <button
                        onClick={() => handleStatusChange(item.id, "UNDER_REVIEW")}
                        className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px]"
                      >
                        Reviewing
                      </button>
                      <button
                        onClick={() => handleStatusChange(item.id, "INSPECTED")}
                        className="px-2.5 py-1 rounded bg-indigo-600 hover:bg-indigo-500 text-white text-[11px]"
                      >
                        Mark Inspected
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: 30s Video Upload & Pipeline Inspector (5 cols) */}
        <div className="lg:col-span-5 p-4 space-y-4 overflow-y-auto bg-slate-950/60">
          {/* Step 1: Upload 30s Video Form */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-5 space-y-4 shadow-lg">
            <div className="flex items-center space-x-2 text-sm font-bold text-white">
              <UploadCloud className="h-5 w-5 text-indigo-400" />
              <span>Step 1: Upload 30-Second Phone Video</span>
            </div>

            <form onSubmit={handleSimulateUpload} className="space-y-3 text-xs">
              <div>
                <label className="block text-slate-400 mb-1">Select Video File (.mp4 / .mov)</label>
                <input
                  type="file"
                  accept="video/*"
                  className="w-full text-slate-300 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:bg-indigo-600 file:text-white hover:file:bg-indigo-500 cursor-pointer"
                />
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-slate-400 mb-1">GPS Latitude</label>
                  <input
                    type="text"
                    value={lat}
                    onChange={(e) => setLat(e.target.value)}
                    className="w-full px-2.5 py-1.5 rounded bg-slate-800 border border-slate-700 text-slate-200"
                  />
                </div>
                <div>
                  <label className="block text-slate-400 mb-1">GPS Longitude</label>
                  <input
                    type="text"
                    value={lng}
                    onChange={(e) => setLng(e.target.value)}
                    className="w-full px-2.5 py-1.5 rounded bg-slate-800 border border-slate-700 text-slate-200"
                  />
                </div>
              </div>

              <div>
                <label className="block text-slate-400 mb-1">SAM 3.1 Structural Target Prompt</label>
                <input
                  type="text"
                  value={targetPrompt}
                  onChange={(e) => setTargetPrompt(e.target.value)}
                  placeholder="e.g. corner of concrete pillar"
                  className="w-full px-2.5 py-1.5 rounded bg-slate-800 border border-slate-700 text-slate-200"
                />
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Stationary Background Reference Prompt</label>
                <input
                  type="text"
                  value={bgPrompt}
                  onChange={(e) => setBgPrompt(e.target.value)}
                  placeholder="e.g. stationary ground bedrock"
                  className="w-full px-2.5 py-1.5 rounded bg-slate-800 border border-slate-700 text-slate-200"
                />
              </div>

              <button
                type="submit"
                disabled={isUploading}
                className="w-full mt-2 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold transition flex items-center justify-center space-x-2 disabled:opacity-50"
              >
                {isUploading ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    <span>Running SETU 7-Stage Pipeline...</span>
                  </>
                ) : (
                  <>
                    <Waves className="h-4 w-4" />
                    <span>Run Frequency Extraction & Triage</span>
                  </>
                )}
              </button>
            </form>
          </div>

          {/* Selected Item Telemetry Details */}
          {selectedItem && (
            <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-5 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-indigo-400 uppercase tracking-wider">
                  Detailed Structural Telemetry
                </span>
                <span className="text-xs text-slate-400">{selectedItem.bridgeName}</span>
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/60">
                  <span className="text-slate-400 block text-[10px]">DETECTED FUNDAMENTAL</span>
                  <span className="text-xl font-mono font-bold text-white mt-1 block">
                    {selectedItem.fundamentalHz > 0 ? `${selectedItem.fundamentalHz} Hz` : "Inconclusive"}
                  </span>
                </div>
                <div className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/60">
                  <span className="text-slate-400 block text-[10px]">NOMINAL EXPECTED (OSM)</span>
                  <span className="text-sm font-mono font-semibold text-slate-300 mt-1.5 block">
                    {selectedItem.expectedHz}
                  </span>
                </div>
              </div>

              {/* Assessment Narrative */}
              <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 text-xs space-y-2">
                <div className="flex items-center space-x-1.5 text-indigo-400 font-semibold">
                  <Eye className="h-3.5 w-3.5" />
                  <span>Muse Spark 1.3 Cautious Assessment</span>
                </div>
                <p className="text-slate-300 leading-relaxed text-xs">{selectedItem.summary}</p>
                <div className="pt-2 border-t border-slate-800 text-[11px] text-amber-300 font-medium">
                  <strong>Recommended Action:</strong> On-site physical engineering survey required. Do not assume structural safety.
                </div>
              </div>

              {/* 7-Stage Pipeline Visual Indicator */}
              <div className="space-y-1.5 pt-2">
                <span className="text-[11px] font-semibold text-slate-400 block">7-Stage Execution Pipeline:</span>
                <div className="text-[11px] text-slate-400 space-y-1">
                  <div className="flex items-center space-x-2 text-emerald-400">
                    <CheckCircle2 className="h-3 w-3 shrink-0" />
                    <span>1. 30s Video + GPS Upload Verified</span>
                  </div>
                  <div className="flex items-center space-x-2 text-emerald-400">
                    <CheckCircle2 className="h-3 w-3 shrink-0" />
                    <span>2. Meta SAM 3.1 Promptable Concept Segmentation</span>
                  </div>
                  <div className="flex items-center space-x-2 text-emerald-400">
                    <CheckCircle2 className="h-3 w-3 shrink-0" />
                    <span>3. Subpixel Target Tracking - Micro-Jitter Subtraction</span>
                  </div>
                  <div className="flex items-center space-x-2 text-emerald-400">
                    <CheckCircle2 className="h-3 w-3 shrink-0" />
                    <span>4. Welch PSD & Modal Peak Identification</span>
                  </div>
                  <div className="flex items-center space-x-2 text-emerald-400">
                    <CheckCircle2 className="h-3 w-3 shrink-0" />
                    <span>5. Eulerian Video Magnification (Display Only)</span>
                  </div>
                  <div className="flex items-center space-x-2 text-emerald-400">
                    <CheckCircle2 className="h-3 w-3 shrink-0" />
                    <span>6. OSM Overpass Attributes & Muse Spark 1.3 Synthesis</span>
                  </div>
                  <div className="flex items-center space-x-2 text-emerald-400">
                    <CheckCircle2 className="h-3 w-3 shrink-0" />
                    <span>7. Regional Map & Civil Engineer Triage Queue</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
