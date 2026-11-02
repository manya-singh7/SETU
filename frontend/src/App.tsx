import React, { useState, useEffect } from "react";
import {
  Activity,
  Camera,
  MapPin,
  ShieldAlert,
  BarChart3,
} from "lucide-react";
import ScanPage from "./pages/ScanPage";
import ResultPage from "./pages/ResultPage";
import EngineerMapPage from "./pages/EngineerMapPage";

export default function App() {
  const [currentPage, setCurrentPage] = useState<"scan" | "result" | "map">("scan");
  const [selectedScanId, setSelectedScanId] = useState<string>("scan-chn-001");
  const [backendHealth, setBackendHealth] = useState<string>("Checking backend...");

  useEffect(() => {
    fetch("http://127.0.0.1:8000/health")
      .then((res) => res.json())
      .then((data) => {
        setBackendHealth(data.status === "ok" ? "FastAPI Online (:8000)" : "Degraded");
      })
      .catch(() => {
        setBackendHealth("FastAPI Standby (Mock Telemetry Active)");
      });
  }, []);

  const handleNavigate = (page: "scan" | "result" | "map", scanId?: string) => {
    if (scanId) {
      setSelectedScanId(scanId);
    }
    setCurrentPage(page);
  };

  const handleScanComplete = (newScanId: string) => {
    setSelectedScanId(newScanId);
    setCurrentPage("result");
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Strict Safety Philosophy Banner (Fixed Requirement) */}
      <div className="bg-amber-950/80 border-b border-amber-800/80 px-4 py-2 text-xs text-amber-200 flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center space-x-2">
          <ShieldAlert className="h-4 w-4 text-amber-400 shrink-0" />
          <span>
            <strong>SETU SAFETY PRINCIPLE:</strong> SETU does not assess safety. Flagged readings
            are for engineer review. Mobile video vibration telemetry is an early screening heuristic only.
          </span>
        </div>
        <div className="flex items-center space-x-2 text-[11px] font-mono text-amber-300">
          <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
          <span>{backendHealth}</span>
        </div>
      </div>

      {/* Main Top Header & Navigation Bar */}
      <header className="border-b border-slate-800 bg-slate-900/90 px-4 sm:px-6 py-3 flex items-center justify-between sticky top-0 z-50 backdrop-blur-md">
        <div className="flex items-center space-x-3">
          <div className="h-9 w-9 rounded-lg bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-600/30">
            <Activity className="h-5 w-5 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-extrabold text-lg sm:text-xl tracking-wider text-white">SETU</span>
              <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-indigo-950 text-indigo-400 border border-indigo-800 hidden sm:inline-block">
                SHM Telemetry Core
              </span>
            </div>
            <p className="text-[11px] text-slate-400 hidden sm:block">
              Structural Evaluation & Telemetry Utility
            </p>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav className="flex items-center space-x-1 sm:space-x-2 text-xs">
          <button
            onClick={() => setCurrentPage("scan")}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg font-medium transition ${
              currentPage === "scan"
                ? "bg-indigo-600 text-white shadow-sm"
                : "text-slate-300 hover:bg-slate-800"
            }`}
          >
            <Camera className="h-3.5 w-3.5" />
            <span>Scan</span>
          </button>

          <button
            onClick={() => setCurrentPage("result")}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg font-medium transition ${
              currentPage === "result"
                ? "bg-indigo-600 text-white shadow-sm"
                : "text-slate-300 hover:bg-slate-800"
            }`}
          >
            <BarChart3 className="h-3.5 w-3.5" />
            <span>Result</span>
          </button>

          <button
            onClick={() => setCurrentPage("map")}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg font-medium transition ${
              currentPage === "map"
                ? "bg-indigo-600 text-white shadow-sm"
                : "text-slate-300 hover:bg-slate-800"
            }`}
          >
            <MapPin className="h-3.5 w-3.5" />
            <span>Engineer Map</span>
          </button>
        </nav>
      </header>

      {/* Main Page Content Body */}
      <main className="flex-1 p-4 sm:p-6 max-w-7xl w-full mx-auto">
        {currentPage === "scan" && (
          <ScanPage onScanComplete={handleScanComplete} />
        )}

        {currentPage === "result" && (
          <ResultPage
            scanId={selectedScanId}
            onNavigate={handleNavigate}
          />
        )}

        {currentPage === "map" && (
          <EngineerMapPage
            initialScanId={selectedScanId}
            onSelectScan={(id) => handleNavigate("result", id)}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 px-6 py-4 text-center text-xs text-slate-500">
        Project SETU • Bridge Modal Vibration Telemetry • Chennai & Mumbai Benchmarks
      </footer>
    </div>
  );
}
