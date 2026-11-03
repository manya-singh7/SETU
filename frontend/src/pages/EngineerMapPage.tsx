import React, { useEffect, useState, useRef, useMemo, Suspense, lazy } from "react";
import { useTranslation } from "react-i18next";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import {
  Activity,
  ChevronRight,
  Filter,
  ExternalLink,
} from "lucide-react";
import { getBridges } from "../api";
import { BridgeSummary } from "../types";
import { formatLocaleNumber, formatLocaleDate } from "../i18n";

// Code-split chart component so Recharts is loaded asynchronously
const BridgeFrequencyHistoryChart = lazy(() => import("../components/BridgeFrequencyHistoryChart"));

// Fix Leaflet default marker icons for bundlers
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

delete (L.Icon.Default.prototype as unknown as { _getIconUrl?: unknown })._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

const SEVERITY_WEIGHT: Record<string, number> = {
  high: 4,
  medium: 3,
  low: 2,
  none: 1,
};

// Four clearly distinct colours for usable scans from Indian road signage palette:
// High is hazard red (most prominent), Medium is milestone yellow, Low is light green, None is highway green.
const SEVERITY_COLORS: Record<string, string> = {
  high: "#B91C1C",    // IRC hazard red (most prominent)
  medium: "#F2C230",  // milestone yellow
  low: "#4A8B6B",     // signpost light green
  none: "#0B6E4F",    // highway green (nominal)
};

interface EngineerMapPageProps {
  onSelectScan: (scanId: string) => void;
  initialScanId?: string;
}

export default function EngineerMapPage({ onSelectScan, initialScanId }: EngineerMapPageProps) {
  const { t } = useTranslation();

  const [bridges, setBridges] = useState<BridgeSummary[]>([]);
  const [selectedBridge, setSelectedBridge] = useState<BridgeSummary | null>(null);
  const [severityFilter, setSeverityFilter] = useState<string>("all");
  const [dateRangeFilter, setDateRangeFilter] = useState<string>("all");
  const [cityFilter, setCityFilter] = useState<string>("all");

  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const markersRef = useRef<Record<string, L.CircleMarker>>({});

  // 1. Fetch bridges on mount
  useEffect(() => {
    getBridges().then((data) => {
      setBridges(data);
      if (data.length > 0) {
        if (initialScanId) {
          const match = data.find((b) => b.id === initialScanId);
          setSelectedBridge(match || data[0]);
        } else {
          setSelectedBridge(data[0]);
        }
      }
    });
  }, [initialScanId]);

  // Never show demo results as real scans
  const isSampleData = useMemo(() => {
    return bridges.length > 0 && bridges.some((b) => b.is_sample_data || b.is_demo);
  }, [bridges]);

  // 2. Filter and Sort Triage Queue
  const filteredAndSortedBridges = useMemo(() => {
    const now = Date.now();

    return bridges
      .filter((b) => {
        const isWeakOrFailed = b.verdict_level === "weak" || b.verdict_level === "failed" || !b.fundamental_hz;

        if (severityFilter === "not_enough_data") {
          if (!isWeakOrFailed) return false;
        } else if (severityFilter !== "all") {
          if (isWeakOrFailed) return false;
          if (b.severity.toLowerCase() !== severityFilter) return false;
        }

        if (cityFilter !== "all" && (b.city || "").toLowerCase() !== cityFilter.toLowerCase()) {
          return false;
        }

        if (dateRangeFilter !== "all") {
          const checkTime = new Date(b.timestamp).getTime();
          const diffDays = (now - checkTime) / (1000 * 3600 * 24);
          if (dateRangeFilter === "7d" && diffDays > 7) return false;
          if (dateRangeFilter === "30d" && diffDays > 30) return false;
          if (dateRangeFilter === "gt30d" && diffDays <= 30) return false;
        }

        return true;
      })
      .sort((a, b) => {
        const weightA = SEVERITY_WEIGHT[a.severity.toLowerCase()] || 0;
        const weightB = SEVERITY_WEIGHT[b.severity.toLowerCase()] || 0;
        if (weightB !== weightA) {
          return weightB - weightA;
        }
        const timeA = new Date(a.timestamp).getTime();
        const timeB = new Date(b.timestamp).getTime();
        return timeA - timeB;
      });
  }, [bridges, severityFilter, dateRangeFilter, cityFilter]);

  // Severity counts
  const counts = useMemo(() => {
    const res = { high: 0, medium: 0, low: 0, none: 0, notEnoughData: 0 };
    bridges.forEach((b) => {
      const isWeakOrFailed = b.verdict_level === "weak" || b.verdict_level === "failed" || !b.fundamental_hz;
      if (isWeakOrFailed) {
        res.notEnoughData++;
      } else {
        const s = b.severity?.toLowerCase();
        if (s === "high") res.high++;
        else if (s === "medium") res.medium++;
        else if (s === "low") res.low++;
        else res.none++;
      }
    });
    return res;
  }, [bridges]);

  // 3. Initialize Leaflet Map with CARTO Positron muted light tiles
  useEffect(() => {
    if (!mapContainerRef.current) return;

    if (!mapInstanceRef.current) {
      const map = L.map(mapContainerRef.current, {
        center: [16.0, 76.5],
        zoom: 6,
        zoomControl: true,
      });

      // Muted light tiles: CARTO Positron
      L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
        attribution: '&copy; <a href="https://carto.com/">CARTO</a>, &copy; OpenStreetMap',
        maxZoom: 18,
        subdomains: "abcd",
      }).addTo(map);

      mapInstanceRef.current = map;
    }

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  // 4. Update Map Markers
  // Usable scans: Solid coloured dots by assessment severity (none, low, medium, high - high being most prominent red).
  // Weak and failed scans: Hollow grey rings. Never use the failed grey for high severity.
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    Object.values(markersRef.current).forEach((marker) => marker.remove());
    markersRef.current = {};

    filteredAndSortedBridges.forEach((bridge) => {
      const isSelected = selectedBridge?.id === bridge.id;
      const isWeakOrFailed = bridge.verdict_level === "weak" || bridge.verdict_level === "failed" || !bridge.fundamental_hz;

      let circle: L.CircleMarker;

      if (isWeakOrFailed) {
        // Hollow grey ring for weak and failed scans
        circle = L.circleMarker([bridge.lat, bridge.lon], {
          radius: isSelected ? 11 : 8,
          fillColor: "transparent",
          color: isSelected ? "#22262A" : "#737982",
          weight: isSelected ? 3 : 2,
          opacity: 1,
          fillOpacity: 0,
        }).addTo(map);
      } else {
        // Solid coloured dot by assessment severity
        const color = SEVERITY_COLORS[bridge.severity?.toLowerCase()] || "#0B6E4F";
        circle = L.circleMarker([bridge.lat, bridge.lon], {
          radius: isSelected ? 11 : 8,
          fillColor: color,
          color: isSelected ? "#22262A" : "#FFFFFF",
          weight: isSelected ? 3 : 2,
          opacity: 1,
          fillOpacity: 0.95,
        }).addTo(map);
      }

      // Demo/sample indicator
      const isDemoBridge = bridge.is_demo || bridge.is_sample_data;
      const demoLabel = isDemoBridge
        ? `<div style="color: #22262A; background-color: #F2C230; font-size: 10px; font-weight: 700; padding: 1px 4px; display: inline-block; margin-top: 2px;">SAMPLE DATA</div>`
        : `<div style="color: #0B6E4F; font-size: 10px; font-weight: 700;">LIVE SCAN</div>`;

      const statusText = isWeakOrFailed
        ? `<div style="color: #737982; font-weight: 700; margin-top: 2px;">NOT ENOUGH DATA</div>`
        : `<div style="margin-top: 3px;">
             <span style="font-weight: 700;">${bridge.severity.toUpperCase()}</span> | 
             <span style="font-family: monospace;">${bridge.fundamental_hz ? `${bridge.fundamental_hz.toFixed(2)} Hz` : "N/A"}</span>
           </div>`;

      circle.bindTooltip(
        `<div style="font-family: Hind, sans-serif; font-size: 12px; padding: 4px; color: #22262A;">
          <strong>${bridge.bridge_name}</strong>
          ${demoLabel}
          ${statusText}
        </div>`,
        { direction: "top", offset: [0, -6] }
      );

      circle.on("click", () => {
        setSelectedBridge(bridge);
      });

      markersRef.current[bridge.id] = circle;
    });
  }, [filteredAndSortedBridges, selectedBridge]);

  const handleZoomCity = (cityName: "all" | "chennai" | "mumbai") => {
    const map = mapInstanceRef.current;
    if (!map) return;
    if (cityName === "chennai") {
      map.flyTo([13.0674, 80.25], 12);
      setCityFilter("Chennai");
    } else if (cityName === "mumbai") {
      map.flyTo([19.05, 72.88], 11);
      setCityFilter("Mumbai");
    } else {
      map.flyTo([16.0, 76.5], 6);
      setCityFilter("all");
    }
  };

  const getSeverityBadgeClass = (bridge: BridgeSummary) => {
    const isWeakOrFailed = bridge.verdict_level === "weak" || bridge.verdict_level === "failed" || !bridge.fundamental_hz;
    if (isWeakOrFailed) {
      return "border border-[#737982] text-[#737982] bg-white";
    }
    switch (bridge.severity?.toLowerCase()) {
      case "high":
        return "bg-[#B91C1C] text-white border border-[#B91C1C]"; // IRC hazard red
      case "medium":
        return "bg-[#F2C230] text-[#22262A] border border-[#22262A]"; // milestone yellow
      case "low":
        return "bg-[#4A8B6B] text-white border border-[#4A8B6B]"; // signpost light green
      case "none":
      default:
        return "bg-[#0B6E4F] text-white border border-[#0B6E4F]"; // highway green
    }
  };

  const historyChartData = useMemo(() => {
    const hist = selectedBridge?.history || [];
    return hist.map((item) => {
      const dateObj = new Date(item.timestamp);
      return {
        dateStr: `${dateObj.getMonth() + 1}/${dateObj.getDate()}/${String(dateObj.getFullYear()).slice(2)}`,
        fullDate: dateObj.toLocaleDateString(),
        freq_hz: item.freq_hz,
      };
    });
  }, [selectedBridge]);

  return (
    <div className="space-y-4 py-2 font-sans text-[#22262A]">
      {/* Sample Data Notice Banner if backend is empty/offline */}
      {isSampleData && (
        <div className="border border-[#22262A] bg-white p-3 flex items-center justify-between flex-wrap gap-2 text-xs text-[#22262A] rounded-[4px]">
          <div className="flex items-center space-x-2.5">
            <span className="px-2 py-0.5 rounded-[4px] bg-[#F2C230] text-[#22262A] border border-[#22262A] font-bold text-[11px] uppercase">
              {t("map.sample_data_badge", "Sample data")}
            </span>
            <span className="font-sans font-medium">
              {t(
                "map.sample_data_banner",
                "Live server returned no data or is offline. Displaying benchmark sample dataset with clear labeling."
              )}
            </span>
          </div>
          <span className="text-[11px] font-mono text-[#475569] font-bold">
            {t("map.offline_mode", "Offline Fallback Mode")}
          </span>
        </div>
      )}

      {/* Header & Region Switcher (NO developer text) */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#CCCCCC] pb-3">
        <div>
          <div className="flex items-center space-x-2.5">
            <h1 className="text-xl sm:text-2xl font-heading font-bold text-[#22262A] tracking-tight">
              {t("map.title", "Engineer bridge map")}
            </h1>
            {isSampleData && (
              <span className="px-2 py-0.5 rounded-[4px] bg-[#F2C230] text-[#22262A] border border-[#22262A] text-[10px] font-bold uppercase">
                {t("map.sample_data_badge", "Sample data")}
              </span>
            )}
          </div>
          <p className="text-xs text-[#22262A]/80 font-sans mt-0.5">
            {t(
              "map.subtitle",
              "Modal frequency screening across regional bridge networks"
            )}
          </p>
        </div>

        <div className="flex items-center space-x-2 text-xs">
          <span className="text-[#22262A]/70 font-semibold">{t("map.region", "Region:")}</span>
          <button
            type="button"
            onClick={() => handleZoomCity("all")}
            className={`px-3 py-1.5 rounded-[4px] border font-semibold transition ${
              cityFilter === "all"
                ? "bg-[#0B6E4F] border-[#0B6E4F] text-white"
                : "bg-white border-[#22262A]/30 text-[#22262A] hover:bg-[#E9E7E2]"
            }`}
          >
            {t("map.all_india", "All India")}
          </button>
          <button
            type="button"
            onClick={() => handleZoomCity("chennai")}
            className={`px-3 py-1.5 rounded-[4px] border font-semibold transition ${
              cityFilter === "Chennai"
                ? "bg-[#0B6E4F] border-[#0B6E4F] text-white"
                : "bg-white border-[#22262A]/30 text-[#22262A] hover:bg-[#E9E7E2]"
            }`}
          >
            {t("map.chennai", "Chennai")}
          </button>
          <button
            type="button"
            onClick={() => handleZoomCity("mumbai")}
            className={`px-3 py-1.5 rounded-[4px] border font-semibold transition ${
              cityFilter === "Mumbai"
                ? "bg-[#0B6E4F] border-[#0B6E4F] text-white"
                : "bg-white border-[#22262A]/30 text-[#22262A] hover:bg-[#E9E7E2]"
            }`}
          >
            {t("map.mumbai", "Mumbai")}
          </button>
        </div>
      </div>

      {/* Main Grid: Map on Left, Triage Queue on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Map Column */}
        <div className="lg:col-span-7 border border-[#22262A]/30 bg-white p-3 rounded-[4px] flex flex-col space-y-2.5">
          {/* Muted Legend Header with Distinct Coloured Dots & Hollow Grey Ring */}
          <div className="flex items-center justify-between text-xs px-1 text-[#22262A] flex-wrap gap-2">
            <div className="flex items-center space-x-3 flex-wrap gap-y-1">
              <span className="flex items-center space-x-1.5">
                <span className="h-3 w-3 rounded-full bg-[#B91C1C] border border-[#22262A]" />
                <span className="tabular-nums font-semibold">High ({counts.high})</span>
              </span>
              <span className="flex items-center space-x-1.5">
                <span className="h-3 w-3 rounded-full bg-[#F2C230] border border-[#22262A]" />
                <span className="tabular-nums font-semibold">Med ({counts.medium})</span>
              </span>
              <span className="flex items-center space-x-1.5">
                <span className="h-3 w-3 rounded-full bg-[#4A8B6B] border border-[#22262A]" />
                <span className="tabular-nums font-semibold">Low ({counts.low})</span>
              </span>
              <span className="flex items-center space-x-1.5">
                <span className="h-3 w-3 rounded-full bg-[#0B6E4F] border border-[#22262A]" />
                <span className="tabular-nums font-semibold">None ({counts.none})</span>
              </span>
              <span className="flex items-center space-x-1.5">
                <span className="h-3 w-3 rounded-full border-2 border-[#737982] bg-transparent" />
                <span className="tabular-nums font-semibold">{t("map.not_enough_data", "Not enough data")} ({counts.notEnoughData})</span>
              </span>
            </div>
            <span className="text-[11px] text-[#22262A]/60">{t("map.click_to_inspect", "Click marker to inspect")}</span>
          </div>

          <div className="h-[460px] w-full rounded-[4px] overflow-hidden border border-[#CCCCCC]">
            <div ref={mapContainerRef} className="h-full w-full" />
          </div>
        </div>

        {/* Triage Queue & Bridge Detail Column */}
        <div className="lg:col-span-5 flex flex-col space-y-3.5">
          {/* Filters Card */}
          <div className="border border-[#22262A]/30 bg-white p-3.5 space-y-2.5 text-xs rounded-[4px]">
            <div className="flex items-center space-x-2 font-heading font-bold text-[#22262A]">
              <Filter className="h-4 w-4 text-[#0B6E4F]" />
              <span>{t("map.triage_filters", "Filters")}</span>
            </div>

            <div className="grid grid-cols-2 gap-2.5">
              <div>
                <label className="block text-[10px] text-[#22262A]/70 uppercase font-semibold mb-1">
                  {t("map.severity_urgency", "Urgency")}
                </label>
                <select
                  value={severityFilter}
                  onChange={(e) => setSeverityFilter(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded-[4px] bg-[#E9E7E2]/50 border border-[#22262A]/30 text-[#22262A] font-medium"
                >
                  <option value="all">{t("map.all_severities", "All severities")} ({bridges.length})</option>
                  <option value="high">High ({counts.high})</option>
                  <option value="medium">Medium ({counts.medium})</option>
                  <option value="low">Low ({counts.low})</option>
                  <option value="none">None ({counts.none})</option>
                  <option value="not_enough_data">{t("map.not_enough_data", "Not enough data")} ({counts.notEnoughData})</option>
                </select>
              </div>

              <div>
                <label className="block text-[10px] text-[#22262A]/70 uppercase font-semibold mb-1">
                  {t("map.last_checked", "Date range")}
                </label>
                <select
                  value={dateRangeFilter}
                  onChange={(e) => setDateRangeFilter(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded-[4px] bg-[#E9E7E2]/50 border border-[#22262A]/30 text-[#22262A] font-medium"
                >
                  <option value="all">{t("map.all_history", "All dates")}</option>
                  <option value="7d">{t("map.last_7d", "Last 7 days")}</option>
                  <option value="30d">{t("map.last_30d", "Last 30 days")}</option>
                  <option value="gt30d">{t("map.gt_30d", "Older than 30 days")}</option>
                </select>
              </div>
            </div>
          </div>

          {/* Selected Bridge Inspection */}
          {selectedBridge && (
            <div className="border border-[#22262A]/30 bg-white p-3.5 space-y-3 rounded-[4px]">
              <div className="flex items-start justify-between gap-2 border-b border-[#CCCCCC] pb-2.5">
                <div>
                  <div className="flex items-center space-x-2 flex-wrap gap-y-1">
                    <h3 className="font-heading font-bold text-sm text-[#22262A]">{selectedBridge.bridge_name}</h3>
                    {selectedBridge.verdict_level === "weak" || selectedBridge.verdict_level === "failed" || !selectedBridge.fundamental_hz ? (
                      <span className="text-[10px] font-bold px-2 py-0.5 rounded-[4px] uppercase border border-[#737982] text-[#737982] bg-white">
                        {t("map.not_enough_data", "Not enough data")}
                      </span>
                    ) : (
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded-[4px] uppercase ${getSeverityBadgeClass(selectedBridge)}`}>
                        {selectedBridge.severity}
                      </span>
                    )}
                    {(selectedBridge.is_sample_data || selectedBridge.is_demo) && (
                      <span className="text-[9px] font-bold px-1.5 py-0.5 rounded-[4px] uppercase bg-[#F2C230] text-[#22262A] border border-[#22262A]">
                        {t("map.sample_data_badge", "Sample data")}
                      </span>
                    )}
                  </div>
                  <div className="text-[11px] text-[#22262A]/70 mt-0.5 tabular-nums">
                    {selectedBridge.city} • Checked: {formatLocaleDate(selectedBridge.timestamp, { dateStyle: "medium" })}
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => onSelectScan(selectedBridge.id)}
                  className="inline-flex items-center space-x-1 px-3 py-1.5 rounded-[4px] bg-[#0B6E4F] hover:bg-[#08523b] text-white text-xs font-semibold shrink-0"
                >
                  <span>{t("map.inspect", "Inspect scan")}</span>
                  <ExternalLink className="h-3 w-3" />
                </button>
              </div>

              {/* Fundamental Frequency History (Loaded asynchronously via Suspense) */}
              {historyChartData.length > 0 && (
                <div className="space-y-1.5 pt-1">
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-semibold text-[#22262A] flex items-center space-x-1.5 font-heading">
                      <Activity className="h-3.5 w-3.5 text-[#0B6E4F]" />
                      <span>{t("map.freq_history", "Frequency history")}</span>
                    </span>
                    <span className="font-mono text-[#22262A]/80 text-[11px] tabular-nums">
                      {t("map.current_freq", "Current")}: {selectedBridge.fundamental_hz ? `${selectedBridge.fundamental_hz.toFixed(2)} Hz` : "N/A"}
                    </span>
                  </div>

                  <Suspense fallback={<div className="h-28 flex items-center justify-center text-xs text-[#22262A]/60 font-sans">Loading history chart...</div>}>
                    <BridgeFrequencyHistoryChart data={historyChartData} />
                  </Suspense>
                </div>
              )}

              <div className="text-[11px] text-[#22262A] bg-[#E9E7E2]/40 p-2.5 rounded-[4px] border-l-4 border-[#0B6E4F]">
                <strong className="text-[#22262A]">{t("map.triage_directive", "Action:")} </strong>
                {selectedBridge.recommended_action || "Schedule routine physical inspection per matrix."}
              </div>
            </div>
          )}

          {/* Triage Queue List */}
          <div className="border border-[#22262A]/30 bg-white p-3.5 space-y-2.5 flex-1 overflow-hidden flex flex-col rounded-[4px]">
            <div className="flex justify-between items-center text-xs px-1 border-b border-[#CCCCCC] pb-2">
              <span className="font-heading font-bold text-[#22262A] tabular-nums">
                {t("map.triage_queue_title", "Bridges")} ({filteredAndSortedBridges.length})
              </span>
              <span className="text-[10px] text-[#22262A]/70 font-sans">
                {t("map.sorted_notice", "Sorted by urgency, then date")}
              </span>
            </div>

            <div className="space-y-1.5 overflow-y-auto max-h-56 pr-1 divide-y divide-[#E9E7E2]">
              {filteredAndSortedBridges.length === 0 ? (
                <div className="text-center py-6 text-xs text-[#22262A]/70">
                  {t("map.no_bridges_match", "No bridges match current filters.")}
                </div>
              ) : (
                filteredAndSortedBridges.map((b) => {
                  const isSelected = selectedBridge?.id === b.id;
                  const isDemo = b.is_demo || b.is_sample_data;
                  const isWeakOrFailed = b.verdict_level === "weak" || b.verdict_level === "failed" || !b.fundamental_hz;
                  return (
                    <div
                      key={b.id}
                      onClick={() => setSelectedBridge(b)}
                      className={`p-2.5 rounded-[4px] border cursor-pointer transition flex items-center justify-between text-xs ${
                        isSelected
                          ? "border-[#22262A] bg-[#E9E7E2]"
                          : "border-transparent hover:border-[#CCCCCC] hover:bg-[#E9E7E2]/40"
                      }`}
                    >
                      <div>
                        <div className="flex items-center space-x-2 flex-wrap gap-y-0.5">
                          <span className="font-heading font-semibold text-[#22262A]">{b.bridge_name}</span>
                          {isWeakOrFailed ? (
                            <span className="text-[9px] font-bold px-1.5 py-0.5 rounded-[4px] uppercase border border-[#737982] text-[#737982] bg-white">
                              {t("map.not_enough_data", "Not enough data")}
                            </span>
                          ) : (
                            <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded-[4px] uppercase ${getSeverityBadgeClass(b)}`}>
                              {b.severity}
                            </span>
                          )}
                          {isDemo && (
                            <span className="text-[9px] font-bold px-1.5 py-0.5 rounded-[4px] uppercase bg-[#F2C230] text-[#22262A] border border-[#22262A]">
                              {t("map.sample_data_badge", "Sample data")}
                            </span>
                          )}
                        </div>
                        <div className="flex items-center space-x-2 text-[10px] text-[#22262A]/70 mt-1 tabular-nums">
                          <span>{b.city}</span>
                          <span>•</span>
                          <span className="font-mono font-semibold text-[#22262A]">
                            {b.fundamental_hz ? `${formatLocaleNumber(b.fundamental_hz, { minimumFractionDigits: 2 })} Hz` : t("map.not_enough_data", "Not enough data")}
                          </span>
                          <span>•</span>
                          <span>Checked: {formatLocaleDate(b.timestamp, { dateStyle: "short" })}</span>
                        </div>
                        {b.bridge_key && <div className="mt-2 flex flex-wrap gap-3 text-[10px] font-semibold">
                          <a href={`/api/bridges/${encodeURIComponent(b.bridge_key)}/report.csv`} download className="underline text-[#0B6E4F]" onClick={(event) => event.stopPropagation()}>{t("map.download_csv", "Download report (CSV)")}</a>
                          <a href={`/report/${encodeURIComponent(b.bridge_key)}`} className="underline text-[#0B6E4F]" onClick={(event) => event.stopPropagation()}>{t("map.printable_report", "Printable report")}</a>
                        </div>}
                      </div>

                      <ChevronRight className={`h-4 w-4 ${isSelected ? "text-[#22262A]" : "text-[#22262A]/40"}`} />
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}