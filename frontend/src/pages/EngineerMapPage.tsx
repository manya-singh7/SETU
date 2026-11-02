import React, { useEffect, useState, useRef, useMemo } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import {
  Activity,
  ChevronRight,
  Filter,
  MapPin,
  ExternalLink,
} from "lucide-react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
import { getBridges, getScan } from "../api";
import { BridgeSummary, ScanRecord, TriageSeverity } from "../types";

// Fix Leaflet's default marker icon issue in bundlers
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

const SEVERITY_COLORS: Record<string, string> = {
  high: "#ef4444",
  medium: "#f97316",
  low: "#eab308",
  none: "#10b981",
};

interface EngineerMapPageProps {
  onSelectScan: (scanId: string) => void;
  initialScanId?: string;
}

export default function EngineerMapPage({ onSelectScan, initialScanId }: EngineerMapPageProps) {
  const [bridges, setBridges] = useState<BridgeSummary[]>([]);
  const [selectedBridge, setSelectedBridge] = useState<BridgeSummary | null>(null);
  const [selectedScanDetail, setSelectedScanDetail] = useState<ScanRecord | null>(null);
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

  // 2. Fetch full scan details when selected bridge changes
  useEffect(() => {
    if (selectedBridge?.id) {
      getScan(selectedBridge.id)
        .then((detail) => setSelectedScanDetail(detail))
        .catch(() => setSelectedScanDetail(null));
    }
  }, [selectedBridge]);

  // 3. Filter and Sort Triage Queue
  // Rule: Sorted by severity (high to none), then oldest last-checked date
  const filteredAndSortedBridges = useMemo(() => {
    const now = Date.now();

    return bridges
      .filter((b) => {
        if (severityFilter !== "all" && b.severity.toLowerCase() !== severityFilter) {
          return false;
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
        // Primary sort: Severity priority
        const weightA = SEVERITY_WEIGHT[a.severity.toLowerCase()] || 0;
        const weightB = SEVERITY_WEIGHT[b.severity.toLowerCase()] || 0;
        if (weightB !== weightA) {
          return weightB - weightA;
        }
        // Secondary sort: Oldest last-checked date first
        const timeA = new Date(a.timestamp).getTime();
        const timeB = new Date(b.timestamp).getTime();
        return timeA - timeB;
      });
  }, [bridges, severityFilter, dateRangeFilter, cityFilter]);

  // Severity counts
  const counts = useMemo(() => {
    const res: Record<string, number> = { high: 0, medium: 0, low: 0, none: 0 };
    bridges.forEach((b) => {
      const s = b.severity?.toLowerCase();
      if (res[s] !== undefined) res[s]++;
    });
    return res;
  }, [bridges]);

  // 4. Initialize Leaflet Map
  useEffect(() => {
    if (!mapContainerRef.current) return;

    if (!mapInstanceRef.current) {
      // Center between Chennai (13.08) and Mumbai (19.07)
      const map = L.map(mapContainerRef.current, {
        center: [16.0, 76.5],
        zoom: 6,
        zoomControl: true,
      });

      L.tileLayer("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png", {
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

  // 5. Update Map Markers
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    Object.values(markersRef.current).forEach((marker) => marker.remove());
    markersRef.current = {};

    filteredAndSortedBridges.forEach((bridge) => {
      const color = SEVERITY_COLORS[bridge.severity?.toLowerCase()] || "#38bdf8";
      const isSelected = selectedBridge?.id === bridge.id;

      const circle = L.circleMarker([bridge.lat, bridge.lon], {
        radius: isSelected ? 12 : 9,
        fillColor: color,
        color: isSelected ? "#ffffff" : color,
        weight: isSelected ? 3 : 1.5,
        opacity: 1,
        fillOpacity: 0.85,
      }).addTo(map);

      circle.bindTooltip(
        `<div style="font-family: Inter, sans-serif; font-size: 12px; padding: 2px;">
          <strong>${bridge.bridge_name}</strong><br/>
          <span style="color: ${color}; font-weight: 600;">${bridge.severity.toUpperCase()}</span> | 
          ${bridge.fundamental_hz ? `${bridge.fundamental_hz.toFixed(2)} Hz` : "N/A"}
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

  const getSeverityBadgeClass = (sev: string) => {
    switch (sev?.toLowerCase()) {
      case "high":
        return "bg-rose-950 text-rose-300 border border-rose-800";
      case "medium":
        return "bg-orange-950 text-orange-300 border border-orange-800";
      case "low":
        return "bg-amber-950 text-amber-300 border border-amber-800";
      default:
        return "bg-emerald-950 text-emerald-300 border border-emerald-800";
    }
  };

  // Prepare history points for Recharts
  const historyChartData = useMemo(() => {
    const hist = selectedScanDetail?.history || selectedBridge?.history || [];
    return hist.map((item) => {
      const dateObj = new Date(item.timestamp);
      return {
        dateStr: `${dateObj.getMonth() + 1}/${dateObj.getDate()}/${String(dateObj.getFullYear()).slice(2)}`,
        fullDate: dateObj.toLocaleDateString(),
        freq_hz: item.freq_hz,
      };
    });
  }, [selectedScanDetail, selectedBridge]);

  return (
    <div className="space-y-4">
      {/* Header & Focus Region Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-white tracking-tight">
            Civil Engineer Infrastructure Triage Map
          </h1>
          <p className="text-xs text-slate-400">
            Real-time modal frequency screening across Chennai and Mumbai bridge networks
          </p>
        </div>

        <div className="flex items-center space-x-2 text-xs">
          <span className="text-slate-500 text-[11px]">Region:</span>
          <button
            onClick={() => handleZoomCity("all")}
            className={`px-3 py-1.5 rounded-lg border font-medium ${
              cityFilter === "all"
                ? "bg-indigo-600 border-indigo-500 text-white"
                : "bg-slate-900 border-slate-800 text-slate-300 hover:bg-slate-800"
            }`}
          >
            All India
          </button>
          <button
            onClick={() => handleZoomCity("chennai")}
            className={`px-3 py-1.5 rounded-lg border font-medium ${
              cityFilter === "Chennai"
                ? "bg-indigo-600 border-indigo-500 text-white"
                : "bg-slate-900 border-slate-800 text-slate-300 hover:bg-slate-800"
            }`}
          >
            Chennai (6)
          </button>
          <button
            onClick={() => handleZoomCity("mumbai")}
            className={`px-3 py-1.5 rounded-lg border font-medium ${
              cityFilter === "Mumbai"
                ? "bg-indigo-600 border-indigo-500 text-white"
                : "bg-slate-900 border-slate-800 text-slate-300 hover:bg-slate-800"
            }`}
          >
            Mumbai (6)
          </button>
        </div>
      </div>

      {/* Main Grid: Map on Left (7 cols), Triage Queue on Right (5 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Map Column */}
        <div className="lg:col-span-7 rounded-xl border border-slate-800 bg-slate-900/60 p-3 flex flex-col space-y-2">
          {/* Severity Legend Header */}
          <div className="flex items-center justify-between text-xs px-1 text-slate-300">
            <div className="flex items-center space-x-3">
              <span className="flex items-center space-x-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-rose-500" />
                <span>High ({counts.high})</span>
              </span>
              <span className="flex items-center space-x-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-orange-500" />
                <span>Med ({counts.medium})</span>
              </span>
              <span className="flex items-center space-x-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-amber-500" />
                <span>Low ({counts.low})</span>
              </span>
              <span className="flex items-center space-x-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
                <span>None ({counts.none})</span>
              </span>
            </div>
            <span className="text-[11px] text-slate-500">Click marker to inspect</span>
          </div>

          <div className="h-[460px] w-full rounded-lg overflow-hidden border border-slate-800">
            <div ref={mapContainerRef} className="h-full w-full" />
          </div>
        </div>

        {/* Triage Queue & Bridge Detail Column */}
        <div className="lg:col-span-5 flex flex-col space-y-3">
          {/* Filters Card */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-3.5 space-y-2.5 text-xs">
            <div className="flex items-center space-x-2 font-semibold text-slate-200">
              <Filter className="h-3.5 w-3.5 text-indigo-400" />
              <span>Triage Queue Filters</span>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-[10px] text-slate-400 uppercase mb-1">
                  Severity Urgency
                </label>
                <select
                  value={severityFilter}
                  onChange={(e) => setSeverityFilter(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-200"
                >
                  <option value="all">All Severities ({bridges.length})</option>
                  <option value="high">High ({counts.high})</option>
                  <option value="medium">Medium ({counts.medium})</option>
                  <option value="low">Low ({counts.low})</option>
                  <option value="none">None ({counts.none})</option>
                </select>
              </div>

              <div>
                <label className="block text-[10px] text-slate-400 uppercase mb-1">
                  Last Checked Range
                </label>
                <select
                  value={dateRangeFilter}
                  onChange={(e) => setDateRangeFilter(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-200"
                >
                  <option value="all">All History</option>
                  <option value="7d">Last 7 Days</option>
                  <option value="30d">Last 30 Days</option>
                  <option value="gt30d">&gt; 30 Days Ago</option>
                </select>
              </div>
            </div>
          </div>

          {/* Selected Bridge Inspection & Frequency-over-Time Chart */}
          {selectedBridge && (
            <div className="rounded-xl border border-indigo-500/70 bg-slate-900/90 p-4 space-y-3 shadow-lg">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <div className="flex items-center space-x-2">
                    <h3 className="font-bold text-sm text-white">{selectedBridge.bridge_name}</h3>
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase ${getSeverityBadgeClass(selectedBridge.severity)}`}>
                      {selectedBridge.severity}
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-400 mt-0.5">
                    {selectedBridge.city} • Checked: {new Date(selectedBridge.timestamp).toLocaleDateString()}
                  </div>
                </div>

                <button
                  onClick={() => onSelectScan(selectedBridge.id)}
                  className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shrink-0"
                >
                  <span>Inspect</span>
                  <ExternalLink className="h-3 w-3" />
                </button>
              </div>

              {/* Frequency-over-Time Chart */}
              <div className="space-y-1.5 pt-1">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-medium text-indigo-300 flex items-center space-x-1.5">
                    <Activity className="h-3.5 w-3.5 text-indigo-400" />
                    <span>Fundamental Frequency History</span>
                  </span>
                  <span className="font-mono text-slate-400 text-[11px]">
                    Current: {selectedBridge.fundamental_hz ? `${selectedBridge.fundamental_hz.toFixed(2)} Hz` : "N/A"}
                  </span>
                </div>

                <div className="w-full h-28 bg-slate-950/60 rounded-lg p-1.5 border border-slate-800">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={historyChartData} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="dateStr" stroke="#64748b" tick={{ fontSize: 10 }} />
                      <YAxis
                        stroke="#64748b"
                        tick={{ fontSize: 10 }}
                        domain={["auto", "auto"]}
                        unit="Hz"
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", borderRadius: 6, fontSize: 11 }}
                        formatter={(val) => [`${val} Hz`, "Frequency"]}
                        labelFormatter={(_label, payload) => payload?.[0]?.payload?.fullDate || ""}
                      />
                      <Line
                        type="monotone"
                        dataKey="freq_hz"
                        stroke="#818cf8"
                        strokeWidth={2}
                        dot={{ r: 3, fill: "#818cf8", strokeWidth: 1, stroke: "#fff" }}
                        activeDot={{ r: 5 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="text-[11px] text-slate-300 bg-slate-950 p-2.5 rounded-lg border-l-2 border-indigo-500">
                <strong className="text-indigo-400">Triage Directive: </strong>
                {selectedBridge.recommended_action || "Schedule routine physical inspection per matrix."}
              </div>
            </div>
          )}

          {/* Triage Queue List */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3 space-y-2 flex-1 overflow-hidden flex flex-col">
            <div className="flex justify-between items-center text-xs px-1">
              <span className="font-semibold text-slate-300">
                TRIAGE QUEUE ({filteredAndSortedBridges.length} BRIDGES)
              </span>
              <span className="text-[10px] text-slate-500">
                Sorted: Severity &gt; Oldest Date
              </span>
            </div>

            <div className="space-y-1.5 overflow-y-auto max-h-56 pr-1">
              {filteredAndSortedBridges.length === 0 ? (
                <div className="text-center py-8 text-xs text-slate-500">
                  No bridges match the current filter criteria.
                </div>
              ) : (
                filteredAndSortedBridges.map((b) => {
                  const isSelected = selectedBridge?.id === b.id;
                  return (
                    <div
                      key={b.id}
                      onClick={() => setSelectedBridge(b)}
                      className={`p-2.5 rounded-lg border cursor-pointer transition flex items-center justify-between text-xs ${
                        isSelected
                          ? "border-indigo-500 bg-indigo-950/40"
                          : "border-slate-800 bg-slate-950/60 hover:border-slate-700"
                      }`}
                    >
                      <div>
                        <div className="flex items-center space-x-2">
                          <span className="font-semibold text-white">{b.bridge_name}</span>
                          <span className={`text-[9px] font-bold px-1.5 py-0.2 rounded uppercase ${getSeverityBadgeClass(b.severity)}`}>
                            {b.severity}
                          </span>
                        </div>
                        <div className="flex items-center space-x-2 text-[10px] text-slate-400 mt-1">
                          <span>{b.city}</span>
                          <span>•</span>
                          <span className="font-mono text-indigo-400">
                            {b.fundamental_hz ? `${b.fundamental_hz.toFixed(2)} Hz` : "N/A"}
                          </span>
                          <span>•</span>
                          <span>Checked: {new Date(b.timestamp).toLocaleDateString()}</span>
                        </div>
                      </div>

                      <ChevronRight className={`h-4 w-4 ${isSelected ? "text-indigo-400" : "text-slate-600"}`} />
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
