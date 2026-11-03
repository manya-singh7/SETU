import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

interface HistoryDataPoint {
  dateStr: string;
  fullDate: string;
  freq_hz: number;
}

interface BridgeFrequencyHistoryChartProps {
  data: HistoryDataPoint[];
}

export default function BridgeFrequencyHistoryChart({
  data,
}: BridgeFrequencyHistoryChartProps) {
  return (
    <div className="w-full h-28 bg-[#E9E7E2]/30 rounded-[4px] p-1.5 border border-[#CCCCCC]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#CCCCCC" />
          <XAxis dataKey="dateStr" stroke="#22262A" tick={{ fontSize: 10 }} />
          <YAxis dataKey="freq_hz" stroke="#22262A" tick={{ fontSize: 10 }} domain={["auto", "auto"]} unit="Hz" />
          <Tooltip
            contentStyle={{ backgroundColor: "#FFFFFF", borderColor: "#22262A", borderRadius: 4, fontSize: 11 }}
            formatter={(val) => [`${val} Hz`, "Frequency"]}
          />
          <Line
            type="monotone"
            dataKey="freq_hz"
            stroke="#0B6E4F"
            strokeWidth={2}
            dot={{ r: 3, fill: "#0B6E4F", strokeWidth: 1, stroke: "#FFFFFF" }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
