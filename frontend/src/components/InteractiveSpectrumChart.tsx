import React from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ReferenceDot,
  CartesianGrid,
} from "recharts";

interface InteractiveSpectrumChartProps {
  data: Array<{ freq_hz: number; power: number }>;
  primaryPeak?: { frequency_hz: number; relative_power: number };
}

export default function InteractiveSpectrumChart({
  data,
  primaryPeak,
}: InteractiveSpectrumChartProps) {
  return (
    <div className="w-full h-64 pt-2">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 15, right: 20, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#D5D2CA" />
          <XAxis
            dataKey="freq_hz"
            stroke="#555B62"
            tick={{ fontSize: 11 }}
            unit=" Hz"
            label={{
              value: "Frequency (Hz)",
              position: "insideBottom",
              offset: -2,
              fill: "#22262A",
              fontSize: 11,
            }}
          />
          <YAxis
            stroke="#555B62"
            tick={{ fontSize: 11 }}
            label={{
              value: "Power",
              angle: -90,
              position: "insideLeft",
              fill: "#22262A",
              fontSize: 11,
            }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#FFFFFF",
              borderColor: "#C4C1B8",
              borderRadius: 4,
              fontSize: 12,
            }}
            formatter={(val) => [Number(val).toFixed(4), "Power"]}
            labelFormatter={(label) => `${label} Hz`}
          />
          {primaryPeak && (
            <ReferenceLine
              x={primaryPeak.frequency_hz}
              stroke="#0B6E4F"
              strokeDasharray="4 4"
              strokeWidth={2}
              label={{
                value: `Peak: ${primaryPeak.frequency_hz.toFixed(2)} Hz`,
                position: "top",
                fill: "#0B6E4F",
                fontSize: 11,
                fontWeight: 700,
              }}
            />
          )}
          {primaryPeak && (
            <ReferenceDot
              x={primaryPeak.frequency_hz}
              y={primaryPeak.relative_power}
              r={5}
              fill="#0B6E4F"
              stroke="#fff"
              strokeWidth={2}
            />
          )}
          <Area
            type="monotone"
            dataKey="power"
            stroke="#0B6E4F"
            strokeWidth={2}
            fill="#0B6E4F"
            fillOpacity={0.15}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
