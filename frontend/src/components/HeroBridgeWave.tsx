import React from "react";

export default function HeroBridgeWave() {
  return (
    <div className="relative w-full max-w-2xl mx-auto py-2 px-4 select-none">
      <svg
        viewBox="0 0 640 180"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-auto drop-shadow-sm"
      >
        {/* Background baseline */}
        <line x1="20" y1="100" x2="620" y2="100" stroke="#1F2A44" strokeWidth="2.5" strokeLinecap="round" />
        <line x1="20" y1="105" x2="620" y2="105" stroke="#1F2A44" strokeWidth="1" strokeOpacity="0.4" />

        {/* Pier foundations */}
        <path d="M70 105 L60 150 L100 150 L90 105 Z" fill="#1F2A44" fillOpacity="0.08" stroke="#1F2A44" strokeWidth="1.5" />
        <path d="M230 105 L220 150 L260 150 L250 105 Z" fill="#1F2A44" fillOpacity="0.08" stroke="#1F2A44" strokeWidth="1.5" />
        <path d="M390 105 L380 150 L420 150 L410 105 Z" fill="#1F2A44" fillOpacity="0.08" stroke="#1F2A44" strokeWidth="1.5" />
        <path d="M550 105 L540 150 L580 150 L570 105 Z" fill="#1F2A44" fillOpacity="0.08" stroke="#1F2A44" strokeWidth="1.5" />

        {/* Bridge Piers & Truss Spans */}
        {/* Span 1 (Arch-Truss) */}
        <path d="M80 100 Q160 30 240 100" stroke="#1F2A44" strokeWidth="2" fill="none" />
        <line x1="120" y1="65" x2="120" y2="100" stroke="#1F2A44" strokeWidth="1" strokeOpacity="0.6" />
        <line x1="160" y1="48" x2="160" y2="100" stroke="#1F2A44" strokeWidth="1.25" strokeOpacity="0.7" />
        <line x1="200" y1="65" x2="200" y2="100" stroke="#1F2A44" strokeWidth="1" strokeOpacity="0.6" />
        <line x1="80" y1="100" x2="160" y2="48" stroke="#1F2A44" strokeWidth="0.75" strokeOpacity="0.4" />
        <line x1="240" y1="100" x2="160" y2="48" stroke="#1F2A44" strokeWidth="0.75" strokeOpacity="0.4" />

        {/* Span 2 (Center Span with Cable Stays or Arch) */}
        <path d="M240 100 Q320 20 400 100" stroke="#1F2A44" strokeWidth="2.5" fill="none" />
        <line x1="280" y1="60" x2="280" y2="100" stroke="#1F2A44" strokeWidth="1" strokeOpacity="0.6" />
        <line x1="320" y1="40" x2="320" y2="100" stroke="#E8912D" strokeWidth="1.5" />
        <line x1="360" y1="60" x2="360" y2="100" stroke="#1F2A44" strokeWidth="1" strokeOpacity="0.6" />
        <circle cx="320" cy="40" r="3" fill="#E8912D" />

        {/* Span 3 (Arch-Truss) */}
        <path d="M400 100 Q480 30 560 100" stroke="#1F2A44" strokeWidth="2" fill="none" />
        <line x1="440" y1="65" x2="440" y2="100" stroke="#1F2A44" strokeWidth="1" strokeOpacity="0.6" />
        <line x1="480" y1="48" x2="480" y2="100" stroke="#1F2A44" strokeWidth="1.25" strokeOpacity="0.7" />
        <line x1="520" y1="65" x2="520" y2="100" stroke="#1F2A44" strokeWidth="1" strokeOpacity="0.6" />

        {/* Micro-vibration modal wave (Deflection mode 1: sine wave) */}
        {/* Animated gentle wave beneath the bridge */}
        <g className="animate-wave">
          <path
            d="M20 135 C100 120, 160 150, 240 135 C320 120, 400 150, 480 135 C540 120, 580 145, 620 135"
            stroke="#E8912D"
            strokeWidth="1.75"
            strokeLinecap="round"
            fill="none"
            opacity="0.85"
          />
          <path
            d="M20 138 C100 125, 160 152, 240 138 C320 125, 400 152, 480 138 C540 125, 580 147, 620 138"
            stroke="#0D766E"
            strokeWidth="1"
            strokeDasharray="4 4"
            fill="none"
            opacity="0.45"
          />
        </g>

        {/* Resonant vibration node labels */}
        <circle cx="320" cy="135" r="4" fill="#E8912D" className="animate-pulse" />
        <text
          x="320"
          y="165"
          textAnchor="middle"
          fill="#1F2A44"
          fontSize="10"
          fontFamily="Inter, sans-serif"
          fontWeight="500"
          opacity="0.75"
        >
          f₀ Fundamental Resonance Mode
        </text>
      </svg>
    </div>
  );
}
