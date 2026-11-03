import React from "react";

interface BridgeArtProps {
  type?: "truss" | "stone-arch" | "cable-stayed" | "corridor";
  className?: string;
  opacity?: number;
}

export default function BridgeArt({
  type = "truss",
  className = "",
  opacity = 0.06,
}: BridgeArtProps) {
  if (type === "stone-arch") {
    return (
      <svg
        viewBox="0 0 600 160"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        style={{ opacity }}
        className={`pointer-events-none select-none ${className}`}
      >
        {/* Pier bases */}
        <rect x="80" y="80" width="40" height="70" stroke="#1F2A44" strokeWidth="1.5" />
        <rect x="240" y="80" width="40" height="70" stroke="#1F2A44" strokeWidth="1.5" />
        <rect x="400" y="80" width="40" height="70" stroke="#1F2A44" strokeWidth="1.5" />
        {/* Road deck */}
        <line x1="20" y1="40" x2="580" y2="40" stroke="#1F2A44" strokeWidth="2" />
        <line x1="20" y1="46" x2="580" y2="46" stroke="#1F2A44" strokeWidth="1" />
        {/* Stone arches */}
        <path
          d="M120 150 L120 90 Q180 30 240 90 L240 150"
          stroke="#1F2A44"
          strokeWidth="1.75"
        />
        <path
          d="M280 150 L280 90 Q340 30 400 90 L400 150"
          stroke="#1F2A44"
          strokeWidth="1.75"
        />
        {/* Parapet balustrade */}
        <line x1="40" y1="32" x2="560" y2="32" stroke="#1F2A44" strokeWidth="1" strokeDasharray="4 4" />
      </svg>
    );
  }

  if (type === "cable-stayed") {
    return (
      <svg
        viewBox="0 0 600 180"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        style={{ opacity }}
        className={`pointer-events-none select-none ${className}`}
      >
        {/* Pylon tower */}
        <line x1="300" y1="10" x2="285" y2="170" stroke="#1F2A44" strokeWidth="2.5" />
        <line x1="300" y1="10" x2="315" y2="170" stroke="#1F2A44" strokeWidth="2.5" />
        <line x1="288" y1="110" x2="312" y2="110" stroke="#1F2A44" strokeWidth="1.5" />
        {/* Deck */}
        <line x1="40" y1="110" x2="560" y2="110" stroke="#1F2A44" strokeWidth="2" />
        {/* Left stay cables */}
        <line x1="300" y1="30" x2="100" y2="110" stroke="#1F2A44" strokeWidth="1" />
        <line x1="300" y1="45" x2="140" y2="110" stroke="#1F2A44" strokeWidth="1" />
        <line x1="300" y1="60" x2="180" y2="110" stroke="#1F2A44" strokeWidth="1" />
        <line x1="300" y1="75" x2="220" y2="110" stroke="#1F2A44" strokeWidth="1" />
        <line x1="300" y1="90" x2="260" y2="110" stroke="#1F2A44" strokeWidth="1" />
        {/* Right stay cables */}
        <line x1="300" y1="30" x2="500" y2="110" stroke="#1F2A44" strokeWidth="1" />
        <line x1="300" y1="45" x2="460" y2="110" stroke="#1F2A44" strokeWidth="1" />
        <line x1="300" y1="60" x2="420" y2="110" stroke="#1F2A44" strokeWidth="1" />
        <line x1="300" y1="75" x2="380" y2="110" stroke="#1F2A44" strokeWidth="1" />
        <line x1="300" y1="90" x2="340" y2="110" stroke="#1F2A44" strokeWidth="1" />
      </svg>
    );
  }

  // Default: Warren/Pratt truss bridge
  return (
    <svg
      viewBox="0 0 600 140"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      style={{ opacity }}
      className={`pointer-events-none select-none ${className}`}
    >
      {/* Top Chord */}
      <line x1="80" y1="30" x2="520" y2="30" stroke="#1F2A44" strokeWidth="2" />
      {/* Bottom Chord (Deck) */}
      <line x1="40" y1="90" x2="560" y2="90" stroke="#1F2A44" strokeWidth="2.5" />
      {/* End Posts */}
      <line x1="40" y1="90" x2="80" y2="30" stroke="#1F2A44" strokeWidth="2" />
      <line x1="560" y1="90" x2="520" y2="30" stroke="#1F2A44" strokeWidth="2" />
      {/* Truss Verticals & Diagonals */}
      <line x1="140" y1="30" x2="140" y2="90" stroke="#1F2A44" strokeWidth="1.25" />
      <line x1="200" y1="30" x2="200" y2="90" stroke="#1F2A44" strokeWidth="1.25" />
      <line x1="260" y1="30" x2="260" y2="90" stroke="#1F2A44" strokeWidth="1.25" />
      <line x1="320" y1="30" x2="320" y2="90" stroke="#1F2A44" strokeWidth="1.25" />
      <line x1="380" y1="30" x2="380" y2="90" stroke="#1F2A44" strokeWidth="1.25" />
      <line x1="440" y1="30" x2="440" y2="90" stroke="#1F2A44" strokeWidth="1.25" />
      <line x1="500" y1="30" x2="500" y2="90" stroke="#1F2A44" strokeWidth="1.25" />

      {/* Diagonals */}
      <line x1="80" y1="30" x2="140" y2="90" stroke="#1F2A44" strokeWidth="1" />
      <line x1="200" y1="30" x2="140" y2="90" stroke="#1F2A44" strokeWidth="1" />
      <line x1="200" y1="30" x2="260" y2="90" stroke="#1F2A44" strokeWidth="1" />
      <line x1="320" y1="30" x2="260" y2="90" stroke="#1F2A44" strokeWidth="1" />
      <line x1="320" y1="30" x2="380" y2="90" stroke="#1F2A44" strokeWidth="1" />
      <line x1="440" y1="30" x2="380" y2="90" stroke="#1F2A44" strokeWidth="1" />
      <line x1="440" y1="30" x2="500" y2="90" stroke="#1F2A44" strokeWidth="1" />

      {/* Piers */}
      <rect x="35" y="90" width="20" height="40" fill="#1F2A44" fillOpacity="0.1" stroke="#1F2A44" strokeWidth="1.5" />
      <rect x="255" y="90" width="20" height="40" fill="#1F2A44" fillOpacity="0.1" stroke="#1F2A44" strokeWidth="1.5" />
      <rect x="545" y="90" width="20" height="40" fill="#1F2A44" fillOpacity="0.1" stroke="#1F2A44" strokeWidth="1.5" />
    </svg>
  );
}
