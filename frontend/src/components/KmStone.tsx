import React from "react";
import { formatLocaleNumber } from "../i18n";

interface KmStoneProps {
  count: number;
  className?: string;
  label?: string;
}

export default function KmStone({ count, className = "", label = "BRIDGES" }: KmStoneProps) {
  if (count <= 0) return null;

  return (
    <div className={`inline-flex flex-col items-center select-none ${className}`}>
      <svg
        viewBox="0 0 140 180"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-28 sm:w-32 h-auto"
        role="img"
        aria-label={`${count} ${label}`}
      >
        {/* Milestone Base Shadow (Flat ground line) */}
        <line x1="10" y1="176" x2="130" y2="176" stroke="#C4C1B8" strokeWidth="3" strokeLinecap="round" />

        {/* Milestone Body (White stone with sharp black outline) */}
        <path
          d="M15 65 C15 25 35 10 70 10 C105 10 125 25 125 65 L125 174 L15 174 Z"
          fill="#FFFFFF"
          stroke="#22262A"
          strokeWidth="3.5"
        />

        {/* Milestone Cap (Yellow #F2C230) */}
        <path
          d="M15 65 C15 25 35 10 70 10 C105 10 125 25 125 65 Z"
          fill="#F2C230"
          stroke="#22262A"
          strokeWidth="3.5"
        />

        {/* Dividing line between cap and body */}
        <line x1="15" y1="65" x2="125" y2="65" stroke="#22262A" strokeWidth="2.5" />

        {/* Route marker on yellow cap */}
        <text
          x="70"
          y="48"
          textAnchor="middle"
          fill="#22262A"
          fontFamily="Hind, sans-serif"
          fontWeight="700"
          fontSize="15"
          letterSpacing="1"
        >
          SETU
        </text>

        {/* Count in tabular figures */}
        <text
          x="70"
          y="125"
          textAnchor="middle"
          fill="#22262A"
          fontFamily="Hind, sans-serif"
          fontWeight="700"
          fontSize="48"
          className="tabular-nums"
        >
          {formatLocaleNumber(count)}
        </text>

        {/* Label */}
        <text
          x="70"
          y="152"
          textAnchor="middle"
          fill="#555B62"
          fontFamily="Hind, sans-serif"
          fontWeight="600"
          fontSize="11"
          letterSpacing="1"
        >
          {label}
        </text>
      </svg>
    </div>
  );
}
