import React from "react";

interface JaaliDividerProps {
  className?: string;
  opacity?: number;
}

export default function JaaliDivider({ className = "", opacity = 0.25 }: JaaliDividerProps) {
  return (
    <div className={`w-full flex items-center justify-center my-6 overflow-hidden ${className}`}>
      <svg
        className="w-full h-4"
        style={{ opacity }}
        preserveAspectRatio="repeat-x"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <pattern
            id="jaali-pattern"
            width="32"
            height="16"
            patternUnits="userSpaceOnUse"
          >
            {/* Traditional 8-point geometric jaali lattice unit */}
            <path
              d="M16 0 L24 8 L16 16 L8 8 Z"
              fill="none"
              stroke="#1F2A44"
              strokeWidth="0.75"
            />
            <path
              d="M0 8 L8 0 L8 16 Z"
              fill="none"
              stroke="#1F2A44"
              strokeWidth="0.75"
            />
            <path
              d="M32 8 L24 0 L24 16 Z"
              fill="none"
              stroke="#1F2A44"
              strokeWidth="0.75"
            />
            <circle cx="16" cy="8" r="1.5" fill="#E8912D" opacity="0.6" />
          </pattern>
        </defs>
        <rect width="100%" height="16" fill="url(#jaali-pattern)" />
      </svg>
    </div>
  );
}
