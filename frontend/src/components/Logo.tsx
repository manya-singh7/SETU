import React from "react";

interface LogoProps {
  size?: "sm" | "md" | "lg";
  onClick?: () => void;
}

export default function Logo({ size = "md", onClick }: LogoProps) {
  const isSm = size === "sm";
  const isLg = size === "lg";

  return (
    <button
      type="button"
      onClick={onClick}
      className={`inline-flex flex-col text-left select-none p-1 -m-1 rounded-[4px] hover:bg-concrete-dark/30 transition focus-visible:outline-highway ${
        onClick ? "cursor-pointer" : ""
      }`}
    >
      <div className="flex items-center space-x-1.5 leading-none">
        <span
          className={`font-heading text-asphalt tracking-tight ${
            isSm ? "text-lg font-bold" : isLg ? "text-3xl font-bold" : "text-2xl font-bold"
          }`}
        >
          SETU
        </span>
        <span className="h-2 w-2 rounded-[2px] bg-highway inline-block" />
      </div>
      <span
        className={`text-asphalt-muted font-medium leading-none font-sans ${
          isSm ? "text-[11px] mt-0.5" : isLg ? "text-sm mt-1" : "text-xs mt-0.5"
        }`}
      >
        सेतु
      </span>
    </button>
  );
}
