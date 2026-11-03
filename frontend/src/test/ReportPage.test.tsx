import React from "react";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import ReportPage from "../pages/ReportPage";
import { BridgeReport } from "../types";

const reportFixture: BridgeReport = {
  bridge_key: "bridge-river-7",
  bridge: { bridge_key: "bridge-river-7", name: "River Bridge", bridge_type: "Cable-stayed", material: "Steel", location: { lat: 13.08, lon: 80.27 } },
  scans: [
    {
      scan_id: "scan-a",
      date: "2026-10-01T10:00:00Z",
      verdict: "usable",
      frequency_hz: 2.4,
      ci_low_hz: 2.2,
      ci_high_hz: 2.6,
      cable_tension_change_pct: 3.5,
      cable_tension_change_low_pct: 2.5,
      cable_tension_change_high_pct: 4.5,
      field_note_tags: [{ tag: "rust", evidence: "Rust near anchor" }],
    },
    { scan_id: "scan-b", date: "2026-11-01T10:00:00Z", verdict: "weak", frequency_hz: null },
  ],
};

describe("Printable bridge report", () => {
  it("renders a scan table from report data", () => {
    render(<ReportPage bridgeKey="bridge-river-7" initialReport={reportFixture} />);
    expect(screen.getByRole("heading", { name: "River Bridge" })).toBeInTheDocument();
    expect(screen.getByText("usable")).toBeInTheDocument();
    expect(screen.getByText(/2.4 Hz/)).toBeInTheDocument();
    expect(screen.getByText(/Rust near anchor/)).toBeInTheDocument();
    expect(screen.getByText(/3.5%/)).toBeInTheDocument();
  });

  it("print CSS hides navigation and uses A4 black text", () => {
    const css = readFileSync("src/index.css", "utf8");
    expect(css).toContain("@page");
    expect(css).toContain("size: A4");
    expect(css).toMatch(/\.print-hidden,\s*\.app-navigation\s*\{\s*display:\s*none\s*!important;/);
    expect(css).toContain("color: #000 !important");
  });
});
