import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ResultPage, { makeWhatsAppText } from "../pages/ResultPage";
import {
  mockUsableScan,
  mockWeakScan,
  mockFailedScan,
  mockDemoScan,
  mockQueuedScan,
  mockRunningScan,
  mockErrorScan,
} from "./fixtures";

// Mock API call getScan
vi.mock("../api", async () => {
  const actual = await vi.importActual("../api");
  return {
    ...actual,
    getScan: vi.fn().mockImplementation((id: string) => {
      if (id === "scan_usable_001") return Promise.resolve(mockUsableScan);
      if (id === "scan_weak_002") return Promise.resolve(mockWeakScan);
      if (id === "scan_failed_003") return Promise.resolve(mockFailedScan);
      if (id === "scan_demo_004") return Promise.resolve(mockDemoScan);
      if (id === "scan_queued_005") return Promise.resolve(mockQueuedScan);
      if (id === "scan_running_006") return Promise.resolve(mockRunningScan);
      if (id === "scan_error_007") return Promise.resolve(mockErrorScan);
      return Promise.reject(new Error("Scan not found"));
    }),
    getFileUrl: (scanId: string, name: string) => `/api/scans/${scanId}/files/${name}`,
  };
});

describe("ResultPage Smoke Tests", () => {
  const mockNavigate = vi.fn();

  it("renders field note transcript, English translation, and evidence tags", () => {
    const scan = { ...mockUsableScan, field_notes: [{ note_id: "n1", created_at: "2026-01-01T00:00:00Z", language: "hi", transcript: "Visible crack on the bridge", english: "A crack is visible on the bridge", tags: [{ tag: "crack" as const, evidence: "Visible crack" }] }] };
    const { container } = render(<ResultPage scanId="scan_usable_001" onNavigate={mockNavigate} initialScan={scan} />);
    expect(screen.getByText("Tell us what you see: cracks, sounds, sagging")).toBeInTheDocument();
    expect(screen.getByText("Visible crack on the bridge")).toBeInTheDocument();
    expect(screen.getByText("A crack is visible on the bridge")).toBeInTheDocument();
    expect(Array.from(container.querySelectorAll("article span")).some((el) => el.textContent?.includes("Visible crack"))).toBe(true);
  });

  it("shows setup advice and recording issues for drivers and setup source for engineers", () => {
    const scan = { ...mockUsableScan, setup: {
      source: "muse_spark" as const,
      structure_type: "deck",
      recording_issues: ["The camera moved during the clip."],
      advice: "Rest the phone on a firm surface.",
    } };
    render(<ResultPage scanId="scan_usable_001" onNavigate={mockNavigate} initialScan={scan} />);
    expect(screen.getByText("Rest the phone on a firm surface.")).toBeInTheDocument();
    expect(screen.getByText("The camera moved during the clip.")).toBeInTheDocument();
    fireEvent.click(screen.getByText("For engineers (technical details)"));
    expect(screen.getByText("muse_spark")).toBeInTheDocument();
  });

  it("renders sonification, device agreement, traffic plot, weather, and real mode-shape points", () => {
    const scan = {
      ...mockUsableScan,
      outputs: { ...mockUsableScan.outputs, sound: "vibration.wav" },
      sonification: { speedup: 20, pitch_hz: 3.7 },
      device_check: { status: "agrees" as const, device_frequency_hz: 3.7, device_ci_hz: [3.6, 3.8] as [number, number], sample_rate_hz: 50, note: null },
      weather: { temperature_c: 24, humidity_pct: 65, wind_speed_ms: 2.5, source: "open-meteo" },
      traffic: { events: [], response_plot: "traffic.png" },
      measurement: { ...mockUsableScan.measurement, mode_shape: [
        { label: "Pier A", position_norm: 0, amplitude_px: 0.4 },
        { label: "Pier B", position_norm: 1, amplitude_px: 0.8 },
      ] },
    };
    render(<ResultPage scanId="scan_usable_001" onNavigate={mockNavigate} initialScan={scan} />);

    expect(screen.getByRole("button", { name: "Listen to the bridge" })).toBeInTheDocument();
    expect(screen.getByText("The real vibration, sped up 20x so you can hear it.")).toBeInTheDocument();
    expect(screen.getByText("Your phone's motion sensor agreed with the camera")).toBeInTheDocument();
    fireEvent.click(screen.getByText("For engineers (technical details)"));
    expect(screen.getByText("24 °C")).toBeInTheDocument();
    expect(screen.getByText("65%")).toBeInTheDocument();
    expect(screen.getByAltText("Bridge vibration response during vehicle crossings")).toBeInTheDocument();
    expect(screen.getByText("Shaded: vehicles crossing.")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Bridge mode shape" })).toBeInTheDocument();
  });

  it("shows the magnification preparation message for pending work and hides unrecorded device checks", () => {
    const scan = { ...mockUsableScan, magnification_status: "running", device_check: { status: "not_recorded" as const, device_frequency_hz: null, device_ci_hz: null, sample_rate_hz: null, note: null } };
    render(<ResultPage scanId="scan_usable_001" onNavigate={mockNavigate} initialScan={scan} />);
    expect(screen.getByText("Preparing the slowed-down movement video...")).toBeInTheDocument();
    expect(screen.queryByText("Your phone's motion sensor agreed with the camera")).not.toBeInTheDocument();
  });

  it("shows a clear message when microphone permission is refused", async () => {
    Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: { getUserMedia: vi.fn().mockRejectedValue(new Error("denied")) } });
    render(<ResultPage scanId="scan_usable_001" onNavigate={mockNavigate} initialScan={mockUsableScan} />);
    fireEvent.click(screen.getByRole("button", { name: /Tell us what you see/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Microphone access denied");
  });

  it("shows cable tension details only when cable change data exists", () => {
    const withCable = { ...mockUsableScan, cable: { is_cable: true, tension_change_pct: 4.2, tension_change_range_pct: [3.1, 5.3] as [number, number], compared_to_scan_id: "old-1", compared_to_date: "2026-10-01T00:00:00Z" } };
    const { rerender } = render(<ResultPage scanId="scan_usable_001" onNavigate={mockNavigate} initialScan={withCable} />);
    expect(screen.getByText("Compared with an earlier scan of this cable")).toBeInTheDocument();
    fireEvent.click(screen.getByText("For engineers (technical details)"));
    expect(screen.getByText(/Estimated change in cable tension since/)).toBeInTheDocument();
    expect(screen.getByText(/Based on the change in vibration frequency/)).toBeInTheDocument();
    rerender(<ResultPage scanId="scan_usable_001" onNavigate={mockNavigate} initialScan={mockUsableScan} />);
    expect(screen.queryByText("Compared with an earlier scan of this cable")).not.toBeInTheDocument();
  });

  it("creates WhatsApp summaries with frequency only for usable readings", () => {
    const labels = { bridge: "Bridge", reading: "Reading", frequency: "Frequency", link: "Page" };
    const usable = makeWhatsAppText({ bridge: "River Bridge", verdict: "reading available", frequency: "3.7 Hz (3.6-3.8 Hz)", link: "https://setu.test/?scan=x", labels });
    expect(usable).toContain("Frequency: 3.7 Hz (3.6-3.8 Hz)");
    const weak = makeWhatsAppText({ bridge: "River Bridge", verdict: "reading not clear enough", frequency: null, link: "https://setu.test/?scan=x", labels });
    expect(weak).toContain("reading not clear enough");
    expect(weak).not.toContain("Frequency:");
    expect(weak).not.toContain("Hz");
  });

  it("renders usable scan with dominant frequency and usable verdict badge", async () => {
    render(
      <ResultPage
        scanId="scan_usable_001"
        onNavigate={mockNavigate}
        initialScan={mockUsableScan}
      />
    );

    // Checks verdict badge
    expect(screen.getByText("USABLE")).toBeInTheDocument();

    // Checks dominant frequency
    expect(screen.getByText("3.70")).toBeInTheDocument();
    expect(screen.getByText("How fast the bridge vibrates")).toBeInTheDocument();

    // Checks motion section
    expect(screen.getByText("Bridge movement in video")).toBeInTheDocument();
    expect(screen.getByText("Live vibration wave")).toBeInTheDocument();
  });

  it("checks that failed scan shows 'No reliable frequency detected' and no frequency number", async () => {
    render(
      <ResultPage
        scanId="scan_failed_003"
        onNavigate={mockNavigate}
        initialScan={mockFailedScan}
      />
    );

    // Verdict is FAILED
    expect(screen.getByText("FAILED")).toBeInTheDocument();

    // Must show 'No reliable frequency detected'
    expect(screen.getByText("No reliable frequency detected")).toBeInTheDocument();

    // Must NOT render any frequency number
    expect(screen.queryByText("3.70")).not.toBeInTheDocument();
    expect(screen.queryByText("2.10")).not.toBeInTheDocument();
    expect(screen.queryByText("Vibrates about")).not.toBeInTheDocument();
  });

  it("checks that weak scan NEVER shows severity or anomaly wording anywhere", async () => {
    const { container } = render(
      <ResultPage
        scanId="scan_weak_002"
        onNavigate={mockNavigate}
        initialScan={mockWeakScan}
      />
    );

    // Verdict is WEAK
    expect(screen.getByText("WEAK")).toBeInTheDocument();

    // Check whole container text content: MUST NOT contain 'severity' or 'anomal'
    const fullText = container.textContent || "";
    expect(fullText).not.toMatch(/severity/i);
    expect(fullText).not.toMatch(/anomal/i);

    // Expand "For engineers" section and verify again
    const engineerToggle = screen.getByText("For engineers (technical details)");
    fireEvent.click(engineerToggle);

    const fullExpandedText = container.textContent || "";
    expect(fullExpandedText).not.toMatch(/severity/i);
    expect(fullExpandedText).not.toMatch(/anomal/i);
  });

  it("checks that demo scan shows the prominent demo banner", async () => {
    render(
      <ResultPage
        scanId="scan_demo_004"
        onNavigate={mockNavigate}
        initialScan={mockDemoScan}
      />
    );

    expect(
      screen.getByText("Demo result: sample data, not a real measurement")
    ).toBeInTheDocument();
  });

  it("renders status queued state", async () => {
    render(
      <ResultPage
        scanId="scan_queued_005"
        onNavigate={mockNavigate}
        initialScan={mockQueuedScan}
      />
    );

    expect(screen.getByText("Status: waiting in line")).toBeInTheDocument();
    expect(screen.getByText("Checking every 2 seconds")).toBeInTheDocument();
  });

  it("renders status running state", async () => {
    render(
      <ResultPage
        scanId="scan_running_006"
        onNavigate={mockNavigate}
        initialScan={mockRunningScan}
      />
    );

    expect(screen.getByText("Status: measuring bridge motion")).toBeInTheDocument();
    expect(screen.getByText("Checking every 2 seconds")).toBeInTheDocument();
  });

  it("renders error state with a retry option", async () => {
    render(
      <ResultPage
        scanId="scan_error_007"
        onNavigate={mockNavigate}
        initialScan={mockErrorScan}
      />
    );

    // Shows error text
    expect(
      screen.getByText("Pipeline error: tracking lost due to sudden illumination change.")
    ).toBeInTheDocument();

    // Shows a clear Retry option
    const retryBtn = screen.getByRole("button", { name: /retry/i });
    expect(retryBtn).toBeInTheDocument();
  });

  it("verifies that every image rendered on the page has non-empty alt text", async () => {
    const { container } = render(
      <ResultPage
        scanId="scan_usable_001"
        onNavigate={mockNavigate}
        initialScan={mockUsableScan}
      />
    );

    const images = container.querySelectorAll("img");
    expect(images.length).toBeGreaterThan(0);

    images.forEach((img) => {
      const alt = img.getAttribute("alt");
      expect(alt).toBeTruthy();
      expect(alt?.trim().length).toBeGreaterThan(0);
    });
  });
});
