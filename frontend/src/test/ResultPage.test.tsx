import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ResultPage from "../pages/ResultPage";
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
