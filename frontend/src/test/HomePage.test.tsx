import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import HomePage from "../pages/HomePage";
import { getBridges, getShowcaseScans } from "../api";
import { ScanResult } from "../types";

vi.mock("../api", () => ({
  getBridges: vi.fn(),
  getShowcaseScans: vi.fn(),
  getFileUrl: (scanId: string, name: string) => `/api/scans/${scanId}/files/${name}`,
}));

afterEach(() => vi.clearAllMocks());

function scan(outputs: ScanResult["outputs"]): ScanResult {
  return {
    scan_id: "showcase-real-1",
    is_showcase: true,
    is_demo: false,
    status: "complete",
    verdict: { level: "usable", reasons: [] },
    measurement: { dominant_frequency_hz: 2.34 } as ScanResult["measurement"],
    outputs,
  } as unknown as ScanResult;
}

describe("home showcase hero", () => {
  it("shows a real magnified video, its frequency and verdict, and links to its result", async () => {
    vi.mocked(getBridges).mockResolvedValue([]);
    vi.mocked(getShowcaseScans).mockResolvedValue([scan({ magnified_video: "movement.mp4", spectrum_plot: "spectrum.png" })]);
    const onNavigate = vi.fn();
    const { container } = render(<HomePage onNavigate={onNavigate} />);

    expect(await screen.findByText("Actual scan result")).toBeInTheDocument();
    expect(screen.getByText("2.34 Hz")).toBeInTheDocument();
    expect(screen.getByText("USABLE")).toBeInTheDocument();
    expect(container.querySelector("video")).toHaveAttribute("src", "/api/scans/showcase-real-1/files/movement.mp4");
    expect(screen.queryByRole("img", { name: "Vibration spectrum plot" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "See a real scan" }));
    expect(onNavigate).toHaveBeenCalledWith("result", "showcase-real-1");
  });

  it("uses the spectrum when no magnified video exists", async () => {
    vi.mocked(getBridges).mockResolvedValue([]);
    vi.mocked(getShowcaseScans).mockResolvedValue([scan({ spectrum_plot: "spectrum.png" })]);
    render(<HomePage onNavigate={vi.fn()} />);

    expect(await screen.findByRole("img", { name: "Vibration spectrum plot" })).toHaveAttribute(
      "src",
      "/api/scans/showcase-real-1/files/spectrum.png",
    );
  });

  it("shows plain explanatory text and no scan content when no showcase exists", async () => {
    vi.mocked(getBridges).mockResolvedValue([]);
    vi.mocked(getShowcaseScans).mockResolvedValue([]);
    render(<HomePage onNavigate={vi.fn()} />);

    expect(await screen.findByText(/No completed scan is currently loaded/)).toBeInTheDocument();
    expect(screen.queryByText("3.70 Hz")).not.toBeInTheDocument();
    expect(screen.queryByText(/scan_20261103/)).not.toBeInTheDocument();
    expect(document.querySelector("video")).not.toBeInTheDocument();
    expect(screen.queryByRole("img", { name: "Vibration spectrum plot" })).not.toBeInTheDocument();
  });
});
