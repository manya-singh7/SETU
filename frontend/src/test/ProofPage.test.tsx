import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import ProofPage from "../pages/ProofPage";

afterEach(() => vi.unstubAllGlobals());

function jsonResponse(value: unknown) {
  return { ok: true, json: async () => value } as Response;
}

describe("Proof page", () => {
  it("shows the coming soon message for an empty validation index", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse([])));
    render(<ProofPage />);
    expect(await screen.findByText("Validation results coming soon")).toBeInTheDocument();
  });

  it("renders real comparison frequencies, spectra, difference, and agreement", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(["comparison.json"]))
      .mockResolvedValueOnce(jsonResponse({
        camera_frequency_hz: 3.7,
        camera_frequency_ci_hz: [3.6, 3.8],
        accelerometer_frequency_hz: 3.72,
        accelerometer_frequency_ci_hz: [3.65, 3.79],
        camera_spectrum: { available: true, points: [{ frequency_hz: 3.7, normalized_power: 1 }] },
        accelerometer_spectrum: { available: true, points: [{ frequency_hz: 3.72, normalized_power: 1 }] },
        difference_hz: 0.02,
        difference_pct: 0.54,
        agreement: true,
        agreement_verdict: "PASS",
      }));
    vi.stubGlobal("fetch", fetchMock);
    render(<ProofPage />);
    expect(await screen.findByText("The sensors agree within the comparison criteria")).toBeInTheDocument();
    expect(screen.getByText("Normalized spectra")).toBeInTheDocument();
    expect(screen.getByText("0.54%")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenNthCalledWith(1, "/validation/index.json");
    expect(fetchMock).toHaveBeenNthCalledWith(2, "/validation/comparison.json");
  });
});
