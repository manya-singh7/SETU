import { afterEach, describe, expect, it, vi } from "vitest";
import { createScan, getScan, getShowcaseScans } from "../api";
import { ScanResult } from "../types";

afterEach(() => vi.unstubAllGlobals());

describe("createScan", () => {
  it("sends the language code and empty prompts for AI-directed setup", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ scan_id: "scan-1" }) });
    vi.stubGlobal("fetch", fetchMock);

    const deviceMotion = "[{\"t\":0.25,\"ax\":1.2,\"ay\":-2.5,\"az\":9.81}]";
    await createScan({ lat: 13, lon: 80, target_prompt: "", background_prompt: "", language: "ta", device_motion: deviceMotion });

    const body = fetchMock.mock.calls[0][1].body as FormData;
    expect(body.get("language")).toBe("ta");
    expect(body.get("target_prompt")).toBe("");
    expect(body.get("background_prompt")).toBe("");
    expect(body.get("device_motion")).toBe(deviceMotion);
  });

  it("continues polling a completed scan while magnification is pending", async () => {
    const pending = { status: "complete", magnification_status: "pending" } as ScanResult;
    const finished = { status: "complete", magnification_status: "done" } as ScanResult;
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => pending })
      .mockResolvedValueOnce({ ok: true, json: async () => finished });
    vi.stubGlobal("fetch", fetchMock);
    vi.useFakeTimers();

    const resultPromise = getScan("scan-magnifying", undefined, true);
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    await vi.advanceTimersByTimeAsync(2000);
    const result = await resultPromise;

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(result.magnification_status).toBe("done");
    vi.useRealTimers();
  });
});

describe("getShowcaseScans", () => {
  it("loads only explicitly marked, non-demo showcase scans", async () => {
    const showcase = { scan_id: "real-1", is_showcase: true, is_demo: false };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [showcase, { scan_id: "demo", is_showcase: true, is_demo: true }, { scan_id: "other" }],
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(getShowcaseScans()).resolves.toEqual([showcase]);
    expect(fetchMock).toHaveBeenCalledWith("/api/showcase");
  });
});
