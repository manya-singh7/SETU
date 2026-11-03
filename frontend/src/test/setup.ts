import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";
import "../i18n"; // initialize i18next

// Mock window.scrollTo
window.scrollTo = () => {};

// Mock ResizeObserver for Recharts and layout observers
(globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};

// Mock HTMLCanvasElement 2D context for RealWaveform
HTMLCanvasElement.prototype.getContext = (() => ({
  fillRect: () => {},
  clearRect: () => {},
  getImageData: () => ({ data: [] }),
  putImageData: () => {},
  createImageData: () => [],
  setTransform: () => {},
  drawImage: () => {},
  save: () => {},
  fillText: () => {},
  restore: () => {},
  beginPath: () => {},
  moveTo: () => {},
  lineTo: () => {},
  closePath: () => {},
  stroke: () => {},
  translate: () => {},
  scale: () => {},
  rotate: () => {},
  arc: () => {},
  fill: () => {},
  measureText: () => ({ width: 0 }),
  transform: () => {},
  rect: () => {},
  clip: () => {},
})) as unknown as typeof HTMLCanvasElement.prototype.getContext;

// Mock global fetch for relative URLs in JSDOM
const originalFetch = globalThis.fetch;
globalThis.fetch = vi.fn().mockImplementation((input: RequestInfo | URL) => {
  const url = typeof input === "string" ? input : input.toString();
  if (url.includes("/health")) {
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ status: "ok" }),
    });
  }
  if (url.includes("/api/bridges")) {
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve([]),
    });
  }
  return Promise.resolve({
    ok: true,
    json: () => Promise.resolve({}),
    text: () => Promise.resolve(""),
  });
}) as unknown as typeof originalFetch;

// Automatically cleanup DOM after each test
afterEach(() => {
  cleanup();
  localStorage.clear();
});
