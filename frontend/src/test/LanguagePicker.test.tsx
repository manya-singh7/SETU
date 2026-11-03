import React from "react";
import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import App from "../App";
import { getStoredLanguage } from "../i18n";

describe("LanguagePicker Smoke Tests", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("checks that the full-screen language picker appears on first visit", () => {
    expect(localStorage.getItem("setu_user_language")).toBeNull();

    render(<App />);

    // Must show language picker modal title
    expect(
      screen.getByText("अपनी भाषा चुनें · Choose your language")
    ).toBeInTheDocument();

    // Must list languages like English, हिन्दी, தமிழ், etc.
    expect(screen.getAllByText("English").length).toBeGreaterThan(0);
    expect(screen.getByText("हिन्दी")).toBeInTheDocument();
    expect(screen.getByText("தமிழ்")).toBeInTheDocument();
  });

  it("persists chosen language in localStorage and does not reopen on subsequent visit", async () => {
    expect(localStorage.getItem("setu_user_language")).toBeNull();

    const { unmount } = render(<App />);

    // Click Hindi button
    const hindiBtn = screen.getByText("हिन्दी");
    fireEvent.click(hindiBtn);

    // Verify persisted in localStorage
    expect(getStoredLanguage()).toBe("hi");
    expect(localStorage.getItem("setu_user_language")).toBe("hi");

    // Unmount and re-render App (subsequent visit)
    unmount();

    render(<App />);

    // Modal should no longer appear on subsequent visit
    await waitFor(() => {
      expect(
        screen.queryByText("अपनी भाषा चुनें · Choose your language")
      ).not.toBeInTheDocument();
    });
  });
});
