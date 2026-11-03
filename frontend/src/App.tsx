import React, { useState, useEffect, Suspense, lazy } from "react";
import { useTranslation } from "react-i18next";
import {
  Camera,
  MapPin,
  ShieldAlert,
  BarChart3,
  BadgeCheck,
  Home as HomeIcon,
} from "lucide-react";
import HomePage from "./pages/HomePage";
import ScanPage from "./pages/ScanPage";
import Logo from "./components/Logo";
import LanguagePickerModal from "./components/LanguagePickerModal";
import LanguageSwitcher from "./components/LanguageSwitcher";
import ReportPage from "./pages/ReportPage";
import ProofPage from "./pages/ProofPage";
import { hasUserChosenLanguage, SUPPORTED_LANGUAGES } from "./i18n";

// Code-split heavy map (Leaflet) and chart (Recharts) libraries so initial page load is fast
const EngineerMapPage = lazy(() => import("./pages/EngineerMapPage"));
const ResultPage = lazy(() => import("./pages/ResultPage"));

type AppPage = "home" | "scan" | "result" | "map" | "proof";

function AppShell({ initialPage = "home" }: { initialPage?: AppPage }) {
  const { t, i18n } = useTranslation();
  const [currentPage, setCurrentPage] = useState<AppPage>(initialPage);
  const [selectedScanId, setSelectedScanId] = useState<string>("scan_20261103_a1b2c3");
  const [backendHealth, setBackendHealth] = useState<string>("Connecting...");
  const [showLanguageModal, setShowLanguageModal] = useState<boolean>(false);

  useEffect(() => {
    // Show full-screen language picker on first visit
    if (!hasUserChosenLanguage()) {
      setShowLanguageModal(true);
    }

    // Health check via proxied /health
    fetch("/health")
      .then((res) => res.json())
      .then((data) => {
        setBackendHealth(
          data.status === "ok"
            ? t("common.fastapi_online", "System online")
            : "Degraded"
        );
      })
      .catch(() => {
        setBackendHealth(t("common.fastapi_standby", "Offline mode (Sample data)"));
      });
  }, [t]);

  const handleNavigate = (page: AppPage, scanId?: string) => {
    if (scanId) {
      setSelectedScanId(scanId);
    }
    setCurrentPage(page);
    if (page === "proof") window.history.pushState({}, "", "/proof");
    else if (window.location.pathname === "/proof") window.history.pushState({}, "", "/");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleScanComplete = (newScanId: string) => {
    setSelectedScanId(newScanId);
    setCurrentPage("result");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="min-h-screen bg-[#E9E7E2] text-[#22262A] flex flex-col font-sans selection:bg-[#0B6E4F] selection:text-white">
      {/* First-visit Full Screen Language Picker */}
      <LanguagePickerModal
        isOpen={showLanguageModal}
        onClose={() => setShowLanguageModal(false)}
      />

      {/* Safety Principle Banner */}
      <div className="bg-white border-b border-[#22262A]/20 px-4 py-2 text-xs text-[#22262A] flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center space-x-2">
          <ShieldAlert className="h-4 w-4 text-[#0B6E4F] shrink-0" />
          <span className="text-[11px] sm:text-xs">
            <strong>{t("common.app_name", "SETU")} SAFETY PRINCIPLE:</strong>{" "}
            {t(
              "common.safety_principle_full",
              "SETU does not assess safety. Flagged readings are for engineer review. Mobile video vibration telemetry is an early screening heuristic only."
            )}
          </span>
        </div>
        <div className="flex items-center space-x-2 text-[11px] font-mono text-[#22262A]/70">
          <span className="h-2 w-2 rounded-full bg-[#0B6E4F]" />
          <span>{backendHealth}</span>
        </div>
      </div>

      {/* Top Header & Navigation Bar */}
      <header className="app-navigation border-b border-[#22262A]/20 bg-white px-4 sm:px-6 py-3 flex items-center justify-between sticky top-0 z-40">
        {/* Logo: "SETU" with "सेतु" beneath it */}
        <Logo size="md" onClick={() => handleNavigate("home")} />

        {/* Navigation & Language Switcher */}
        <div className="flex items-center space-x-2 sm:space-x-3">
          <nav className="flex items-center space-x-1 sm:space-x-1.5 text-xs font-semibold">
            {/* Home Tab */}
            <button
              type="button"
              onClick={() => handleNavigate("home")}
              className={`flex items-center space-x-1.5 px-3 py-2 rounded-[4px] transition ${
                currentPage === "home"
                  ? "bg-[#0B6E4F] text-white"
                  : "text-[#22262A] hover:bg-[#E9E7E2]"
              }`}
            >
              <HomeIcon className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">{t("nav.home", "Home")}</span>
            </button>

            {/* Scan Tab */}
            <button
              type="button"
              onClick={() => handleNavigate("scan")}
              className={`flex items-center space-x-1.5 px-3 py-2 rounded-[4px] transition ${
                currentPage === "scan"
                  ? "bg-[#0B6E4F] text-white"
                  : "text-[#22262A] hover:bg-[#E9E7E2]"
              }`}
            >
              <Camera className="h-3.5 w-3.5" />
              <span>{t("nav.scan", "Film a bridge")}</span>
            </button>

            {/* Result Tab */}
            <button
              type="button"
              onClick={() => handleNavigate("result")}
              className={`flex items-center space-x-1.5 px-3 py-2 rounded-[4px] transition ${
                currentPage === "result"
                  ? "bg-[#0B6E4F] text-white"
                  : "text-[#22262A] hover:bg-[#E9E7E2]"
              }`}
            >
              <BarChart3 className="h-3.5 w-3.5" />
              <span>{t("nav.result", "Results")}</span>
            </button>

            {/* Map Tab */}
            <button
              type="button"
              onClick={() => handleNavigate("map")}
              className={`flex items-center space-x-1.5 px-3 py-2 rounded-[4px] transition ${
                currentPage === "map"
                  ? "bg-[#0B6E4F] text-white"
                  : "text-[#22262A] hover:bg-[#E9E7E2]"
              }`}
            >
              <MapPin className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">{t("nav.map", "Engineer Map")}</span>
              <span className="sm:hidden">Map</span>
            </button>
            <button
              type="button"
              onClick={() => handleNavigate("proof")}
              className={`flex items-center space-x-1.5 px-3 py-2 rounded-[4px] transition ${currentPage === "proof" ? "bg-[#0B6E4F] text-white" : "text-[#22262A] hover:bg-[#E9E7E2]"}`}
            >
              <BadgeCheck className="h-3.5 w-3.5" />
              <span>{t("nav.proof", "Proof")}</span>
            </button>
          </nav>

          {/* Language Switcher Dropdown */}
          <LanguageSwitcher />
        </div>
      </header>

      {/* Machine translation note for regional Indian scripts */}
      {SUPPORTED_LANGUAGES.find((l) => l.code === i18n.language)?.isCarefullyCurated === false && (
        <div className="bg-[#FFFFFF] border-b border-[#22262A]/20 px-4 py-1.5 text-center text-[11px] text-[#22262A]/80 font-sans">
          {t("common.machine_translation_note", "Machine translated; please report errors")}
        </div>
      )}

      {/* Main Page Content with Suspense for code-split pages */}
      <main className="flex-1 p-3 sm:p-6 max-w-7xl w-full mx-auto">
        <Suspense
          fallback={
            <div className="p-12 text-center text-xs font-semibold text-[#22262A] font-sans">
              Loading module...
            </div>
          }
        >
          {currentPage === "home" && <HomePage onNavigate={handleNavigate} />}

          {currentPage === "scan" && <ScanPage onScanComplete={handleScanComplete} />}

          {currentPage === "result" && (
            <ResultPage scanId={selectedScanId} onNavigate={handleNavigate} />
          )}

          {currentPage === "map" && (
            <EngineerMapPage
              initialScanId={selectedScanId}
              onSelectScan={(id) => handleNavigate("result", id)}
            />
          )}

          {currentPage === "proof" && <ProofPage />}
        </Suspense>
      </main>

      {/* Footer */}
      <footer className="border-t border-[#22262A]/20 bg-white px-6 py-6 text-center text-xs text-[#22262A]/80 space-y-1">
        <div className="font-heading font-semibold text-[#22262A]">
          {t("common.footer_text", "Project SETU • Civic Infrastructure Telemetry • Made in India")}
        </div>
        <p className="text-[11px] text-[#22262A]/70 font-sans">
          {t("common.safety_principle", "SETU does not assess safety. Flagged readings are for engineer review.")}
        </p>
      </footer>
    </div>
  );
}

export default function App() {
  const match = window.location.pathname.match(/^\/report\/([^/]+)\/?$/);
  if (match) return <ReportPage bridgeKey={decodeURIComponent(match[1])} />;
  if (window.location.pathname === "/proof") return <AppShell initialPage="proof" />;
  return <AppShell />;
}
