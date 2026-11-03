import React from "react";
import { useTranslation } from "react-i18next";
import { SUPPORTED_LANGUAGES, setStoredLanguage } from "../i18n";
import { Check } from "lucide-react";
import Logo from "./Logo";

interface LanguagePickerModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function LanguagePickerModal({ isOpen, onClose }: LanguagePickerModalProps) {
  const { i18n } = useTranslation();

  if (!isOpen) return null;

  const handleSelectLanguage = (code: string) => {
    i18n.changeLanguage(code);
    setStoredLanguage(code);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-asphalt/80">
      <div className="w-full max-w-xl bg-concrete border-2 border-asphalt rounded-[4px] p-6 sm:p-8 flex flex-col space-y-6 max-h-[90vh] overflow-y-auto">
        <div className="space-y-2">
          <Logo size="md" />
          <h2 className="text-xl sm:text-2xl font-heading font-bold text-asphalt tracking-tight pt-2 border-b border-concrete-rule pb-2">
            अपनी भाषा चुनें · Choose your language
          </h2>
          <p className="text-xs text-asphalt-muted font-sans">
            Select your language for bridge vibration telemetry. You can change this anytime from the header.
          </p>
        </div>

        {/* 10 Languages Grid in their own script */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1 font-sans">
          {SUPPORTED_LANGUAGES.map((lang) => {
            const isSelected = i18n.language === lang.code;

            return (
              <button
                key={lang.code}
                onClick={() => handleSelectLanguage(lang.code)}
                type="button"
                className={`p-3 rounded-[4px] border text-left transition flex items-center justify-between ${
                  isSelected
                    ? "bg-highway text-white border-highway font-bold"
                    : "bg-white border-concrete-rule text-asphalt hover:border-asphalt hover:bg-concrete-light"
                }`}
              >
                <div>
                  <div className="text-base leading-tight font-medium">
                    {lang.nativeName}
                  </div>
                  <div
                    className={`text-[11px] mt-0.5 ${
                      isSelected ? "text-white/80" : "text-asphalt-muted"
                    }`}
                  >
                    {lang.label}
                  </div>
                </div>

                <div className="flex items-center space-x-1.5">
                  {!lang.isCarefullyCurated && (
                    <span
                      className={`text-[10px] px-1.5 py-0.2 rounded-[2px] uppercase font-mono ${
                        isSelected
                          ? "bg-highway-dark text-white"
                          : "bg-concrete text-asphalt-muted border border-concrete-rule"
                      }`}
                    >
                      AI
                    </span>
                  )}
                  {isSelected && <Check className="h-4 w-4 text-white" />}
                </div>
              </button>
            );
          })}
        </div>

        <div className="border-t border-concrete-rule pt-3 text-xs text-asphalt-muted font-sans">
          Hindi & English are human-verified. Other Indic languages include automated assistance.
        </div>
      </div>
    </div>
  );
}
