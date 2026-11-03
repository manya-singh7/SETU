import React, { useState, useRef, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { ChevronDown, Check } from "lucide-react";
import { SUPPORTED_LANGUAGES, setStoredLanguage } from "../i18n";

export default function LanguageSwitcher() {
  const { i18n, t } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const currentLang =
    SUPPORTED_LANGUAGES.find((l) => l.code === i18n.language) || SUPPORTED_LANGUAGES[0];

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelect = (code: string) => {
    i18n.changeLanguage(code);
    setStoredLanguage(code);
    setIsOpen(false);
  };

  return (
    <div className="relative inline-block text-left" ref={dropdownRef}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-[4px] border border-concrete-rule bg-white hover:bg-concrete-light text-asphalt text-xs font-medium transition focus-visible:outline-highway"
        aria-haspopup="true"
        aria-expanded={isOpen}
      >
        <span className="font-semibold">{currentLang.nativeName}</span>
        <ChevronDown className="h-3.5 w-3.5 text-asphalt-muted" />
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-1 w-52 rounded-[4px] bg-white border border-asphalt z-50 py-1 font-sans">
          <div className="px-3 py-1.5 border-b border-concrete-rule text-[11px] font-semibold text-asphalt-muted">
            {t("common.choose_language", "Choose your language")}
          </div>

          <div className="max-h-60 overflow-y-auto py-1">
            {SUPPORTED_LANGUAGES.map((lang) => {
              const isSelected = i18n.language === lang.code;
              return (
                <button
                  key={lang.code}
                  type="button"
                  onClick={() => handleSelect(lang.code)}
                  className={`w-full px-3 py-1.5 text-left text-xs flex items-center justify-between hover:bg-concrete-light transition ${
                    isSelected ? "font-bold text-highway bg-concrete" : "text-asphalt"
                  }`}
                >
                  <div className="flex flex-col">
                    <span className="text-xs">{lang.nativeName}</span>
                    <span className="text-[10px] text-asphalt-muted">{lang.label}</span>
                  </div>
                  {isSelected && <Check className="h-3.5 w-3.5 text-highway" />}
                </button>
              );
            })}
          </div>

          {currentLang.isCarefullyCurated === false && (
            <div className="px-3 py-1.5 bg-concrete-light border-t border-concrete-rule text-[11px] text-asphalt-muted">
              {t("common.machine_translation_note", "Machine translated; please report errors")}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
