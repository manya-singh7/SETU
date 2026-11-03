import i18n from "i18next";
import { initReactI18next } from "react-i18next";

import en from "./locales/en.json";
import hi from "./locales/hi.json";
import ta from "./locales/ta.json";
import bn from "./locales/bn.json";
import te from "./locales/te.json";
import mr from "./locales/mr.json";
import kn from "./locales/kn.json";
import gu from "./locales/gu.json";
import pa from "./locales/pa.json";
import ml from "./locales/ml.json";

export const SUPPORTED_LANGUAGES = [
  { code: "en", label: "English", nativeName: "English", isCarefullyCurated: true },
  { code: "hi", label: "Hindi", nativeName: "हिन्दी", isCarefullyCurated: true },
  { code: "ta", label: "Tamil", nativeName: "தமிழ்", isCarefullyCurated: false },
  { code: "bn", label: "Bengali", nativeName: "বাংলা", isCarefullyCurated: false },
  { code: "te", label: "Telugu", nativeName: "తెలుగు", isCarefullyCurated: false },
  { code: "mr", label: "Marathi", nativeName: "मराठी", isCarefullyCurated: false },
  { code: "kn", label: "Kannada", nativeName: "ಕನ್ನಡ", isCarefullyCurated: false },
  { code: "gu", label: "Gujarati", nativeName: "ગુજરાતી", isCarefullyCurated: false },
  { code: "pa", label: "Punjabi", nativeName: "ਪੰਜਾਬੀ", isCarefullyCurated: false },
  { code: "ml", label: "Malayalam", nativeName: "മലയാളം", isCarefullyCurated: false },
] as const;

export type SupportedLanguageCode = (typeof SUPPORTED_LANGUAGES)[number]["code"];

const STORAGE_KEY = "setu_user_language";

export function getStoredLanguage(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch (err) {
    console.warn("Could not read language from localStorage:", err);
    return null;
  }
}

export function setStoredLanguage(langCode: string): void {
  try {
    localStorage.setItem(STORAGE_KEY, langCode);
  } catch (err) {
    console.warn("Could not save language to localStorage:", err);
  }
}

export function hasUserChosenLanguage(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) !== null;
  } catch (err) {
    return false;
  }
}

const initialLang = getStoredLanguage() || "en";

i18n.use(initReactI18next).init({
  resources: {
    en: { translation: en },
    hi: { translation: hi },
    ta: { translation: ta },
    bn: { translation: bn },
    te: { translation: te },
    mr: { translation: mr },
    kn: { translation: kn },
    gu: { translation: gu },
    pa: { translation: pa },
    ml: { translation: ml },
  },
  lng: initialLang,
  fallbackLng: "en",
  interpolation: {
    escapeValue: false,
  },
});

export default i18n;

/**
 * Format number according to active locale
 */
export function formatLocaleNumber(
  num: number | null | undefined,
  options?: Intl.NumberFormatOptions
): string {
  if (num == null || isNaN(num)) return "–";
  const lang = i18n.language || "en";
  try {
    return new Intl.NumberFormat(lang, options).format(num);
  } catch {
    return num.toLocaleString();
  }
}

/**
 * Format date according to active locale
 */
export function formatLocaleDate(
  dateInput: string | Date | null | undefined,
  options?: Intl.DateTimeFormatOptions
): string {
  if (!dateInput) return "–";
  const date = typeof dateInput === "string" ? new Date(dateInput) : dateInput;
  if (isNaN(date.getTime())) return "–";
  const lang = i18n.language || "en";
  try {
    return new Intl.DateTimeFormat(lang, options || { dateStyle: "medium", timeStyle: "short" }).format(date);
  } catch {
    return date.toLocaleDateString();
  }
}
