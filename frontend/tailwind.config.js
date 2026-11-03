/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Indian road & engineering signage palette
        concrete: "#E9E7E2",     // background concrete grey
        "concrete-light": "#F2F0EB",
        "concrete-dark": "#D5D2CA",
        "concrete-rule": "#C4C1B8",
        highway: "#0B6E4F",      // highway sign green (primary)
        "highway-dark": "#08573E",
        asphalt: "#22262A",      // text / dark elements
        "asphalt-muted": "#555B62",
        "asphalt-light": "#737982",
        milestone: "#F2C230",    // milestone yellow (highlight / weak)
        "milestone-dark": "#D4A517",
        // Verdict colors
        verdict: {
          usable: "#0B6E4F",     // green
          weak: "#F2C230",       // milestone yellow
          failed: "#737982",     // slate grey (NEVER red!)
        },
      },
      fontFamily: {
        heading: ["Tiro Devanagari Hindi", "serif"],
        sans: [
          "Hind",
          "Noto Sans Devanagari",
          "Noto Sans Tamil",
          "Noto Sans Bengali",
          "Noto Sans Telugu",
          "Noto Sans Kannada",
          "Noto Sans Gujarati",
          "Noto Sans Gurmukhi",
          "Noto Sans Malayalam",
          "sans-serif",
        ],
      },
      borderRadius: {
        DEFAULT: "4px",
        none: "0",
        sm: "4px",
        md: "4px",
        lg: "4px",
        xl: "4px",
        "2xl": "4px",
        full: "9999px",
      },
      boxShadow: {
        none: "none",
        DEFAULT: "none",
        sm: "none",
        md: "none",
        lg: "none",
        xl: "none",
        "2xl": "none",
        inner: "none",
      },
    },
  },
  plugins: [],
};
