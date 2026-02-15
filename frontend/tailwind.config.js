/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        cream: "#fef6e6",
        sand: "#f4f0ea",
        clay: "#de5b2f",
        ink: "#1f1b16",
        moss: "#1f6a44",
        gold: "#8a6f00",
        rose: "#a8322a"
      },
      fontFamily: {
        display: ["'Space Grotesk'", "'Trebuchet MS'", "sans-serif"],
        mono: ["'Space Mono'", "'Courier New'", "monospace"]
      },
      boxShadow: {
        hero: "0 20px 40px rgba(31, 27, 22, 0.12)",
        card: "0 12px 24px rgba(31, 27, 22, 0.08)"
      }
    }
  },
  plugins: []
};
