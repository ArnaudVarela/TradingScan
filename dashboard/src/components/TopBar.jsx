// dashboard/src/components/TopBar.jsx
import { useEffect, useState } from "react";
import { Sun, Moon } from "lucide-react";
import FearGreedGauge from "./FearGreedGauge.jsx";
import LogoRadar from "./LogoRadar.jsx";
import { fetchFearGreedLive } from "../lib/fng.js"; // ⬅️ chemin corrigé

export default function TopBar({ lastRefreshed, onRefresh, fear }) {
  // thème
  const [darkMode, setDarkMode] = useState(() => {
    if (typeof window === "undefined") return false;
    const saved = localStorage.getItem("darkMode");
    if (saved === "1") return true;
    if (saved === "0") return false;
    return window.matchMedia?.("(prefers-color-scheme: dark)")?.matches ?? false;
  });

  useEffect(() => {
    const root = document.documentElement;
    if (darkMode) root.classList.add("dark");
    else root.classList.remove("dark");
    if (typeof window !== "undefined") {
      localStorage.setItem("darkMode", darkMode ? "1" : "0");
    }
  }, [darkMode]);

  // Fear & Greed (live si non passé en prop)
  const [fearLive, setFearLive] = useState(null);
  useEffect(() => {
    let stop = false;
    if (!fear) {
      fetchFearGreedLive().then((v) => {
        if (!stop) setFearLive(v);
      });
    } else {
      setFearLive(null); // on respecte la prop si fournie
    }
    return () => { stop = true; };
  }, [fear]);

  const fearData = fear || fearLive; // priorité à la prop (ex: historique + streak)

  return (
    <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-6">
      {/* Logo + titre */}
      <div className="flex items-center gap-3 min-w-0">
        <LogoRadar dark={darkMode} size={38} className="flex-shrink-0" />
        <div className="min-w-0">
          <h1 className="text-2xl font-bold truncate">Signal Intelligence Dashboard</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Confirmed / Anticipative / Event-driven signals — powered by cross-analysis
          </p>
        </div>
      </div>

      {/* Fear & Greed Gauge */}
      <div className="flex items-center gap-4">
        {fearData ? (
          <FearGreedGauge
            score={Number(fearData.score)}
            label={String(fearData.label || "")}
            // l’API alt.me ne donne pas de streak → 0 par défaut si non fourni
            streak={Number(fearData.streak_days ?? fearData.streak ?? 0)}
            asof={fearData.asof}
          />
        ) : (
          <div className="text-xs text-slate-400">Fear &amp; Greed: N/A</div>
        )}
      </div>

      {/* Boutons à droite */}
      <div className="flex gap-2 items-center">
        <span className="text-xs text-slate-500 dark:text-slate-400">
          Last refresh: {lastRefreshed || "-"}
        </span>
        <button
          type="button"
          className="px-3 py-1 rounded bg-slate-800 text-white text-sm dark:bg-slate-200 dark:text-black"
          onClick={onRefresh}
          aria-label="Refresh data"
        >
          Refresh
        </button>
        <button
          type="button"
          onClick={() => setDarkMode((v) => !v)}
          className="p-2 rounded bg-slate-200 dark:bg-slate-700"
          aria-label={darkMode ? "Activer le thème clair" : "Activer le thème sombre"}
          title={darkMode ? "Light mode" : "Dark mode"}
        >
          {darkMode ? <Sun size={18} /> : <Moon size={18} />}
        </button>
      </div>
    </div>
  );
}
