// dashboard/src/components/TopBar.jsx
import { useEffect, useState } from "react";
import { Sun, Moon, RefreshCw } from "lucide-react";
import FearGreedGauge from "./FearGreedGauge.jsx";
import LogoRadar from "./LogoRadar.jsx";
import { fetchFearGreedLive } from "../lib/fng.js";

export default function TopBar({ lastRefreshed, onRefresh, fear, loading }) {
  const [darkMode, setDarkMode] = useState(() => {
    if (typeof window === "undefined") return true;
    const saved = localStorage.getItem("darkMode");
    if (saved === "1") return true;
    if (saved === "0") return false;
    return true; // dark par défaut (dashboard "lux")
  });

  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle("dark", darkMode);
    if (typeof window !== "undefined") localStorage.setItem("darkMode", darkMode ? "1" : "0");
  }, [darkMode]);

  const [fearLive, setFearLive] = useState(null);
  useEffect(() => {
    let stop = false;
    if (!fear) fetchFearGreedLive().then((v) => { if (!stop) setFearLive(v); });
    else setFearLive(null);
    return () => { stop = true; };
  }, [fear]);

  const fearData = fear || fearLive;

  return (
    <header className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-6">
      <div className="flex items-center gap-3 min-w-0">
        <div className="relative flex-shrink-0">
          <div className="absolute inset-0 blur-xl bg-cyan-500/20 rounded-full" aria-hidden />
          <LogoRadar dark={darkMode} size={42} className="relative" />
        </div>
        <div className="min-w-0">
          <h1 className="text-xl sm:text-2xl font-extrabold tracking-tight leading-none">
            <span className="brand-grad">TradingScan</span>
            <span className="text-slate-400 font-semibold"> · Setups thématiques</span>
          </h1>
          <p className="mt-1 text-xs sm:text-sm text-slate-500 dark:text-slate-400">
            Screener hard-tech · pré-explosion scorée <span className="font-semibold text-slate-600 dark:text-slate-300">/100</span> · MAJ quotidienne (EOD)
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {fearData ? (
          <FearGreedGauge
            score={Number(fearData.score)}
            label={String(fearData.label || "")}
            streak={Number(fearData.streak_days ?? fearData.streak ?? 0)}
            asof={fearData.asof}
          />
        ) : (
          <div className="text-xs text-slate-400">Fear &amp; Greed: —</div>
        )}

        <div className="flex items-center gap-2">
          <span className="hidden lg:inline text-[11px] text-slate-500 dark:text-slate-500">
            MAJ&nbsp;{lastRefreshed || "—"}
          </span>
          <button
            type="button"
            onClick={onRefresh}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium
                       bg-slate-900 text-white hover:bg-slate-800
                       dark:bg-white/10 dark:hover:bg-white/15 dark:ring-1 dark:ring-white/10 transition"
            aria-label="Rafraîchir"
          >
            <RefreshCw size={15} className={loading ? "animate-spin" : ""} />
            <span className="hidden sm:inline">Refresh</span>
          </button>
          <button
            type="button"
            onClick={() => setDarkMode((v) => !v)}
            className="p-2 rounded-lg bg-slate-200 text-slate-700 hover:bg-slate-300
                       dark:bg-white/10 dark:text-slate-200 dark:hover:bg-white/15 dark:ring-1 dark:ring-white/10 transition"
            aria-label={darkMode ? "Thème clair" : "Thème sombre"}
            title={darkMode ? "Light mode" : "Dark mode"}
          >
            {darkMode ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
      </div>
    </header>
  );
}
