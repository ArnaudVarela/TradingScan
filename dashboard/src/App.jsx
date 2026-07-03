// dashboard/src/App.jsx — dashboard épuré & redesign "terminal lux".
import { useCallback, useEffect, useMemo, useState } from "react";
import { Boxes, Flame, Gauge, Layers } from "lucide-react";
import TopBar from "./components/TopBar.jsx";
import ThematicSetups from "./components/ThematicSetups.jsx";
import ScoreReliability from "./components/ScoreReliability.jsx";
import SectorHeat from "./components/SectorHeat.jsx";
import SectorRotation from "./components/SectorRotation.jsx";
import ErrorBoundary from "./components/ErrorBoundary.jsx";
import { fetchCSVFresh, toNumber } from "./lib/csv.js";

function StatCard({ icon: Icon, label, value, accent }) {
  return (
    <div className="panel p-4 flex items-center gap-3">
      <div className={`h-11 w-11 rounded-xl grid place-items-center ring-1 ring-inset ${accent}`}>
        <Icon size={20} />
      </div>
      <div className="min-w-0">
        <div className="text-2xl font-bold num leading-none">{value}</div>
        <div className="text-xs text-slate-500 dark:text-slate-400 mt-1 truncate">{label}</div>
      </div>
    </div>
  );
}

export default function App() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [last, setLast] = useState("—");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const r = await fetchCSVFresh("thematic_setups.csv");
      setRows(Array.isArray(r) ? r : []);
      setLast(new Date().toLocaleString());
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const stats = useMemo(() => {
    const themes = new Set();
    rows.forEach((r) => (r.themes || "").split("|").forEach((t) => t && themes.add(t)));
    const nanoMicro = rows.filter((r) => {
      const n = toNumber(r.mcap_usd);
      return Number.isFinite(n) && n > 0 && n < 2e9;
    }).length;
    const hot = rows.filter((r) => (toNumber(r.score) ?? 0) >= 70).length;
    return { total: rows.length, nanoMicro, hot, themes: themes.size };
  }, [rows]);

  return (
    <div className="min-h-screen">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
        <TopBar lastRefreshed={last} onRefresh={load} fear={null} loading={loading} />

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
          <StatCard icon={Boxes} label="Setups scannés" value={stats.total} accent="bg-cyan-500/15 text-cyan-600 dark:text-cyan-300 ring-cyan-500/25" />
          <StatCard icon={Flame} label="Micro/Nano ≤ $2B" value={stats.nanoMicro} accent="bg-violet-500/15 text-violet-600 dark:text-violet-300 ring-violet-500/25" />
          <StatCard icon={Gauge} label="Score ≥ 70" value={stats.hot} accent="bg-emerald-500/15 text-emerald-600 dark:text-emerald-300 ring-emerald-500/25" />
          <StatCard icon={Layers} label="Thèmes couverts" value={stats.themes} accent="bg-sky-500/15 text-sky-600 dark:text-sky-300 ring-sky-500/25" />
        </div>

        <ErrorBoundary fallback="La rotation sectorielle n'a pas pu s'afficher.">
          <SectorRotation />
        </ErrorBoundary>

        <ErrorBoundary fallback="La chaleur sectorielle n'a pas pu s'afficher.">
          <SectorHeat />
        </ErrorBoundary>

        <ErrorBoundary fallback="La vue Setups thématiques n'a pas pu s'afficher.">
          <ThematicSetups rows={rows} loading={loading} />
        </ErrorBoundary>

        <div className="mt-6">
          <ErrorBoundary fallback="Le rapport de fiabilité n'a pas pu s'afficher.">
            <ScoreReliability />
          </ErrorBoundary>
        </div>

        <footer className="mt-8 text-center text-[11px] leading-relaxed text-slate-500 dark:text-slate-500">
          Screener thématique hard-tech · données prix/volume via yfinance (EOD).<br />
          Le score /100 est un <span className="font-medium">filtre de timing/qualité</span>, pas un conseil d'investissement.
        </footer>
      </div>
    </div>
  );
}
