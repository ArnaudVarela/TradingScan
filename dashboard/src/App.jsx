// dashboard/src/App.jsx
import { useEffect, useMemo, useState } from "react";
import TopBar from "./components/TopBar.jsx";
import MetricCard from "./components/MetricCard.jsx";
import DataTable from "./components/DataTable.jsx";
import SectorHeatmap from "./components/SectorHeatmap.jsx";
import SectorTimeline from "./components/SectorTimeline.jsx";
import { fetchCSV, rawUrl } from "./lib/csv.js";
import { ChevronDown, ChevronUp } from "lucide-react";
import EquityCurve from "./components/EquityCurve.jsx";

// ---------- Coordonnées repo (fallback hors Vercel)
const OWNER  = "ArnaudVarela";
const REPO   = "TradingScan";
const BRANCH = "main";

// Détection Vercel : on sert /public à la racine du site
const isBrowser = typeof window !== "undefined";
const USE_LOCAL = isBrowser && window.location.hostname.endsWith(".vercel.app");

// ---------- Fichiers CSV
const FILES = {
  confirmed: "confirmed_STRONGBUY.csv",
  pre:       "anticipative_pre_signals.csv",
  events:    "event_driven_signals.csv",
  all:       "candidates_all_ranked.csv",
  history:   "sector_history.csv",
  breadth:   "sector_breadth.csv",
  equity10: "backtest_equity_10d.csv",
};

// URL finale + cache-buster court (1 min)
function urlFor(file) {
  const bucketMs = 60_000;
  const bust = `?t=${Math.floor(Date.now() / bucketMs)}`;
  return USE_LOCAL ? `/${file}${bust}` : rawUrl(OWNER, REPO, BRANCH, file) + bust;
}

export default function App() {
  const [data, setData] = useState({
    confirmed: [],
    pre:       [],
    events:    [],
    all:       [],
    history:   [],
    breadth:   [],
    equity10:  [], 
  });
  const [last, setLast] = useState("-");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [warningFiles, setWarningFiles] = useState([]); // fichiers qui ont échoué
  const [selectedSectors, setSelectedSectors] = useState([]); // filtre heatmap -> tables

  // états pliables (mémorisés)
  const [showHeatmap, setShowHeatmap] = useState(() => {
    if (typeof window === "undefined") return true;
    return localStorage.getItem("showHeatmap") !== "0";
  });
  const [showTimeline, setShowTimeline] = useState(() => {
    if (typeof window === "undefined") return true;
    return localStorage.getItem("showTimeline") !== "0";
  });

  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem("showHeatmap", showHeatmap ? "1" : "0");
    }
  }, [showHeatmap]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem("showTimeline", showTimeline ? "1" : "0");
    }
  }, [showTimeline]);

  // ---------- Chargement robuste de tous les CSV
  async function loadAll() {
    setLoading(true);
    setError("");
    setWarningFiles([]);
        
    const tasks = [
      ["confirmed", FILES.confirmed],
      ["pre",       FILES.pre],
      ["events",    FILES.events],
      ["all",       FILES.all],
      ["history",   FILES.history],
      ["breadth",   FILES.breadth],
      ["equity10",  FILES.equity10],
    ];

    try {
      const results = await Promise.allSettled(
        tasks.map(([_, file]) => fetchCSV(urlFor(file)))
      );

      const next = { confirmed: [], pre: [], events: [], all: [], history: [], breadth: [], equity10: [] };
      const failed = [];

      results.forEach((res, idx) => {
        const [key, file] = tasks[idx];
        if (res.status === "fulfilled") {
          const rows = Array.isArray(res.value) ? res.value : [];
          next[key] = rows;
        } else {
          console.warn(`[CSV] Échec de chargement: ${file}`, res.reason);
          failed.push(file);
          next[key] = [];
        }
      });

      setData(next);
      setLast(new Date().toLocaleString());
      setWarningFiles(failed);

      const allEmpty =
        next.confirmed.length === 0 &&
        next.pre.length === 0 &&
        next.events.length === 0 &&
        next.all.length === 0 &&
        next.history.length === 0 &&
        next.breadth.length === 0;
        next.equity10.length === 0;

      if (allEmpty) {
        setError("Impossible de charger les données. Réessaie plus tard.");
      }
    } catch (e) {
      console.error(e);
      setError("Impossible de charger les données (exception). Vérifie la console.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { loadAll(); }, []);

  // ---------- KPIs
  const totals = useMemo(() => ({
    confirmed: data.confirmed.length,
    pre:       data.pre.length,
    events:    data.events.length,
    universe:  data.all.length,
  }), [data]);

  // ---------- Filtre sectoriel (sélection depuis heatmap)
  const sectorFilterActive = selectedSectors.length > 0;
  const filterBySectors = (rows) => {
    if (!sectorFilterActive) return rows;
    const set = new Set(selectedSectors.map(s => (s || "").toLowerCase()));
    return rows.filter(r => set.has((r?.sector || "Unknown").toLowerCase()));
  };

  const rowsConfirmed = useMemo(() => filterBySectors(data.confirmed), [data, selectedSectors]);
  const rowsPre       = useMemo(() => filterBySectors(data.pre),       [data, selectedSectors]);
  const rowsEvents    = useMemo(() => filterBySectors(data.events),    [data, selectedSectors]);

  const handleToggleSector = (sector, additive) => {
    if (!sector) return;
    setSelectedSectors(prev => {
      if (!additive) {
        return prev.length === 1 && prev[0] === sector ? [] : [sector];
      }
      return prev.includes(sector) ? prev.filter(s => s !== sector) : [...prev, sector];
    });
  };

  const clearSectorFilter = () => setSelectedSectors([]);

  return (
    <div className="max-w-7xl mx-auto p-6 text-slate-900 dark:text-slate-100">
      <TopBar lastRefreshed={last} onRefresh={loadAll} />

      {/* Info source */}
      <div className="mb-2 text-xs text-slate-500">
        Source CSV : {USE_LOCAL ? "fichiers statiques (public/) sur Vercel" : "raw.githubusercontent.com (repo GitHub)"}
      </div>

      {/* Bandeau avertissements (certains fichiers KO) */}
      {warningFiles.length > 0 && (
        <div className="mb-4 text-sm text-yellow-800 bg-yellow-50 border border-yellow-200 rounded p-2">
          Certains fichiers n’ont pas pu être chargés :{" "}
          <span className="font-mono">{warningFiles.join(", ")}</span>
          {" "} (détails dans la console du navigateur).
        </div>
      )}

      {/* KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <MetricCard label="Confirmed"    value={totals.confirmed} />
        <MetricCard label="Pre-signals"  value={totals.pre} />
        <MetricCard label="Event-driven" value={totals.events} />
        <MetricCard label="Universe"     value={totals.universe} />
      </div>

      {/* Heatmap pliable */}
      <div className="mb-4 bg-white dark:bg-slate-900 rounded shadow p-4">
        <button
          onClick={() => setShowHeatmap(!showHeatmap)}
          className="flex items-center justify-between w-full text-left font-semibold"
          aria-expanded={showHeatmap}
        >
          <span>Sector signals (heatmap)</span>
          {showHeatmap ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </button>

        {showHeatmap && (
          <div className="mt-4">
            <SectorHeatmap
              confirmed={data.confirmed}
              pre={data.pre}
              events={data.events}
              breadth={data.breadth}
              selectedSectors={selectedSectors}
              onToggleSector={handleToggleSector}
            />
          </div>
        )}
      </div>

      {/* Affichage du filtre actif */}
      {sectorFilterActive && (
        <div className="mb-4 flex items-center gap-2 flex-wrap">
          <span className="text-xs opacity-70">Sector filter:</span>
          {selectedSectors.map(s => (
            <span key={s} className="text-xs px-2 py-1 bg-slate-200 dark:bg-slate-700 rounded">
              {s}
            </span>
          ))}
          <button
            onClick={clearSectorFilter}
            className="text-xs px-2 py-1 rounded border dark:border-slate-600 hover:bg-slate-100 dark:hover:bg-slate-700"
          >
            Clear
          </button>
        </div>
      )}

      {/* Timeline (weekly history) pliable */}
      <div className="mb-6 bg-white dark:bg-slate-900 rounded shadow p-4">
        <button
          onClick={() => setShowTimeline(!showTimeline)}
          className="flex items-center justify-between w-full text-left font-semibold"
          aria-expanded={showTimeline}
        >
          <span>Sector signals timeline (weekly)</span>
          {showTimeline ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </button>

        {showTimeline && (
          <div className="mt-4">
            <SectorTimeline
              history={data.history}
              selectedSectors={new Set(selectedSectors)}
            />
          </div>
        )}
      </div>

      {/* Equity curve (10-day hold) */}
      <div className="mb-6">
        <EquityCurve data={data.equity10} title="Equity curve — 10 trading days hold" />
      </div>

      {/* États */}
      {loading && <div className="mb-4 text-sm text-slate-600">Chargement…</div>}
      {!!error && <div className="mb-4 text-sm text-red-600">{error}</div>}

      {/* Tables */}
      <div className="grid grid-cols-1 gap-6">
        <section>
          <h2 className="h1 mb-3">Confirmed STRONG_BUY</h2>
          <DataTable rows={rowsConfirmed} />
        </section>

        <section>
          <h2 className="h1 mb-3">Anticipative pre-signals</h2>
          <DataTable rows={rowsPre} />
        </section>

        <section>
          <h2 className="h1 mb-3">Event-driven signals</h2>
          <DataTable rows={rowsEvents} />
        </section>
      </div>
    </div>
  );
}
