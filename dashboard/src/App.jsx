// dashboard/src/App.jsx
import { useEffect, useMemo, useState } from "react";
import TopBar from "./components/TopBar.jsx";
import MetricCard from "./components/MetricCard.jsx";
import DataTable from "./components/DataTable.jsx";
import SectorHeatmap from "./components/SectorHeatmap.jsx";
import { fetchCSV, rawUrl } from "./lib/csv.js";
import SectorTimeline from "./components/SectorTimeline.jsx";

// Repo (fallback quand pas sur Vercel)
const OWNER  = "ArnaudVarela";
const REPO   = "TradingScan";
const BRANCH = "main";

// Détection Vercel : serve /public en prod
const isBrowser = typeof window !== "undefined";
const USE_LOCAL = isBrowser && window.location.hostname.endsWith(".vercel.app");

const FILES = {
  confirmed: "confirmed_STRONGBUY.csv",
  pre: "anticipative_pre_signals.csv",
  events: "event_driven_signals.csv",
  all: "candidates_all_ranked.csv",
  history: "sector_history.csv", 
};

function urlFor(file) {
  const bucketMs = 60_000;
  const bust = `?t=${Math.floor(Date.now() / bucketMs)}`;
  return USE_LOCAL ? `/${file}${bust}` : rawUrl(OWNER, REPO, BRANCH, file) + bust;
}

export default function App() {
  const [data, setData] = useState({ confirmed: [], pre: [], events: [], all: [], history: [] });
  const [last, setLast] = useState("-");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedSectors, setSelectedSectors] = useState([]); // NEW

  async function loadAll() {
    setLoading(true);
    setError("");
    try {
      const [confirmed, pre, events, all] = await Promise.all([
        fetchCSV(urlFor(FILES.confirmed)).catch(() => []),
        fetchCSV(urlFor(FILES.pre)).catch(() => []),
        fetchCSV(urlFor(FILES.events)).catch(() => []),
        fetchCSV(urlFor(FILES.all)).catch(() => []),
        fetchCSV(urlFor(FILES.history)).catch(() => []),
      ]);
      setData({
        confirmed: Array.isArray(confirmed) ? confirmed : [],
        pre: Array.isArray(pre) ? pre : [],
        events: Array.isArray(events) ? events : [],
        all: Array.isArray(all) ? all : [],
        history: Array.isArray(history) ? history : [],
      });
      setLast(new Date().toLocaleString());
    } catch (e) {
      console.error(e);
      setError("Impossible de charger les données. Réessaie plus tard.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { loadAll(); }, []);

  const totals = useMemo(() => ({
    confirmed: data.confirmed.length,
    pre: data.pre.length,
    events: data.events.length,
    universe: data.all.length,
  }), [data]);

  // --- Filtrage sectoriel appliqué aux tables
  const sectorFilterActive = selectedSectors.length > 0;
  const filterBySectors = (rows) => {
    if (!sectorFilterActive) return rows;
    const set = new Set(selectedSectors.map(s => s.toLowerCase()));
    return rows.filter(r => (r?.sector || "Unknown").toLowerCase() && set.has((r?.sector || "Unknown").toLowerCase()));
  };

  const rowsConfirmed = useMemo(() => filterBySectors(data.confirmed), [data, selectedSectors]);
  const rowsPre       = useMemo(() => filterBySectors(data.pre),       [data, selectedSectors]);
  const rowsEvents    = useMemo(() => filterBySectors(data.events),    [data, selectedSectors]);

  const handleToggleSector = (sector, additive) => {
    if (!sector) return;
    setSelectedSectors((prev) => {
      // reset si pas additive et on clique un nouveau
      if (!additive) {
        return prev.length === 1 && prev[0] === sector ? [] : [sector];
      }
      // additive
      const has = prev.includes(sector);
      if (has) return prev.filter((s) => s !== sector);
      return [...prev, sector];
    });
  };

  const clearSectorFilter = () => setSelectedSectors([]);

  return (
    <div className="max-w-7xl mx-auto p-6 text-slate-900 dark:text-slate-100">
      <TopBar lastRefreshed={last} onRefresh={loadAll} />

      <div className="mb-4 text-xs text-slate-500">
        Source CSV : {USE_LOCAL ? "fichiers statiques (public/) sur Vercel" : "raw.githubusercontent.com (repo GitHub)"}
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <MetricCard label="Confirmed" value={totals.confirmed} />
        <MetricCard label="Pre-signals" value={totals.pre} />
        <MetricCard label="Event-driven" value={totals.events} />
        <MetricCard label="Universe" value={totals.universe} />
      </div>

      {/* Heatmap + filtre */}
      <div className="mb-3">
        <SectorHeatmap
          confirmed={data.confirmed}
          pre={data.pre}
          events={data.events}
          selectedSectors={selectedSectors}
          onToggleSector={handleToggleSector}
        />
      </div>

      {sectorFilterActive && (
        <div className="mb-4 flex items-center gap-2 flex-wrap">
          <span className="text-xs opacity-70">Sector filter:</span>
          {selectedSectors.map((s) => (
            <span key={s} className="text-xs px-2 py-1 bg-slate-200 dark:bg-slate-700 rounded">
              {s}
            </span>
          ))}
          <button
            onClick={clearSectorFilter}
            className="text-xs px-2 py-1 rounded border dark:border-slate-600 hover:bg-slate-100 dark:hover:bg-slate-700">
            Clear
          </button>
        </div>
      )}

        <div className="mb-6">
          <SectorTimeline
             historyRows={data.history}
            selectedSectors={selectedSectors}  // <- ta sélection depuis la heatmap
            />
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
