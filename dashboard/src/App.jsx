// dashboard/src/App.jsx
import { useEffect, useMemo, useState } from "react";
import TopBar from "./components/TopBar.jsx";
import MetricCard from "./components/MetricCard.jsx";
import DataTable from "./components/DataTable.jsx";
import SectorHeatmap from "./components/SectorHeatmap.jsx";
import { fetchCSV, rawUrl } from "./lib/csv.js";
import SectorTimeline from "./components/SectorTimeline.jsx";
import { ChevronDown, ChevronUp } from "lucide-react"; // icons pliable/dépliable

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
  breadth: "sector_breadth.csv",
};

function urlFor(file) {
  const bucketMs = 60_000; // 1 min de cache-busting
  const bust = `?t=${Math.floor(Date.now() / bucketMs)}`;
  return USE_LOCAL ? `/${file}${bust}` : rawUrl(OWNER, REPO, BRANCH, file) + bust;
}

export default function App() {
  const [data, setData] = useState({ confirmed: [], pre: [], events: [], all: [], history: [], breadth: [] });
  const [last, setLast] = useState("-");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedSectors, setSelectedSectors] = useState([]); // array de strings
  const [showHeatmap, setShowHeatmap] = useState(true);

  async function loadAll() {
    setLoading(true);
    setError("");
    try {
      // ✅ récupère bien les 5 résultats (dont history)
      const [confirmed, pre, events, all, history] = await Promise.allSettled([
        fetchCSV(urlFor(FILES.confirmed)).catch(() => []),
        fetchCSV(urlFor(FILES.pre)).catch(() => []),
        fetchCSV(urlFor(FILES.events)).catch(() => []),
        fetchCSV(urlFor(FILES.all)).catch(() => []),
        fetchCSV(urlFor(FILES.history)).catch(() => []),
        fetchCSV(urlFor(FILES.breadth)).catch(() => []),
      ]);

      setData({
        confirmed: Array.isArray(confirmed) ? confirmed : [],
        pre: Array.isArray(pre) ? pre : [],
        events: Array.isArray(events) ? events : [],
        all: Array.isArray(all) ? all : [],
        history: Array.isArray(history) ? history : [],
        breadth: Array.isArray(breadth) ? breadth : [],
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
    return rows.filter(r => set.has((r?.sector || "Unknown").toLowerCase())); // ✅ simple et net
  };

  const rowsConfirmed = useMemo(() => filterBySectors(data.confirmed), [data, selectedSectors]);
  const rowsPre       = useMemo(() => filterBySectors(data.pre),       [data, selectedSectors]);
  const rowsEvents    = useMemo(() => filterBySectors(data.events),    [data, selectedSectors]);

  const handleToggleSector = (sector, additive) => {
    if (!sector) return;
    setSelectedSectors((prev) => {
      if (!additive) {
        return prev.length === 1 && prev[0] === sector ? [] : [sector];
      }
      const has = prev.includes(sector);
      return has ? prev.filter((s) => s !== sector) : [...prev, sector];
    });
  };

  const clearSectorFilter = () => setSelectedSectors([]);

  return (
    <div className="max-w-7xl mx-auto p-6 text-slate-900 dark:text-slate-100">
      <TopBar lastRefreshed={last} onRefresh={loadAll} />

      {/* Bandeau d'info */}
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

      {/* Section Heatmap pliable */}
      <div className="mb-4 bg-white dark:bg-slate-900 rounded shadow p-4">
        <button
          onClick={() => setShowHeatmap(!showHeatmap)}
          className="flex items-center justify-between w-full text-left font-semibold"
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
              selectedSectors={selectedSectors}
              onToggleSector={handleToggleSector}
            />
          </div>
        )}
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

      {/* Timeline (pense bien au bon nom de prop) */}
      <div className="mb-6">
        <SectorTimeline
          history={data.history}               // ✅ prop attendue par le composant
          selectedSectors={new Set(selectedSectors)} // optionnel si tu filtres par secteurs dans le composant
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
