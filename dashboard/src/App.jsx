// dashboard/src/App.jsx
import { useEffect, useMemo, useState } from "react";
import TopBar from "./components/TopBar.jsx";
import MetricCard from "./components/MetricCard.jsx";
import DataTable from "./components/DataTable.jsx";
import { fetchCSV, rawUrl } from "./lib/csv.js";

// --- Coordonnées du repo (utilisés quand on n'est pas sur Vercel) ---
const OWNER  = "ArnaudVarela";
const REPO   = "TradingScan";
const BRANCH = "main";

// Détection Vercel : sur *.vercel.app on sert les CSV depuis /public
const isBrowser = typeof window !== "undefined";
const USE_LOCAL = isBrowser && window.location.hostname.endsWith(".vercel.app");

// Fichiers CSV produits par le workflow
const FILES = {
  confirmed: "confirmed_STRONGBUY.csv",
  pre: "anticipative_pre_signals.csv",
  events: "event_driven_signals.csv",
  all: "candidates_all_ranked.csv",
};

// Construit l'URL finale pour un fichier CSV, avec cache-buster (1 min)
function urlFor(file) {
  const bucketMs = 60_000; // change à 300_000 pour 5 minutes si tu veux
  const bust = `?t=${Math.floor(Date.now() / bucketMs)}`;
  return USE_LOCAL ? `/${file}${bust}` : rawUrl(OWNER, REPO, BRANCH, file) + bust;
}

export default function App() {
  const [data, setData] = useState({ confirmed: [], pre: [], events: [], all: [] });
  const [last, setLast] = useState("-");         // horodatage du dernier refresh
  const [loading, setLoading] = useState(false); // état de chargement
  const [error, setError] = useState("");        // message d'erreur user-friendly

  async function loadAll() {
    setLoading(true);
    setError("");
    try {
      const [confirmed, pre, events, all] = await Promise.all([
        fetchCSV(urlFor(FILES.confirmed)).catch(() => []),
        fetchCSV(urlFor(FILES.pre)).catch(() => []),
        fetchCSV(urlFor(FILES.events)).catch(() => []),
        fetchCSV(urlFor(FILES.all)).catch(() => []),
      ]);
      setData({
        confirmed: Array.isArray(confirmed) ? confirmed : [],
        pre: Array.isArray(pre) ? pre : [],
        events: Array.isArray(events) ? events : [],
        all: Array.isArray(all) ? all : [],
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

  const totals = useMemo(
    () => ({
      confirmed: data.confirmed.length,
      pre: data.pre.length,
      events: data.events.length,
      universe: data.all.length,
    }),
    [data]
  );

  return (
    <div className="max-w-7xl mx-auto p-6 text-slate-900 dark:text-slate-100">
      <TopBar lastRefreshed={last} onRefresh={loadAll} />

      {/* Bandeau d'info sur la source des CSV */}
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

      {/* États */}
      {loading && <div className="mb-4 text-sm text-slate-600">Chargement…</div>}
      {!!error && <div className="mb-4 text-sm text-red-600">{error}</div>}

      {/* Tables */}
      <div className="grid grid-cols-1 gap-6">
        <section>
          <h2 className="h1 mb-3">Confirmed STRONG_BUY</h2>
          <DataTable rows={data.confirmed} />
        </section>

        <section>
          <h2 className="h1 mb-3">Anticipative pre-signals</h2>
          <DataTable rows={data.pre} />
        </section>

        <section>
          <h2 className="h1 mb-3">Event-driven signals</h2>
          <DataTable rows={data.events} />
        </section>
      </div>
    </div>
  );
}
