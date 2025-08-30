import { useEffect, useMemo, useState } from "react";
import TopBar from "./components/TopBar.jsx";
import MetricCard from "./components/MetricCard.jsx";
import DataTable from "./components/DataTable.jsx";
import { fetchCSV, rawUrl } from "./lib/csv.js";

const OWNER  = "ArnaudVarela";   
const REPO   = "TradingScan";          
const BRANCH = "main";                 

const FILES = {
  confirmed: "confirmed_STRONGBUY.csv",
  pre: "anticipative_pre_signals.csv",
  events: "event_driven_signals.csv",
  all: "candidates_all_ranked.csv"
};

export default function App() {
  const [data, setData] = useState({confirmed: [], pre: [], events: [], all: []});
  const [last, setLast] = useState("-");

  async function loadAll() {
    const urls = Object.fromEntries(
      Object.entries(FILES).map(([k, f]) => [k, rawUrl(OWNER, REPO, BRANCH, f)])
    );
    const [confirmed, pre, events, all] = await Promise.all([
      fetchCSV(urls.confirmed).catch(()=>[]),
      fetchCSV(urls.pre).catch(()=>[]),
      fetchCSV(urls.events).catch(()=>[]),
      fetchCSV(urls.all).catch(()=>[])
    ]);
    setData({ confirmed, pre, events, all });
    setLast(new Date().toLocaleString());
  }

  useEffect(() => { loadAll(); }, []);

  const totals = useMemo(() => ({
    confirmed: data.confirmed.length,
    pre: data.pre.length,
    events: data.events.length,
    universe: data.all.length
  }), [data]);

  return (
    <div className="max-w-7xl mx-auto p-6">
      <TopBar lastRefreshed={last} onRefresh={loadAll} />
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <MetricCard label="Confirmed" value={totals.confirmed} />
        <MetricCard label="Pre-signals" value={totals.pre} />
        <MetricCard label="Event-driven" value={totals.events} />
        <MetricCard label="Universe" value={totals.universe} />
      </div>

      <div className="grid grid-cols-1 gap-6">
        <DataTable rows={data.confirmed} />
        <DataTable rows={data.pre} />
        <DataTable rows={data.events} />
      </div>
    </div>
  );
}
