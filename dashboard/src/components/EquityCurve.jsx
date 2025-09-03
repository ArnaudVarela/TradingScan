// dashboard/src/components/EquityCurve.jsx
import React, { useEffect, useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid,
} from "recharts";
import { ChevronDown, ChevronUp } from "lucide-react";

/* ---------- helpers ---------- */
// rows: [{date, equity}]  ->  [{date, <key>: number}]
function normSeries(rows, key) {
  if (!Array.isArray(rows)) return [];
  return rows
    .filter(r => r && r.date != null && r.equity != null && r.equity !== "")
    .map(r => ({ date: String(r.date), [key]: Number(r.equity) }))
    .filter(r => Number.isFinite(r[key]));
}

function mergeOnDate(series) {
  const map = new Map();
  for (const arr of series) {
    for (const row of arr) {
      const d = row.date;
      if (!map.has(d)) map.set(d, { date: d });
      Object.assign(map.get(d), row);
    }
  }
  return Array.from(map.values()).sort((a, b) => a.date.localeCompare(b.date));
}

// au moins 1 point pour la clé ?
const hasKey = (rows, key) =>
  rows.some(r => Object.prototype.hasOwnProperty.call(r, key));

/* ---------- component ---------- */
export default function EquityCurve({
  data,     // auto model 10d
  bench,    // SPY 10d
  p3,       // P3_confirmed 10d
  p2,       // P2_highconv 10d
  user,     // Mes Picks 10d
  title = "Equity curve",
  storageKey = "eqcurve_open", // pour mémoriser l'état (unique si plusieurs courbes)
  defaultOpen = true,
}) {
  /* état plié/déplié avec mémoire */
  const [open, setOpen] = useState(() => {
    if (typeof window === "undefined") return defaultOpen;
    const v = localStorage.getItem(storageKey);
    return v == null ? defaultOpen : v !== "0";
  });
  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem(storageKey, open ? "1" : "0");
    }
  }, [open, storageKey]);

  /* séries & merge */
  const modelS = useMemo(() => normSeries(data, "Model"), [data]);
  const spyS   = useMemo(() => normSeries(bench, "SPY"), [bench]);
  const p3S    = useMemo(() => normSeries(p3, "P3_confirmed"), [p3]);
  const p2S    = useMemo(() => normSeries(p2, "P2_highconv"), [p2]);
  const userS  = useMemo(() => normSeries(user, "User_picks"), [user]);

  const merged = useMemo(
    () => mergeOnDate([modelS, spyS, p3S, p2S, userS]),
    [modelS, spyS, p3S, p2S, userS]
  );
  const hasAny = merged.length > 0;

  /* petit résumé quand c’est replié (dernier point de chaque série dispo) */
  const mini = useMemo(() => {
    const last = merged.at(-1) || {};
    const fmt = (v) => (Number.isFinite(v) ? v.toFixed(1) : "—");
    return {
      Model:        fmt(last.Model),
      SPY:          fmt(last.SPY),
      P3_confirmed: fmt(last.P3_confirmed),
      P2_highconv:  fmt(last.P2_highconv),
      User_picks:   fmt(last.User_picks),
      date:         last.date || "",
    };
  }, [merged]);

  return (
    <div className="bg-white dark:bg-slate-900 rounded shadow">
      {/* header pliable */}
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center justify-between px-4 py-3 text-left font-semibold border-b border-slate-200 dark:border-slate-700"
        aria-expanded={open}
      >
        <span>{title}</span>
        <div className="flex items-center gap-4 text-xs text-slate-500 dark:text-slate-400">
          {!open && hasAny && (
            <span className="hidden md:inline">
              {mini.date && <span className="mr-2">{mini.date}</span>}
              {hasKey(merged, "Model")        && <span className="mr-2">Model {mini.Model}</span>}
              {hasKey(merged, "SPY")          && <span className="mr-2">SPY {mini.SPY}</span>}
              {hasKey(merged, "P3_confirmed") && <span className="mr-2">P3 {mini.P3_confirmed}</span>}
              {hasKey(merged, "P2_highconv")  && <span className="mr-2">P2 {mini.P2_highconv}</span>}
              {hasKey(merged, "User_picks")   && <span>User {mini.User_picks}</span>}
            </span>
          )}
          {open ? <ChevronUp size={18}/> : <ChevronDown size={18}/>}
        </div>
      </button>

      {/* body (chart) */}
      {open && (
        <div className="p-4">
          {hasAny ? (
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={merged} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Legend />

                  {hasKey(merged, "Model")        && <Line type="monotone" dataKey="Model"        dot={false} strokeWidth={2} />}
                  {hasKey(merged, "SPY")          && <Line type="monotone" dataKey="SPY"          dot={false} strokeWidth={2} />}
                  {hasKey(merged, "P3_confirmed") && <Line type="monotone" dataKey="P3_confirmed" dot={false} strokeWidth={2} />}
                  {hasKey(merged, "P2_highconv")  && <Line type="monotone" dataKey="P2_highconv"  dot={false} strokeWidth={2} />}
                  {hasKey(merged, "User_picks")   && <Line type="monotone" dataKey="User_picks"   dot={false} strokeWidth={2} />}
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="text-sm text-slate-500">
              Aucune donnée d’équity à afficher.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
