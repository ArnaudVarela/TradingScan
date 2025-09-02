// dashboard/src/components/EquityCurve.jsx
import React, { useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid,
} from "recharts";

// rows: [{date, equity}]
function normSeries(rows, key) {
  if (!Array.isArray(rows)) return [];
  return rows
    .filter(r => r && r.date != null && r.equity != null && r.equity !== "")
    .map(r => ({ date: String(r.date), [key]: Number(r.equity) }))
    .filter(r => Number.isFinite(r[key]));
}

function mergeOnDate(series) {
  // series: array of arrays already shaped as {date, keyX}
  const map = new Map();
  for (const arr of series) {
    for (const row of arr) {
      const d = row.date;
      if (!map.has(d)) map.set(d, { date: d });
      Object.assign(map.get(d), row);
    }
  }
  return Array.from(map.values()).sort((a,b) => a.date.localeCompare(b.date));
}

export default function EquityCurve({
  data,     // auto model 10d
  bench,    // spy 10d
  p3,       // cohort
  p2,       // cohort
  user,     // ⬅️ Mes Picks 10d
  title = "Equity curve",
}) {
  const modelS = useMemo(() => normSeries(data, "Model"), [data]);
  const spyS   = useMemo(() => normSeries(bench, "SPY"), [bench]);
  const p3S    = useMemo(() => normSeries(p3, "P3_confirmed"), [p3]);
  const p2S    = useMemo(() => normSeries(p2, "P2_highconv"), [p2]);
  const userS  = useMemo(() => normSeries(user, "User_picks"), [user]);

  const merged = useMemo(() => mergeOnDate([modelS, spyS, p3S, p2S, userS]), [modelS, spyS, p3S, p2S, userS]);

  const hasAny = merged.length > 0;

  return (
    <div className="bg-white dark:bg-slate-900 rounded shadow p-4">
      <h3 className="font-semibold mb-3">{title}</h3>
      {hasAny ? (
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={merged} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              {/* Laisse Recharts choisir les couleurs par défaut */}
              {"Model" in merged[0] && <Line type="monotone" dataKey="Model" dot={false} strokeWidth={2} />}
              {"SPY" in merged[0] && <Line type="monotone" dataKey="SPY" dot={false} strokeWidth={2} />}
              {"P3_confirmed" in merged[0] && <Line type="monotone" dataKey="P3_confirmed" dot={false} strokeWidth={2} />}
              {"P2_highconv" in merged[0] && <Line type="monotone" dataKey="P2_highconv" dot={false} strokeWidth={2} />}
              {"User_picks" in merged[0] && <Line type="monotone" dataKey="User_picks" dot={false} strokeWidth={2} />}
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="text-sm text-slate-500">Aucune donnée d’équity à afficher.</div>
      )}
    </div>
  );
}
