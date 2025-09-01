import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from "recharts";

// Affiche une courbe d’equity principale (data),
// + SPY (bench), + cohortes P3 / P2 si fournies.
export default function EquityCurve({
  data = [],
  bench = [],   // SPY: [{date, equity}]
  p3 = [],      // P3  : [{date, equity}]
  p2 = [],      // P2  : [{date, equity}]
  title = "Equity curve",
}) {
  const key = (d) => {
    try {
      return new Date(d.date).toISOString().slice(0, 10);
    } catch {
      return String(d.date).slice(0, 10);
    }
  };

  // merge on ISO date
  const idx = new Map();
  (data || []).forEach(d => idx.set(key(d), { date: key(d), model: d.equity }));
  (bench || []).forEach(d => {
    const k = key(d);
    idx.set(k, { ...(idx.get(k) || { date: k }), spy: d.equity });
  });
  (p3 || []).forEach(d => {
    const k = key(d);
    idx.set(k, { ...(idx.get(k) || { date: k }), p3: d.equity });
  });
  (p2 || []).forEach(d => {
    const k = key(d);
    idx.set(k, { ...(idx.get(k) || { date: k }), p2: d.equity });
  });

  const series = Array.from(idx.values()).sort((a, b) =>
    a.date.localeCompare(b.date)
  );

  const hasModel = series.some(s => Number.isFinite(s.model));
  const hasSpy   = series.some(s => Number.isFinite(s.spy));
  const hasP3    = series.some(s => Number.isFinite(s.p3));
  const hasP2    = series.some(s => Number.isFinite(s.p2));

  return (
    <div className="w-full bg-white dark:bg-slate-900 rounded shadow p-4">
      <div className="font-semibold mb-2">{title}</div>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={series} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 10 }} domain={["dataMin", "dataMax"]} />
            <Tooltip />
            <Legend />
            {/* Si P3/P2 existent, on les montre; sinon on garde la courbe 'model' */}
            {hasP3    && <Line type="monotone" dataKey="p3"    name="P3 (3/3 pillars)" dot={false} strokeWidth={2} />}
            {hasP2    && <Line type="monotone" dataKey="p2"    name="P2 (2/3 + votes≥15)" dot={false} strokeWidth={2} />}
            {!hasP3 && !hasP2 && hasModel &&
              <Line type="monotone" dataKey="model" name="Model" dot={false} strokeWidth={2} />
            }
            {hasSpy   && <Line type="monotone" dataKey="spy"   name="SPY (buy & hold)" dot={false} strokeWidth={2} />}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
