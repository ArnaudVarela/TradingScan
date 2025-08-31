// dashboard/src/components/EquityCurve.jsx
import { useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid,
} from "recharts";

export default function EquityCurve({ data = [], bench = [], title = "Equity curve" }) {
  // Normalisation dates + merge
  const merged = useMemo(() => {
    const parse = (rows, key) =>
      (Array.isArray(rows) ? rows : [])
        .filter(r => r?.date && r?.equity != null)
        .map(r => ({
          date: String(r.date).slice(0, 10),
          [key]: Number(r.equity),
        }));

    const a = parse(data, "model");
    const b = parse(bench, "spy");

    // index par date
    const map = new Map();
    for (const r of a) {
      map.set(r.date, { date: r.date, model: r.model });
    }
    for (const r of b) {
      const cur = map.get(r.date) || { date: r.date };
      cur.spy = r.spy;
      map.set(r.date, cur);
    }

    return Array.from(map.values()).sort((x, y) => x.date.localeCompare(y.date));
  }, [data, bench]);

  if (!merged.length) {
    return (
      <div className="bg-white dark:bg-slate-900 rounded shadow p-4">
        <div className="text-sm opacity-70">{title}</div>
        <div className="text-xs mt-2">Pas encore de points d’equity à afficher.</div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-slate-900 rounded shadow p-4">
      <div className="text-sm opacity-70 mb-2">{title}</div>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={merged}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
          <XAxis dataKey="date" tick={{ fontSize: 12 }} minTickGap={24} />
          <YAxis tick={{ fontSize: 12 }} domain={["dataMin", "dataMax"]} />
          <Tooltip />
          <Legend />
          {/* Stratégie */}
          <Line type="monotone" dataKey="model" dot={false} strokeWidth={2} name="Model (10d)" />
          {/* SPY si dispo */}
          <Line type="monotone" dataKey="spy" dot={false} strokeWidth={2} strokeOpacity={0.8} name="SPY (Buy & Hold)" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
