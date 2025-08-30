// dashboard/src/components/SectorHeatmap.jsx
import React, { useMemo } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";

function buildSectors(confirmed = [], pre = [], events = []) {
  const map = new Map();

  const bump = (row, bucket) => {
    const sector = (row?.sector || "Unknown").trim() || "Unknown";
    if (!map.has(sector)) map.set(sector, { sector, confirmed: 0, pre: 0, events: 0, total: 0 });
    const s = map.get(sector);
    s[bucket] += 1;
    s.total += 1;
  };

  confirmed.forEach((r) => bump(r, "confirmed"));
  pre.forEach((r) => bump(r, "pre"));
  events.forEach((r) => bump(r, "events"));

  return Array.from(map.values())
    .sort((a, b) => b.total - a.total)
    .slice(0, 20); // limite aux 20 secteurs les plus actifs pour lisibilité
}

function SectorTip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  const x = payload.reduce(
    (acc, p) => ({ ...acc, [p.dataKey]: p.value }),
    { confirmed: 0, pre: 0, events: 0 }
  );
  const total = (x.confirmed || 0) + (x.pre || 0) + (x.events || 0);
  return (
    <div className="rounded border bg-white text-slate-900 p-2 text-xs shadow dark:bg-slate-800 dark:text-slate-100 dark:border-slate-700">
      <div className="font-semibold mb-1">{label}</div>
      <div>Total: {total}</div>
      <div>Confirmed: {x.confirmed || 0}</div>
      <div>Pre-signals: {x.pre || 0}</div>
      <div>Event-driven: {x.events || 0}</div>
      {total > 0 && (
        <div className="mt-1 opacity-80">
          Confirm ratio: {Math.round(100 * (x.confirmed || 0) / total)}%
        </div>
      )}
    </div>
  );
}

export default function SectorHeatmap({ confirmed, pre, events }) {
  const data = useMemo(() => buildSectors(confirmed, pre, events), [confirmed, pre, events]);

  if (!data.length) {
    return (
      <div className="rounded border p-4 text-sm text-slate-500 dark:text-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800">
        No sector data yet.
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-slate-800 rounded shadow border dark:border-slate-700 p-4">
      <div className="flex justify-between items-center mb-3">
        <h2 className="text-lg font-bold">Sector signals (stacked)</h2>
        <div className="text-xs text-slate-500 dark:text-slate-400">
          Bars = Confirmed + Pre + Event • Sorted by total
        </div>
      </div>
      <div style={{ width: "100%", height: 360 }}>
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 10, right: 16, left: 0, bottom: 26 }}>
            <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.3} />
            <XAxis dataKey="sector" angle={-25} textAnchor="end" interval={0} height={50} />
            <YAxis allowDecimals={false} />
            <Tooltip content={<SectorTip />} />
            <Legend />
            <Bar dataKey="confirmed" stackId="a" fill="#16a34a" /> {/* green-600 */}
            <Bar dataKey="pre"       stackId="a" fill="#f59e0b" /> {/* amber-500 */}
            <Bar dataKey="events"    stackId="a" fill="#3b82f6" /> {/* blue-500 */}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
