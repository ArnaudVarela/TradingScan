// dashboard/src/components/SectorTimeline.jsx
import React, { useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid,
} from "recharts";

/**
 * Props:
 * - history OR historyRows: [{ date: 'YYYY-Www', bucket: 'confirmed'|'pre'|'events', sector: '...', count: number }]
 * - selectedSectors: [] | Set<string>
 */
export default function SectorTimeline(props) {
  const rawRows = props.history ?? props.historyRows ?? [];
  const selected = props.selectedSectors ?? [];

  const selectedSet = useMemo(() => {
    if (selected instanceof Set) return selected;
    return new Set(Array.isArray(selected) ? selected : []);
  }, [selected]);

  const data = useMemo(() => {
    if (!Array.isArray(rawRows) || rawRows.length === 0) return [];

    // Normalize and filter
    const rows = rawRows
      .map((r) => ({
        date: String(r.date || "").trim(),
        bucket: String(r.bucket || "").toLowerCase().trim(),
        sector: String(r.sector || "Unknown").trim(),
        count: Number(r.count ?? r.Count ?? r.COUNT ?? 0),
      }))
      .filter((r) => r.date && r.bucket && !Number.isNaN(r.count));

    // If sectors are selected, keep only those
    const filtered = selectedSet.size
      ? rows.filter((r) => selectedSet.has(r.sector))
      : rows;

    // Aggregate by (date, bucket) summing counts across sectors
    const key = (d, b) => `${d}__${b}`;
    const agg = new Map();
    for (const r of filtered) {
      const k = key(r.date, r.bucket);
      agg.set(k, (agg.get(k) || 0) + r.count);
    }

    // Build a wide table per date with columns confirmed/pre/events
    const byDate = new Map();
    for (const [k, v] of agg) {
      const [d, b] = k.split("__");
      if (!byDate.has(d)) byDate.set(d, { date: d, confirmed: 0, pre: 0, events: 0 });
      if (b === "confirmed") byDate.get(d).confirmed = v;
      else if (b === "pre") byDate.get(d).pre = v;
      else if (b === "events") byDate.get(d).events = v;
    }

    // Sort by date label (string), so weeks appear in order of label
    // If you want chrono ordering across years, you can later parse and sort properly.
    const out = Array.from(byDate.values()).sort((a, b) =>
      String(a.date).localeCompare(String(b.date))
    );

    return out;
  }, [rawRows, selectedSet]);

  if (!data.length) {
    return (
      <div className="text-sm text-slate-500 dark:text-slate-400">
        Aucune donnée historique exploitable pour le moment.
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-slate-800 shadow rounded p-4 mb-6">
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-semibold">Sector signals timeline (weekly)</h3>
        <div className="text-xs text-slate-500">Somme des secteurs (tous) par bucket</div>
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={data} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis allowDecimals={false} />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="confirmed" stroke="#16a34a" dot={false} name="confirmed" />
          <Line type="monotone" dataKey="pre" stroke="#f59e0b" dot={false} name="pre" />
          <Line type="monotone" dataKey="events" stroke="#2563eb" dot={false} name="events" />
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-2 text-xs text-slate-500">
        Conseil: clique des secteurs dans la heatmap du dessus pour voir leur évolution (multi-sélection Ctrl/Cmd).
      </div>
    </div>
  );
}
