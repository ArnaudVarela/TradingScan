// dashboard/src/components/SectorTimeline.jsx
import React, { useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid,
} from "recharts";

// util: récupère la première clé existante parmi plusieurs candidats
function pick(obj, keys) {
  for (const k of keys) {
    if (obj[k] !== undefined && obj[k] !== null && String(obj[k]).trim() !== "") {
      return obj[k];
    }
  }
  return undefined;
}

/**
 * Props possibles:
 * - history OU historyRows: [{ date, bucket, sector, count }]
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

    // normalisation ultra-robuste
    const rows = rawRows.map((r) => {
      const date = pick(r, ["date", "Date", "DATE", "\ufeffdate", "﻿date"]);
      const bucketRaw = pick(r, ["bucket", "Bucket", "BUCKET"]);
      const sector = pick(r, ["sector", "Sector", "SECTOR"]) ?? "Unknown";
      const countRaw = pick(r, ["count", "Count", "COUNT"]);

      const bucket = (bucketRaw ?? "").toString().toLowerCase().trim();
      const count = Number.parseFloat(countRaw ?? 0);

      return {
        date: String(date ?? "").trim(),
        bucket,
        sector: String(sector).trim(),
        count: Number.isFinite(count) ? count : 0,
      };
    })
    .filter((r) => r.date && r.bucket && r.count >= 0);

    const filtered = selectedSet.size
      ? rows.filter((r) => selectedSet.has(r.sector))
      : rows;

    // agrégation par (date,bucket) → somme des secteurs
    const byDate = new Map();
    for (const r of filtered) {
      if (!byDate.has(r.date)) {
        byDate.set(r.date, { date: r.date, confirmed: 0, pre: 0, events: 0 });
      }
      const cur = byDate.get(r.date);
      if (r.bucket === "confirmed") cur.confirmed += r.count;
      else if (r.bucket === "pre") cur.pre += r.count;
      else if (r.bucket === "events") cur.events += r.count;
    }

    // tri par label (suffisant pour 1–2 semaines; on fera un tri chrono si besoin multi-années)
    return Array.from(byDate.values()).sort((a, b) =>
      String(a.date).localeCompare(String(b.date))
    );
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
        Conseil : clique des secteurs dans la heatmap du dessus pour voir leur évolution (multi-sélection Ctrl/Cmd).
      </div>
    </div>
  );
}
