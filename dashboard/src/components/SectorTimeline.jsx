import React, { useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid,
} from "recharts";

// Try multiple possible header names (case-insensitive)
function pickKey(row, variants) {
  const keys = Object.keys(row);
  const map = {};
  for (const k of keys) map[k.toLowerCase()] = k;
  for (const v of variants) {
    const found = map[v.toLowerCase()];
    if (found) return found;
  }
  return null;
}

function coerceNum(v) {
  if (v === null || v === undefined || v === "") return 0;
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

/**
 * Props:
 *  - history: array of rows from sector_history.csv
 * Optional:
 *  - selectedSectors: Set<string> to filter (you can pass it later from the heatmap)
 */
export default function SectorTimeline({ history = [], selectedSectors }) {
  // Normalize headers + aggregate per week across selected sectors
  const data = useMemo(() => {
    if (!history || history.length === 0) return [];

    // Resolve header names once
    const sample = history[0] || {};
    const weekKey  = pickKey(sample, ["week", "date", "period"]);
    const sectKey  = pickKey(sample, ["sector", "gics_sector", "industry"]);
    const confKey  = pickKey(sample, ["confirmed", "confirm", "confirmed_count"]);
    const preKey   = pickKey(sample, ["pre", "pre_signals", "anticipative"]);
    const evtKey   = pickKey(sample, ["events", "event", "event_driven"]);

    if (!weekKey || !sectKey || !confKey || !preKey || !evtKey) {
      // Headers not recognized — return empty so we show the “no data” box
      return [];
    }

    // Aggregate: week -> { confirmed, pre, events }
    const agg = new Map(); // weekStr -> totals
    for (const row of history) {
      const week = String(row[weekKey] ?? "").trim();
      if (!week) continue;

      const sector = String(row[sectKey] ?? "Unknown").trim();
      if (selectedSectors && selectedSectors.size > 0 && !selectedSectors.has(sector)) {
        continue;
      }

      const c = coerceNum(row[confKey]);
      const p = coerceNum(row[preKey]);
      const e = coerceNum(row[evtKey]);

      const cur = agg.get(week) || { week, confirmed: 0, pre: 0, events: 0 };
      cur.confirmed += c;
      cur.pre       += p;
      cur.events    += e;
      agg.set(week, cur);
    }

    // Sort by week ascending if possible (YYYY-Www or YYYY-MM-DD both sort well lexicographically)
    return Array.from(agg.values()).sort((a, b) => String(a.week).localeCompare(String(b.week)));
  }, [history, selectedSectors]);

  if (!data.length) {
    return (
      <div className="bg-white dark:bg-slate-900 shadow rounded p-4">
        <div className="text-sm text-slate-500">Aucune donnée historique exploitable pour le moment.</div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-slate-900 shadow rounded p-4">
      <div className="flex items-baseline justify-between mb-2">
        <h3 className="text-base font-semibold">Sector signals timeline (weekly)</h3>
        <div className="text-xs text-slate-500">
          Somme des secteurs {selectedSectors && selectedSectors.size > 0 ? "(filtrés)" : "(tous)"} par bucket
        </div>
      </div>

      <div className="w-full h-64">
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 6, right: 24, bottom: 6, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="week" />
            <YAxis />
            <Tooltip />
            <Legend />
            {/* Colors are left to defaults so your theme can style them; change if you want fixed colors */}
            <Line type="monotone" dataKey="confirmed" name="confirmed" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="pre"        name="pre"        dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="events"     name="events"     dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-2 text-xs text-slate-500">
        Conseil: clique des secteurs dans la heatmap au-dessus pour voir leur évolution seule (multi-sélection Ctrl/Cmd).
      </div>
    </div>
  );
}
