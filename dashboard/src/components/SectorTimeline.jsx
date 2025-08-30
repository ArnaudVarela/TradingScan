import React, { useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";

/**
 * history: array d'objets { date: "2025-W35", bucket: "confirmed"|"pre"|"events", sector: "...", count: <string|number> }
 * selectedSectors: Set<string> OU Array<string> (optionnel)
 */
export default function SectorTimeline({ history = [], selectedSectors }) {
  // normalise la sélection en Set lowercase
  const selectedSet = useMemo(() => {
    if (!selectedSectors) return new Set();
    if (selectedSectors instanceof Set) {
      return new Set([...selectedSectors].map((s) => String(s).toLowerCase()));
    }
    if (Array.isArray(selectedSectors)) {
      return new Set(selectedSectors.map((s) => String(s).toLowerCase()));
    }
    return new Set();
  }, [selectedSectors]);

  const data = useMemo(() => {
    if (!Array.isArray(history) || history.length === 0) return [];

    // Agrégation par semaine et par bucket (confirmed / pre / events)
    const byWeek = new Map(); // week -> {confirmed, pre, events}

    for (const row of history) {
      if (!row) continue;
      const week = String(row.date || "").trim(); // ex: 2025-W35
      if (!week) continue;

      const bucket = String(row.bucket || "").toLowerCase(); // "confirmed" | "pre" | "events"
      if (!["confirmed", "pre", "events"].includes(bucket)) continue;

      const sector = String(row.sector || "Unknown");
      const count = Number(row.count ?? row.Count ?? row.COUNT ?? 0) || 0;

      // filtre par secteurs si sélection active
      if (selectedSet.size > 0 && !selectedSet.has(sector.toLowerCase())) continue;

      if (!byWeek.has(week)) byWeek.set(week, { week, confirmed: 0, pre: 0, events: 0 });
      byWeek.get(week)[bucket] += count;
    }

    // Tri chronologique simple (les chaînes "YYYY-Www" se trient correctement)
    const out = Array.from(byWeek.values()).sort((a, b) => (a.week < b.week ? -1 : a.week > b.week ? 1 : 0));
    return out;
  }, [history, selectedSet]);

  if (!data.length) {
    return (
      <div className="p-4 rounded border border-slate-200 dark:border-slate-700 text-sm opacity-80">
        Aucune donnée historique exploitable pour le moment.
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-slate-900 rounded shadow p-4">
      <div className="flex items-center justify-between mb-3 text-xs opacity-70">
        <span>Sector signals timeline (weekly)</span>
        <span>Somme des secteurs {selectedSet.size ? "(filtrés)" : "(tous)"} par bucket</span>
      </div>

      <div style={{ width: "100%", height: 320 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="week" />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="confirmed" name="confirmed" stroke="#16a34a" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="pre"        name="pre"        stroke="#f59e0b" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="events"     name="events"     stroke="#2563eb" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-2 text-xs opacity-70">
        Conseil: clique des secteurs dans la heatmap du dessus pour voir leur évolution seule (multi-sélection Ctrl/Cmd).
      </div>
    </div>
  );
}
