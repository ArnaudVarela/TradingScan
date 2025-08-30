import React, { useMemo, useState } from "react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts";

// couleurs cohérentes avec le heatmap
const COLORS = {
  confirmed: "#16a34a",
  pre:       "#f59e0b",
  events:    "#3b82f6",
};

function toSeries(rows, focusSectors) {
  // rows: [{date,bucket,sector,count}]
  // On agrège par date/bucket, en filtrant eventuellement sur focusSectors (set ou null)
  const map = new Map(); // key=date -> {date, confirmed, pre, events}
  const useFilter = focusSectors && focusSectors.size > 0;

  rows.forEach(r => {
    const date = r.date;
    const bucket = (r.bucket || "").toLowerCase();
    const sector = r.sector || "Unknown";
    const n = Number(r.count || 0);
    if (useFilter && !focusSectors.has(sector)) return;

    if (!map.has(date)) map.set(date, { date, confirmed: 0, pre: 0, events: 0 });
    const o = map.get(date);
    if (bucket === "confirmed") o.confirmed += n;
    else if (bucket === "pre") o.pre += n;
    else if (bucket === "events") o.events += n;
  });

  // tri chronologique sur date ISO "YYYY-Www" — on peut convertir en clé triable
  const toKey = (d) => {
    const [y, w] = d.split("-W");
    return Number(y) * 100 + Number(w);
  };

  return Array.from(map.values()).sort((a, b) => toKey(a.date) - toKey(b.date));
}

function Tip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  const vals = payload.reduce((acc, p) => ({ ...acc, [p.dataKey]: p.value }), {});
  const total = (vals.confirmed || 0) + (vals.pre || 0) + (vals.events || 0);
  return (
    <div className="rounded border bg-white text-slate-900 p-2 text-xs shadow
                    dark:bg-slate-800 dark:text-slate-100 dark:border-slate-700">
      <div className="font-semibold mb-1">{label}</div>
      <div style={{ color: COLORS.confirmed }}>Confirmed: {vals.confirmed || 0}</div>
      <div style={{ color: COLORS.pre       }}>Pre-signals: {vals.pre || 0}</div>
      <div style={{ color: COLORS.events    }}>Event-driven: {vals.events || 0}</div>
      <div className="mt-1 opacity-80">Total: {total}</div>
    </div>
  );
}

export default function SectorTimeline({ historyRows = [], selectedSectors = [] }) {
  const focusSet = useMemo(
    () => new Set((selectedSectors || []).map(s => (s || "").trim())),
    [selectedSectors]
  );

  const data = useMemo(() => toSeries(historyRows, focusSet), [historyRows, focusSet]);

  return (
    <div className="bg-white dark:bg-slate-800 rounded shadow border dark:border-slate-700 p-4">
      <div className="flex items-center justify-between mb-3 gap-3">
        <h2 className="text-lg font-bold">Sector signals timeline (weekly)</h2>
        <span className="text-xs opacity-70">Somme des secteurs {focusSet.size ? "(filtrés)" : "(tous)"} par bucket</span>
      </div>

      <div style={{ width: "100%", height: 340 }}>
        <ResponsiveContainer>
          <AreaChart data={data} margin={{ top: 10, right: 16, left: 0, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.25} />
            <XAxis dataKey="date" />
            <YAxis allowDecimals={false} />
            <Tooltip content={<Tip />} />
            <Legend />
            <Area type="monotone" dataKey="confirmed" stackId="1" stroke={COLORS.confirmed} fill={COLORS.confirmed} fillOpacity={0.25} />
            <Area type="monotone" dataKey="pre"       stackId="1" stroke={COLORS.pre}       fill={COLORS.pre}       fillOpacity={0.25} />
            <Area type="monotone" dataKey="events"    stackId="1" stroke={COLORS.events}    fill={COLORS.events}    fillOpacity={0.25} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-2 text-xs opacity-70">
        Conseil: clique des secteurs dans la heatmap du dessus pour voir leur évolution seule (multi-sélection Ctrl/Cmd).
      </div>
    </div>
  );
}
