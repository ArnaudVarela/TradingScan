// dashboard/src/components/SectorHeatmap.jsx
import React, { useMemo, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";

const COLORS = {
  confirmed: "#16a34a", // green-600
  pre:       "#f59e0b", // amber-500
  events:    "#3b82f6", // blue-500
};

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
    .slice(0, 20);
}

function SectorTip({ active, payload, label, percentMode }) {
  if (!active || !payload || !payload.length) return null;

  const vals = payload.reduce(
    (acc, p) => ({ ...acc, [p.dataKey]: p.value }),
    { confirmed: 0, pre: 0, events: 0 }
  );
  const total = (vals.confirmed || 0) + (vals.pre || 0) + (vals.events || 0);
  const pct = (k) => total ? Math.round((100 * (vals[k] || 0)) / total) : 0;

  return (
    <div className="rounded border bg-white text-slate-900 p-2 text-xs shadow
                    dark:bg-slate-800 dark:text-slate-100 dark:border-slate-700">
      <div className="font-semibold mb-1">{label}</div>
      {!percentMode && <div>Total: {total}</div>}
      <div style={{ color: COLORS.confirmed }}>Confirmed: {vals.confirmed || 0} {!percentMode && `(${pct("confirmed")}%)`}</div>
      <div style={{ color: COLORS.pre       }}>Pre-signals: {vals.pre || 0} {!percentMode && `(${pct("pre")}%)`}</div>
      <div style={{ color: COLORS.events    }}>Event-driven: {vals.events || 0} {!percentMode && `(${pct("events")}%)`}</div>
      {percentMode && (
        <div className="mt-1 opacity-80">
          (stack 100% • {vals.confirmed + vals.pre + vals.events === 0 ? "no data" :
            `${Math.round(vals.confirmed)}% / ${Math.round(vals.pre)}% / ${Math.round(vals.events)}%`})
        </div>
      )}
    </div>
  );
}

/**
 * Props:
 *  - confirmed, pre, events: arrays
 *  - selectedSectors: string[] (optionnel)
 *  - onToggleSector: (sector: string, additive: boolean) => void
 */
export default function SectorHeatmap({
  confirmed = [],
  pre = [],
  events = [],
  selectedSectors = [],
  onToggleSector,
}) {
  const [percentMode, setPercentMode] = useState(false);
  const dataAbs = useMemo(() => buildSectors(confirmed, pre, events), [confirmed, pre, events]);

  // transforme en % (sum -> 100)
  const dataPct = useMemo(() => {
    return dataAbs.map((d) => {
      const sum = d.total || (d.confirmed + d.pre + d.events);
      if (!sum) return { ...d, confirmed: 0, pre: 0, events: 0 };
      return {
        ...d,
        confirmed: (100 * d.confirmed) / sum,
        pre:       (100 * d.pre) / sum,
        events:    (100 * d.events) / sum,
      };
    });
  }, [dataAbs]);

  const data = percentMode ? dataPct : dataAbs;

  const isSelected = (sector) =>
    Array.isArray(selectedSectors) && selectedSectors.length > 0
      ? selectedSectors.includes(sector)
      : true;

  const handleBarClick = (entry) => {
    if (!entry?.activeLabel || !onToggleSector) return;
    const additive = !!(window?.event?.metaKey || window?.event?.ctrlKey);
    onToggleSector(entry.activeLabel, additive);
  };

  return (
    <div className="bg-white dark:bg-slate-800 rounded shadow border dark:border-slate-700 p-4">
      <div className="flex items-center justify-between mb-3 gap-3 flex-wrap">
        <h2 className="text-lg font-bold">Sector signals</h2>
        <div className="flex items-center gap-2 text-xs">
          <button
            onClick={() => setPercentMode(false)}
            className={`px-2 py-1 rounded border dark:border-slate-600
                       ${!percentMode ? "bg-slate-200 dark:bg-slate-700" : "bg-transparent"}`}>
            Absolute
          </button>
          <button
            onClick={() => setPercentMode(true)}
            className={`px-2 py-1 rounded border dark:border-slate-600
                       ${percentMode ? "bg-slate-200 dark:bg-slate-700" : "bg-transparent"}`}>
            % stacked
          </button>
          <span className="opacity-60 ml-2">
            {percentMode ? "Bars sum to 100% per sector" : "Bars = Confirmed + Pre + Event • Sorted by total"}
          </span>
        </div>
      </div>

      <div style={{ width: "100%", height: 380 }}>
        <ResponsiveContainer>
          <BarChart
            data={data}
            margin={{ top: 10, right: 16, left: 0, bottom: 64 }}
            onClick={handleBarClick}
          >
            <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.25} />
            <XAxis
              dataKey="sector"
              angle={-25}
              textAnchor="end"
              interval={0}
              height={64}
              tickFormatter={(s) => (isSelected(s) ? s : `• ${s}`)}
            />
            <YAxis allowDecimals={false} domain={percentMode ? [0, 100] : ["auto", "auto"]} />
            <Tooltip content={<SectorTip percentMode={percentMode} />} />
            <Legend />
            <Bar dataKey="confirmed" stackId="a" fill={COLORS.confirmed} />
            <Bar dataKey="pre"       stackId="a" fill={COLORS.pre} />
            <Bar dataKey="events"    stackId="a" fill={COLORS.events} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-2 text-xs opacity-70">
        Astuce: clique un secteur pour filtrer les tables (Ctrl/Cmd = multi-sélection).
      </div>
    </div>
  );
}
