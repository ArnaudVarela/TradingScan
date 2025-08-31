import React, { useMemo, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  CartesianGrid, ComposedChart, Line
} from "recharts";

/**
 * props :
 * - confirmed: []  (lignes avec { sector, ... })
 * - pre: []        (idem)
 * - events: []     (idem)
 * - breadth: []    (lignes sector_breadth.csv avec { sector, universe, ... })
 * - selectedSectors: string[]
 * - onToggleSector: (sector: string, additive: boolean) => void
 */
export default function SectorHeatmap({
  confirmed = [],
  pre = [],
  events = [],
  breadth = [],
  selectedSectors = [],
  onToggleSector = () => {},
}) {
  const [stackMode, setStackMode] = useState("abs"); // "abs" | "pct"
  const [showBreadth, setShowBreadth] = useState(true);

  // --- petits helpers
  const norm = (s) => (s || "Unknown").trim();

  const countBySector = (rows) => {
    const m = new Map();
    rows.forEach(r => {
      const k = norm(r.sector);
      m.set(k, (m.get(k) || 0) + 1);
    });
    return m;
  };

  // Comptages
  const mConfirmed = useMemo(() => countBySector(confirmed), [confirmed]);
  const mPre       = useMemo(() => countBySector(pre), [pre]);
  const mEvents    = useMemo(() => countBySector(events), [events]);

  // Breadth overlay : universe par secteur (si dispo), sinon 0
  const mUniverse  = useMemo(() => {
    const m = new Map();
    breadth.forEach(r => {
      // sector_breadth.csv => { sector, universe, ... }
      m.set(norm(r.sector), Number(r.universe || 0));
    });
    return m;
  }, [breadth]);

  // Liste ordonnée des secteurs (par total desc)
  const sectors = useMemo(() => {
    const set = new Set([
      ...mConfirmed.keys(), ...mPre.keys(), ...mEvents.keys(), ...mUniverse.keys(),
    ]);
    const arr = Array.from(set);
    return arr.sort((a, b) => {
      const ta = (mConfirmed.get(a) || 0) + (mPre.get(a) || 0) + (mEvents.get(a) || 0);
      const tb = (mConfirmed.get(b) || 0) + (mPre.get(b) || 0) + (mEvents.get(b) || 0);
      return tb - ta;
    });
  }, [mConfirmed, mPre, mEvents, mUniverse]);

  // Dataset pour le chart
  const data = useMemo(() => {
    return sectors.map(sec => {
      const c  = mConfirmed.get(sec) || 0;
      const p  = mPre.get(sec) || 0;
      const ev = mEvents.get(sec) || 0;
      const uni = mUniverse.get(sec) || 0;
      const total = c + p + ev;

      if (stackMode === "pct" && total > 0) {
        return {
          sector: sec,
          confirmed: (c / total) * 100,
          pre: (p / total) * 100,
          events: (ev / total) * 100,
          universe: uni, // overlay reste en absolu (axe droit)
          __total: total,
        };
      }
      return { sector: sec, confirmed: c, pre: p, events: ev, universe: uni, __total: total };
    });
  }, [sectors, mConfirmed, mPre, mEvents, mUniverse, stackMode]);

  // Gestion click secteur pour filtrer les tables
  const handleClick = (entry, e) => {
    if (!entry || !entry.activePayload || !entry.activeLabel) return;
    const sector = entry.activeLabel;
    const additive = e && (e.ctrlKey || e.metaKey);
    onToggleSector(sector, additive);
  };

  const yLabel = stackMode === "pct" ? "Pourcentage (%)" : "Comptes";

  return (
    <div className="rounded border border-slate-200 dark:border-slate-700">
      <div className="flex items-center justify-between p-3">
        <div className="font-semibold">Sector signals (heatmap)</div>
        <div className="flex items-center gap-2 text-xs">
          <button
            onClick={() => setStackMode("abs")}
            className={`px-2 py-1 rounded border ${stackMode === "abs" ? "bg-slate-200 dark:bg-slate-800" : ""}`}
          >Absolu</button>
          <button
            onClick={() => setStackMode("pct")}
            className={`px-2 py-1 rounded border ${stackMode === "pct" ? "bg-slate-200 dark:bg-slate-800" : ""}`}
          >% stacked</button>

          <label className="inline-flex items-center gap-1 ml-3 cursor-pointer">
            <input
              type="checkbox"
              checked={showBreadth}
              onChange={e => setShowBreadth(e.target.checked)}
            />
            <span>Overlay breadth</span>
          </label>
        </div>
      </div>

      <div className="px-2 pb-3">
        <ResponsiveContainer width="100%" height={320}>
          <ComposedChart
            data={data}
            onClick={handleClick}
            margin={{ top: 10, right: 20, left: 0, bottom: 30 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="sector"
              interval={0}
              angle={-25}
              textAnchor="end"
              height={60}
            />
            <YAxis yAxisId="left" label={{ value: yLabel, angle: -90, position: "insideLeft" }} />
            <YAxis yAxisId="right" orientation="right" hide={!showBreadth} />
            <Tooltip formatter={(val, name) => {
              if (name === "confirmed" || name === "pre" || name === "events") {
                return stackMode === "pct" ? [`${val.toFixed(1)}%`, name] : [val, name];
              }
              if (name === "universe") return [val, "Universe"];
              return [val, name];
            }} />
            <Legend />

            {/* Barres (empilées) */}
            <Bar yAxisId="left" dataKey="confirmed" stackId="a" name="confirmed" fill="#22c55e" />   {/* vert */}
            <Bar yAxisId="left" dataKey="pre"       stackId="a" name="pre"       fill="#f59e0b" />   {/* orange */}
            <Bar yAxisId="left" dataKey="events"    stackId="a" name="events"    fill="#3b82f6" />   {/* bleu */}

            {/* Overlay breadth = universe (axe droit) */}
            {showBreadth && (
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="universe"
                name="Universe"
                dot={false}
                strokeDasharray="4 4"
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>

        <div className="text-xs text-slate-500 mt-2">
          Astuce : clique un secteur pour filtrer les tables (Ctrl/Cmd = multi-sélection).
        </div>
      </div>
    </div>
  );
}
