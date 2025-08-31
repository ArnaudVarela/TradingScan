// SectorHeatmap.jsx
import React, { useMemo, useState } from "react";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid, Brush,
} from "recharts";

// util: compte par secteur
function countBySector(rows) {
  const out = {};
  for (const r of rows || []) {
    const s = (r?.sector || "Unknown").trim() || "Unknown";
    out[s] = (out[s] || 0) + 1;
  }
  return out;
}

// util: transforme {sec: n} en [{sector, value}]
function dictToSeries(dict, keyName) {
  return Object.entries(dict)
    .map(([sector, value]) => ({ sector, [keyName]: value }));
}

export default function SectorHeatmap({
  confirmed = [],
  pre = [],
  events = [],
  breadth = [],                 // <-- NEW (sector_breadth.csv)
  selectedSectors = [],
  onToggleSector,
}) {
  // états d’affichage
  const [mode, setMode] = useState("abs");   // "abs" | "pct"
  const [showBreadth, setShowBreadth] = useState(true);

  // map breadth: secteur -> taille univers (nb tickers)
  const breadthMap = useMemo(() => {
    // sector_breadth.csv attendus: sector,total (ou columns sector, count)
    const out = {};
    for (const r of breadth || []) {
      const s = (r?.sector || "Unknown").trim() || "Unknown";
      const n = Number(r?.total ?? r?.count ?? r?.value ?? 0);
      out[s] = (out[s] || 0) + (Number.isFinite(n) ? n : 0);
    }
    return out;
  }, [breadth]);

  // Comptages par secteur
  const c = useMemo(() => countBySector(confirmed), [confirmed]);
  const p = useMemo(() => countBySector(pre), [pre]);
  const e = useMemo(() => countBySector(events), [events]);

  // union des secteurs
  const sectors = useMemo(() => {
    const s = new Set([...Object.keys(c), ...Object.keys(p), ...Object.keys(e), ...Object.keys(breadthMap)]);
    return [...s];
  }, [c, p, e, breadthMap]);

  // dataset
  const data = useMemo(() => {
    const rows = sectors.map((sec) => {
      const cc = c[sec] || 0;
      const pp = p[sec] || 0;
      const ee = e[sec] || 0;
      const tot = cc + pp + ee;

      if (mode === "pct") {
        const d = tot || 1;
        return {
          sector: sec,
          confirmed: (cc * 100) / d,
          pre: (pp * 100) / d,
          events: (ee * 100) / d,
          breadth: breadthMap[sec] || 0,
          __abs_total: tot,
        };
      }
      // mode absolu
      return {
        sector: sec,
        confirmed: cc,
        pre: pp,
        events: ee,
        breadth: breadthMap[sec] || 0,
        __abs_total: tot,
      };
    });

    // tri par total (abs)
    rows.sort((a, b) => (b.__abs_total - a.__abs_total));
    return rows;
  }, [sectors, c, p, e, mode, breadthMap]);

  const maxBreadth = useMemo(
    () => Math.max(1, ...data.map(d => d.breadth || 0)),
    [data]
  );

  // tooltip custom
  const tooltip = ({ active, payload, label }) => {
    if (!active || !payload || payload.length === 0) return null;
    const row = payload.reduce((acc, p) => ({ ...acc, [p.dataKey]: p.value }), { sector: label });
    const totalAbs = data.find(d => d.sector === label)?.__abs_total ?? 0;
    return (
      <div className="rounded border bg-white/90 dark:bg-slate-900/90 px-3 py-2 text-xs shadow">
        <div className="font-semibold mb-1">{label}</div>
        <div>Confirmed: <b>{row.confirmed?.toFixed?.(mode === "pct" ? 0 : 0)}</b>{mode === "pct" ? "%" : ""}</div>
        <div>Pre-signals: <b>{row.pre?.toFixed?.(mode === "pct" ? 0 : 0)}</b>{mode === "pct" ? "%" : ""}</div>
        <div>Event-driven: <b>{row.events?.toFixed?.(mode === "pct" ? 0 : 0)}</b>{mode === "pct" ? "%" : ""}</div>
        <div className="mt-1 opacity-70">Total (abs): {totalAbs}</div>
        {showBreadth && <div className="opacity-70">Universe size: {row.breadth}</div>}
      </div>
    );
  };

  // clic pour filtrer les tables
  const onBarClick = (e) => {
    if (!onToggleSector || !e?.activeLabel) return;
    const additive = (e?.event?.ctrlKey || e?.event?.metaKey) ?? false;
    onToggleSector(e.activeLabel, additive);
  };

  return (
    <div>
      {/* contrôles */}
      <div className="flex items-center justify-end gap-2 text-xs mb-2">
        <button
          className={`px-2 py-1 rounded border ${mode === "abs" ? "bg-slate-200 dark:bg-slate-700" : ""}`}
          onClick={() => setMode("abs")}
        >
          Absolu
        </button>
        <button
          className={`px-2 py-1 rounded border ${mode === "pct" ? "bg-slate-200 dark:bg-slate-700" : ""}`}
          onClick={() => setMode("pct")}
        >
          % stacked
        </button>
        <label className="ml-3 flex items-center gap-1 cursor-pointer">
          <input type="checkbox" checked={showBreadth} onChange={(e) => setShowBreadth(e.target.checked)} />
          <span>Overlay breadth</span>
        </label>
      </div>

      <div style={{ width: "100%", height: 360 }}>
        <ResponsiveContainer>
          <BarChart
            data={data}
            onClick={onBarClick}
            margin={{ top: 10, right: 10, left: 0, bottom: 40 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="sector" angle={-25} textAnchor="end" interval={0} height={60} />
            <YAxis yAxisId="left" tickFormatter={(v) => (mode === "pct" ? `${v}%` : v)} />
            {showBreadth && (
              <YAxis yAxisId="right" orientation="right" hide domain={[0, maxBreadth]} />
            )}
            <Tooltip content={tooltip} />
            <Legend verticalAlign="top" height={24} />

            {/* Overlay breadth (gris clair, derrière) */}
            {showBreadth && (
              <Bar
                yAxisId="right"
                dataKey="breadth"
                name="Universe"
                fill="#cbd5e1"
                opacity={0.45}
                barSize={14}
              />
            )}

            {/* signaux */}
            <Bar yAxisId="left" dataKey="confirmed" name="confirmed" stackId={mode === "pct" ? "s" : undefined} fill="#10b981" />
            <Bar yAxisId="left" dataKey="pre"       name="pre"       stackId={mode === "pct" ? "s" : undefined} fill="#f59e0b" />
            <Bar yAxisId="left" dataKey="events"    name="events"    stackId={mode === "pct" ? "s" : undefined} fill="#3b82f6" />

            <Brush height={18} y={320} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="text-[11px] mt-2 opacity-70">
        Astuce : clique un secteur pour filtrer les tables (Ctrl/Cmd = multi-sélection).
      </div>
    </div>
  );
}
