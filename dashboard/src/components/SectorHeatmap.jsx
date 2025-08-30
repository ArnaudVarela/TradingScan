import React, { useMemo } from "react";
import { Treemap, Tooltip, ResponsiveContainer } from "recharts";

// tiny helpers
const norm = (x, min, max) => (max === min ? 0 : (x - min) / (max - min));
const clamp01 = (v) => Math.max(0, Math.min(1, v));

// green intensity by confirmed ratio
function colorFromRatio(r, dark = false) {
  const t = clamp01(r);
  const g = Math.round(180 + 60 * t); // 180..240
  const a = dark ? 0.8 : 1;
  return `rgba(34, ${g}, 94, ${a})`; // teal-ish → stronger when confirmed ratio high
}

// build sector → counts
function buildSectors(confirmed = [], pre = [], events = []) {
  const map = new Map();

  const bump = (row, bucket) => {
    const sector = (row.sector || "Unknown").trim() || "Unknown";
    if (!map.has(sector)) map.set(sector, { name: sector, confirmed: 0, pre: 0, events: 0, total: 0 });
    const s = map.get(sector);
    s[bucket] += 1;
    s.total += 1;
  };

  confirmed.forEach((r) => bump(r, "confirmed"));
  pre.forEach((r) => bump(r, "pre"));
  events.forEach((r) => bump(r, "events"));

  return Array.from(map.values()).sort((a, b) => b.total - a.total);
}

function SectorTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0].payload;
  return (
    <div className="rounded border bg-white text-slate-900 p-2 text-xs shadow dark:bg-slate-800 dark:text-slate-100 dark:border-slate-700">
      <div className="font-semibold mb-1">{p.name}</div>
      <div>Total: {p.total}</div>
      <div>Confirmed: {p.confirmed}</div>
      <div>Pre-signals: {p.pre}</div>
      <div>Event-driven: {p.events}</div>
      {p.total > 0 && <div className="mt-1 opacity-80">Confirm ratio: {(100 * p.confirmed / p.total).toFixed(0)}%</div>}
    </div>
  );
}

export default function SectorHeatmap({ confirmed, pre, events }) {
  const isDark = typeof window !== "undefined" && document.documentElement.classList.contains("dark");

  const data = useMemo(() => {
    const rows = buildSectors(confirmed, pre, events);
    const totals = rows.map((r) => r.total);
    const minT = Math.min(...totals, 0);
    const maxT = Math.max(...totals, 1);
    // enrich with size + fill color
    return rows.map((r) => {
      const ratio = r.total ? r.confirmed / r.total : 0;
      const size = 40 + 260 * norm(r.total, minT, maxT); // keep minimum tile visible
      return { ...r, size, fill: colorFromRatio(ratio, isDark) };
    });
  }, [confirmed, pre, events, isDark]);

  if (!data.length) {
    return (
      <div className="rounded border p-4 text-sm text-slate-500 dark:text-slate-300 dark:border-slate-700">
        No sector data yet.
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-slate-800 rounded shadow border dark:border-slate-700 p-4">
      <div className="flex justify-between items-center mb-3">
        <h2 className="text-lg font-bold">Sector heatmap</h2>
        <div className="text-xs text-slate-500 dark:text-slate-400">
          Size = total signals • Color = confirmed ratio
        </div>
      </div>
      <div style={{ width: "100%", height: 340 }}>
        <ResponsiveContainer>
          <Treemap
            data={data}
            dataKey="size"
            aspectRatio={4 / 3}
            stroke={isDark ? "#1f2937" : "#ffffff"}
            content={<CustomCell />}
          >
            <Tooltip content={<SectorTooltip />} />
          </Treemap>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// custom cell to render label with contrast
function CustomCell(props) {
  const {
    x, y, width, height, name, fill, payload,
  } = props;
  const pad = 4;
  const label = `${name} (${payload.total})`;
  const textColor = "rgba(255,255,255,0.95)";
  return (
    <g>
      <rect x={x} y={y} width={width} height={height} fill={fill} stroke="rgba(0,0,0,0.06)" />
      {width > 80 && height > 26 && (
        <text x={x + pad} y={y + 18} fill={textColor} fontSize={12} fontWeight={700}>
          {label}
        </text>
      )}
    </g>
  );
}
