import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  Line,
} from "recharts";
import { useMemo, useState } from "react";

export default function SectorHeatmap({
  confirmed = [],
  pre = [],
  events = [],
  breadth = [],
  selectedSectors = [],
  onToggleSector = () => {},
}) {
  const [mode, setMode] = useState("absolute"); // "absolute" | "percent"
  const [overlayBreadth, setOverlayBreadth] = useState(false);

  // --- préparation des données agrégées
  const data = useMemo(() => {
    const counts = {};

    const add = (rows, key) => {
      rows.forEach((r) => {
        const sec = r.sector || "Unknown";
        if (!counts[sec]) counts[sec] = { sector: sec, confirmed: 0, pre: 0, events: 0 };
        counts[sec][key] += 1;
      });
    };

    add(confirmed, "confirmed");
    add(pre, "pre");
    add(events, "events");

    // merge breadth (universe)
    breadth.forEach((b) => {
      const sec = b.sector || "Unknown";
      if (!counts[sec]) counts[sec] = { sector: sec, confirmed: 0, pre: 0, events: 0 };
      counts[sec].universe = b.universe || 0;
    });

    let arr = Object.values(counts);

    // en % si demandé
    if (mode === "percent") {
      arr = arr.map((r) => {
        const total = r.confirmed + r.pre + r.events;
        return {
          ...r,
          confirmed: total ? (r.confirmed / total) * 100 : 0,
          pre: total ? (r.pre / total) * 100 : 0,
          events: total ? (r.events / total) * 100 : 0,
        };
      });
    }

    return arr.sort((a, b) => (b.confirmed + b.pre + b.events) - (a.confirmed + a.pre + a.events));
  }, [confirmed, pre, events, breadth, mode]);

  const colors = {
    confirmed: "#22c55e", // green-500
    pre: "#f59e0b",       // amber-500
    events: "#3b82f6",    // blue-500
    universe: "#94a3b8",  // slate-400 (overlay)
  };

  return (
    <div>
      <div className="flex justify-end gap-2 mb-2 text-xs">
        <button
          onClick={() => setMode("absolute")}
          className={`px-2 py-1 border rounded ${mode === "absolute" ? "bg-slate-200 dark:bg-slate-700" : ""}`}
        >
          Absolu
        </button>
        <button
          onClick={() => setMode("percent")}
          className={`px-2 py-1 border rounded ${mode === "percent" ? "bg-slate-200 dark:bg-slate-700" : ""}`}
        >
          % stacked
        </button>
        <label className="flex items-center gap-1 cursor-pointer">
          <input
            type="checkbox"
            checked={overlayBreadth}
            onChange={(e) => setOverlayBreadth(e.target.checked)}
          />
          Overlay breadth
        </label>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={data}
          margin={{ top: 10, right: 20, left: 0, bottom: 40 }}
          onClick={(e) => {
            if (e && e.activeLabel) onToggleSector(e.activeLabel, e.ctrlKey || e.metaKey);
          }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="sector" angle={-35} textAnchor="end" interval={0} height={60} />
          <YAxis />
          <Tooltip />
          <Legend />

          <Bar dataKey="confirmed" stackId="a" fill={colors.confirmed} stroke="#fff" strokeWidth={0.5} />
          <Bar dataKey="pre"       stackId="a" fill={colors.pre}       stroke="#fff" strokeWidth={0.5} />
          <Bar dataKey="events"    stackId="a" fill={colors.events}    stroke="#fff" strokeWidth={0.5} />

          {overlayBreadth && (
            <Line
              type="monotone"
              dataKey="universe"
              stroke={colors.universe}
              strokeWidth={2}
              dot={false}
            />
          )}
        </BarChart>
      </ResponsiveContainer>

      <p className="mt-2 text-xs text-slate-500">
        Astuce : clique un secteur pour filtrer les tables (Ctrl/Cmd = multi-sélection).
      </p>
    </div>
  );
}
