// dashboard/src/components/BacktestSummary.jsx
import React, { useMemo } from "react";

const fmtPct = (x) =>
  x === null || x === undefined || x === "" ? "—" : `${Number(x).toFixed(2)}%`;
const fmtInt = (x) =>
  x === null || x === undefined || x === "" ? "—" : Number(x).toLocaleString();

// Petits badges colorés (vert = bon, rouge = mauvais, gris = neutre)
function Chip({ value, type }) {
  // type: "winrate" | "avg"
  if (value === null || value === undefined || value === "") {
    return <span className="px-2 py-0.5 text-xs rounded bg-slate-200 dark:bg-slate-700">—</span>;
  }
  const v = Number(value);

  let cls = "bg-slate-200 text-slate-900 dark:bg-slate-700 dark:text-slate-100";
  if (type === "winrate") {
    if (v >= 65) cls = "bg-green-600 text-white";
    else if (v >= 55) cls = "bg-green-300 text-black";
    else if (v <= 45) cls = "bg-red-500 text-white";
    else if (v <= 50) cls = "bg-orange-400 text-black";
  } else if (type === "avg") {
    if (v >= 1.0) cls = "bg-green-600 text-white";
    else if (v >= 0.3) cls = "bg-green-300 text-black";
    else if (v <= -1.0) cls = "bg-red-600 text-white";
    else if (v < 0) cls = "bg-orange-400 text-black";
  }

  const text = `${v.toFixed(2)}%`;
  return <span className={`px-2 py-0.5 text-xs rounded ${cls}`}>{text}</span>;
}

export default function BacktestSummary({ rows }) {
  const byHorizon = useMemo(() => {
    const g = new Map();
    (rows || []).forEach((r) => {
      const h = String(r.horizon_days ?? r.horizon ?? "").trim();
      if (!g.has(h)) g.set(h, []);
      g.get(h).push(r);
    });
    return Array.from(g.entries()).sort((a, b) => Number(a[0]) - Number(b[0]));
  }, [rows]);

  if (!rows || rows.length === 0) {
    return (
      <div className="text-sm text-slate-500">
        Aucun résumé de backtest disponible (fichier <code className="font-mono">backtest_summary.csv</code> manquant ou vide).
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {byHorizon.map(([h, subset]) => (
        <div key={h} className="bg-white dark:bg-slate-900 rounded shadow p-4">
          <div className="mb-3 flex items-baseline gap-2">
            <h3 className="text-base font-semibold">Horizon {h} jours</h3>
            <span className="text-xs text-slate-500">
              {fmtInt(
                subset.reduce((acc, r) => acc + Number(r.n_trades || 0), 0)
              )}{" "}
              trades total
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left bg-slate-50 dark:bg-slate-800">
                  <th className="p-2">Bucket</th>
                  <th className="p-2">Trades</th>
                  <th className="p-2">Winrate</th>
                  <th className="p-2">Moyenne</th>
                  <th className="p-2">Médiane</th>
                  <th className="p-2">P95</th>
                  <th className="p-2">P05</th>
                </tr>
              </thead>
              <tbody>
                {subset
                  .sort((a, b) => (b.winrate ?? 0) - (a.winrate ?? 0))
                  .map((r, i) => (
                    <tr
                      key={i}
                      className="border-t border-slate-100 dark:border-slate-800"
                    >
                      <td className="p-2">{r.bucket || "—"}</td>
                      <td className="p-2">{fmtInt(r.n_trades)}</td>
                      <td className="p-2">
                        <Chip value={r.winrate} type="winrate" />
                      </td>
                      <td className="p-2">
                        <Chip value={r.avg_ret} type="avg" />
                      </td>
                      <td className="p-2">{fmtPct(r.median_ret)}</td>
                      <td className="p-2">{fmtPct(r.p95_ret)}</td>
                      <td className="p-2">{fmtPct(r.p05_ret)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>

          {/* Légende rapide */}
          <div className="mt-3 text-xs text-slate-500 flex flex-wrap gap-2">
            <span>🟩 très bon</span>
            <span className="opacity-70">🟨 passable</span>
            <span>🟥 faible</span>
          </div>
        </div>
      ))}
    </div>
  );
}
