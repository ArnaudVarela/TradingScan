// dashboard/src/components/SectorRotation.jsx — rotation sectorielle GICS (ETFs SPDR, RRG-lite) pour anticiper.
import { useEffect, useMemo, useState } from "react";
import { TrendingUp } from "lucide-react";
import { fetchCSVFresh, toNumber } from "../lib/csv.js";

const QUAD = {
  Leading:   { label: "Leading",   emoji: "🔵", cls: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-300 ring-emerald-500/30" },
  Improving: { label: "Improving", emoji: "🟢", cls: "bg-cyan-500/15 text-cyan-600 dark:text-cyan-300 ring-cyan-500/30" },
  Weakening: { label: "Weakening", emoji: "🟠", cls: "bg-amber-500/15 text-amber-600 dark:text-amber-300 ring-amber-500/30" },
  Lagging:   { label: "Lagging",   emoji: "🔴", cls: "bg-rose-500/15 text-rose-600 dark:text-rose-300 ring-rose-500/30" },
};
const ORDER = { Leading: 0, Improving: 1, Weakening: 2, Lagging: 3 };
const pctCls = (v) => (v > 0 ? "text-emerald-600 dark:text-emerald-400" : v < 0 ? "text-rose-600 dark:text-rose-400" : "text-slate-500");
const fmt = (v) => (Number.isFinite(v) ? (v >= 0 ? "+" : "") + v.toFixed(1) + "%" : "—");

export default function SectorRotation() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let a = true;
    fetchCSVFresh("sector_perf.csv").then((r) => { if (!a) return; setRows(Array.isArray(r) ? r : []); setLoading(false); });
    return () => { a = false; };
  }, []);

  const data = useMemo(
    () => rows
      .map((r) => ({ ...r, _1m: toNumber(r.chg_1m), _3m: toNumber(r.chg_3m), _mom: toNumber(r.rs_mom) }))
      .sort((a, b) => (ORDER[a.quadrant] ?? 9) - (ORDER[b.quadrant] ?? 9) || (b._mom ?? -99) - (a._mom ?? -99)),
    [rows]
  );

  if (!loading && data.length === 0) return null;

  return (
    <div className="panel p-4 sm:p-5 mb-6">
      <div className="flex items-center gap-2 mb-1">
        <TrendingUp size={18} className="text-cyan-500" />
        <h2 className="text-base sm:text-lg font-bold">Rotation sectorielle</h2>
        <span className="text-[11px] text-slate-400">— GICS via ETFs SPDR (prix) · RRG-lite</span>
      </div>
      <p className="text-[11px] text-slate-400 mb-3 leading-relaxed">
        🟢 <b>Improving</b> = rentre en rotation (à anticiper) · 🔵 Leading · 🟠 Weakening = sort · 🔴 Lagging.
        Tes thèmes : semis/IA/quantique → <b>IT</b> · défense/élec/espace → <b>Industrials</b> · énergie → <b>Utilities/Energy</b>.
      </p>
      {loading ? (
        <div className="text-sm text-slate-500 py-4 text-center">Chargement…</div>
      ) : (
        <div className="overflow-x-auto thin-scroll">
          <table className="w-full text-sm">
            <thead className="text-slate-500 border-b dark:border-white/10">
              <tr>
                <th className="th">Secteur</th>
                <th className="th">Rotation</th>
                <th className="th text-right">1 mois</th>
                <th className="th text-right">3 mois</th>
                <th className="th text-right">Momentum RS</th>
              </tr>
            </thead>
            <tbody>
              {data.map((d) => {
                const q = QUAD[d.quadrant] || { label: d.quadrant || "—", emoji: "", cls: "bg-slate-500/10 text-slate-500 ring-white/10" };
                return (
                  <tr key={d.etf} className="border-b border-slate-100 dark:border-white/5">
                    <td className="px-3 py-2 font-medium whitespace-nowrap">{d.sector} <span className="text-[10px] text-slate-400">{d.etf}</span></td>
                    <td className="px-3 py-2"><span className={`text-[10px] px-1.5 py-0.5 rounded-full ring-1 ring-inset ${q.cls}`}>{q.emoji} {q.label}</span></td>
                    <td className={`px-3 py-2 text-right num ${pctCls(d._1m)}`}>{fmt(d._1m)}</td>
                    <td className={`px-3 py-2 text-right num ${pctCls(d._3m)}`}>{fmt(d._3m)}</td>
                    <td className={`px-3 py-2 text-right num ${pctCls(d._mom)}`}>{Number.isFinite(d._mom) ? (d._mom >= 0 ? "↗ +" : "↘ ") + d._mom.toFixed(1) : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
