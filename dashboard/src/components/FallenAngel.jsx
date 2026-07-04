// dashboard/src/components/FallenAngel.jsx — Module "Fallen Angel" : candidats HAUTE-VARIANCE (structure validée
// par backtest, edge MODESTE ~PF 1,3 net). L'edge est dans le STOP, pas la sélection. Framing honnête, paper-first.
import { useEffect, useMemo, useState } from "react";
import { Feather, AlertTriangle } from "lucide-react";
import { fetchCSVFresh, toNumber } from "../lib/csv.js";

const tvUrl = (t) => `https://www.tradingview.com/chart/?symbol=${encodeURIComponent(String(t).trim().replace(/-/g, "."))}`;

function fmtMcap(n) {
  if (!Number.isFinite(n) || n <= 0) return "—";
  if (n >= 1e9) return "$" + (n / 1e9).toFixed(1) + "B";
  if (n >= 1e6) return "$" + (n / 1e6).toFixed(0) + "M";
  return "$" + n.toFixed(0);
}

export default function FallenAngel() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [onlyStrict, setOnlyStrict] = useState(false);

  useEffect(() => {
    let a = true;
    fetchCSVFresh("fallen_angel.csv").then((r) => {
      if (!a) return;
      setRows(Array.isArray(r) ? r : []);
      setLoading(false);
    });
    return () => { a = false; };
  }, []);

  const data = useMemo(() => {
    let v = rows.map((r) => ({
      ...r, _dd: toNumber(r.dd_from_hi_pct), _vg: toNumber(r.vol_growth_pct),
      _dr: toNumber(r.dist_resist_pct), _mc: toNumber(r.mcap_usd),
    }));
    if (onlyStrict) v = v.filter((r) => r.level === "strict");
    return v.sort((a, b) => (a.level === b.level ? (b._vg ?? 0) - (a._vg ?? 0) : a.level === "strict" ? -1 : 1));
  }, [rows, onlyStrict]);

  const nStrict = useMemo(() => rows.filter((r) => r.level === "strict").length, [rows]);
  if (!loading && rows.length === 0) return null;

  return (
    <div className="panel overflow-hidden mt-6">
      <div className="p-4 sm:p-5 border-b border-slate-200/70 dark:border-white/10">
        <div className="flex flex-wrap items-center gap-2 mb-1">
          <Feather size={18} className="text-amber-500" />
          <h2 className="text-base sm:text-lg font-bold">Fallen Angel</h2>
          <span className="text-[11px] text-slate-400">— défoncé + base tendue + volume en hausse + re-test de résistance</span>
          <div className="ml-auto flex items-center gap-2">
            <span className="text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-500 dark:bg-white/10 dark:text-slate-300 num">{data.length}</span>
            <button
              onClick={() => setOnlyStrict((v) => !v)}
              className={`text-[11px] px-2.5 py-1 rounded-full ring-1 ring-inset transition ${
                onlyStrict ? "bg-amber-500/15 text-amber-600 dark:text-amber-300 ring-amber-500/40 font-semibold"
                           : "bg-slate-500/5 text-slate-500 dark:text-slate-400 ring-white/10 hover:bg-slate-500/10"}`}
            >
              strict uniquement ({nStrict})
            </button>
          </div>
        </div>
        <div className="mt-2 flex items-start gap-2 text-[11px] leading-relaxed text-amber-700 dark:text-amber-300/90 bg-amber-500/10 ring-1 ring-inset ring-amber-500/25 rounded-lg px-3 py-2">
          <AlertTriangle size={14} className="mt-0.5 shrink-0" />
          <span>
            <b>Candidats haute-variance, pas des signaux d'achat.</b> Backtest sur 6 ans / 1 100+ trades &lt;$5B :
            edge <b>modeste</b> (profit factor ~1,3 net de coûts réalistes), winrate ~25-35% mais gagnants 4-5× les perdants.
            <b> L'edge est dans le STOP</b> (couper vite, laisser courir) — pas dans la sélection seule.
            Biais de survivorship + régime favorable non exclus → <b>paper-trade d'abord</b>.
          </span>
        </div>
      </div>

      {loading ? (
        <div className="p-12 text-center text-sm text-slate-500">Chargement…</div>
      ) : data.length === 0 ? (
        <div className="p-12 text-center text-sm text-slate-500">Aucun candidat aujourd'hui.</div>
      ) : (
        <div className="overflow-x-auto max-h-[60vh] thin-scroll">
          <table className="w-full text-sm">
            <thead className="sticky top-0 z-10 bg-white/90 dark:bg-slate-950/70 backdrop-blur">
              <tr className="border-b border-slate-200 dark:border-white/10 text-slate-500">
                <th className="th w-10">#</th>
                <th className="th">Ticker</th>
                <th className="th text-right">Prix</th>
                <th className="th text-right">MCap</th>
                <th className="th text-right" title="Baisse depuis le pic ~3 ans">Depuis pic</th>
                <th className="th text-right" title="Croissance du volume récent vs base">Volume</th>
                <th className="th text-right" title="Distance au re-test de la résistance de la base">Résistance</th>
                <th className="th text-right" title="Stop initial de référence (−10%)">Stop réf.</th>
                <th className="th text-right" title="Score technique /100 (contexte)">Score</th>
              </tr>
            </thead>
            <tbody>
              {data.map((r, i) => (
                <tr key={r.ticker + "_" + i} className="border-b border-slate-100 dark:border-white/5 hover:bg-slate-50 dark:hover:bg-white/5 transition-colors">
                  <td className="px-3 py-2 text-slate-400 num">{i + 1}</td>
                  <td className="px-3 py-2">
                    <div className="flex items-center gap-2">
                      <a href={tvUrl(r.ticker)} target="_blank" rel="noreferrer"
                         className="font-semibold tracking-tight hover:text-amber-600 dark:hover:text-amber-300 hover:underline decoration-dotted underline-offset-2"
                         title={`Ouvrir ${r.ticker} sur TradingView`}>{r.ticker}</a>
                      <span className={`text-[10px] px-1.5 py-0.5 rounded ring-1 ring-inset ${
                        r.level === "strict" ? "bg-amber-500/15 text-amber-600 dark:text-amber-300 ring-amber-500/30"
                                             : "bg-slate-500/10 text-slate-500 dark:text-slate-400 ring-white/10"}`}>
                        {r.level}
                      </span>
                    </div>
                  </td>
                  <td className="px-3 py-2 text-right num text-slate-600 dark:text-slate-300">{r.price}</td>
                  <td className="px-3 py-2 text-right num text-slate-500">{fmtMcap(r._mc)}</td>
                  <td className="px-3 py-2 text-right num text-rose-600 dark:text-rose-400">−{r._dd}%</td>
                  <td className="px-3 py-2 text-right num text-emerald-600 dark:text-emerald-400">+{r._vg}%</td>
                  <td className="px-3 py-2 text-right num text-slate-600 dark:text-slate-300">{r._dr}%</td>
                  <td className="px-3 py-2 text-right num text-slate-500" title="Stop initial −10%">{r.stop_ref}</td>
                  <td className="px-3 py-2 text-right num text-slate-400">{toNumber(r.score) ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
