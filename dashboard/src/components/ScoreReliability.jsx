// dashboard/src/components/ScoreReliability.jsx — fiabilité MESURÉE du score (lit score_reliability.csv).
import { useEffect, useMemo, useState } from "react";
import { ShieldCheck } from "lucide-react";
import { fetchCSVFresh, toNumber } from "../lib/csv.js";

const BUCKETS = ["70+", "60-70", "50-60"];
const BUCKET_CLS = {
  "70+": "text-emerald-600 dark:text-emerald-300",
  "60-70": "text-lime-600 dark:text-lime-300",
  "50-60": "text-amber-600 dark:text-amber-300",
};
const exCls = (v) => (v > 0 ? "text-emerald-600 dark:text-emerald-400" : v < 0 ? "text-rose-600 dark:text-rose-400" : "text-slate-500");

export default function ScoreReliability() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [h, setH] = useState(10);

  useEffect(() => {
    let alive = true;
    fetchCSVFresh("score_reliability.csv").then((r) => {
      if (!alive) return;
      setRows(Array.isArray(r) ? r : []);
      setLoading(false);
    });
    return () => { alive = false; };
  }, []);

  const horizons = useMemo(() => {
    const s = new Set(rows.map((r) => toNumber(r.horizon)).filter((x) => x != null));
    return Array.from(s).sort((a, b) => a - b);
  }, [rows]);

  const cards = useMemo(() => {
    const byB = {};
    rows.filter((r) => toNumber(r.horizon) === h).forEach((r) => { byB[String(r.bucket)] = r; });
    return BUCKETS.map((b) => ({ bucket: b, row: byB[b] })).filter((x) => x.row);
  }, [rows, h]);

  return (
    <div className="panel p-4 sm:p-5">
      <div className="flex items-center justify-between gap-3 mb-3">
        <div className="flex items-center gap-2">
          <ShieldCheck size={18} className="text-cyan-500" />
          <h2 className="text-base sm:text-lg font-bold">Fiabilité mesurée du score</h2>
          <span className="text-[11px] text-slate-400">— outcome réel des setups loggés, net de frais, vs SPY</span>
        </div>
        {horizons.length > 0 && (
          <div className="flex items-center gap-1">
            {horizons.map((hh) => (
              <button key={hh} className={`chip ${h === hh ? "chip-on" : ""}`} onClick={() => setH(hh)}>{hh}j</button>
            ))}
          </div>
        )}
      </div>

      {loading ? (
        <div className="text-sm text-slate-500 py-6 text-center">Chargement…</div>
      ) : cards.length === 0 ? (
        <div className="text-sm text-slate-500 py-6 text-center leading-relaxed">
          📈 En cours d'accumulation.<br />
          Le rapport se remplit à mesure que les fenêtres forward (5/10/20 j) des setups loggés s'écoulent.
          <span className="block text-xs text-slate-400 mt-1">
            (Chaque scan quotidien enregistre les setups ; leur rendement réel est résolu quelques jours plus tard.)
          </span>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {cards.map(({ bucket, row }) => {
              const wr = toNumber(row.winrate);
              const avg = toNumber(row.avg_ret);
              const ex = toNumber(row.avg_excess_spy);
              const beat = toNumber(row.pct_beat_spy);
              const n = toNumber(row.n);
              return (
                <div key={bucket} className="rounded-xl border border-slate-200 dark:border-white/10 p-3">
                  <div className="flex items-baseline justify-between">
                    <span className={`font-bold ${BUCKET_CLS[bucket] || ""}`}>Score {bucket}</span>
                    <span className="text-[11px] text-slate-400 num">n={n ?? "—"}</span>
                  </div>
                  <div className="mt-2 grid grid-cols-2 gap-y-1 text-sm">
                    <span className="text-slate-500">Winrate</span><span className="text-right num font-medium">{wr != null ? wr.toFixed(0) + "%" : "—"}</span>
                    <span className="text-slate-500">Rendement moy.</span><span className="text-right num font-medium">{avg != null ? (avg >= 0 ? "+" : "") + avg.toFixed(2) + "%" : "—"}</span>
                    <span className="text-slate-500">Excès vs SPY</span><span className={`text-right num font-semibold ${ex != null ? exCls(ex) : ""}`}>{ex != null ? (ex >= 0 ? "+" : "") + ex.toFixed(2) + "%" : "—"}</span>
                    <span className="text-slate-500">% bat SPY</span><span className="text-right num font-medium">{beat != null ? beat.toFixed(0) + "%" : "—"}</span>
                  </div>
                </div>
              );
            })}
          </div>
          <p className="mt-3 text-[11px] text-slate-400 leading-relaxed">
            Lecture : si <b>« Score 70+ »</b> a un excès vs SPY et un winrate supérieurs aux autres tranches, le score
            trie utilement. Sinon, le /100 n'ajoute pas de valeur — et le tableau le dit honnêtement (fenêtre glissante).
          </p>
        </>
      )}
    </div>
  );
}
