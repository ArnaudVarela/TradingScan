// dashboard/src/components/ThematicSetups.jsx
// Setups thématiques de pré-explosion, score /100. Lit thematic_setups.csv (local-first).
import { useEffect, useMemo, useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { fetchCSVFresh, toNumber } from "../lib/csv.js";

const THEME_LABELS = {
  semiconducteurs: "🔲 Semi-conducteurs",
  quantique: "⚛️ Quantique",
  ia: "🤖 IA",
  laser_photonique: "🔦 Laser/Photonique",
  robotique: "🦾 Robotique",
  equipement_electrique: "⚡ Équip. électrique",
  production_energie: "🔋 Énergie",
  gestion_thermique: "🌡️ Thermique",
  espace: "🚀 Espace",
  defense: "🛡️ Défense",
  new_tech: "🆕 New tech",
};

function fmtMcap(v) {
  const n = Number(v);
  if (!Number.isFinite(n) || n <= 0) return "—";
  if (n >= 1e12) return "$" + (n / 1e12).toFixed(2) + "T";
  if (n >= 1e9) return "$" + (n / 1e9).toFixed(1) + "B";
  if (n >= 1e6) return "$" + (n / 1e6).toFixed(0) + "M";
  return "$" + n.toFixed(0);
}

function scoreClasses(s) {
  if (s >= 70) return "bg-emerald-500 text-white";
  if (s >= 60) return "bg-lime-500 text-white";
  if (s >= 50) return "bg-amber-400 text-slate-900";
  return "bg-slate-300 text-slate-700 dark:bg-slate-700 dark:text-slate-200";
}

function Chip({ active, onClick, children }) {
  return (
    <button
      onClick={onClick}
      className={`text-xs px-2 py-1 rounded border ${
        active
          ? "bg-slate-800 text-white dark:bg-slate-200 dark:text-slate-900"
          : "border-slate-300 dark:border-slate-600 hover:bg-slate-100 dark:hover:bg-slate-800"
      }`}
    >
      {children}
    </button>
  );
}

export default function ThematicSetups() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [theme, setTheme] = useState("all");
  const [open, setOpen] = useState(true);

  useEffect(() => {
    let alive = true;
    fetchCSVFresh("thematic_setups.csv").then((r) => {
      if (!alive) return;
      setRows(Array.isArray(r) ? r : []);
      setLoading(false);
    });
    return () => { alive = false; };
  }, []);

  const themes = useMemo(() => {
    const set = new Set();
    rows.forEach((r) => (r.themes || "").split("|").forEach((t) => t && set.add(t)));
    return Array.from(set).sort();
  }, [rows]);

  const filtered = useMemo(
    () =>
      rows
        .map((r) => ({ ...r, _score: toNumber(r.score) ?? 0 }))
        .filter((r) => theme === "all" || (r.themes || "").split("|").includes(theme))
        .sort((a, b) => b._score - a._score),
    [rows, theme]
  );

  return (
    <div className="mb-6 bg-white dark:bg-slate-900 rounded shadow p-4">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center justify-between w-full text-left font-semibold"
        aria-expanded={open}
      >
        <span>🎯 Setups thématiques — score /100 (pré-explosion)</span>
        {open ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
      </button>

      {open && (
        <>
          <div className="mt-2 mb-3 text-xs text-slate-600 dark:text-slate-400 bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 rounded p-2 leading-relaxed">
            <b>Comment lire ce score.</b> Le /100 mesure la <i>qualité de timing</i> d'un setup
            (tendance saine, base tendue près du pivot, contraction de volatilité, volume qui
            s'assèche, RSI/MACD constructifs). Backtesté honnêtement : les scores élevés{" "}
            <b>gagnent ~62% du temps et battent SPY</b> à 20&nbsp;jours — mais ce n'est <b>pas</b> un
            prédicteur de la <i>taille</i> des mouvements. Le vrai moteur reste l'
            <b>appartenance thématique</b> (biais de survivance à garder en tête). À utiliser comme
            filtre de découverte, <b>pas</b> comme signal d'achat automatique.
          </div>

          <div className="flex flex-wrap gap-1 mb-3">
            <Chip active={theme === "all"} onClick={() => setTheme("all")}>
              Tous ({rows.length})
            </Chip>
            {themes.map((t) => (
              <Chip key={t} active={theme === t} onClick={() => setTheme(t)}>
                {THEME_LABELS[t] || t}
              </Chip>
            ))}
          </div>

          {loading ? (
            <div className="text-sm text-slate-500">Chargement…</div>
          ) : filtered.length === 0 ? (
            <div className="text-sm text-slate-500">
              Aucun setup à afficher (fichier <span className="font-mono">thematic_setups.csv</span> absent ?).
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-left text-slate-500 border-b dark:border-slate-700">
                  <tr>
                    <th className="py-1 pr-2">#</th>
                    <th className="pr-2">Ticker</th>
                    <th className="pr-2">Thème(s)</th>
                    <th className="pr-2">Score</th>
                    <th className="pr-2">Setup</th>
                    <th className="pr-2 text-right">Prix</th>
                    <th className="pr-2 text-right">MCap</th>
                    <th className="pr-2 text-right">RSI</th>
                    <th className="pr-2 text-right">Dist. haut</th>
                    <th className="pr-2 text-right">Base</th>
                    <th className="pr-2 text-right">Vol dry-up</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.slice(0, 60).map((r, i) => (
                    <tr key={r.ticker} className="border-b border-slate-100 dark:border-slate-800">
                      <td className="py-1 pr-2 text-slate-400">{i + 1}</td>
                      <td className="pr-2 font-semibold">{r.ticker}</td>
                      <td className="pr-2 text-xs text-slate-500">
                        {(r.themes || "").split("|").map((t) => THEME_LABELS[t] || t).join(", ")}
                      </td>
                      <td className="pr-2">
                        <span className={`inline-block px-2 py-0.5 rounded font-semibold ${scoreClasses(r._score)}`}>
                          {r._score.toFixed(0)}
                        </span>
                      </td>
                      <td className="pr-2">{r.setup}</td>
                      <td className="pr-2 text-right">{r.price}</td>
                      <td className="pr-2 text-right">{fmtMcap(r.mcap_usd)}</td>
                      <td className="pr-2 text-right">{r.rsi}</td>
                      <td className="pr-2 text-right">{r.dist_to_high_pct}%</td>
                      <td className="pr-2 text-right">{r.base_depth_pct}%</td>
                      <td className="pr-2 text-right">{r.vol_dryup}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="mt-2 text-xs text-slate-400">
                {filtered.length} setups · affichage top 60 · mise à jour quotidienne (EOD)
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
