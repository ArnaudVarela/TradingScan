// dashboard/src/components/SectorHeat.jsx — Chaleur sectorielle × Rotation, en CLASSEMENT RELATIF cross-secteurs.
// Le marché a toujours des leaders/retardataires relatifs : on classe les secteurs les uns par rapport aux autres
// sur 2 axes (buzz news + force prix RS via rs_ratio & rs_mom), et "signal fort" = haut des DEUX classements.
// -> plus de "9/11 en signal fort" : par construction seuls les vrais standouts ressortent.
// Toggle : "Secteurs GICS" (couverture totale) ↔ "Thèmes hard-tech" (détail).
import { useEffect, useMemo, useState } from "react";
import { Newspaper } from "lucide-react";
import { fetchCSVFresh, toNumber } from "../lib/csv.js";

const TL = {
  semiconducteurs: ["Semi-conducteurs", "bg-cyan-500"],
  quantique: ["Quantique", "bg-violet-500"],
  ia: ["IA", "bg-fuchsia-500"],
  laser_photonique: ["Laser/Photonique", "bg-sky-500"],
  robotique: ["Robotique", "bg-teal-500"],
  equipement_electrique: ["Équip. électrique", "bg-amber-500"],
  production_energie: ["Énergie", "bg-lime-500"],
  gestion_thermique: ["Thermique", "bg-orange-500"],
  espace: ["Espace", "bg-indigo-500"],
  defense: ["Défense", "bg-rose-500"],
  new_tech: ["New tech", "bg-slate-500"],
};

const THEME_GICS = {
  semiconducteurs: "Information Technology", quantique: "Information Technology", ia: "Information Technology",
  laser_photonique: "Information Technology", new_tech: "Information Technology",
  robotique: "Industrials", equipement_electrique: "Industrials", gestion_thermique: "Industrials",
  espace: "Industrials", defense: "Industrials", production_energie: "Energy",
};

const GICS_FR = {
  "Information Technology": "Tech (IT)", "Health Care": "Santé", "Financials": "Finance",
  "Industrials": "Industrie", "Consumer Discretionary": "Conso cyclique", "Consumer Staples": "Conso défensive",
  "Energy": "Énergie", "Utilities": "Utilities", "Materials": "Matériaux",
  "Real Estate": "Immobilier", "Communication Services": "Communication",
};
const GICS_COLOR = {
  "Information Technology": "bg-cyan-500", "Health Care": "bg-emerald-500", "Financials": "bg-indigo-500",
  "Industrials": "bg-amber-500", "Consumer Discretionary": "bg-fuchsia-500", "Consumer Staples": "bg-lime-500",
  "Energy": "bg-orange-500", "Utilities": "bg-teal-500", "Materials": "bg-rose-500",
  "Real Estate": "bg-sky-500", "Communication Services": "bg-violet-500",
};

const QUAD = {
  Leading:   { emoji: "🔵", label: "Leading" },
  Improving: { emoji: "🟢", label: "Improving" },
  Weakening: { emoji: "🟠", label: "Weakening" },
  Lagging:   { emoji: "🔴", label: "Lagging" },
};

const CONF = {
  fort:      { emoji: "🔥", label: "signal fort",    cls: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-300 ring-emerald-500/30", rank: 0 },
  anticiper: { emoji: "⚡", label: "à anticiper",    cls: "bg-cyan-500/15 text-cyan-600 dark:text-cyan-300 ring-cyan-500/30",             rank: 1 },
  radar:     { emoji: "🔍", label: "sous le radar",  cls: "bg-sky-500/15 text-sky-600 dark:text-sky-300 ring-sky-500/25",                 rank: 2 },
  hype:      { emoji: "⚠️", label: "hype sans prix", cls: "bg-amber-500/15 text-amber-600 dark:text-amber-300 ring-amber-500/30",         rank: 3 },
  neutre:    { emoji: "",   label: "",               cls: "",                                                                             rank: 4 },
};

// Seuil unique : "haut de classement" = percentile >= T. Baisser -> plus permissif ; monter -> plus sélectif.
const T = 0.6;

// Rang percentile [0,1] (0 = plus bas, 1 = plus haut), ties = rang moyen ; non-finis -> 0.5 (neutre).
function pranks(values) {
  const idx = values.map((v, i) => [v, i]).filter(([v]) => Number.isFinite(v)).sort((a, b) => a[0] - b[0]);
  const k = idx.length;
  const out = values.map(() => 0.5);
  if (k <= 1) return out;
  let i = 0;
  while (i < k) {
    let j = i;
    while (j + 1 < k && idx[j + 1][0] === idx[i][0]) j++;   // groupe d'égalités
    const avgPos = (i + j) / 2;
    for (let t = i; t <= j; t++) out[idx[t][1]] = avgPos / (k - 1);
    i = j + 1;
  }
  return out;
}

// heatRank (H), priceRank (P = moyenne rangs rs_ratio & rs_mom), momRank (A = rang rs_mom).
function confLabel(H, P, A) {
  if (H >= T && P >= T) return "fort";                 // buzz ET prix en haut de classement
  if (P >= T && H < T) return A >= 0.5 ? "anticiper" : "radar"; // prix fort, news calme (accélère -> anticiper, stable -> radar)
  if (H >= T && P < T) return "hype";                  // buzz en haut, prix pas au niveau
  return "neutre";
}

export default function SectorHeat() {
  const [mode, setMode] = useState("gics");
  const [themeRows, setThemeRows] = useState([]);
  const [gicsRows, setGicsRows] = useState([]);
  const [perfBySector, setPerfBySector] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    Promise.all([
      fetchCSVFresh("sector_heat.csv"),
      fetchCSVFresh("sector_heat_gics.csv"),
      fetchCSVFresh("sector_perf.csv"),
    ]).then(([themes, gics, perf]) => {
      if (!alive) return;
      const m = {};
      (Array.isArray(perf) ? perf : []).forEach((x) => {
        if (x.sector) m[String(x.sector).trim()] = { quadrant: x.quadrant, rs_ratio: x.rs_ratio, rs_mom: x.rs_mom };
      });
      setPerfBySector(m);
      setThemeRows(Array.isArray(themes) ? themes : []);
      setGicsRows(Array.isArray(gics) ? gics : []);
      setLoading(false);
    });
    return () => { alive = false; };
  }, []);

  // Fallback : si la vue GICS n'a pas encore de données, on retombe sur les thèmes.
  const effMode = mode === "gics" && gicsRows.length === 0 ? "themes" : mode;

  const data = useMemo(() => {
    const src = effMode === "gics" ? gicsRows : themeRows;
    const base = src.map((r) => {
      let label, color, gics, key;
      if (effMode === "gics") {
        gics = String(r.sector || "").trim();
        label = GICS_FR[gics] || gics; color = GICS_COLOR[gics] || "bg-slate-500"; key = gics;
      } else {
        const tl = TL[r.theme] || [r.theme, "bg-slate-500"];
        label = tl[0]; color = tl[1]; gics = THEME_GICS[r.theme] || ""; key = r.theme;
      }
      const perf = perfBySector[gics] || {};
      return {
        key, label, color, gics, _h: toNumber(r.heat) ?? 0,
        quad: perf.quadrant || "", rs_ratio: toNumber(perf.rs_ratio), rs_mom: toNumber(perf.rs_mom),
        top_headline: r.top_headline,
      };
    });
    const max = Math.max(1, ...base.map((d) => d._h));
    const H = pranks(base.map((d) => d._h));
    const RR = pranks(base.map((d) => d.rs_ratio));
    const RM = pranks(base.map((d) => d.rs_mom));
    return base
      .map((d, i) => ({ ...d, _max: max, conf: confLabel(H[i], (RR[i] + RM[i]) / 2, RM[i]) }))
      .sort((a, b) => (CONF[a.conf].rank - CONF[b.conf].rank) || (b._h - a._h));
  }, [effMode, gicsRows, themeRows, perfBySector]);

  if (!loading && data.length === 0) return null;

  return (
    <div className="panel p-4 sm:p-5 mb-6">
      <div className="flex flex-wrap items-center gap-2 mb-1">
        <Newspaper size={18} className="text-cyan-500" />
        <h2 className="text-base sm:text-lg font-bold">Chaleur sectorielle</h2>
        <span className="text-[11px] text-slate-400">— news 48h × force prix (classement relatif)</span>
        <div className="ml-auto flex gap-1.5">
          {[["gics", "Secteurs GICS"], ["themes", "Thèmes hard-tech"]].map(([k, l]) => (
            <button
              key={k}
              onClick={() => setMode(k)}
              className={`text-[11px] px-2.5 py-1 rounded-full ring-1 ring-inset transition ${
                effMode === k
                  ? "bg-cyan-500/15 text-cyan-600 dark:text-cyan-300 ring-cyan-500/40 font-semibold"
                  : "bg-slate-500/5 text-slate-500 dark:text-slate-400 ring-white/10 hover:bg-slate-500/10"
              }`}
            >
              {l}
            </button>
          ))}
        </div>
      </div>
      <p className="text-[11px] text-slate-400 mb-3 leading-relaxed">
        Classement <b>relatif</b> des secteurs entre eux (news + force prix RS). 🔥 <b>signal fort</b> = haut des deux ·
        ⚡ <b>à anticiper</b> = prix fort qui accélère, news calme · 🔍 <b>sous le radar</b> = prix fort mais stable, news calme ·
        ⚠️ <b>hype sans prix</b> = news chaudes, prix pas au niveau.
      </p>
      {loading ? (
        <div className="text-sm text-slate-500 py-4 text-center">Chargement…</div>
      ) : (
        <div className="space-y-2">
          {data.map((d) => {
            const q = QUAD[d.quad];
            const c = CONF[d.conf];
            return (
              <div key={d.key} className="flex items-center gap-2 sm:gap-3 text-sm">
                <div className="w-28 sm:w-32 shrink-0 font-medium truncate" title={d.gics ? `Secteur GICS : ${d.gics}` : undefined}>{d.label}</div>
                <div className="w-6 shrink-0 text-center" title={q ? `Rotation ${d.gics} : ${q.label}` : "Rotation inconnue"}>
                  {q ? q.emoji : <span className="text-slate-300 dark:text-slate-600">·</span>}
                </div>
                <div className="flex-1 min-w-[40px] h-2 rounded-full bg-slate-100 dark:bg-white/5 overflow-hidden">
                  <div className={`h-full ${d.color} rounded-full`} style={{ width: `${Math.round((d._h / d._max) * 100)}%` }} />
                </div>
                <div className="w-8 text-right num text-slate-500 shrink-0">{d._h}</div>
                <div className="w-28 sm:w-32 shrink-0">
                  {c.label ? (
                    <span className={`text-[10px] px-1.5 py-0.5 rounded-full ring-1 ring-inset whitespace-nowrap ${c.cls}`}>{c.emoji} {c.label}</span>
                  ) : null}
                </div>
                <div className="hidden xl:block flex-[2] truncate text-[11px] text-slate-400" title={d.top_headline}>
                  {d.top_headline}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
