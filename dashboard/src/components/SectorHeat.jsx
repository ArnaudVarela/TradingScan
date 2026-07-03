// dashboard/src/components/SectorHeat.jsx — Chaleur sectorielle croisée avec la Rotation (confluence news × prix).
// Toggle : "Secteurs GICS" (11 secteurs, couverture totale, jointure 1:1 avec la Rotation)
//        ↔ "Thèmes hard-tech" (détail semis/IA/quantique via un pont thème→GICS).
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

// Pont thème -> secteur GICS parent (libellés EXACTS de sector_perf.py) pour la vue "Thèmes".
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
  hype:      { emoji: "⚠️", label: "hype sans prix", cls: "bg-amber-500/15 text-amber-600 dark:text-amber-300 ring-amber-500/30",         rank: 2 },
  radar:     { emoji: "🔍", label: "sous le radar",  cls: "bg-sky-500/15 text-sky-600 dark:text-sky-300 ring-sky-500/25",                 rank: 3 },
  neutre:    { emoji: "",   label: "",               cls: "",                                                                             rank: 4 },
};

function confKey(quad, hot) {
  const rotIn = quad === "Leading" || quad === "Improving";
  const rotOut = quad === "Weakening" || quad === "Lagging";
  if (rotIn && hot) return "fort";
  if (quad === "Improving") return "anticiper";
  if (rotOut && hot) return "hype";
  if (quad === "Leading") return "radar";
  return "neutre";
}

export default function SectorHeat() {
  const [mode, setMode] = useState("gics");
  const [themeRows, setThemeRows] = useState([]);
  const [gicsRows, setGicsRows] = useState([]);
  const [quadBySector, setQuadBySector] = useState({});
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
      (Array.isArray(perf) ? perf : []).forEach((x) => { if (x.sector) m[String(x.sector).trim()] = x.quadrant; });
      setQuadBySector(m);
      setThemeRows(Array.isArray(themes) ? themes : []);
      setGicsRows(Array.isArray(gics) ? gics : []);
      setLoading(false);
    });
    return () => { alive = false; };
  }, []);

  // Si la vue GICS n'a pas encore de données (avant le 1er scan backend), on retombe sur les thèmes.
  const effMode = mode === "gics" && gicsRows.length === 0 ? "themes" : mode;

  const data = useMemo(() => {
    const src = effMode === "gics" ? gicsRows : themeRows;
    const base = src.map((r) => {
      if (effMode === "gics") {
        const gics = String(r.sector || "").trim();
        return { key: gics, label: GICS_FR[gics] || gics, color: GICS_COLOR[gics] || "bg-slate-500", gics, _h: toNumber(r.heat) ?? 0, top_headline: r.top_headline };
      }
      const th = r.theme;
      const [label, color] = TL[th] || [th, "bg-slate-500"];
      return { key: th, label, color, gics: THEME_GICS[th] || "", _h: toNumber(r.heat) ?? 0, top_headline: r.top_headline };
    });
    const max = Math.max(1, ...base.map((d) => d._h));
    return base
      .map((d) => {
        const quad = quadBySector[d.gics] || "";
        const hot = d._h >= 0.4 * max;
        return { ...d, _max: max, quad, conf: confKey(quad, hot) };
      })
      .sort((a, b) => (CONF[a.conf].rank - CONF[b.conf].rank) || (b._h - a._h));
  }, [effMode, gicsRows, themeRows, quadBySector]);

  if (!loading && data.length === 0) return null;

  return (
    <div className="panel p-4 sm:p-5 mb-6">
      <div className="flex flex-wrap items-center gap-2 mb-1">
        <Newspaper size={18} className="text-cyan-500" />
        <h2 className="text-base sm:text-lg font-bold">Chaleur sectorielle</h2>
        <span className="text-[11px] text-slate-400">— news 48h (Google) × rotation prix</span>
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
        🔥 <b>signal fort</b> = buzz + secteur en rotation entrante · ⚡ <b>à anticiper</b> = secteur qui rentre, buzz pas encore là ·
        ⚠️ <b>hype sans prix</b> = buzz mais prix qui décroche · 🔍 <b>sous le radar</b> = prix leader, news calme.
        {effMode === "gics" ? " Vue 11 secteurs GICS (couverture totale)." : " Vue thèmes hard-tech (détail)."}
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
