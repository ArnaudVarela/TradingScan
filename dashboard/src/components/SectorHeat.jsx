// dashboard/src/components/SectorHeat.jsx — Chaleur sectorielle (buzz news par thème, Google News 48h)
// CROISÉE avec la Rotation sectorielle (prix) : chaque thème -> son secteur GICS parent -> quadrant de rotation.
// La confluence news×prix prime le tri : chaud + secteur qui rentre = signal fort.
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

// Pont thème -> secteur GICS parent (libellés EXACTS de sector_perf.py / panneau Rotation).
const THEME_GICS = {
  semiconducteurs: "Information Technology",
  quantique: "Information Technology",
  ia: "Information Technology",
  laser_photonique: "Information Technology",
  new_tech: "Information Technology",
  robotique: "Industrials",
  equipement_electrique: "Industrials",
  gestion_thermique: "Industrials",
  espace: "Industrials",
  defense: "Industrials",
  production_energie: "Energy",
};

const QUAD = {
  Leading:   { emoji: "🔵", label: "Leading",   cls: "text-emerald-600 dark:text-emerald-300" },
  Improving: { emoji: "🟢", label: "Improving", cls: "text-cyan-600 dark:text-cyan-300" },
  Weakening: { emoji: "🟠", label: "Weakening", cls: "text-amber-600 dark:text-amber-300" },
  Lagging:   { emoji: "🔴", label: "Lagging",   cls: "text-rose-600 dark:text-rose-300" },
};

const CONF = {
  fort:      { emoji: "🔥", label: "signal fort",    cls: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-300 ring-emerald-500/30", rank: 0 },
  anticiper: { emoji: "⚡", label: "à anticiper",    cls: "bg-cyan-500/15 text-cyan-600 dark:text-cyan-300 ring-cyan-500/30",             rank: 1 },
  hype:      { emoji: "⚠️", label: "hype sans prix", cls: "bg-amber-500/15 text-amber-600 dark:text-amber-300 ring-amber-500/30",         rank: 2 },
  radar:     { emoji: "🔍", label: "sous le radar",  cls: "bg-sky-500/15 text-sky-600 dark:text-sky-300 ring-sky-500/25",                 rank: 3 },
  neutre:    { emoji: "",   label: "",               cls: "",                                                                             rank: 4 },
};

// news chaude + secteur qui rentre => fort ; entrant sans buzz => à anticiper ;
// chaud mais prix qui décroche => hype ; prix leader mais news calme => sous le radar.
function confKey(quad, hot) {
  const rotIn = quad === "Leading" || quad === "Improving";
  const rotOut = quad === "Weakening" || quad === "Lagging";
  if (rotIn && hot) return "fort";
  if (quad === "Improving") return "anticiper";   // ici forcément !hot
  if (rotOut && hot) return "hype";
  if (quad === "Leading") return "radar";         // ici forcément !hot
  return "neutre";                                 // Weakening/Lagging sans buzz, ou pas de rotation connue
}

export default function SectorHeat() {
  const [rows, setRows] = useState([]);
  const [quadBySector, setQuadBySector] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    Promise.all([fetchCSVFresh("sector_heat.csv"), fetchCSVFresh("sector_perf.csv")]).then(([heat, perf]) => {
      if (!alive) return;
      const m = {};
      (Array.isArray(perf) ? perf : []).forEach((x) => { if (x.sector) m[String(x.sector).trim()] = x.quadrant; });
      setQuadBySector(m);
      setRows(Array.isArray(heat) ? heat : []);
      setLoading(false);
    });
    return () => { alive = false; };
  }, []);

  const data = useMemo(() => {
    const base = rows.map((r) => ({ ...r, _h: toNumber(r.heat) ?? 0 }));
    const max = Math.max(1, ...base.map((d) => d._h));
    return base
      .map((d) => {
        const gics = THEME_GICS[d.theme] || "";
        const quad = quadBySector[gics] || "";
        const hot = d._h >= 0.4 * max;
        const conf = confKey(quad, hot);
        return { ...d, _max: max, gics, quad, conf };
      })
      .sort((a, b) => (CONF[a.conf].rank - CONF[b.conf].rank) || (b._h - a._h));
  }, [rows, quadBySector]);

  if (!loading && data.length === 0) return null;

  return (
    <div className="panel p-4 sm:p-5 mb-6">
      <div className="flex items-center gap-2 mb-1">
        <Newspaper size={18} className="text-cyan-500" />
        <h2 className="text-base sm:text-lg font-bold">Chaleur sectorielle</h2>
        <span className="text-[11px] text-slate-400">— news 48h (Google) × rotation prix</span>
      </div>
      <p className="text-[11px] text-slate-400 mb-3 leading-relaxed">
        🔥 <b>signal fort</b> = buzz + secteur en rotation entrante · ⚡ <b>à anticiper</b> = secteur qui rentre, buzz pas encore là ·
        ⚠️ <b>hype sans prix</b> = buzz mais prix qui décroche · 🔍 <b>sous le radar</b> = prix leader, news calme.
      </p>
      {loading ? (
        <div className="text-sm text-slate-500 py-4 text-center">Chargement…</div>
      ) : (
        <div className="space-y-2">
          {data.map((d) => {
            const [label, color] = TL[d.theme] || [d.theme, "bg-slate-500"];
            const q = QUAD[d.quad];
            const c = CONF[d.conf];
            return (
              <div key={d.theme} className="flex items-center gap-2 sm:gap-3 text-sm">
                <div className="w-28 sm:w-32 shrink-0 font-medium truncate" title={d.gics ? `Secteur GICS : ${d.gics}` : undefined}>{label}</div>
                <div className="w-6 shrink-0 text-center" title={q ? `Rotation ${d.gics} : ${q.label}` : "Rotation inconnue"}>
                  {q ? q.emoji : <span className="text-slate-300 dark:text-slate-600">·</span>}
                </div>
                <div className="flex-1 min-w-[40px] h-2 rounded-full bg-slate-100 dark:bg-white/5 overflow-hidden">
                  <div className={`h-full ${color} rounded-full`} style={{ width: `${Math.round((d._h / d._max) * 100)}%` }} />
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
