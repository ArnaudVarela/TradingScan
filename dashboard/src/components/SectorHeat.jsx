// dashboard/src/components/SectorHeat.jsx — chaleur sectorielle (buzz news par thème, Google News 48h).
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

export default function SectorHeat() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    fetchCSVFresh("sector_heat.csv").then((r) => {
      if (!alive) return;
      setRows(Array.isArray(r) ? r : []);
      setLoading(false);
    });
    return () => { alive = false; };
  }, []);

  const data = useMemo(
    () => rows.map((r) => ({ ...r, _h: toNumber(r.heat) ?? 0 })).sort((a, b) => b._h - a._h),
    [rows]
  );
  const max = useMemo(() => Math.max(1, ...data.map((d) => d._h)), [data]);

  if (!loading && data.length === 0) return null;

  return (
    <div className="panel p-4 sm:p-5 mb-6">
      <div className="flex items-center gap-2 mb-3">
        <Newspaper size={18} className="text-cyan-500" />
        <h2 className="text-base sm:text-lg font-bold">Chaleur sectorielle</h2>
        <span className="text-[11px] text-slate-400">— buzz news par thème (48 h · Google News)</span>
      </div>
      {loading ? (
        <div className="text-sm text-slate-500 py-4 text-center">Chargement…</div>
      ) : (
        <div className="space-y-2">
          {data.map((d) => {
            const [label, color] = TL[d.theme] || [d.theme, "bg-slate-500"];
            return (
              <div key={d.theme} className="flex items-center gap-3 text-sm">
                <div className="w-32 shrink-0 font-medium truncate">{label}</div>
                <div className="flex-1 h-2 rounded-full bg-slate-100 dark:bg-white/5 overflow-hidden">
                  <div className={`h-full ${color} rounded-full`} style={{ width: `${Math.round((d._h / max) * 100)}%` }} />
                </div>
                <div className="w-8 text-right num text-slate-500">{d._h}</div>
                <div className="hidden lg:block flex-[2] truncate text-[11px] text-slate-400" title={d.top_headline}>
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
