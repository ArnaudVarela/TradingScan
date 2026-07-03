// dashboard/src/components/ThematicSetups.jsx — tableau des setups (redesign "terminal lux").
import { useEffect, useMemo, useState } from "react";
import { Search, Info, ArrowUpDown, ChevronUp, ChevronDown } from "lucide-react";
import { fetchCSVFresh, toNumber } from "../lib/csv.js";

const THEME_META = {
  semiconducteurs:       { label: "Semis",     dot: "bg-cyan-400",    chip: "bg-cyan-500/15 text-cyan-600 dark:text-cyan-300 ring-cyan-500/25" },
  quantique:             { label: "Quantique", dot: "bg-violet-400",  chip: "bg-violet-500/15 text-violet-600 dark:text-violet-300 ring-violet-500/25" },
  ia:                    { label: "IA",        dot: "bg-fuchsia-400", chip: "bg-fuchsia-500/15 text-fuchsia-600 dark:text-fuchsia-300 ring-fuchsia-500/25" },
  laser_photonique:      { label: "Laser",     dot: "bg-sky-400",     chip: "bg-sky-500/15 text-sky-600 dark:text-sky-300 ring-sky-500/25" },
  robotique:             { label: "Robotique", dot: "bg-teal-400",    chip: "bg-teal-500/15 text-teal-600 dark:text-teal-300 ring-teal-500/25" },
  equipement_electrique: { label: "Élec.",     dot: "bg-amber-400",   chip: "bg-amber-500/15 text-amber-600 dark:text-amber-300 ring-amber-500/25" },
  production_energie:    { label: "Énergie",   dot: "bg-lime-400",    chip: "bg-lime-500/15 text-lime-600 dark:text-lime-300 ring-lime-500/25" },
  gestion_thermique:     { label: "Thermique", dot: "bg-orange-400",  chip: "bg-orange-500/15 text-orange-600 dark:text-orange-300 ring-orange-500/25" },
  espace:                { label: "Espace",    dot: "bg-indigo-400",  chip: "bg-indigo-500/15 text-indigo-600 dark:text-indigo-300 ring-indigo-500/25" },
  defense:               { label: "Défense",   dot: "bg-rose-400",    chip: "bg-rose-500/15 text-rose-600 dark:text-rose-300 ring-rose-500/25" },
  new_tech:              { label: "New tech",  dot: "bg-slate-400",   chip: "bg-slate-500/15 text-slate-600 dark:text-slate-300 ring-slate-500/25" },
};

// Rotation (panneau SectorRotation) : même taxonomie GICS que sector_tag.py -> croisement setup <-> rotation.
const QUAD_EMOJI = { Leading: "🔵", Improving: "🟢", Weakening: "🟠", Lagging: "🔴" };
const SECTOR_LABEL = {
  "Information Technology": "Tech", "Health Care": "Santé", "Financials": "Finance",
  "Industrials": "Industrie", "Consumer Discretionary": "Conso cyc.", "Consumer Staples": "Conso déf.",
  "Energy": "Énergie", "Utilities": "Utilities", "Materials": "Matériaux",
  "Real Estate": "Immo.", "Communication Services": "Comm.",
};

// Lien graphe TradingView. yfinance utilise 'BRK-B' ; TradingView attend 'BRK.B'.
const tvUrl = (t) => `https://www.tradingview.com/chart/?symbol=${encodeURIComponent(String(t).trim().replace(/-/g, "."))}`;

function fmtMcap(n) {
  if (!Number.isFinite(n) || n <= 0) return "—";
  if (n >= 1e12) return "$" + (n / 1e12).toFixed(2) + "T";
  if (n >= 1e9) return "$" + (n / 1e9).toFixed(1) + "B";
  if (n >= 1e6) return "$" + (n / 1e6).toFixed(0) + "M";
  return "$" + n.toFixed(0);
}

function tierBadge(n) {
  if (!Number.isFinite(n) || n <= 0) return null;
  if (n < 3e8) return { t: "nano", c: "bg-fuchsia-500/15 text-fuchsia-600 dark:text-fuchsia-300" };
  if (n < 2e9) return { t: "micro", c: "bg-violet-500/15 text-violet-600 dark:text-violet-300" };
  if (n < 1e10) return { t: "small", c: "bg-sky-500/15 text-sky-600 dark:text-sky-300" };
  return { t: "large", c: "bg-slate-500/15 text-slate-500 dark:text-slate-400" };
}

function scoreCls(s) {
  const b = "inline-flex items-center justify-center min-w-[2.4rem] px-2 py-0.5 rounded-md text-xs font-bold num ring-1 ring-inset";
  if (s >= 70) return b + " bg-emerald-500/15 text-emerald-600 dark:text-emerald-300 ring-emerald-500/30";
  if (s >= 60) return b + " bg-lime-500/15 text-lime-600 dark:text-lime-300 ring-lime-500/30";
  if (s >= 50) return b + " bg-amber-500/15 text-amber-600 dark:text-amber-300 ring-amber-500/30";
  return b + " bg-slate-500/10 text-slate-500 dark:text-slate-400 ring-slate-500/20";
}

function rsiCls(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "text-slate-400";
  if (n >= 70) return "text-amber-600 dark:text-amber-400";
  if (n < 40) return "text-slate-400";
  return "text-slate-600 dark:text-slate-300";
}

function hasCatalyst(r) {
  const c = (r.catalyst ?? "").toString().trim();
  return c && c.toLowerCase() !== "nan";
}
function CatalystBadge({ r }) {
  if (!hasCatalyst(r)) return <span className="text-slate-300 dark:text-slate-600">—</span>;
  const c = r.catalyst.toString().trim();
  const days = Number(r.catalyst_days);
  const recent = Number.isFinite(days) && days <= 7;
  const cls = recent
    ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-300 ring-emerald-500/30"
    : "bg-slate-500/10 text-slate-600 dark:text-slate-300 ring-white/10";
  return (
    <span className={`inline-flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 rounded ring-1 ring-inset ${cls}`} title="Dépôt SEC 8-K récent">
      {recent ? "🔥 " : ""}{c}{Number.isFinite(days) ? ` · ${days}j` : ""}
    </span>
  );
}
function BuzzBadge({ r }) {
  const b = toNumber(r.buzz);
  if (!Number.isFinite(b) || b <= 0) return null;
  return (
    <span className="text-[10px] px-1.5 py-0.5 rounded bg-sky-500/15 text-sky-600 dark:text-sky-300 ring-1 ring-inset ring-sky-500/25"
          title="Articles de presse récents (Finnhub, 7j)">
      📰 {b}
    </span>
  );
}

const inSize = (mc, size) => {
  if (size === "all") return true;
  const n = Number(mc);
  if (!Number.isFinite(n) || n <= 0) return false;
  if (size === "nanomicro") return n < 2e9;
  if (size === "small") return n >= 2e9 && n < 1e10;
  if (size === "large") return n >= 1e10;
  return true;
};

function SortHeader({ label, k, sort, onSort, align = "left" }) {
  const active = sort.key === k;
  return (
    <th className={`th ${align === "right" ? "text-right" : ""}`}>
      <button
        onClick={() => onSort(k)}
        className={`inline-flex items-center gap-1 ${align === "right" ? "flex-row-reverse" : ""} hover:text-slate-700 dark:hover:text-slate-200 ${active ? "text-slate-700 dark:text-slate-200" : ""}`}
      >
        {label}
        {active ? (sort.dir === "desc" ? <ChevronDown size={13} /> : <ChevronUp size={13} />) : <ArrowUpDown size={12} className="opacity-40" />}
      </button>
    </th>
  );
}

export default function ThematicSetups({ rows = [], loading = false }) {
  const [theme, setTheme] = useState("all");
  const [size, setSize] = useState("all");
  const [minScore, setMinScore] = useState(0);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState({ key: "score", dir: "desc" });
  const [showInfo, setShowInfo] = useState(false);
  const [onlyCatalyst, setOnlyCatalyst] = useState(false);
  const [sector, setSector] = useState("all");
  const [hotOnly, setHotOnly] = useState(false);
  const [sectorQuad, setSectorQuad] = useState({});

  useEffect(() => {
    let a = true;
    fetchCSVFresh("sector_perf.csv").then((rr) => {
      if (!a) return;
      const m = {};
      (Array.isArray(rr) ? rr : []).forEach((x) => { if (x.sector) m[String(x.sector).trim()] = x.quadrant; });
      setSectorQuad(m);
    });
    return () => { a = false; };
  }, []);

  const onSort = (k) => setSort((s) => (s.key === k ? { key: k, dir: s.dir === "desc" ? "asc" : "desc" } : { key: k, dir: "desc" }));

  const themes = useMemo(() => {
    const s = new Set();
    rows.forEach((r) => (r.themes || "").split("|").forEach((t) => t && s.add(t)));
    return Array.from(s).sort();
  }, [rows]);

  const hasSectors = useMemo(() => rows.some((r) => (r.sector || "").toString().trim()), [rows]);
  const sectors = useMemo(() => {
    const s = new Set();
    rows.forEach((r) => { const x = (r.sector || "").toString().trim(); if (x) s.add(x); });
    return Array.from(s).sort();
  }, [rows]);

  // Si l'univers courant n'a pas de secteur (vue thématique), on purge le filtre secteur pour ne pas vider la table.
  useEffect(() => { if (!hasSectors) { setSector("all"); setHotOnly(false); } }, [hasSectors]);

  const view = useMemo(() => {
    let v = rows.map((r) => ({ ...r, _s: toNumber(r.score) ?? 0, _mc: toNumber(r.mcap_usd) }));
    if (theme !== "all") v = v.filter((r) => (r.themes || "").split("|").includes(theme));
    if (hasSectors && sector !== "all") v = v.filter((r) => (r.sector || "").toString().trim() === sector);
    if (hasSectors && hotOnly) v = v.filter((r) => { const qd = sectorQuad[(r.sector || "").toString().trim()]; return qd === "Leading" || qd === "Improving"; });
    v = v.filter((r) => inSize(r._mc, size));
    if (minScore > 0) v = v.filter((r) => r._s >= minScore);
    const q = query.trim().toUpperCase();
    if (q) v = v.filter((r) => String(r.ticker).toUpperCase().includes(q));
    if (onlyCatalyst) v = v.filter((r) => hasCatalyst(r));
    const { key, dir } = sort;
    const mul = dir === "desc" ? -1 : 1;
    v.sort((a, b) => {
      if (key === "ticker") {
        const av = String(a.ticker), bv = String(b.ticker);
        return av < bv ? -1 * mul : av > bv ? 1 * mul : 0;
      }
      const av = toNumber(a[key]) ?? -Infinity;
      const bv = toNumber(b[key]) ?? -Infinity;
      return (av - bv) * mul;
    });
    return v;
  }, [rows, hasSectors, theme, sector, hotOnly, sectorQuad, size, minScore, query, sort, onlyCatalyst]);

  const active = theme !== "all" || (hasSectors && (sector !== "all" || hotOnly)) || size !== "all" || minScore > 0 || query.trim() || onlyCatalyst;

  return (
    <div className="panel overflow-hidden">
      {/* En-tête */}
      <div className="p-4 sm:p-5 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3 border-b border-slate-200/70 dark:border-white/10">
        <div className="flex items-center gap-2">
          <span className="text-lg">🎯</span>
          <h2 className="text-base sm:text-lg font-bold">Setups pré-explosion</h2>
          <span className="text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-500 dark:bg-white/10 dark:text-slate-300 num">{view.length}</span>
          <button onClick={() => setShowInfo((v) => !v)} className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200" title="Comment lire le score">
            <Info size={15} />
          </button>
        </div>
        <div className="relative w-full lg:w-72">
          <Search size={15} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400" />
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Rechercher un ticker…" className="input-lux pl-8" />
        </div>
      </div>

      {showInfo && (
        <div className="px-4 sm:px-5 py-3 text-xs text-slate-500 dark:text-slate-400 bg-slate-50 dark:bg-white/5 border-b border-slate-200/70 dark:border-white/10 leading-relaxed">
          Le <b>/100</b> mesure la qualité de <i>timing</i> (tendance saine, base tendue près du pivot, contraction de volatilité,
          volume qui s'assèche, RSI/MACD). Backtesté : un score élevé ≈ <b>62% de winrate</b> et bat SPY à 20 j — mais ce n'est
          <b> pas</b> un prédicteur de la <i>taille</i> des mouvements. Le vrai moteur reste l'appartenance thématique.
          Outil de découverte, <b>pas</b> un signal d'achat automatique.
        </div>
      )}

      {/* Filtres */}
      <div className="p-4 sm:p-5 space-y-3 border-b border-slate-200/70 dark:border-white/10">
        <div className="flex flex-wrap gap-1.5">
          <button className={`chip ${theme === "all" ? "chip-on" : ""}`} onClick={() => setTheme("all")}>Tous les thèmes</button>
          {themes.map((t) => (
            <button key={t} className={`chip ${theme === t ? "chip-on" : ""}`} onClick={() => setTheme(theme === t ? "all" : t)}>
              <span className={`h-2 w-2 rounded-full ${THEME_META[t]?.dot || "bg-slate-400"}`} />
              {THEME_META[t]?.label || t}
            </button>
          ))}
        </div>
        {hasSectors && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] uppercase tracking-wide text-slate-400 mr-0.5">Secteur</span>
            <button className={`chip ${sector === "all" ? "chip-on" : ""}`} onClick={() => setSector("all")}>Tous</button>
            {sectors.map((s) => (
              <button key={s} className={`chip ${sector === s ? "chip-on" : ""}`} onClick={() => setSector(sector === s ? "all" : s)} title={s}>
                {sectorQuad[s] ? QUAD_EMOJI[sectorQuad[s]] + " " : ""}{SECTOR_LABEL[s] || s}
              </button>
            ))}
            <button className={`chip ${hotOnly ? "chip-on" : ""}`} onClick={() => setHotOnly((v) => !v)} title="Secteurs Leading/Improving du panneau Rotation">
              🔥 En rotation
            </button>
          </div>
        )}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] uppercase tracking-wide text-slate-400 mr-0.5">Taille</span>
            {[["all", "Toutes"], ["nanomicro", "Nano+Micro ≤ $2B"], ["small", "Small $2–10B"], ["large", "Large > $10B"]].map(([k, l]) => (
              <button key={k} className={`chip ${size === k ? "chip-on" : ""}`} onClick={() => setSize(k)}>{l}</button>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] uppercase tracking-wide text-slate-400 mr-0.5">Score</span>
            {[[0, "Tous"], [50, "≥50"], [60, "≥60"], [70, "≥70"]].map(([k, l]) => (
              <button key={k} className={`chip ${minScore === k ? "chip-on" : ""}`} onClick={() => setMinScore(k)}>{l}</button>
            ))}
          </div>
          <button className={`chip ${onlyCatalyst ? "chip-on" : ""}`} onClick={() => setOnlyCatalyst((v) => !v)} title="Titres avec un dépôt SEC 8-K récent (< 30j)">
            🔥 Catalyseur
          </button>
          {active && (
            <button className="text-xs text-cyan-600 dark:text-cyan-400 hover:underline ml-auto" onClick={() => { setTheme("all"); setSector("all"); setHotOnly(false); setSize("all"); setMinScore(0); setQuery(""); setOnlyCatalyst(false); }}>
              Réinitialiser
            </button>
          )}
        </div>
      </div>

      {/* Tableau */}
      {loading ? (
        <div className="p-12 text-center text-sm text-slate-500">Chargement des setups…</div>
      ) : view.length === 0 ? (
        <div className="p-12 text-center text-sm text-slate-500">Aucun setup ne correspond aux filtres.</div>
      ) : (
        <div className="overflow-x-auto max-h-[72vh] thin-scroll">
          <table className="w-full text-sm">
            <thead className="sticky top-0 z-10 bg-white/90 dark:bg-slate-950/70 backdrop-blur">
              <tr className="border-b border-slate-200 dark:border-white/10">
                <th className="th w-10">#</th>
                <SortHeader label="Ticker" k="ticker" sort={sort} onSort={onSort} />
                <th className="th">Thèmes</th>
                {hasSectors && <th className="th">Secteur</th>}
                <SortHeader label="Score" k="score" sort={sort} onSort={onSort} />
                <th className="th">Setup</th>
                <th className="th">Catalyseur</th>
                <SortHeader label="MCap" k="mcap_usd" sort={sort} onSort={onSort} align="right" />
                <SortHeader label="Prix" k="price" sort={sort} onSort={onSort} align="right" />
                <SortHeader label="RSI" k="rsi" sort={sort} onSort={onSort} align="right" />
                <SortHeader label="Dist. haut" k="dist_to_high_pct" sort={sort} onSort={onSort} align="right" />
              </tr>
            </thead>
            <tbody>
              {view.map((r, i) => {
                const tb = tierBadge(r._mc);
                const tl = (r.themes || "").split("|").filter(Boolean);
                return (
                  <tr key={r.ticker + "_" + i} className="border-b border-slate-100 dark:border-white/5 hover:bg-slate-50 dark:hover:bg-white/5 transition-colors">
                    <td className="px-3 py-2 text-slate-400 num">{i + 1}</td>
                    <td className="px-3 py-2">
                      <div className="flex items-center gap-2">
                        <a
                          href={tvUrl(r.ticker)}
                          target="_blank"
                          rel="noreferrer"
                          className="font-semibold tracking-tight hover:text-cyan-600 dark:hover:text-cyan-300 hover:underline decoration-dotted underline-offset-2"
                          title={`Ouvrir ${r.ticker} sur TradingView`}
                        >
                          {r.ticker}
                        </a>
                        {tb && <span className={`text-[10px] px-1.5 py-0.5 rounded ${tb.c}`}>{tb.t}</span>}
                      </div>
                    </td>
                    <td className="px-3 py-2">
                      <div className="flex flex-wrap gap-1">
                        {tl.slice(0, 3).map((t) => (
                          <span key={t} className={`text-[10px] px-1.5 py-0.5 rounded-full ring-1 ring-inset ${THEME_META[t]?.chip || "bg-slate-500/10 text-slate-400 ring-white/10"}`}>
                            {THEME_META[t]?.label || t}
                          </span>
                        ))}
                        {tl.length > 3 && <span className="text-[10px] text-slate-400 self-center">+{tl.length - 3}</span>}
                      </div>
                    </td>
                    {hasSectors && (
                      <td className="px-3 py-2 whitespace-nowrap text-xs text-slate-500 dark:text-slate-400">
                        {(r.sector || "").toString().trim() ? (
                          <span title={r.sector}>
                            {sectorQuad[(r.sector || "").toString().trim()] ? QUAD_EMOJI[sectorQuad[(r.sector || "").toString().trim()]] + " " : ""}
                            {SECTOR_LABEL[r.sector] || r.sector}
                          </span>
                        ) : (
                          <span className="text-slate-300 dark:text-slate-600">—</span>
                        )}
                      </td>
                    )}
                    <td className="px-3 py-2"><span className={scoreCls(r._s)}>{r._s.toFixed(0)}</span></td>
                    <td className="px-3 py-2 text-slate-600 dark:text-slate-300 whitespace-nowrap">{r.setup}</td>
                    <td className="px-3 py-2 whitespace-nowrap"><div className="flex items-center gap-1"><CatalystBadge r={r} /><BuzzBadge r={r} /></div></td>
                    <td className="px-3 py-2 text-right num font-medium">{fmtMcap(r._mc)}</td>
                    <td className="px-3 py-2 text-right num text-slate-600 dark:text-slate-300">{r.price}</td>
                    <td className={`px-3 py-2 text-right num ${rsiCls(r.rsi)}`}>{r.rsi}</td>
                    <td className="px-3 py-2 text-right num text-slate-500">{r.dist_to_high_pct}%</td>
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
