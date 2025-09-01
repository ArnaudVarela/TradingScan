// dashboard/src/components/DataTable.jsx
import React, { useMemo, useState } from "react";

/* ---------- utils numériques & formats ---------- */
const toNum = (v) => {
  const n = Number(String(v ?? "").replaceAll(",", "").trim());
  return Number.isFinite(n) ? n : null;
};

const formatPrice = (val) => {
  const n = toNum(val);
  return n === null ? "—" : `$${n.toFixed(2)}`;
};

const formatMCap = (val) => {
  const n = toNum(val);
  if (n === null) return "—";
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T$`;
  if (n >= 1e9)  return `${(n / 1e9).toFixed(1)}B$`;
  if (n >= 1e6)  return `${(n / 1e6).toFixed(1)}M$`;
  return n.toLocaleString();
};

/* ---------- couleurs badges ---------- */
const recoBadge = (label) => {
  const v = String(label ?? "").toUpperCase();
  if (v === "STRONG BUY" || v === "STRONG_BUY")  return "bg-green-600 text-white";
  if (v === "BUY")                                return "bg-green-300 text-black dark:bg-green-500/60 dark:text-white";
  if (v === "HOLD")                               return "bg-gray-300 text-black dark:bg-gray-600/60 dark:text-white";
  if (v === "SELL")                               return "bg-orange-400 text-black dark:bg-orange-500/70 dark:text-white";
  if (v === "STRONG SELL" || v === "STRONG_SELL") return "bg-red-600 text-white";
  return "bg-gray-200 text-black dark:bg-gray-700/60 dark:text-white";
};

const votesClass = (v) => {
  const n = toNum(v);
  if (n === null) return "";
  if (n >= 15) return "text-green-600 dark:text-green-400";
  if (n >= 5)  return "text-amber-600 dark:text-amber-400";
  return "text-red-600 dark:text-red-400";
};

const mcapPill = (v) => {
  const n = toNum(v);
  if (n === null) return "bg-gray-200 dark:bg-gray-700";
  if (n < 3e9)  return "bg-emerald-200 dark:bg-emerald-700";
  if (n < 20e9) return "bg-sky-200 dark:bg-sky-700";
  return "bg-purple-200 dark:bg-purple-700";
};

/* ---------- helpers data ---------- */
function pick(row, key, alts = []) {
  if (!row) return undefined;
  if (row[key] !== undefined && row[key] !== null) return row[key];
  for (const k of alts) {
    if (row[k] !== undefined && row[k] !== null) return row[k];
  }
  return undefined;
}

/* ---------- “pillars” (Tech / TV / Analyst) ---------- */
const isStrongTech   = (v) => String(v ?? "").toUpperCase().includes("STRONG") || String(v ?? "").toUpperCase() === "STRONG BUY";
const isStrongTV     = (v) => String(v ?? "").toUpperCase().includes("STRONG_BUY");
const isBullAnalyst  = (v) => {
  const x = String(v ?? "").toUpperCase();
  return x === "BUY" || x === "STRONG BUY" || x === "OUTPERFORM" || x === "OVERWEIGHT";
};

function computePillars(row) {
  const t = pick(row, "technical_local");
  const tv = pick(row, "tv_reco");
  const an = pick(row, "analyst_bucket");
  let ok = 0;
  if (isStrongTech(t)) ok += 1;
  if (isStrongTV(tv)) ok += 1;
  if (isBullAnalyst(an)) ok += 1;
  return ok; // 0..3
}

function pillarsBadgeClass(n) {
  if (n >= 3) return "bg-emerald-600 text-white";
  if (n === 2) return "bg-amber-500 text-black dark:text-white";
  return "bg-slate-400 text-white";
}

/* ---------- colonnes ---------- */
const columns = [
  { label: "Ticker",   key: "ticker_tv", alts: ["ticker_yf","ticker","symbol","Symbol","Ticker"] },
  { label: "Pillars",  key: "__pillars_num" }, // virtuel pour tri
  { label: "Price",    key: "price" },
  { label: "MCap",     key: "market_cap", alts: ["mcap","MarketCap"] },
  { label: "Tech",     key: "technical_local" },
  { label: "TV",       key: "tv_reco" },
  { label: "Analyst",  key: "analyst_bucket" },
  { label: "Votes",    key: "analyst_votes" },
  { label: "Sector",   key: "sector" },
  { label: "Industry", key: "industry" },
];

export default function DataTable({ rows }) {
  const safeRows = Array.isArray(rows) ? rows : [];
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState({ label: null, dir: "asc" });

  /* enrichi avec champs calculés (pour tri “Pillars”) */
  const enriched = useMemo(() => {
    return safeRows.map((r) => {
      const p = computePillars(r);
      return { ...r, __pillars_num: p };
    });
  }, [safeRows]);

  /* filtre texte */
  const filtered = useMemo(() => {
    if (!query) return enriched;
    const q = query.toLowerCase();
    return enriched.filter((r) => {
      try {
        const fields = [
          pick(r,"ticker_tv",["ticker_yf","ticker","symbol"]) ?? "",
          r.sector ?? "",
          r.industry ?? "",
          r.technical_local ?? "",
          r.tv_reco ?? "",
          r.analyst_bucket ?? "",
        ].join(" ").toLowerCase();
        return fields.includes(q);
      } catch {
        return false;
      }
    });
  }, [enriched, query]);

  /* tri */
  const sorted = useMemo(() => {
    if (!sort.label) return filtered;
    const col = columns.find((c) => c.label === sort.label);
    const getter = (r) => (col ? pick(r, col.key, col.alts) : undefined);
    return [...filtered].sort((a, b) => {
      try {
        // “Pillars” -> tri numérique sur __pillars_num
        if (col?.key === "__pillars_num") {
          const A = toNum(getter(a));
          const B = toNum(getter(b));
          if (A === null || B === null) return 0;
          return sort.dir === "asc" ? A - B : B - A;
        }
        // sinon: numérique si possible, sinon string
        const A = getter(a), B = getter(b);
        const nA = toNum(A), nB = toNum(B);
        if (nA !== null && nB !== null) return sort.dir === "asc" ? nA - nB : nB - nA;
        const sA = (A ?? "").toString(), sB = (B ?? "").toString();
        return sort.dir === "asc" ? sA.localeCompare(sB) : sB.localeCompare(sA);
      } catch {
        return 0;
      }
    });
  }, [filtered, sort]);

  const onSort = (label) =>
    setSort((prev) =>
      prev.label === label ? { label, dir: prev.dir === "asc" ? "desc" : "asc" } : { label, dir: "asc" }
    );

  return (
    <div className="bg-white dark:bg-slate-800 shadow rounded p-4 mb-6">
      {/* Search */}
      <div className="mb-3">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search ticker / sector / industry…"
          className="w-full md:w-80 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 px-3 py-2 text-sm"
        />
      </div>

      <div className="overflow-auto rounded">
        <table className="min-w-full table-fixed text-sm">
          <thead className="sticky top-0 z-10 bg-gray-100 dark:bg-slate-700">
            <tr>
              {columns.map((c, idx) => (
                <th
                  key={c.label}
                  onClick={() => onSort(c.label)}
                  className={`
                    p-2 text-left font-semibold cursor-pointer select-none
                    ${idx === 0 ? "w-[10%]" : ""}
                    ${idx === 1 ? "w-[8%]"  : ""}
                    ${idx === 2 ? "w-[10%]" : ""}
                    ${idx === 3 ? "w-[10%]" : ""}
                    ${idx === 4 ? "w-[10%]" : ""}
                    ${idx === 5 ? "w-[10%]" : ""}
                    ${idx === 6 ? "w-[10%]" : ""}
                    ${idx === 7 ? "w-[8%]"  : ""}
                    ${idx === 8 ? "w-[14%]" : ""}
                    ${idx === 9 ? "w-[20%]" : ""}
                  `}
                >
                  {c.label}
                  {sort.label === c.label ? (sort.dir === "asc" ? " ▲" : " ▼") : ""}
                </th>
              ))}
            </tr>
          </thead>

          <tbody>
            {sorted.map((r, i) => {
              try {
                const ticker = pick(r, "ticker_tv", ["ticker_yf", "ticker", "symbol"]);
                const price  = r.price;
                const mcap   = pick(r, "market_cap", ["mcap","MarketCap"]);
                const tech   = r.technical_local;
                const tv     = r.tv_reco;
                const ab     = r.analyst_bucket;
                const votes  = r.analyst_votes;
                const sector = r.sector ?? "Unknown";
                const industry = r.industry ?? "Unknown";
                const pillars = r.__pillars_num ?? 0;

                const tvUrl = ticker ? `https://www.tradingview.com/symbols/${String(ticker).toUpperCase()}/` : undefined;
                const yfUrl = ticker ? `https://finance.yahoo.com/quote/${String(ticker).toUpperCase().replace(".","-")}` : undefined;

                return (
                  <tr key={i} className="border-t border-slate-200 dark:border-slate-700">
                    {/* Ticker + liens */}
                    <td className="p-2 font-mono">
                      {ticker ? (
                        <>
                          <a className="underline mr-2" href={tvUrl} target="_blank" rel="noreferrer">{ticker}</a>
                          <a className="text-slate-500 underline" href={yfUrl} target="_blank" rel="noreferrer">YF</a>
                        </>
                      ) : "—"}
                    </td>

                    {/* Pillars badge */}
                    <td className="p-2">
                      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs ${pillarsBadgeClass(pillars)}`}>
                        {pillars}/3
                      </span>
                    </td>

                    {/* Price */}
                    <td className="p-2">{formatPrice(price)}</td>

                    {/* MCap */}
                    <td className="p-2">
                      <span className={`px-2 py-0.5 rounded ${mcapPill(mcap)}`}>{formatMCap(mcap)}</span>
                    </td>

                    {/* Reco badges */}
                    <td className="p-2">
                      <span className={`px-2 py-0.5 rounded ${recoBadge(tech)}`}>{tech || "—"}</span>
                    </td>
                    <td className="p-2">
                      <span className={`px-2 py-0.5 rounded ${recoBadge(tv)}`}>{tv || "—"}</span>
                    </td>
                    <td className="p-2">
                      <span className={`px-2 py-0.5 rounded ${recoBadge(ab)}`}>{ab || "—"}</span>
                    </td>

                    {/* Votes */}
                    <td className={`p-2 ${votesClass(votes)}`}>
                      {toNum(votes) !== null ? `${toNum(votes)} votes` : "—"}
                    </td>

                    {/* Sector / Industry */}
                    <td className="p-2">{sector}</td>
                    <td className="p-2">{industry}</td>
                  </tr>
                );
              } catch {
                return null; // ligne corrompue => on skip sans crasher
              }
            })}

            {sorted.length === 0 && (
              <tr>
                <td colSpan={columns.length} className="p-4 text-slate-500">No rows.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
