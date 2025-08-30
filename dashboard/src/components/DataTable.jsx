import React, { useMemo, useState } from "react";

const formatPrice = (val) => (val || val === 0 ? `$${Number(val).toFixed(2)}` : "—");
const formatMCap = (val) => {
  const n = Number(val);
  if (!Number.isFinite(n)) return "—";
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T$`;
  if (n >= 1e9)  return `${(n / 1e9).toFixed(1)}B$`;
  if (n >= 1e6)  return `${(n / 1e6).toFixed(1)}M$`;
  return n.toLocaleString();
};

const badge = (label) => {
  const v = (label || "").toUpperCase();
  if (v === "STRONG BUY" || v === "STRONG_BUY")  return "bg-green-600 text-white";
  if (v === "BUY")                                return "bg-green-300 text-black dark:bg-green-500/60 dark:text-white";
  if (v === "HOLD")                               return "bg-gray-300 text-black dark:bg-gray-600/60 dark:text-white";
  if (v === "SELL")                               return "bg-orange-400 text-black dark:bg-orange-500/70 dark:text-white";
  if (v === "STRONG SELL" || v === "STRONG_SELL") return "bg-red-600 text-white";
  return "bg-gray-200 text-black dark:bg-gray-700/60 dark:text-white";
};

const columns = [
  { label: "Ticker",   key: "ticker_tv", alts: ["ticker_yf","ticker","symbol","Symbol","Ticker"] },
  { label: "Price",    key: "price" },
  { label: "MCap",     key: "market_cap", alts: ["mcap","MarketCap"] },
  { label: "Tech",     key: "technical_local" },
  { label: "TV",       key: "tv_reco" },
  { label: "Analyst",  key: "analyst_bucket" },
  { label: "Votes",    key: "analyst_votes" },
  { label: "Sector",   key: "sector" },
  { label: "Industry", key: "industry" },
];

function pick(row, key, alts = []) {
  if (!row) return undefined;
  if (row[key] != null) return row[key];
  for (const k of alts) if (row[k] != null) return row[k];
  return undefined;
}

const votesClass = (v) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return "";
  if (n >= 15) return "text-green-600 dark:text-green-400";
  if (n >= 5)  return "text-amber-600 dark:text-amber-400";
  return "text-red-600 dark:text-red-400";
};

const mcapPill = (v) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return "bg-gray-200 dark:bg-gray-700";
  if (n < 3e9)  return "bg-emerald-200 dark:bg-emerald-700"; // small
  if (n < 20e9) return "bg-sky-200 dark:bg-sky-700";         // mid
  return "bg-purple-200 dark:bg-purple-700";                 // large
};

export default function DataTable({ rows }) {
  const safeRows = Array.isArray(rows) ? rows : [];
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState({ label: null, dir: "asc" });

  const filtered = useMemo(() => {
    if (!query) return safeRows;
    const q = query.toLowerCase();
    return safeRows.filter((r) => {
      const fields = [
        pick(r,"ticker_tv",["ticker_yf","ticker","symbol"]) ?? "",
        r.sector ?? "",
        r.industry ?? "",
        r.technical_local ?? "",
        r.tv_reco ?? "",
        r.analyst_bucket ?? "",
      ]
        .join(" ")
        .toLowerCase();
      return fields.includes(q);
    });
  }, [safeRows, query]);

  const sorted = useMemo(() => {
    if (!sort.label) return filtered;
    const col = columns.find((c) => c.label === sort.label);
    const getVal = (r) => (col ? pick(r, col.key, col.alts) : undefined);
    return [...filtered].sort((a, b) => {
      const A = getVal(a), B = getVal(b);
      const nA = Number(A), nB = Number(B);
      const numeric = Number.isFinite(nA) && Number.isFinite(nB);
      if (numeric) return sort.dir === "asc" ? nA - nB : nB - nA;
      const sA = (A ?? "").toString(), sB = (B ?? "").toString();
      return sort.dir === "asc" ? sA.localeCompare(sB) : sB.localeCompare(sA);
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
        <table className="min-w-full text-sm">
          <thead className="sticky top-0 z-10 bg-gray-100 dark:bg-slate-700">
            <tr>
              {columns.map((c) => (
                <th
                  key={c.label}
                  onClick={() => onSort(c.label)}
                  className="p-2 text-left font-semibold cursor-pointer select-none"
                >
                  {c.label}
                  {sort.label === c.label ? (sort.dir === "asc" ? " ▲" : " ▼") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((r, i) => {
              const ticker = pick(r, "ticker_tv", ["ticker_yf", "ticker", "symbol"]);
              const price = r.price;
              const mcap = pick(r, "market_cap", ["mcap","MarketCap"]);
              const tech  = r.technical_local;
              const tv    = r.tv_reco;
              const ab    = r.analyst_bucket;
              const votes = r.analyst_votes;
              const sector = r.sector || "Unknown";
              const industry = r.industry || "Unknown";

              const tvUrl = ticker ? `https://www.tradingview.com/symbols/${String(ticker).toUpperCase()}/` : undefined;
              const yfUrl = ticker ? `https://finance.yahoo.com/quote/${String(ticker).toUpperCase().replace(".","-")}` : undefined;

              return (
                <tr key={i} className="border-t border-slate-200 dark:border-slate-700">
                  <td className="p-2 font-mono">
                    {ticker ? (
                      <>
                        <a className="underline mr-2" href={tvUrl} target="_blank" rel="noreferrer">
                          {ticker}
                        </a>
                        <a className="text-slate-500 underline" href={yfUrl} target="_blank" rel="noreferrer">
                          YF
                        </a>
                      </>
                    ) : "—"}
                  </td>
                  <td className="p-2">{formatPrice(price)}</td>
                  <td className="p-2">
                    <span className={`px-2 py-0.5 rounded ${mcapPill(mcap)}`}>{formatMCap(mcap)}</span>
                  </td>
                  <td className="p-2">
                    <span className={`px-2 py-0.5 rounded ${badge(tech)}`}>{tech || "—"}</span>
                  </td>
                  <td className="p-2">
                    <span className={`px-2 py-0.5 rounded ${badge(tv)}`}>{tv || "—"}</span>
                  </td>
                  <td className="p-2">
                    <span className={`px-2 py-0.5 rounded ${badge(ab)}`}>{ab || "—"}</span>
                  </td>
                  <td className={`p-2 ${votesClass(votes)}`}>{Number.isFinite(Number(votes)) ? `${votes} votes` : "—"}</td>
                  <td className="p-2">{sector}</td>
                  <td className="p-2">{industry}</td>
                </tr>
              );
            })}
            {sorted.length === 0 && (
              <tr>
                <td colSpan={columns.length} className="p-4 text-slate-500">
                  No rows.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
