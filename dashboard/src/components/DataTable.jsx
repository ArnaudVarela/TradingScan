// dashboard/src/components/DataTable.jsx
import React, { useMemo, useState } from "react";

const formatPrice = (val) => (val || val === 0 ? `$${Number(val).toFixed(2)}` : "—");
const formatMCap = (val) => {
  const n = Number(val);
  if (!Number.isFinite(n)) return "—";
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B$`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M$`;
  return n.toLocaleString();
};
const formatVotes = (val) => (val || val === 0 ? `${val} votes` : "—");

const badgeClass = (label) => {
  switch ((label || "").toUpperCase()) {
    case "STRONG BUY":
    case "STRONG_BUY":
      return "bg-green-600 text-white px-2 py-1 rounded";
    case "BUY":
      return "bg-green-300 text-black px-2 py-1 rounded";
    case "HOLD":
      return "bg-gray-300 text-black px-2 py-1 rounded";
    case "SELL":
      return "bg-orange-400 text-black px-2 py-1 rounded";
    case "STRONG SELL":
    case "STRONG_SELL":
      return "bg-red-600 text-white px-2 py-1 rounded";
    default:
      return "bg-gray-200 text-black px-2 py-1 rounded";
  }
};

// colonne -> clé principale + alternatives (tolérant aux CSV variés)
const COLUMNS = [
  { label: "Ticker", key: "ticker_tv", alts: ["ticker_yf", "ticker", "symbol", "Symbol", "Ticker"] },
  { label: "Price", key: "price" },
  { label: "MCap", key: "market_cap", alts: ["mcap", "MarketCap"] },
  { label: "Tech", key: "technical_local" },
  { label: "TV", key: "tv_reco" },
  { label: "Analyst", key: "analyst_bucket" },
  { label: "Votes", key: "analyst_votes" },
  { label: "Sector", key: "sector" },
  { label: "Industry", key: "industry" },
];

// util: récupère la valeur selon clé principale puis alternatives
function pick(row, key, alts = []) {
  if (row == null) return undefined;
  if (row[key] != null) return row[key];
  for (const k of alts) {
    if (row[k] != null) return row[k];
  }
  return undefined;
}

export default function DataTable({ rows, title }) {
  const safeRows = Array.isArray(rows) ? rows : [];

  const [sortConfig, setSortConfig] = useState({ key: null, dir: "asc" });

  const sortedData = useMemo(() => {
    if (!sortConfig.key) return safeRows;
    const col = COLUMNS.find(c => c.label === sortConfig.key);
    const getVal = (r) =>
      col
        ? pick(r, col.key, col.alts)
        : undefined;

    return [...safeRows].sort((a, b) => {
      const A = getVal(a);
      const B = getVal(b);

      const numA = Number(A);
      const numB = Number(B);
      const bothNumeric = Number.isFinite(numA) && Number.isFinite(numB);

      if (bothNumeric) {
        return sortConfig.dir === "asc" ? numA - numB : numB - numA;
      }
      const sa = (A ?? "").toString();
      const sb = (B ?? "").toString();
      return sortConfig.dir === "asc" ? sa.localeCompare(sb) : sb.localeCompare(sa);
    });
  }, [safeRows, sortConfig]);

  const handleSort = (label) => {
    setSortConfig((prev) => {
      if (prev.key === label) {
        return { key: label, dir: prev.dir === "asc" ? "desc" : "asc" };
      }
      return { key: label, dir: "asc" };
    });
  };

  return (
    <div className="bg-white shadow rounded p-4 mb-6">
      {title && <h2 className="text-lg font-bold mb-2">{title}</h2>}
      <table className="min-w-full text-sm">
        <thead>
          <tr className="bg-gray-100">
            {COLUMNS.map((c) => (
              <th
                key={c.label}
                className="p-2 cursor-pointer text-left"
                onClick={() => handleSort(c.label)}
                title={`Sort by ${c.label}`}
              >
                {c.label}
                {sortConfig.key === c.label ? (sortConfig.dir === "asc" ? " ▲" : " ▼") : ""}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedData.map((row, i) => {
            const ticker = pick(row, "ticker_tv", ["ticker_yf", "ticker", "symbol", "Symbol", "Ticker"]);
            const price = pick(row, "price");
            const mcap = pick(row, "market_cap", ["mcap", "MarketCap"]);
            const tech = pick(row, "technical_local");
            const tv = pick(row, "tv_reco");
            const analyst = pick(row, "analyst_bucket");
            const votes = pick(row, "analyst_votes");
            const sector = pick(row, "sector");
            const industry = pick(row, "industry");

            return (
              <tr key={i} className="border-t">
                <td className="p-2 font-mono">{ticker ?? "—"}</td>
                <td className="p-2">{formatPrice(price)}</td>
                <td className="p-2">{formatMCap(mcap)}</td>
                <td className="p-2">
                  <span className={badgeClass(tech)}>{tech || "—"}</span>
                </td>
                <td className="p-2">
                  <span className={badgeClass(tv)}>{tv || "—"}</span>
                </td>
                <td className="p-2">
                  <span className={badgeClass(analyst)}>{analyst || "—"}</span>
                </td>
                <td className="p-2">{formatVotes(votes)}</td>
                <td className="p-2">{sector || "Unknown"}</td>
                <td className="p-2">{industry || "Unknown"}</td>
              </tr>
            );
          })}
          {sortedData.length === 0 && (
            <tr>
              <td className="p-3 text-slate-500" colSpan={COLUMNS.length}>
                No rows.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
