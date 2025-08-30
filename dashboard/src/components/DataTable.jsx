import React, { useState } from "react";

const formatPrice = (val) => (val ? `$${Number(val).toFixed(2)}` : "—");
const formatMCap = (val) => {
  if (!val) return "—";
  const n = Number(val);
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B$`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M$`;
  return n.toLocaleString();
};
const formatVotes = (val) => (val ? `${val} votes` : "—");

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

export default function DataTable({ data, title }) {
  const [sortConfig, setSortConfig] = useState({ key: null, dir: "asc" });

  const sortedData = React.useMemo(() => {
    if (!sortConfig.key) return data;
    return [...data].sort((a, b) => {
      const valA = a[sortConfig.key] ?? "";
      const valB = b[sortConfig.key] ?? "";
      if (typeof valA === "number" && typeof valB === "number") {
        return sortConfig.dir === "asc" ? valA - valB : valB - valA;
      }
      return sortConfig.dir === "asc"
        ? String(valA).localeCompare(String(valB))
        : String(valB).localeCompare(String(valA));
    });
  }, [data, sortConfig]);

  const handleSort = (key) => {
    let dir = "asc";
    if (sortConfig.key === key && sortConfig.dir === "asc") dir = "desc";
    setSortConfig({ key, dir });
  };

  return (
    <div className="bg-white shadow rounded p-4 mb-6">
      <h2 className="text-lg font-bold mb-2">{title}</h2>
      <table className="min-w-full text-sm">
        <thead>
          <tr className="bg-gray-100">
            {["Ticker", "Price", "MCap", "Tech", "TV", "Analyst", "Votes", "Sector", "Industry"].map((h) => (
              <th
                key={h}
                className="p-2 cursor-pointer text-left"
                onClick={() => handleSort(h.toLowerCase())}
              >
                {h}
                {sortConfig.key === h.toLowerCase() ? (sortConfig.dir === "asc" ? " ▲" : " ▼") : ""}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedData.map((row, i) => (
            <tr key={i} className="border-t">
              <td className="p-2 font-mono">{row.ticker_tv || row.ticker_yf}</td>
              <td className="p-2">{formatPrice(row.price)}</td>
              <td className="p-2">{formatMCap(row.market_cap)}</td>
              <td className="p-2">
                <span className={badgeClass(row.technical_local)}>{row.technical_local || "—"}</span>
              </td>
              <td className="p-2">
                <span className={badgeClass(row.tv_reco)}>{row.tv_reco || "—"}</span>
              </td>
              <td className="p-2">
                <span className={badgeClass(row.analyst_bucket)}>{row.analyst_bucket || "—"}</span>
              </td>
              <td className="p-2">{formatVotes(row.analyst_votes)}</td>
              <td className="p-2">{row.sector || "—"}</td>
              <td className="p-2">{row.industry || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
