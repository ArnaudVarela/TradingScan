import { useMemo, useState } from "react";

const currency = v => (v == null ? "-" : Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(v));

export default function DataTable({ rows }) {
  const [q, setQ] = useState("");
  const [onlyTV, setOnlyTV] = useState(false);
  const [onlyAnalyst, setOnlyAnalyst] = useState(false);

  const filtered = useMemo(() => {
    let r = rows || [];
    if (q) {
      const qq = q.toLowerCase();
      r = r.filter(x =>
        (x.ticker_yf || "").toLowerCase().includes(qq) ||
        (x.ticker_tv || "").toLowerCase().includes(qq) ||
        (x.sector || "").toLowerCase().includes(qq) ||
        (x.industry || "").toLowerCase().includes(qq)
      );
    }
    if (onlyTV) r = r.filter(x => x.tv_reco === "STRONG_BUY");
    if (onlyAnalyst) r = r.filter(x => ["Strong Buy", "Buy"].includes(x.analyst_bucket));
    return r;
  }, [rows, q, onlyTV, onlyAnalyst]);

  return (
    <div className="card">
      <div className="flex flex-wrap gap-3 items-center mb-4">
        <input
          className="border rounded-xl px-3 py-2"
          placeholder="Search ticker/sector/industry"
          value={q}
          onChange={e => setQ(e.target.value)}
        />
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={onlyTV} onChange={e => setOnlyTV(e.target.checked)} />
          TV = STRONG_BUY
        </label>
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={onlyAnalyst} onChange={e => setOnlyAnalyst(e.target.checked)} />
          Analyst ∈ {`{Strong Buy, Buy}`}
        </label>
        <div className="ml-auto sub">{filtered.length} rows</div>
      </div>
      <div className="overflow-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="text-left border-b">
              <th className="py-2 pr-4">Ticker</th>
              <th className="py-2 pr-4">Price</th>
              <th className="py-2 pr-4">MCap</th>
              <th className="py-2 pr-4">Tech</th>
              <th className="py-2 pr-4">TV</th>
              <th className="py-2 pr-4">Analyst</th>
              <th className="py-2 pr-4">Votes</th>
              <th className="py-2 pr-4">Sector</th>
              <th className="py-2 pr-4">Industry</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((x, i) => (
              <tr key={i} className="border-b hover:bg-slate-50">
                <td className="py-2 pr-4 font-medium">{x.ticker_yf || x.ticker_tv}</td>
                <td className="py-2 pr-4">{x.price ?? "-"}</td>
                <td className="py-2 pr-4">{currency(x.market_cap)}</td>
                <td className="py-2 pr-4">{x.technical_local}</td>
                <td className="py-2 pr-4">{x.tv_reco}</td>
                <td className="py-2 pr-4">{x.analyst_bucket}</td>
                <td className="py-2 pr-4">{x.analyst_votes ?? "-"}</td>
                <td className="py-2 pr-4">{x.sector ?? "-"}</td>
                <td className="py-2 pr-4">{x.industry ?? "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
