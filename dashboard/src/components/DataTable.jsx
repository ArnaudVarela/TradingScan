import { useMemo, useState } from "react";

// Affiche une table simple avec recherche et tri.
// Ajoute une colonne "Pillars" (+ badge x/3 près du ticker).
export default function DataTable({ rows = [] }) {
  // ---------- helpers: 3 pillars ----------
  const isStrong = (v) => {
    if (!v) return false;
    const s = String(v).trim().toUpperCase();
    return s === "STRONG BUY" || s === "STRONGBUY" || s === "STRONG_BUY";
  };
  const countPillars = (r) => {
    const pTech = isStrong(r?.technical_local);
    const pTv   = isStrong(r?.tv_reco);
    const pAn   = String(r?.analyst_bucket || "").trim().toUpperCase() === "BUY";
    return (pTech ? 1 : 0) + (pTv ? 1 : 0) + (pAn ? 1 : 0);
  };
  const PillarsBadge = ({ n = 0 }) => {
    const map = {
      3: "bg-emerald-600 text-white",
      2: "bg-amber-500 text-black",
      1: "bg-slate-500 text-white",
      0: "bg-slate-700 text-white",
    };
    return (
      <span
        className={`ml-2 px-1.5 py-0.5 rounded text-[10px] ${map[n] || map[0]}`}
        title="Piliers Tech / TV / Analyst atteints"
      >
        {n}/3
      </span>
    );
  };

  // ---------- local UI state ----------
  const [query, setQuery]   = useState("");
  const [sortKey, setSortKey] = useState("_pillars");
  const [sortDir, setSortDir] = useState("desc"); // "asc" | "desc"

  // ---------- derived rows (enrichis + filtrés + triés) ----------
  const enriched = useMemo(() => {
    const arr = Array.isArray(rows) ? rows : [];
    return arr.map(r => ({
      ...r,
      _pillars: countPillars(r),
      _ticker:  r?.ticker_tv || r?.ticker || "",
      _sector:  r?.sector || "",
      _industry:r?.industry || "",
    }));
  }, [rows]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return enriched;
    return enriched.filter(r =>
      String(r._ticker).toLowerCase().includes(q) ||
      String(r._sector).toLowerCase().includes(q) ||
      String(r._industry).toLowerCase().includes(q)
    );
  }, [enriched, query]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    const dir = sortDir === "asc" ? 1 : -1;
    arr.sort((a,b) => {
      const va = a?.[sortKey];
      const vb = b?.[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;

      // numérique vs texte
      const na = typeof va === "number" ? va : Number(va);
      const nb = typeof vb === "number" ? vb : Number(vb);
      if (!Number.isNaN(na) && !Number.isNaN(nb)) return (na - nb) * dir;

      return String(va).localeCompare(String(vb)) * dir;
    });
    return arr;
  }, [filtered, sortKey, sortDir]);

  const onSort = (key) => {
    if (sortKey === key) {
      setSortDir(prev => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const SortHeader = ({ label, k }) => (
    <button
      onClick={() => onSort(k)}
      className="px-3 py-2 text-left w-full select-none hover:underline"
      title="Cliquer pour trier"
    >
      {label}
      {sortKey === k ? (sortDir === "asc" ? " ▲" : " ▼") : ""}
    </button>
  );

  return (
    <div className="w-full">
      <div className="mb-2">
        <input
          className="w-full md:w-96 px-3 py-2 text-sm rounded border dark:border-slate-700 bg-white dark:bg-slate-900"
          placeholder="Search ticker / sector / industry…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>

      <div className="w-full overflow-x-auto">
        <table className="table-auto w-full text-sm">
          <thead>
            <tr className="text-left">
              <th className="px-3 py-2">Ticker</th>
              <th className="px-3 py-2">
                <SortHeader label="Pillars" k="_pillars" />
              </th>
              <th className="px-3 py-2">
                <SortHeader label="Price" k="price" />
              </th>
              <th className="px-3 py-2">
                <SortHeader label="MCap" k="market_cap" />
              </th>
              <th className="px-3 py-2">Tech</th>
              <th className="px-3 py-2">TV</th>
              <th className="px-3 py-2">Analyst</th>
              <th className="px-3 py-2">
                <SortHeader label="Votes" k="analyst_votes" />
              </th>
              <th className="px-3 py-2">Sector</th>
              <th className="px-3 py-2">Industry</th>
            </tr>
          </thead>

          <tbody>
            {sorted.map((r, i) => (
              <tr key={i} className="border-t border-slate-200 dark:border-slate-800">
                <td className="px-3 py-2 font-medium">
                  {r._ticker || "-"}
                  <PillarsBadge n={r._pillars ?? 0} />
                </td>
                <td className="px-3 py-2">{r._pillars ?? 0}</td>
                <td className="px-3 py-2">{r?.price ?? "-"}</td>
                <td className="px-3 py-2">{r?.market_cap ?? "-"}</td>
                <td className="px-3 py-2">{r?.technical_local ?? "-"}</td>
                <td className="px-3 py-2">{r?.tv_reco ?? "-"}</td>
                <td className="px-3 py-2">{r?.analyst_bucket ?? "-"}</td>
                <td className="px-3 py-2">{r?.analyst_votes ?? r?.votes ?? "-"}</td>
                <td className="px-3 py-2">{r?.sector ?? "-"}</td>
                <td className="px-3 py-2">{r?.industry ?? "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
