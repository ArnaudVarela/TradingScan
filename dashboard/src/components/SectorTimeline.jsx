import React, { useMemo } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";

/**
 * Props:
 *  - history: array d’objets issus du CSV
 *  - selectedSectors: [] ou Set<string> (facultatif)
 *
 * CSV accepté (auto-détection):
 *  A) Wide: date, week, sector, confirmed, pre, events
 *  B) Long: date, week, sector, bucket, count (bucket ∈ {confirmed, pre, events})
 */

// --- utils
function toNumber(x) {
  if (x == null || x === "") return 0;
  const n = Number(x);
  return Number.isFinite(n) ? n : 0;
}

// Essaye d’uniformiser un label de bucket
function normBucket(b) {
  const s = String(b || "").trim().toLowerCase();
  if (!s) return null;
  if (s.includes("confirm")) return "confirmed";
  if (s === "pre" || s.includes("anticip")) return "pre";
  if (s.startsWith("event")) return "events";
  if (s === "events") return "events";
  return null;
}

// fabrique un label semaine YYYY-Www depuis date/ISO week ou string déjà au bon format.
// si "week" est déjà fourni: on l’utilise tel quel.
function weekLabel(row) {
  const wk = row.week || row.Week || row.WEEK;
  if (wk) return String(wk);

  const d = row.date || row.Date || row.DATE;
  if (d) {
    // accepte soit déjà "YYYY-W##", soit "YYYY-MM-DD"
    const ds = String(d);
    if (/^\d{4}-W\d{2}$/.test(ds)) return ds;
    // calcul ISO semaine simple
    const date = new Date(ds);
    if (!isNaN(date)) {
      // ISO week calc soft
      const tmp = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
      // jeudi de la semaine
      const dayNum = (tmp.getUTCDay() + 6) % 7;
      tmp.setUTCDate(tmp.getUTCDate() - dayNum + 3);
      const firstThursday = new Date(Date.UTC(tmp.getUTCFullYear(), 0, 4));
      const weekNo = 1 + Math.round(((tmp - firstThursday) / 86400000 - 3) / 7);
      const y = tmp.getUTCFullYear();
      return `${y}-W${String(weekNo).padStart(2, "0")}`;
    }
    // fallback, on renvoie ce qu’on a
    return ds;
  }
  return "Unknown";
}

// tri par ordre chrono sur "YYYY-Www" si possible
function weekSortKey(w) {
  const m = /^(\d{4})-W(\d{2})$/.exec(w);
  if (!m) return w;
  return `${m[1]}${m[2]}`; // ex "2025W35" => "202535" (string triable)
}

export default function SectorTimeline({ history, selectedSectors }) {
  const series = useMemo(() => {
    const rows = Array.isArray(history) ? history : [];
    if (rows.length === 0) return [];

    const cols = Object.keys(rows[0] || {});
    const hasLong = cols.some((c) => c.toLowerCase() === "bucket") && cols.some((c) => c.toLowerCase() === "count");
    const hasWide =
      cols.some((c) => c.toLowerCase() === "confirmed") ||
      cols.some((c) => c.toLowerCase() === "pre") ||
      cols.some((c) => c.toLowerCase() === "events");

    const sectorSet =
      selectedSectors instanceof Set
        ? selectedSectors
        : new Set((selectedSectors || []).map((s) => String(s || "").toLowerCase()));

    // Accumulateur: map week -> { week, confirmed, pre, events }
    const acc = new Map();

    const keepRowBySector = (row) => {
      if (sectorSet.size === 0) return true;
      const sec = String(row.sector || row.Sector || "Unknown").toLowerCase();
      return sectorSet.has(sec);
    };

    if (hasLong) {
      // format long
      for (const r of rows) {
        if (!keepRowBySector(r)) continue;
        const w = weekLabel(r);
        const bucket = normBucket(r.bucket || r.Bucket);
        if (!bucket) continue;
        const count = toNumber(r.count || r.Count);

        const cur = acc.get(w) || { week: w, confirmed: 0, pre: 0, events: 0 };
        cur[bucket] = (cur[bucket] || 0) + count;
        acc.set(w, cur);
      }
    } else if (hasWide) {
      // format large
      for (const r of rows) {
        if (!keepRowBySector(r)) continue;
        const w = weekLabel(r);
        const cur = acc.get(w) || { week: w, confirmed: 0, pre: 0, events: 0 };
        cur.confirmed += toNumber(r.confirmed || r.Confirmed);
        cur.pre       += toNumber(r.pre || r.Pre);
        cur.events    += toNumber(r.events || r.Events);
        acc.set(w, cur);
      }
    } else {
      // colonnes incomprises
      return [];
    }

    // vers array trié
    const out = Array.from(acc.values());
    out.sort((a, b) => String(weekSortKey(a.week)).localeCompare(String(weekSortKey(b.week))));
    return out;
  }, [history, selectedSectors]);

  const hasAny =
    series.length > 0 &&
    series.some((p) => (p.confirmed || 0) + (p.pre || 0) + (p.events || 0) > 0);

  if (!hasAny) {
    return (
      <div className="p-4 rounded border border-slate-200 dark:border-slate-700 text-sm text-slate-600 dark:text-slate-300">
        Aucune donnée historique exploitable pour le moment.
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-slate-900 rounded shadow p-4">
      <div className="flex items-baseline justify-between mb-2">
        <h3 className="font-semibold">Sector signals timeline (weekly)</h3>
        <div className="text-xs opacity-70">Somme des secteurs (filtrés) par bucket</div>
      </div>

      <div style={{ width: "100%", height: 320 }}>
        <ResponsiveContainer>
          <LineChart data={series} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.3} />
            <XAxis dataKey="week" />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="confirmed" name="confirmed" stroke="#16a34a" dot={false} />
            <Line type="monotone" dataKey="pre"        name="pre"        stroke="#f59e0b" dot={false} />
            <Line type="monotone" dataKey="events"     name="events"     stroke="#3b82f6" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-2 text-xs text-slate-500">
        Conseil : clique des secteurs dans la heatmap au-dessus pour voir leur évolution seule (multi-sélection Ctrl/Cmd).
      </div>
    </div>
  );
}
