// dashboard/src/components/CurationPanel.jsx
import { useEffect, useMemo, useRef, useState } from "react";
import {
  getPicks, addPick, removePick, clearPicks, dedupePicks,
  downloadCSV, importCSVFile
} from "../lib/curation.js";
import {
  Plus, Trash2, Upload, Download, History, Check, X, RefreshCw, Layers,
} from "lucide-react";

/* ---------- helpers ---------- */
const todayISO = () => new Date().toISOString().slice(0, 10);
const cls = (...a) => a.filter(Boolean).join(" ");

const COHORTS = [
  { v: "P3_confirmed", label: "P3 confirmed" },
  { v: "P2_highconv",  label: "P2 high-conv" },
  { v: "P1_explore",   label: "P1 explore" },
  { v: "P0_other",     label: "P0 other" },
];
const BUCKETS = [
  { v: "confirmed",  label: "confirmed" },
  { v: "pre_signal", label: "pre-signal" },
  { v: "event",      label: "event" },
  { v: "other",      label: "other" },
];
const H_PRESETS = [1, 3, 5, 10, 20];

export default function CurationPanel() {
  const [picks, setPicks] = useState([]);
  const [msg, setMsg] = useState({ type: "", text: "" });
  const fileRef = useRef(null);
  const dropRef = useRef(null);

  const [form, setForm] = useState({
    date_signal: todayISO(),
    ticker: "",
    cohort: "P3_confirmed",
    bucket: "confirmed",
    horizons: "1|3|5|10|20",
    notes: "",
  });

  /* ---- lifecycle ---- */
  useEffect(() => { setPicks(getPicks()); }, []);

  /* ---- derived ---- */
  const horizonsSet = useMemo(() => {
    const raw = String(form.horizons || "");
    const parts = raw.split("|").map(s => parseInt(s, 10)).filter(n => Number.isFinite(n));
    return new Set(parts);
  }, [form.horizons]);

  /* ---- UX feedback ---- */
  const toast = (type, text) => { setMsg({ type, text }); setTimeout(() => setMsg({ type:"", text:"" }), 2200); };

  /* ---- actions ---- */
  const onAdd = () => {
    const t = form.ticker.trim().toUpperCase();
    if (!t) { toast("err", "Ticker requis"); return; }
    const d = (form.date_signal || "").trim();
    if (!/^\d{4}-\d{2}-\d{2}$/.test(d)) { toast("err", "Date au format YYYY-MM-DD"); return; }

    const payload = { ...form, ticker: t, date_signal: d };
    addPick(payload);
    setPicks(getPicks());
    setForm(s => ({ ...s, ticker: "" }));
    toast("ok", "Ajouté à la liste");
  };

  const onRemove = (i) => {
    removePick(i);
    setPicks(getPicks());
  };

  const onClear = () => {
    if (!confirm("Vider la liste locale ?")) return;
    clearPicks();
    setPicks([]);
    toast("ok", "Liste vidée");
  };

  const onDedupe = () => {
    const d = dedupePicks();
    setPicks(d);
    toast("ok", `Dédup: ${d.length} éléments`);
  };

  const onImport = async (file) => {
    if (!file) return;
    const n = await importCSVFile(file).catch(() => 0);
    setPicks(getPicks());
    toast(n ? "ok" : "err", n ? `Importé: ${n} lignes` : "Import invalide");
  };

  /* ---- drag & drop import ---- */
  useEffect(() => {
    const el = dropRef.current;
    if (!el) return;
    const prevent = (e) => { e.preventDefault(); e.stopPropagation(); };
    const enter = (e) => { prevent(e); el.classList.add("ring-2","ring-sky-400"); };
    const leave = (e) => { prevent(e); el.classList.remove("ring-2","ring-sky-400"); };
    const drop = (e) => {
      prevent(e);
      el.classList.remove("ring-2","ring-sky-400");
      const f = e.dataTransfer.files?.[0];
      if (f) onImport(f);
    };
    ["dragenter","dragover"].forEach(t => el.addEventListener(t, enter));
    ["dragleave","dragend"].forEach(t => el.addEventListener(t, leave));
    el.addEventListener("drop", drop);
    return () => {
      ["dragenter","dragover"].forEach(t => el.removeEventListener(t, enter));
      ["dragleave","dragend"].forEach(t => el.removeEventListener(t, leave));
      el.removeEventListener("drop", drop);
    };
  }, []);

  /* ---- keyboard: Enter to add ---- */
  const onKeyDown = (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      onAdd();
    }
  };

  /* ---- toggle horizons chip ---- */
  const toggleH = (h) => {
    const set = new Set(horizonsSet);
    set.has(h) ? set.delete(h) : set.add(h);
    const next = Array.from(set).sort((a,b)=>a-b).join("|");
    setForm(s => ({ ...s, horizons: next || "" }));
  };

  return (
    <div className="rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900">
      {/* header */}
      <div className="px-4 md:px-6 py-3 flex items-center justify-between border-b border-slate-200 dark:border-slate-700">
        <div className="flex items-center gap-2">
          <Layers size={18} className="opacity-70" />
          <h2 className="text-base md:text-lg font-semibold">Curation backtest (local)</h2>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => downloadCSV()}
            className="inline-flex items-center gap-2 text-sm px-3 py-1.5 rounded-md border border-slate-300 dark:border-slate-600 hover:bg-slate-100 dark:hover:bg-slate-800"
            title="Exporter user_trades.csv"
          >
            <Download size={16} /> Exporter CSV
          </button>
          <label className="inline-flex items-center gap-2 text-sm px-3 py-1.5 rounded-md border border-slate-300 dark:border-slate-600 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer">
            <Upload size={16} />
            Importer
            <input
              ref={fileRef}
              type="file"
              accept=".csv"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onImport(f);
                e.target.value = "";
              }}
            />
          </label>
        </div>
      </div>

      {/* body */}
      <div className="p-4 md:p-6 space-y-5" onKeyDown={onKeyDown}>
        {/* flash */}
        {msg.text && (
          <div
            className={cls(
              "text-sm px-3 py-2 rounded-md inline-flex items-center gap-2",
              msg.type === "ok"
                ? "bg-emerald-50 text-emerald-700 border border-emerald-200 dark:bg-emerald-900/30 dark:text-emerald-200 dark:border-emerald-700"
                : "bg-rose-50 text-rose-700 border border-rose-200 dark:bg-rose-900/30 dark:text-rose-200 dark:border-rose-700"
            )}
          >
            {msg.type === "ok" ? <Check size={16}/> : <X size={16}/>}{msg.text}
          </div>
        )}

        {/* form */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-3">
          <div className="lg:col-span-2">
            <label className="block text-xs mb-1 opacity-70">Date</label>
            <input
              type="date"
              className="w-full text-sm rounded-md px-3 py-2 border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-950"
              value={form.date_signal}
              onChange={(e)=>setForm({...form, date_signal: e.target.value})}
            />
          </div>

          <div className="lg:col-span-2">
            <label className="block text-xs mb-1 opacity-70">Ticker*</label>
            <input
              className="w-full text-sm rounded-md px-3 py-2 border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-950 font-mono"
              placeholder="AAPL"
              value={form.ticker}
              onChange={(e)=>setForm({...form, ticker: e.target.value.toUpperCase()})}
            />
          </div>

          <div className="lg:col-span-3">
            <label className="block text-xs mb-1 opacity-70">Cohort</label>
            <select
              className="w-full text-sm rounded-md px-3 py-2 border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-950"
              value={form.cohort}
              onChange={(e)=>setForm({...form, cohort: e.target.value})}
            >
              {COHORTS.map(c => <option key={c.v} value={c.v}>{c.label}</option>)}
            </select>
          </div>

          <div className="lg:col-span-2">
            <label className="block text-xs mb-1 opacity-70">Bucket</label>
            <select
              className="w-full text-sm rounded-md px-3 py-2 border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-950"
              value={form.bucket}
              onChange={(e)=>setForm({...form, bucket: e.target.value})}
            >
              {BUCKETS.map(b => <option key={b.v} value={b.v}>{b.label}</option>)}
            </select>
          </div>

          <div className="lg:col-span-3">
            <label className="block text-xs mb-1 opacity-70">Horizons</label>
            {/* chips */}
            <div className="flex flex-wrap gap-1.5">
              {H_PRESETS.map(h => (
                <button
                  key={h}
                  type="button"
                  onClick={()=>toggleH(h)}
                  className={cls(
                    "px-2 py-1 rounded-full text-xs border",
                    horizonsSet.has(h)
                      ? "bg-sky-600 text-white border-sky-700"
                      : "bg-slate-100 text-slate-700 border-slate-300 dark:bg-slate-800 dark:text-slate-200 dark:border-slate-600"
                  )}
                >
                  {h}d
                </button>
              ))}
              <input
                className="ml-2 flex-1 min-w-[120px] text-sm rounded-md px-2 py-1 border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-950"
                placeholder="1|3|5|10|20"
                value={form.horizons}
                onChange={(e)=>setForm({...form, horizons: e.target.value})}
              />
            </div>
          </div>

          <div className="lg:col-span-12">
            <label className="block text-xs mb-1 opacity-70">Notes</label>
            <input
              className="w-full text-sm rounded-md px-3 py-2 border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-950"
              placeholder="raison, setup, heure d'entrée, contexte…"
              value={form.notes}
              onChange={(e)=>setForm({...form, notes: e.target.value})}
            />
          </div>

          <div className="lg:col-span-12 flex flex-wrap gap-2">
            <button
              onClick={onAdd}
              className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-slate-900 text-white hover:bg-slate-800 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-white"
              title="Ctrl/⌘ + Enter"
            >
              <Plus size={16}/> Ajouter
            </button>

            <button
              onClick={onDedupe}
              className="inline-flex items-center gap-2 px-3 py-2 rounded-md border border-slate-300 dark:border-slate-600 hover:bg-slate-100 dark:hover:bg-slate-800"
            >
              <RefreshCw size={16}/> Dédup
            </button>

            <button
              onClick={onClear}
              className="inline-flex items-center gap-2 px-3 py-2 rounded-md border border-slate-300 dark:border-slate-600 hover:bg-slate-100 dark:hover:bg-slate-800"
            >
              <Trash2 size={16}/> Vider
            </button>

            <div
              ref={dropRef}
              className="ml-auto w-full md:w-auto text-xs md:text-sm px-3 py-2 rounded-md border border-dashed border-slate-300 dark:border-slate-600 text-slate-600 dark:text-slate-300"
            >
              Glisse un `.csv` ici pour importer
            </div>
          </div>
        </div>

        {/* table */}
        <div className="overflow-x-auto rounded-md border border-slate-200 dark:border-slate-700">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-50 dark:bg-slate-800/70 text-left border-b border-slate-200 dark:border-slate-700">
                <th className="py-2 px-2">#</th>
                <th className="py-2 px-2">Date</th>
                <th className="py-2 px-2">Ticker</th>
                <th className="py-2 px-2">Cohort</th>
                <th className="py-2 px-2">Bucket</th>
                <th className="py-2 px-2">Horizons</th>
                <th className="py-2 px-2">Notes</th>
                <th className="py-2 px-2 w-16"></th>
              </tr>
            </thead>
            <tbody>
              {picks.map((r, i) => (
                <tr key={i} className="border-b border-slate-100 dark:border-slate-800">
                  <td className="py-2 px-2">{i+1}</td>
                  <td className="py-2 px-2 font-mono">{r.date_signal}</td>
                  <td className="py-2 px-2 font-semibold">
                    <span className="px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 font-mono">{r.ticker}</span>
                  </td>
                  <td className="py-2 px-2">
                    <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-200">
                      {r.cohort}
                    </span>
                  </td>
                  <td className="py-2 px-2">
                    <span className="px-2 py-0.5 rounded bg-sky-100 text-sky-800 dark:bg-sky-900/40 dark:text-sky-200">
                      {r.bucket}
                    </span>
                  </td>
                  <td className="py-2 px-2">
                    <span className="px-2 py-0.5 rounded bg-violet-100 text-violet-800 dark:bg-violet-900/40 dark:text-violet-200">
                      {r.horizons}
                    </span>
                  </td>
                  <td className="py-2 px-2 text-slate-600 dark:text-slate-300">{r.notes}</td>
                  <td className="py-2 px-2 text-right">
                    <button
                      onClick={()=>onRemove(i)}
                      className="inline-flex items-center gap-1 px-2 py-1 rounded-md border border-slate-300 dark:border-slate-600 hover:bg-slate-100 dark:hover:bg-slate-800"
                      title="Supprimer"
                    >
                      <Trash2 size={14}/> Suppr
                    </button>
                  </td>
                </tr>
              ))}
              {!picks.length && (
                <tr>
                  <td colSpan={8} className="py-8 text-center text-slate-500 dark:text-slate-400">
                    Aucun pick pour l’instant — ajoute un ticker, importe un CSV, ou glisse un fichier ici.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <p className="text-xs text-slate-500 dark:text-slate-400 flex items-center gap-2">
          <History size={14} className="opacity-70" />
          Quand tu es prêt, clique <span className="font-semibold">“Exporter CSV”</span> puis **uploade `user_trades.csv` dans ton repo** (idéalement dans `dashboard/public/`). Le prochain run GitHub Actions l’inclura dans le backtest.
        </p>
      </div>
    </div>
  );
}
