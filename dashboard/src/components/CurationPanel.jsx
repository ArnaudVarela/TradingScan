// dashboard/src/components/CurationPanel.jsx
import { useEffect, useState } from "react";
import {
  getPicks, addPick, removePick, clearPicks, dedupePicks,
  downloadCSV, importCSVFile
} from "../lib/curation.js";

export default function CurationPanel() {
  const [picks, setPicks] = useState([]);
  const [form, setForm] = useState({
    date_signal: "",
    ticker: "",
    cohort: "P3_confirmed",
    bucket: "confirmed",
    horizons: "1|3|5|10|20",
    notes: "",
  });
  const [msg, setMsg] = useState("");

  useEffect(() => {
    setPicks(getPicks());
  }, []);

  const onAdd = () => {
    if (!form.ticker.trim()) { setMsg("Ticker requis"); return; }
    addPick(form);
    setPicks(getPicks());
    setMsg("Ajouté !");
  };

  const onRemove = (i) => {
    removePick(i);
    setPicks(getPicks());
  };

  const onClear = () => {
    clearPicks();
    setPicks([]);
  };

  const onDedupe = () => {
    const d = dedupePicks();
    setPicks(d);
    setMsg(`Dédup: ${d.length} éléments`);
  };

  const onImport = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const n = await importCSVFile(f);
    setPicks(getPicks());
    setMsg(`Importé: ${n} lignes`);
    e.target.value = "";
  };

  return (
    <div className="p-4 space-y-4 rounded-2xl shadow bg-white">
      <h2 className="text-xl font-semibold">Curation backtest (local)</h2>

      {/* Formulaire rapide */}
      <div className="grid grid-cols-1 md:grid-cols-6 gap-2 items-end">
        <div>
          <label className="text-xs">Date (YYYY-MM-DD)</label>
          <input className="w-full border rounded p-2"
            value={form.date_signal}
            onChange={e=>setForm({...form, date_signal: e.target.value})} />
        </div>
        <div>
          <label className="text-xs">Ticker*</label>
          <input className="w-full border rounded p-2"
            placeholder="AAPL"
            value={form.ticker}
            onChange={e=>setForm({...form, ticker: e.target.value})} />
        </div>
        <div>
          <label className="text-xs">Cohort</label>
          <select className="w-full border rounded p-2"
            value={form.cohort}
            onChange={e=>setForm({...form, cohort: e.target.value})}>
            <option>P3_confirmed</option>
            <option>P2_highconv</option>
            <option>P1_explore</option>
            <option>P0_other</option>
          </select>
        </div>
        <div>
          <label className="text-xs">Bucket</label>
          <select className="w-full border rounded p-2"
            value={form.bucket}
            onChange={e=>setForm({...form, bucket: e.target.value})}>
            <option>confirmed</option>
            <option>pre_signal</option>
            <option>event</option>
            <option>other</option>
          </select>
        </div>
        <div>
          <label className="text-xs">Horizons</label>
          <input className="w-full border rounded p-2"
            placeholder="1|3|5|10|20"
            value={form.horizons}
            onChange={e=>setForm({...form, horizons: e.target.value})} />
        </div>
        <div className="flex gap-2">
          <button className="px-3 py-2 rounded bg-black text-white" onClick={onAdd}>Ajouter</button>
          <button className="px-3 py-2 rounded border" onClick={()=>downloadCSV()}>Exporter CSV</button>
        </div>
        <div className="md:col-span-6">
          <label className="text-xs">Notes</label>
          <input className="w-full border rounded p-2"
            placeholder="raison, setup, etc."
            value={form.notes}
            onChange={e=>setForm({...form, notes: e.target.value})} />
        </div>
      </div>

      {/* Actions globales */}
      <div className="flex items-center gap-2">
        <input type="file" accept=".csv" onChange={onImport} />
        <button className="px-3 py-2 rounded border" onClick={onDedupe}>Dédup</button>
        <button className="px-3 py-2 rounded border" onClick={onClear}>Vider</button>
        {msg && <span className="text-sm text-green-600">{msg}</span>}
      </div>

      {/* Tableau */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left border-b">
              <th className="py-2">#</th>
              <th className="py-2">Date</th>
              <th className="py-2">Ticker</th>
              <th className="py-2">Cohort</th>
              <th className="py-2">Bucket</th>
              <th className="py-2">Horizons</th>
              <th className="py-2">Notes</th>
              <th className="py-2"></th>
            </tr>
          </thead>
          <tbody>
            {picks.map((r, i) => (
              <tr key={i} className="border-b">
                <td className="py-2">{i+1}</td>
                <td className="py-2">{r.date_signal}</td>
                <td className="py-2 font-semibold">{r.ticker}</td>
                <td className="py-2">{r.cohort}</td>
                <td className="py-2">{r.bucket}</td>
                <td className="py-2">{r.horizons}</td>
                <td className="py-2">{r.notes}</td>
                <td className="py-2">
                  <button className="px-2 py-1 rounded border" onClick={()=>onRemove(i)}>Suppr</button>
                </td>
              </tr>
            ))}
            {!picks.length && (
              <tr><td colSpan={8} className="py-6 text-center text-gray-500">Aucun pick ajouté</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <p className="text-xs text-gray-500">
        Quand tu es prêt, clique “Exporter CSV” puis **uploade `user_trades.csv` dans ton repo** (idéalement dans `dashboard/public/`).  
        Le prochain run GitHub Actions l’inclura dans le backtest.
      </p>
    </div>
  );
}
