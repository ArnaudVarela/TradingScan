// dashboard/src/lib/curation.js
const KEY = "curated_picks_v1";

// Helpers
const headers = ["date_signal", "ticker", "cohort", "bucket", "horizons", "notes"];

export function getPicks() {
  try { return JSON.parse(localStorage.getItem(KEY) || "[]"); } catch { return []; }
}

export function setPicks(arr) {
  localStorage.setItem(KEY, JSON.stringify(arr ?? []));
}

export function addPick(pick) {
  const arr = getPicks();
  const row = {
    date_signal: pick.date_signal ?? "",
    ticker: String(pick.ticker || "").trim().toUpperCase(),
    cohort: pick.cohort ?? "",
    bucket: pick.bucket ?? "",
    horizons: pick.horizons ?? "",
    notes: pick.notes ?? "",
  };
  arr.push(row);
  setPicks(arr);
}

export function removePick(index) {
  const arr = getPicks();
  arr.splice(index, 1);
  setPicks(arr);
}

export function clearPicks() {
  setPicks([]);
}

export function dedupePicks() {
  const arr = getPicks();
  const seen = new Set();
  const deduped = [];
  for (const r of arr) {
    const k = [r.date_signal || "", (r.ticker || "").toUpperCase(), r.cohort || "", r.bucket || "", r.horizons || "", r.notes || ""].join("|");
    if (!seen.has(k)) { seen.add(k); deduped.push(r); }
  }
  setPicks(deduped);
  return deduped;
}

export function toCSV(picks = getPicks()) {
  const lines = [headers.join(",")];
  for (const r of picks) {
    const row = headers.map(h => {
      const val = (r[h] ?? "").toString();
      return /[",\n]/.test(val) ? `"${val.replace(/"/g, '""')}"` : val;
    }).join(",");
    lines.push(row);
  }
  return lines.join("\n");
}

export function downloadCSV(filename = "user_trades.csv") {
  const blob = new Blob([toCSV()], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

// Import CSV (string -> picks)
export function parseCSV(text) {
  // Parser simple (gère quotes)
  const rows = [];
  let i = 0, f = "", row = [], inQ = false;
  const pushF = () => { row.push(f); f=""; };
  const pushR = () => { rows.push(row); row=[]; };
  while (i < text.length) {
    const c = text[i];
    if (inQ) {
      if (c === '"') {
        if (text[i+1] === '"') { f+='"'; i+=2; continue; }
        inQ = false; i++; continue;
      }
      f += c; i++; continue;
    }
    if (c === '"') { inQ = true; i++; continue; }
    if (c === ",") { pushF(); i++; continue; }
    if (c === "\n") { pushF(); pushR(); i++; continue; }
    if (c === "\r") { i++; continue; }
    f += c; i++;
  }
  pushF(); if (row.length) pushR();
  return rows;
}

export async function importCSVFile(file) {
  const text = await file.text();
  const rows = parseCSV(text);
  if (!rows.length) return 0;

  // map headers
  const head = rows[0].map(s => s.trim().toLowerCase());
  const idx = (name) => head.indexOf(name);
  const iDate = idx("date_signal");
  const iTicker = idx("ticker");
  const iCohort = idx("cohort");
  const iBucket = idx("bucket");
  const iHorizons = idx("horizons");
  const iNotes = idx("notes");

  const out = [];
  for (let r = 1; r < rows.length; r++) {
    const line = rows[r];
    if (!line || !line.length) continue;
    const ticker = (line[iTicker] || "").trim();
    if (!ticker) continue;
    out.push({
      date_signal: (line[iDate] || "").trim(),
      ticker: ticker.toUpperCase(),
      cohort: (line[iCohort] || "").trim(),
      bucket: (line[iBucket] || "").trim(),
      horizons: (line[iHorizons] || "").trim(),
      notes: (line[iNotes] || "").trim(),
    });
  }
  // merge + save
  const current = getPicks();
  setPicks([...current, ...out]);
  return out.length;
}
