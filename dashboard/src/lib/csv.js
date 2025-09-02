// dashboard/src/lib/csv.js
//
// Chargement des CSV en direct depuis GitHub raw (repo public).
// ➡ Pas besoin de redeployer Vercel : dès qu’un commit met à jour les CSV, le front lit la nouvelle version.
//

// ======================= CONFIG ============================
const OWNER  = import.meta.env.VITE_GH_OWNER   || "ArnaudVarela";
const REPO   = import.meta.env.VITE_GH_REPO    || "TradingScan";
const BRANCH = import.meta.env.VITE_GH_BRANCH  || "main";
const COMMIT_SHA = import.meta.env.VITE_COMMIT_SHA || "";

// Paramètre anti-cache : commit SHA (si injecté) ou timestamp
const bust = () => (COMMIT_SHA ? COMMIT_SHA : Date.now().toString());

// ======================= URL HELPERS =======================
export const rawUrl = (file) =>
  `https://raw.githubusercontent.com/${OWNER}/${REPO}/${BRANCH}/${file}?v=${bust()}`;

// ======================= FETCH =============================
async function fetchText(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status} on ${url}`);
  return res.text();
}

export async function loadCSVText(file) {
  return await fetchText(rawUrl(file));
}

// ======================= PARSER CSV ========================
// Parser simple (gère les champs quotés et retours ligne)
export function parseCSV(text) {
  const rows = [];
  let i = 0, field = "", row = [], inQuotes = false;
  const pushField = () => { row.push(field); field = ""; };
  const pushRow = () => { rows.push(row); row = []; };

  while (i < text.length) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i += 2; continue; }
        inQuotes = false; i++; continue;
      }
      field += c; i++; continue;
    }
    if (c === '"') { inQuotes = true; i++; continue; }
    if (c === ",") { pushField(); i++; continue; }
    if (c === "\n") { pushField(); pushRow(); i++; continue; }
    if (c === "\r") { i++; continue; }
    field += c; i++;
  }
  pushField();
  if (row.length > 1 || (row.length === 1 && row[0] !== "")) pushRow();
  return rows;
}

// Convertit en objets { header: valeur }
export function toObjects(rows) {
  if (!rows || rows.length === 0) return [];
  const headers = rows[0].map((h) => (h ?? "").trim());
  return rows.slice(1).map((line) => {
    const obj = {};
    headers.forEach((h, i) => { obj[h] = (line?.[i] ?? "").trim(); });
    return obj;
  });
}

// ======================= API ===============================
export async function loadCSVObjects(file) {
  const text = await loadCSVText(file);
  return toObjects(parseCSV(text));
}

// Variante "safe" : renvoie [] si le fichier est absent ou invalide
export async function loadCSVObjectsOptional(file) {
  try {
    const text = await loadCSVText(file);
    return toObjects(parseCSV(text));
  } catch (e) {
    console.warn(`[WARN] Impossible de charger ${file}: ${e.message}`);
    return [];
  }
}

// ======================= LISTE DES CSV =====================
export const FILES = {
  // Fear & Greed (root)
  FEAR_GREED: "fear_greed.csv",
  FEAR_GREED_HISTORY: "fear_greed_history.csv",

  // Backtests & signaux
  BACKTEST_SUMMARY: "backtest_summary.csv",
  SIGNALS_HISTORY: "signals_history.csv",

  // Benchmarks / equity (ajoute-les si tes scripts les génèrent)
  BACKTEST_BENCHMARK: "backtest_benchmark.csv",
  BACKTEST_EQUITY: "backtest_equity.csv",

  // Cohortes confirmées (tu as au moins celui-ci)
  CONFIRMED_STRONGBUY: "confirmed_STRONGBUY.csv",
  // CONFIRMED_BUY: "confirmed_BUY.csv",
  // CONFIRMED_SELL: "confirmed_SELL.csv",
  // CONFIRMED_STRONGSELL: "confirmed_STRONGSELL.csv",

  // Univers & secteurs (adapte aux noms réels)
  RUSSELL2000: "russell2000.csv",
  SECTOR_CATALOG: "sector_catalog.csv",
  SECTOR_HISTORY: "sector_history.csv",
  SECTOR_BREADTH: "sector_breadth.csv",

  // Debug / ranking (si utilisés)
  CANDIDATES_ALL_RANKED: "candidates_all_ranked.csv",
  RAW_CANDIDATES: "raw_candidates.csv",
};

// ======================= UTILS =============================
// Conversion string → number safe
export const toNumber = (v) => {
  if (v == null || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

// Égalité insensible à la casse
export const eqI = (a, b) =>
  String(a ?? "").trim().toUpperCase() === String(b ?? "").trim().toUpperCase();
