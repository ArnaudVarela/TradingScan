// dashboard/src/lib/csv.js
//
// Chargement robuste de CSV côté front :
//  - Source par défaut : raw.githubusercontent.com (live, pas besoin de redeploy Vercel)
//  - Fallback : chemin statique local (/dashboard/public/...) pour le dev ou si raw échoue
//  - Cache-busting via commit SHA (si fourni) sinon timestamp
//
// ⚙️ Variables (optionnelles) via Vite :
//   VITE_GH_OWNER, VITE_GH_REPO, VITE_GH_BRANCH, VITE_DATA_SUBDIR, VITE_COMMIT_SHA
//
// Par défaut on suppose : owner=ArnaudVarela, repo=TradingScan, branch=main,
// et les CSV sont dans dashboard/public/
//
// Exemples d’usage :
//   import { loadCSVObjects, FILES } from "@/lib/csv";
//   const rows = await loadCSVObjects(FILES.BACKTEST_SUMMARY);
//   // rows => [{ticker: "...", pillars_met: "3", ...}, ...]
//

// ===== Config par défaut (surchargeable via .env) =============================
const OWNER  = import.meta.env.VITE_GH_OWNER   || "ArnaudVarela";
const REPO   = import.meta.env.VITE_GH_REPO    || "TradingScan";
const BRANCH = import.meta.env.VITE_GH_BRANCH  || "main";

// Sous-dossier des CSV DANS LE REPO
// Si tes CSV sont dans dashboard/public, garde la valeur par défaut.
const DATA_SUBDIR = (import.meta.env.VITE_DATA_SUBDIR || "dashboard/public").replace(/^\/+|\/+$/g, "");

// Commit SHA pour un cache-busting stable (injected par CI). Sinon timestamp.
const COMMIT_SHA = import.meta.env.VITE_COMMIT_SHA || "";

// ===== Générateurs d’URL ======================================================
export const rawUrl = (relativePath) =>
  `https://raw.githubusercontent.com/${OWNER}/${REPO}/${BRANCH}/${DATA_SUBDIR}/${relativePath}`;

// Pour le fallback local (fichiers statiques du déploiement)
export const localUrl = (relativePath) =>
  `/${relativePath}`.replace(/^\/+/, "/"); // on suppose que /public publie à la racine

// Paramètre anti-cache
const bust = () => (COMMIT_SHA ? COMMIT_SHA : Date.now().toString());

// ===== Fetch helpers ==========================================================
async function fetchText(url) {
  const res = await fetch(`${url}?v=${bust()}`, {
    cache: "no-store",
    headers: {
      // Ces headers côté client peuvent aider certains proxies
      "Cache-Control": "no-cache",
      Pragma: "no-cache",
    },
  });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} on ${url}`);
  }
  return res.text();
}

/**
 * Charge un CSV en texte brut.
 * Essaie d'abord GitHub raw, puis retombe sur le chemin local.
 */
export async function loadCSVText(fileName) {
  // 1) RAW GitHub (live)
  try {
    return await fetchText(rawUrl(fileName));
  } catch (e) {
    // 2) Fallback local (utile en dev ou si repo privé)
    // Si ton build copie encore les CSV dans /public, ça marchera sans changer le code.
    return await fetchText(localUrl(fileName));
  }
}

// ===== CSV parser (sans dépendance) ==========================================
// Parser simple mais robuste : gère les champs quotés, les virgules/retours dans guillemets.
export function parseCSV(text) {
  const rows = [];
  let i = 0, field = "", row = [], inQuotes = false;

  const pushField = () => { row.push(field); field = ""; };
  const pushRow = () => { rows.push(row); row = []; };

  while (i < text.length) {
    const c = text[i];

    if (inQuotes) {
      if (c === '"') {
        // double quote inside a quoted field => escape
        if (text[i + 1] === '"') {
          field += '"';
          i += 2;
          continue;
        }
        // closing quote
        inQuotes = false;
        i++;
        continue;
      }
      field += c;
      i++;
      continue;
    }

    if (c === '"') {
      inQuotes = true;
      i++;
      continue;
    }

    if (c === ",") {
      pushField();
      i++;
      continue;
    }

    if (c === "\n") {
      pushField();
      pushRow();
      i++;
      continue;
    }

    if (c === "\r") {
      // ignore CR (handle CRLF)
      i++;
      continue;
    }

    field += c;
    i++;
  }

  // last field/row
  pushField();
  if (row.length > 1 || (row.length === 1 && row[0] !== "")) {
    pushRow();
  }

  return rows;
}

/**
 * Convertit un CSV en tableau d’objets { header1: value, header2: value, ... }.
 * Trim les headers et valeurs, garde les types en string (tu castes après si besoin).
 */
export function toObjects(rows) {
  if (!rows || rows.length === 0) return [];
  const headers = rows[0].map((h) => (h ?? "").trim());
  const out = [];

  for (let r = 1; r < rows.length; r++) {
    const obj = {};
    const line = rows[r];
    for (let c = 0; c < headers.length; c++) {
      const key = headers[c] || `col_${c}`;
      obj[key] = (line?.[c] ?? "").trim();
    }
    out.push(obj);
  }
  return out;
}

/**
 * Charge et parse un CSV en objets.
 * @param {string} fileName ex: "backtest_summary.csv"
 * @returns {Promise<Array<Record<string,string>>>}
 */
export async function loadCSVObjects(fileName) {
  const text = await loadCSVText(fileName);
  const rows = parseCSV(text);
  return toObjects(rows);
}

// ===== Fichiers “connus” (facultatif : centralise les noms) ===================
export const FILES = {
  BACKTEST_SUMMARY: "backtest_summary.csv",
  SIGNALS_HISTORY: "signals_history.csv",
  // exemples côté “indices / indicateurs” :
  CONFIRMED_STRONGBUY: "confirmed_STRONGBUY.csv",
  CONFIRMED_BUY: "confirmed_BUY.csv",
  PILLARS_SUMMARY: "pillars_summary.csv",
  FEAR_GREED: "fear_greed.csv",
  // ajoute ici les autres CSV que ton UI consomme
};

// ===== Utilitaires pratiques ==================================================
/** Cast numérique safe (string -> number | null) */
export const toNumber = (v) => {
  if (v == null || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

/** Equality helper insensible à la casse/espaces */
export const eqI = (a, b) =>
  String(a ?? "").trim().toUpperCase() === String(b ?? "").trim().toUpperCase();
