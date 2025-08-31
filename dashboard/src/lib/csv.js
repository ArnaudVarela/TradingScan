import Papa from "papaparse";

/** Parse un CSV via fetch + Papa. Laisse urlFor() gérer le cache-bust. */
export async function fetchCSV(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`Fetch failed ${res.status}`);
  const text = await res.text();

  const { data, errors } = Papa.parse(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    transformHeader: (h) => h.trim(),
    fastMode: true,   // un poil plus rapide
    worker: false,    // true possible si gros CSV (mais moins portable)
  });

  // Optionnel: log soft des erreurs de parsing
  if (errors?.length) {
    console.warn(`[CSV] ${errors.length} parse warning(s) on ${url}`, errors.slice(0, 3));
  }

  return Array.isArray(data) ? data : [];
}

/** Construit une URL raw GitHub en encodant chaque segment du chemin. */
export function rawUrl(owner, repo, branch, file) {
  const safePath = String(file)
    .split("/")
    .map((seg) => encodeURIComponent(seg))
    .join("/");
  return `https://raw.githubusercontent.com/${owner}/${repo}/${branch}/${safePath}`;
}

/** Fetch JSON simple (no-store) */
export async function fetchJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}
