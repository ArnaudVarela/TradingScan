import Papa from "papaparse";

export async function fetchCSV(url) {
  const cacheBust = `cb=${Date.now()}`;
  const sep = url.includes("?") ? "&" : "?";
  const res = await fetch(`${url}${sep}${cacheBust}`);
  if (!res.ok) throw new Error(`Fetch failed ${res.status}`);
  const text = await res.text();
  const parsed = Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true });
  return parsed.data;
}

/**
 * Helper pour construire l’URL raw des CSV du repo
 * Ex: rawUrl("owner","repo","main","confirmed_STRONGBUY.csv")
 */
export function rawUrl(owner, repo, branch, file) {
  return `https://raw.githubusercontent.com/${owner}/${repo}/${branch}/${encodeURIComponent(file)}`;
}

export async function fetchJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}
