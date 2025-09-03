export async function fetchFearGreedLive() {
  try {
    const r = await fetch("https://api.alternative.me/fng/?limit=1");
    const j = await r.json();
    const it = j?.data?.[0] ?? {};
    return {
      score: Number(it.value),
      label: String(it.value_classification || ""),
      asof: new Date(Number(it.timestamp) * 1000).toISOString().slice(0,10),
    };
  } catch {
    return null;
  }
}
