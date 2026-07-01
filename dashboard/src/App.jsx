// dashboard/src/App.jsx — dashboard épuré : TopBar + Setups thématiques uniquement.
import { useState } from "react";
import TopBar from "./components/TopBar.jsx";
import ThematicSetups from "./components/ThematicSetups.jsx";
import ErrorBoundary from "./components/ErrorBoundary.jsx";

export default function App() {
  const [refreshKey, setRefreshKey] = useState(0);
  const [last, setLast] = useState(() =>
    typeof window !== "undefined" ? new Date().toLocaleString() : "-"
  );

  const onRefresh = () => {
    setRefreshKey((k) => k + 1); // remonte ThematicSetups -> re-fetch
    setLast(new Date().toLocaleString());
  };

  return (
    <div className="max-w-7xl mx-auto p-6 text-slate-900 dark:text-slate-100">
      <TopBar lastRefreshed={last} onRefresh={onRefresh} fear={null} />

      <div className="mb-4 text-xs text-slate-500 dark:text-slate-400">
        Screener thématique d'actions US (hard-tech) · setups de pré-explosion scorés /100 ·
        données prix/volume via yfinance · mise à jour quotidienne (EOD).
      </div>

      <ErrorBoundary fallback="La vue Setups thématiques n'a pas pu s'afficher.">
        <ThematicSetups key={refreshKey} />
      </ErrorBoundary>
    </div>
  );
}
