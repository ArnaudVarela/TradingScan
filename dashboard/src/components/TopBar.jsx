import { useEffect, useState } from "react";
import { Sun, Moon } from "lucide-react";

export default function TopBar({ lastRefreshed, onRefresh }) {
  const [darkMode, setDarkMode] = useState(false);

  // applique la classe dark sur <html>
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [darkMode]);

  return (
    <div className="flex justify-between items-center mb-6">
      <div>
        <h1 className="text-2xl font-bold">Signals Dashboard</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Confirmed / Anticipative / Event-driven
        </p>
      </div>
      <div className="flex gap-2 items-center">
        <span className="text-xs text-slate-500 dark:text-slate-400">
          Last refresh: {lastRefreshed}
        </span>
        <button
          className="px-3 py-1 rounded bg-slate-800 text-white text-sm dark:bg-slate-200 dark:text-black"
          onClick={onRefresh}
        >
          Refresh
        </button>
        <button
          onClick={() => setDarkMode(!darkMode)}
          className="p-2 rounded bg-slate-200 dark:bg-slate-700"
        >
          {darkMode ? <Sun size={18} /> : <Moon size={18} />}
        </button>
      </div>
    </div>
  );
}
