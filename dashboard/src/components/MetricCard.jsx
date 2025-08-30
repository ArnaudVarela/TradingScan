import React from "react";
import { CheckCircle2, Zap, CalendarDays, BarChart3 } from "lucide-react";

const ICONS = {
  Confirmed: CheckCircle2,
  "Pre-signals": Zap,
  "Event-driven": CalendarDays,
  Universe: BarChart3,
};

const COLORS = {
  Confirmed: "bg-green-100 text-green-900 border-green-300 dark:bg-green-900/20 dark:text-green-200 dark:border-green-800",
  "Pre-signals": "bg-yellow-100 text-yellow-900 border-yellow-300 dark:bg-yellow-900/20 dark:text-yellow-200 dark:border-yellow-800",
  "Event-driven": "bg-blue-100 text-blue-900 border-blue-300 dark:bg-blue-900/20 dark:text-blue-200 dark:border-blue-800",
  Universe: "bg-purple-100 text-purple-900 border-purple-300 dark:bg-purple-900/20 dark:text-purple-200 dark:border-purple-800",
};

export default function MetricCard({ label, value }) {
  const Icon = ICONS[label] ?? BarChart3;
  const color = COLORS[label] ?? "bg-gray-100 text-gray-900 border-gray-300 dark:bg-gray-800/40 dark:text-gray-100 dark:border-gray-700";

  return (
    <div className={`rounded-xl shadow border p-4 flex items-center gap-4 ${color}`}>
      <div className="shrink-0 rounded-lg p-2 bg-white/60 dark:bg-white/10">
        <Icon className="w-6 h-6" aria-hidden />
      </div>
      <div>
        <div className="text-2xl font-extrabold leading-tight">{value}</div>
        <div className="text-xs opacity-80">{label}</div>
      </div>
    </div>
  );
}
