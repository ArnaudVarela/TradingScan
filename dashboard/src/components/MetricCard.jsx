// dashboard/src/components/MetricCard.jsx
import React from "react";

const ICONS = {
  Confirmed: "✅",
  "Pre-signals": "⚡",
  "Event-driven": "📅",
  Universe: "📊",
};

const COLORS = {
  Confirmed: "bg-green-100 text-green-800 border-green-300",
  "Pre-signals": "bg-yellow-100 text-yellow-800 border-yellow-300",
  "Event-driven": "bg-blue-100 text-blue-800 border-blue-300",
  Universe: "bg-purple-100 text-purple-800 border-purple-300",
};

export default function MetricCard({ label, value }) {
  const icon = ICONS[label] || "ℹ️";
  const colorClass = COLORS[label] || "bg-gray-100 text-gray-800 border-gray-300";

  return (
    <div
      className={`rounded-lg shadow border p-4 flex flex-col items-center justify-center ${colorClass}`}
    >
      <div className="text-3xl mb-2">{icon}</div>
      <div className="text-2xl font-bold">{value}</div>
      <div className="text-sm">{label}</div>
    </div>
  );
}
