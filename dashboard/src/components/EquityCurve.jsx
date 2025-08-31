import React from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

export default function EquityCurve({ data=[], title="Equity (base 100)" }) {
  if (!Array.isArray(data) || data.length === 0) {
    return <div className="text-sm opacity-60">Aucune donnée d’equity.</div>;
  }
  return (
    <div className="bg-white dark:bg-slate-900 rounded shadow p-4">
      <h3 className="font-semibold mb-2">{title}</h3>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" hide />
          <YAxis domain={["dataMin", "dataMax"]} />
          <Tooltip />
          <Line type="monotone" dataKey="equity" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
