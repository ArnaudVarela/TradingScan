import React from "react";

// palette par régime
const COLORS = {
  "Extreme Fear": "#dc2626", // red-600
  "Fear":         "#f97316", // orange-500
  "Neutral":      "#a3a3a3", // gray-400
  "Greed":        "#10b981", // emerald-500
  "Extreme Greed":"#059669", // emerald-600
};

// mappage “score 0..100” → angle (gauge -100..+100 deg)
const scoreToAngle = (s=50) => -100 + (200 * Math.max(0, Math.min(100, s)) / 100);

export default function FearGreedGauge({ score = null, label = "N/A", streak = 0, asof = null }) {
  const color = COLORS[label] || "#a3a3a3";
  const angle = score == null ? 0 : scoreToAngle(Number(score));

  return (
    <div className="flex items-center gap-3">
      {/* Jauge */}
      <div className="relative w-24 h-12">
        {/* demi-cercle multi-couleurs */}
        <svg viewBox="0 0 200 100" className="w-24 h-12">
          <defs>
            <linearGradient id="fgGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%"   stopColor="#dc2626" />
              <stop offset="25%"  stopColor="#f97316" />
              <stop offset="50%"  stopColor="#a3a3a3" />
              <stop offset="75%"  stopColor="#10b981" />
              <stop offset="100%" stopColor="#059669" />
            </linearGradient>
          </defs>
          <path d="M10,100 A90,90 0 0,1 190,100" fill="none" stroke="url(#fgGrad)" strokeWidth="16" />
          {/* aiguille */}
          <g transform={`rotate(${angle} 100 100)`}>
            <line x1="100" y1="100" x2="100" y2="18" stroke={color} strokeWidth="3" />
            <circle cx="100" cy="100" r="4" fill={color} />
          </g>
        </svg>
      </div>

      {/* Texte */}
      <div className="leading-tight">
        <div className="text-xs opacity-70">Fear &amp; Greed (CNN)</div>
        <div className="text-sm font-semibold" style={{color}}>{label}{score!=null ? ` · ${score}` : ""}</div>
        {!!streak && <div className="text-xs opacity-70">Depuis {streak} jour{streak>1?"s":""}</div>}
        {!!asof &&  <div className="text-[10px] opacity-50">MAJ {asof}</div>}
      </div>
    </div>
  );
}
