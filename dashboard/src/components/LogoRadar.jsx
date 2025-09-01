// LogoRadar.jsx
export default function LogoRadar({ dark = false, size = 32, className = "" }) {
  // Couleurs light/dark (cohérentes avec tes variantes topbar)
  const stroke = dark ? "#8CF5C3" : "#146C54";
  const wedge  = dark ? "#7FEFB9" : "#1FA37A";
  const red    = "#FF2020";

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 200 200"
      role="img"
      aria-label="SID radar animated logo"
      className={className}
    >
      <title>Signal Intelligence Dashboard</title>
      <style>
        {`
        .sid-sweep {
          transform-origin: 100px 100px;
          animation: sid-rotate 3.8s linear infinite;
        }
        .sid-blip-red {
          animation: sid-blink 1.2s ease-in-out infinite;
        }
        .sid-center {
          opacity: 0.7;
        }
        @keyframes sid-rotate {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }
        @keyframes sid-blink {
          0%, 100% { opacity: 0.25; }
          50%      { opacity: 1; }
        }
        /* Respecte le réglage accessibilité */
        @media (prefers-reduced-motion: reduce) {
          .sid-sweep, .sid-blip-red { animation: none; }
        }
      `}
      </style>

      {/* Balayage (wedge) dans un groupe qui tourne */}
      <g className="sid-sweep">
        <path
          d={sectorPath(100, 100, 90, -20, 20)}
          fill={wedge}
          fillOpacity="0.20"
        />
      </g>

      {/* Cercles (épais pour lisibilité petite taille) */}
      <circle cx="100" cy="100" r="90" fill="none" stroke={stroke} strokeWidth="6" />
      <circle cx="100" cy="100" r="70" fill="none" stroke={stroke} strokeWidth="4" />
      <circle cx="100" cy="100" r="45" fill="none" stroke={stroke} strokeWidth="4" />

      {/* Point rouge clignotant */}
      <circle
        className="sid-blip-red"
        cx={100 + 65 * Math.cos((50 * Math.PI) / 180)}
        cy={100 + 65 * Math.sin((50 * Math.PI) / 180)}
        r="8"
        fill={red}
      />

      {/* Point central discret */}
      <circle className="sid-center" cx="100" cy="100" r="5" fill={stroke} />
    </svg>
  );
}

// Petite util utilitaire pour le wedge (dans le même fichier)
function sectorPath(cx, cy, r, startDeg, endDeg, sweep = 1) {
  const toRad = (d) => (d * Math.PI) / 180;
  const x1 = cx + r * Math.cos(toRad(startDeg));
  const y1 = cy + r * Math.sin(toRad(startDeg));
  const x2 = cx + r * Math.cos(toRad(endDeg));
  const y2 = cy + r * Math.sin(toRad(endDeg));
  const largeArc = ((endDeg - startDeg) % 360) > 180 ? 1 : 0;
  return `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${largeArc} ${sweep} ${x2} ${y2} Z`;
}
