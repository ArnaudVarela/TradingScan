import { useMemo, useId } from "react";

/**
 * Radar animé pour la topbar
 * - dark: couleurs adaptées au dark mode
 * - size: px (24–40 conseillé)
 * - sweepDurationSec: durée d’un tour de balayage (ex: 6 pour lent, 3 pour rapide)
 * - blipCount: nombre de petits blips verts aléatoires
 */
export default function LogoRadar({
  dark = false,
  size = 32,
  sweepDurationSec = 6.0,   // ← lent par défaut
  blipCount = 6,
  className = "",
}) {
  const id = useId(); // éviter collisions des <filter> quand multiple logos
  const stroke = dark ? "#8CF5C3" : "#146C54";
  const wedge  = dark ? "#7FEFB9" : "#1FA37A";
  const red    = "#FF2020";
  const view = 200;
  const cx = 100, cy = 100, r = 90;

  // positions aléatoires mais stables (générées 1 seule fois)
  const blips = useMemo(() => {
    const arr = [];
    for (let i = 0; i < blipCount; i++) {
      const angle = Math.random() * 360;
      const radius = 25 + Math.random() * 55; // éviter trop proche du bord
      const x = cx + radius * Math.cos((angle * Math.PI) / 180);
      const y = cy + radius * Math.sin((angle * Math.PI) / 180);
      const delay = Math.random() * 2.0; // décalage d’animation
      arr.push({ x, y, delay });
    }
    return arr;
  }, [blipCount]);

  // cible rouge (fixe)
  const redAngle = 52;
  const redRadius = 62;
  const redX = cx + redRadius * Math.cos((redAngle * Math.PI) / 180);
  const redY = cy + redRadius * Math.sin((redAngle * Math.PI) / 180);

  const wedgePath = sectorPath(cx, cy, r, -20, 20);

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${view} ${view}`}
      role="img"
      aria-label="SID radar animated logo"
      className={className}
    >
      <title>Signal Intelligence Dashboard</title>

      <defs>
        {/* Glow vert */}
        <filter id={`${id}-greenGlow`} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="2" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        {/* Glow rouge */}
        <filter id={`${id}-redGlow`} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="2.2" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <style>{`
        .sid-sweep {
          transform-origin: ${cx}px ${cy}px;
          animation: sid-rotate ${sweepDurationSec}s linear infinite;
        }
        .sid-blip-green {
          animation: sid-pulse 1.8s ease-in-out infinite;
        }
        .sid-blip-red {
          animation: sid-blink 1.2s ease-in-out infinite;
        }
        .sid-center { opacity: .7; }

        @keyframes sid-rotate {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }
        @keyframes sid-pulse {
          0%, 100% { opacity: .15; transform: scale(.8); }
          50%      { opacity: 1;   transform: scale(1); }
        }
        @keyframes sid-blink {
          0%, 100% { opacity: .35; transform: scale(.9); }
          50%      { opacity: 1;   transform: scale(1.08); }
        }

        /* Accessibilité : réduit/stoppe l'anim si l'utilisateur l'a demandé */
        @media (prefers-reduced-motion: reduce) {
          .sid-sweep, .sid-blip-green, .sid-blip-red { animation: none !important; }
        }
      `}</style>

      {/* Balayage (tourne lentement) */}
      <g className="sid-sweep">
        <path d={wedgePath} fill={wedge} fillOpacity="0.20" />
      </g>

      {/* Cercles (lisibles à petite taille) */}
      <circle cx={cx} cy={cy} r={r}   fill="none" stroke={stroke} strokeWidth="6" />
      <circle cx={cx} cy={cy} r={70}  fill="none" stroke={stroke} strokeWidth="4" />
      <circle cx={cx} cy={cy} r={45}  fill="none" stroke={stroke} strokeWidth="4" />

      {/* Petits blips verts (apparition/disparition + glow) */}
      {blips.map((b, i) => (
        <circle
          key={i}
          className="sid-blip-green"
          cx={b.x}
          cy={b.y}
          r={6}
          fill={stroke}
          style={{ animationDelay: `${b.delay}s`, filter: `url(#${id}-greenGlow)` }}
        />
      ))}

      {/* Point rouge (glow + clignote) */}
      <circle
        className="sid-blip-red"
        cx={redX}
        cy={redY}
        r={8}
        fill={red}
        style={{ filter: `url(#${id}-redGlow)` }}
      />

      {/* Centre */}
      <circle className="sid-center" cx={cx} cy={cy} r={5} fill={stroke} />
    </svg>
  );
}

function sectorPath(cx, cy, r, startDeg, endDeg, sweep = 1) {
  const toRad = (d) => (d * Math.PI) / 180;
  const x1 = cx + r * Math.cos(toRad(startDeg));
  const y1 = cy + r * Math.sin(toRad(startDeg));
  const x2 = cx + r * Math.cos(toRad(endDeg));
  const y2 = cy + r * Math.sin(toRad(endDeg));
  const largeArc = ((endDeg - startDeg) % 360) > 180 ? 1 : 0;
  return `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${largeArc} ${sweep} ${x2} ${y2} Z`;
}
