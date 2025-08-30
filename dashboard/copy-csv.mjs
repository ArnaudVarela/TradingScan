import fs from 'fs';
import path from 'path';

const root = path.resolve(process.cwd(), '..');         // repo root
const pub  = path.resolve(process.cwd(), 'public');     // dashboard/public

const FILES = [
  'confirmed_STRONGBUY.csv',
  'anticipative_pre_signals.csv',
  'event_driven_signals.csv',
  'candidates_all_ranked.csv',
  'debug_all_candidates.csv'
];

if (!fs.existsSync(pub)) fs.mkdirSync(pub, { recursive: true });

let copied = 0;
for (const f of FILES) {
  const src = path.join(root, f);
  const dst = path.join(pub, f);
  if (fs.existsSync(src)) {
    fs.copyFileSync(src, dst);
    copied++;
    console.log(`[copy] ${f}`);
  } else {
    // crée un vide si absent pour éviter 404
    fs.writeFileSync(dst, '');
    console.log(`[copy] ${f} (vide)`);
  }
}
console.log(`[copy] done: ${copied}/${FILES.length} copiés`);
