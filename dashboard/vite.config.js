import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const repo = process.env.VITE_GH_REPO || 'TradingScan'   // change si besoin
const useCustomDomain = !!process.env.VITE_CUSTOM_DOMAIN // true si tu mets un domaine
const onVercel = !!process.env.VERCEL                    // Vercel définit VERCEL=1 au build -> sert à la RACINE

// GitHub Pages sert sur /<repo>/ ; Vercel (ou domaine custom) sert sur / -> base '/' sinon page blanche.
export default defineConfig({
  plugins: [react()],
  base: (useCustomDomain || onVercel) ? '/' : `/${repo}/`,
})
