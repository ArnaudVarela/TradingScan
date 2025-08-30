import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vercel définit VERCEL=1 au build
const isVercel = !!process.env.VERCEL;

export default defineConfig({
  plugins: [react()],
  // Vercel: assets à la racine, GitHub Pages: sous-chemin du repo
  base: isVercel ? '/' : '/TradingScan/'
})
