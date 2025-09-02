import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const repo = process.env.VITE_GH_REPO || 'TradingScan'   // change si besoin
const useCustomDomain = !!process.env.VITE_CUSTOM_DOMAIN // true si tu mets un domaine

export default defineConfig({
  plugins: [react()],
  base: useCustomDomain ? '/' : `/${repo}/`,
})
