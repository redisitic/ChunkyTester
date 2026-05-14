import path from 'path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig(({ command }) => ({
  base: command === 'build' ? (process.env.VITE_BASE_PATH ?? '/') : '/',
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      'node:path': path.resolve(__dirname, './src/lib/noop.ts'),
      'node:fs': path.resolve(__dirname, './src/lib/noop.ts'),
    },
  },
  build: {
    chunkSizeWarningLimit: 1600,
  },
}))
