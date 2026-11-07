/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // dev only: forward API calls to the FastAPI app (backend/app, port 8001) so the browser sees one origin, no CORS
  server: {
    proxy: { '/api': 'http://localhost:8001' },
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    include: ['src/**/*.test.{ts,tsx}'], // Playwright specs live in e2e/ and run separately
    css: false,
    coverage: {
      provider: 'v8',
      include: ['src/**/*.{ts,tsx}'],
      exclude: ['src/main.tsx', 'src/types/**', 'src/test/**', '**/*.test.*'],
      reporter: ['text'],
      thresholds: { lines: 80, statements: 80, functions: 80, branches: 75 },
    },
  },
})
