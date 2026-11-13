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
    environment: 'jsdom', // components need DOM
    setupFiles: './src/test/setup.ts', // global mocks and cleanup
    include: ['src/**/*.test.{ts,tsx}'], // Playwright specs live in e2e/ and run separately
    css: false, // not relevant in jsdom
    coverage: {
      provider: 'v8',
      include: ['src/**/*.{ts,tsx}'], // measure all app code
      exclude: ['src/main.tsx', 'src/types/**', 'src/test/**', '**/*.test.*'], // entry point, type-only files and test code don't need to be measured
      reporter: ['text'], 
      thresholds: { lines: 80, statements: 80, functions: 80, branches: 75 },
    },
  },
})
