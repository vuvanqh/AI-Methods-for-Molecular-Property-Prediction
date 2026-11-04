import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // dev only: forward API calls to the FastAPI app (backend/app, port 8001) so the browser sees one origin, no CORS
  server: {
    proxy: { '/api': 'http://localhost:8001' },
  },
})
