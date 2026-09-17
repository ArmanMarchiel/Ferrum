import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The API and the pages are served from the same origin in production: FastAPI
// mounts the built assets. In dev, Vite serves the pages on 5173 and proxies
// everything the backend owns to uvicorn on 8000, so `fetch(location.origin + ...)`
// in the app code works unchanged in both.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      ['/api', '/generate', '/jobs', '/ferrum-logo.png'].map((p) => [
        p,
        { target: 'http://127.0.0.1:8000', changeOrigin: true },
      ]),
    ),
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
