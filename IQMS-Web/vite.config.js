import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

// Development only. Production serves the built app (npm run build -> dist/)
// with Caddy (see Caddyfile); the ngrok tunnel must never point at this dev
// server, which is why it listens on this machine only and accepts no other
// host names.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd());
  // IQMS_API_KEY comes from the project .env (one folder up), like the Python services.
  const rootEnv = loadEnv(mode, path.resolve(process.cwd(), '..'), 'IQMS_');
  const apiTarget = env.VITE_API_TARGET ?? 'http://localhost:8000';
  const apiKey = process.env.IQMS_API_KEY || rootEnv.IQMS_API_KEY || '';

  // Same rule as the Caddyfile: the dev proxy adds the key to the two push
  // calls only. The key is never part of the JavaScript sent to the browser.
  const toApi = { target: apiTarget, changeOrigin: true, rewrite: p => p.replace(/^\/api/, '') };
  const withKey = { ...toApi, headers: apiKey ? { 'X-API-Key': apiKey } : {} };

  return {
    plugins: [react()],
    server: {
      port: 3001,
      strictPort: false,
      host: 'localhost',
      proxy: {
        '/api/push-subscribe': withKey,
        '/api/push-unsubscribe': withKey,
        '/api': toApi,
      },
    },
  };
});
