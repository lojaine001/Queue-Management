import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd());
  // Override via VITE_API_TARGET in a local .env file if needed
  const apiTarget = env.VITE_API_TARGET ?? 'http://localhost:8000';

  return {
    plugins: [react()],
    server: {
      port: 3001,
      strictPort: false,
      // Required to reach this dev server through a tunnel (Cloudflare,
      // ngrok, or anything else): host:true makes it listen on all
      // interfaces, not just localhost, and allowedHosts:true disables
      // Vite 5's check that otherwise blocks any request whose Host header
      // it doesn't recognize -- a tunnel's public hostname would otherwise
      // get "Blocked request" with no other clue. This dev server is
      // intentionally exposed publicly by design here, so there's no
      // meaningful host to pin this to instead.
      host: true,
      allowedHosts: true,
      proxy: {
        '/api': {
          target: apiTarget,
          changeOrigin: true,
          rewrite: path => path.replace(/^\/api/, ''),
        },
      },
    },
  };
});
