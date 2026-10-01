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
      // Required to reach this dev server through a Cloudflare tunnel:
      // host:true makes it listen on all interfaces, not just localhost,
      // and allowedHosts is needed because Vite 5 blocks requests whose
      // Host header it doesn't recognize -- the tunnel's *.trycloudflare.com
      // hostname would otherwise get "Blocked request" with no other clue.
      host: true,
      allowedHosts: ['.trycloudflare.com'],
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
