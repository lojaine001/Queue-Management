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
