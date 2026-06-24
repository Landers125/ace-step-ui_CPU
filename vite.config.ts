import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '');
  const frontendPort = Number(env.FRONTEND_PORT || 3000);
  const backendPort = Number(env.PORT || 3001);
  const backendTarget = `http://127.0.0.1:${backendPort}`;

  return {
    server: {
      port: frontendPort,
      host: '0.0.0.0',
      proxy: {
        '/api': {
          target: backendTarget,
          changeOrigin: true,
        },
        '/audio': {
          target: backendTarget,
          changeOrigin: true,
        },
        '/editor': {
          target: backendTarget,
          changeOrigin: true,
        },
        '/blog': {
          target: backendTarget,
          changeOrigin: true,
        },
        '/demucs-web': {
          target: backendTarget,
          changeOrigin: true,
        },
      },
    },
    optimizeDeps: {
      exclude: ['@ffmpeg/ffmpeg', '@ffmpeg/util'],
    },
    plugins: [react()],
    define: {
      'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
      'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      }
    }
  };
});
