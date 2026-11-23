
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],

  server: {
    port: 5173,

    proxy: {
      '/auth': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true,
      },

      '/documents': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true,
      },

      '/search': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true,
      },

      '/ask': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true,
      },

      '/health': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true,
      },

      '/ready': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true,
      },
    },
  },
})
