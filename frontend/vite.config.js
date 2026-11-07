import { defineConfig } from 'vite';

export default defineConfig({
  base: '/',
  build: {
    outDir: '../app/static',
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {
      output: { manualChunks: { three: ['three', 'three/addons/controls/OrbitControls.js', 'three/addons/loaders/GLTFLoader.js', 'three/addons/loaders/PLYLoader.js'] } }
    }
  },
  server: { proxy: { '/api': 'http://127.0.0.1:7860' } }
});
