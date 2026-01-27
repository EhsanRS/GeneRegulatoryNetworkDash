import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        trajectory: resolve(__dirname, 'trajectory.html'),
        inference: resolve(__dirname, 'inference.html'),
      },
    },
  },
});
