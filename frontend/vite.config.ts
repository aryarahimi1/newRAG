import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

const apiTarget = process.env.VITE_API_PROXY || 'http://127.0.0.1:8000';

export default defineConfig({
	plugins: [sveltekit()],
	css: {
		devSourcemap: false
	},
	server: {
		proxy: {
			'/api': apiTarget
		}
	},
	optimizeDeps: {
		exclude: ['@sveltejs/kit']
	}
});
