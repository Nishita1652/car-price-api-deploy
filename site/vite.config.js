import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/car-price-api-deploy/', // Replace with your exact repo name
})
