/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'Inter', 'system-ui', 'sans-serif'],
        serif: ['"Playfair Display"', 'Georgia', 'serif'],
        display: ['Outfit', '"Plus Jakarta Sans"', 'sans-serif'],
      },
      colors: {
        // Luxury Beige Palette
        beige: {
          50: '#FDFBF7',
          100: '#FAF7F2',
          200: '#F4EFEA',
          300: '#EAE4DC',
          400: '#DED5C9',
          500: '#C8BBAA',
          600: '#A99B87',
          700: '#8A7B68',
          800: '#645849',
          900: '#433A30',
        },
        // Classy Soft Light Pink / Rose Blush
        blush: {
          50: '#FDF2F4',
          100: '#FCE7EC',
          200: '#F8D2DB',
          300: '#F3B4C2',
          400: '#E88FA3',
          500: '#D26C81',
          600: '#BA5369',
          700: '#9B3F52',
          800: '#7F3343',
          900: '#5F2431',
        },
        // Serene Powder Mist / Light Blue
        mist: {
          50: '#F4F8FA',
          100: '#EAF1F6',
          200: '#D6E4EE',
          300: '#BED4E3',
          400: '#96B9D0',
          500: '#6E9EBE',
          600: '#4F82A3',
          700: '#3D6884',
          800: '#2E5066',
          900: '#203949',
        },
        // Refined Charcoal & Warm Greys
        charcoal: {
          50: '#F8FAFC',
          100: '#F1F5F9',
          200: '#E2E8F0',
          300: '#CBD5E1',
          400: '#94A3B8',
          500: '#64748B',
          600: '#475569',
          700: '#334155',
          800: '#1E252B',
          900: '#141A1F',
        }
      },
      boxShadow: {
        'soft-luxury': '0 10px 30px -4px rgba(45, 38, 32, 0.05), 0 4px 12px -2px rgba(45, 38, 32, 0.03)',
        'soft-card': '0 4px 20px -2px rgba(45, 38, 32, 0.04)',
        'soft-hover': '0 16px 40px -6px rgba(45, 38, 32, 0.08), 0 6px 16px -3px rgba(45, 38, 32, 0.04)',
        'glow-blush': '0 10px 30px -5px rgba(210, 108, 129, 0.15)',
        'glow-mist': '0 10px 30px -5px rgba(110, 158, 190, 0.18)',
      },
      animation: {
        'fade-in': 'fadeIn 0.4s ease-out forwards',
        'subtle-pulse': 'subtlePulse 3s infinite ease-in-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        subtlePulse: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.6' },
        }
      }
    },
  },
  plugins: [],
}
