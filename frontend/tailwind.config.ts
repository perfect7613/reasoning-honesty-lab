/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        'display': ['var(--font-space-mono)', 'monospace'],
        'body': ['var(--font-space-grotesk)', 'sans-serif'],
      },
      colors: {
        'lab': {
          bg: '#0a0a0f',
          surface: '#12121a',
          surfaceHover: '#1a1a25',
          border: '#2a2a3a',
          text: '#e8e8f0',
          textMuted: '#6b6b8a',
          accent: '#ff6b35',
          accentDim: '#ff6b3520',
          green: '#22c55e',
          greenDim: '#22c55e15',
          yellow: '#eab308',
          yellowDim: '#eab30815',
          red: '#ef4444',
          redDim: '#ef444415',
        }
      }
    },
  },
  plugins: [],
}
