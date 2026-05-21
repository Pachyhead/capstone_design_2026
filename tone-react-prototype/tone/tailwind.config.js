/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          'Pretendard',
          '-apple-system',
          'BlinkMacSystemFont',
          'system-ui',
          'sans-serif',
        ],
        mono: ['JetBrains Mono', 'SFMono-Regular', 'Consolas', 'monospace'],
      },
      colors: {
        // chrome — light theme
        cream: '#FAF6EF',
        sand: '#EDE7DB',
        ink: '#14130F',
        muted: '#6B5F4F',
        hint: '#9B8E7B',
        line: '#C8BCAA',
        charcoal: '#2C2A26',

        // chrome — dark theme
        'dk-bg': '#14130F',
        'dk-card-recv': '#1F1D1A',
        'dk-card-sent': '#262320',
        'dk-elev': '#2C2A26',

        // 9 emotions × 3 stops + extra deeps
        joy: { light: '#FBF1D6', main: '#F2D89E', deep: '#5C400D', x: '#3D2A0E' },
        sad: { light: '#E5EDF4', main: '#BBCFE5', deep: '#283F66', x: '#1B2945' },
        angry: { light: '#FBE3DB', main: '#F2B5A5', deep: '#94402C', x: '#7B341F' },
        surprise: { light: '#DEF0E8', main: '#B0DECD', deep: '#2D6852', x: '#1F4838' },
        fear: { light: '#ECE7F2', main: '#CDC4DE', deep: '#3B2E5E', x: '#2D1F47' },
        disgust: { light: '#ECEFD9', main: '#CDD5AE', deep: '#545E33', x: '#3E4424' },
        contempt: { light: '#F7E7EE', main: '#E8C5D2', deep: '#5C2C40', x: '#7E445A' },
        neutral: { light: '#ECE7DE', main: '#CFC5B5', deep: '#4A4232', x: '#5D5241' },
        other: { light: '#FAEAD9', main: '#F0CCB5', deep: '#8A4828', x: '#6E371F' },
      },
    },
  },
  plugins: [],
};
