import base from '../tone/tailwind.config.js';

// 실제 Tone 화면 코드를 그대로 쓰므로 그쪽 클래스도 훑어야 한다.
/** @type {import('tailwindcss').Config} */
export default {
  ...base,
  content: ['./index.html', './src/**/*.{ts,tsx}', '../tone/src/**/*.{ts,tsx}'],
};
