import type { Emotion } from '@/types';

export interface EmotionPalette {
  light: string; // bubble background
  main: string; // accent (waveform, chip bg)
  deep: string; // text on light bg, play button bg
  x: string; // extra-deep text variant
}

export const EMOTION_PALETTE: Record<Emotion, EmotionPalette> = {
  happy: { light: '#FBF1D6', main: '#F2D89E', deep: '#5C400D', x: '#3D2A0E' },
  sad: { light: '#E5EDF4', main: '#BBCFE5', deep: '#283F66', x: '#1B2945' },
  angry: { light: '#FBE3DB', main: '#F2B5A5', deep: '#94402C', x: '#7B341F' },
  surprised: { light: '#DEF0E8', main: '#B0DECD', deep: '#2D6852', x: '#1F4838' },
  fearful: { light: '#ECE7F2', main: '#CDC4DE', deep: '#3B2E5E', x: '#2D1F47' },
  disgusted: { light: '#ECEFD9', main: '#CDD5AE', deep: '#545E33', x: '#3E4424' },
  unk: { light: '#F7E7EE', main: '#E8C5D2', deep: '#5C2C40', x: '#7E445A' },
  neutral: { light: '#ECE7DE', main: '#CFC5B5', deep: '#4A4232', x: '#5D5241' },
  other: { light: '#FAEAD9', main: '#F0CCB5', deep: '#8A4828', x: '#6E371F' },
};

export function paletteFor(emotion: Emotion): EmotionPalette {
  return EMOTION_PALETTE[emotion];
}
