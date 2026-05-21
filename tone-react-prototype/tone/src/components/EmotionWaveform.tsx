import type { Emotion } from '@/types';
import { paletteFor } from '@/tokens/emotions';

interface Props {
  emotion: Emotion;
  energy: number[]; // normalized 0..1
  height?: number;
  variant?: 'light' | 'dark';
}

/**
 * Single-emotion-color waveform.
 * Color = the emotion of the utterance (constant across all bars).
 * Height = audio energy at that time slice.
 */
export function EmotionWaveform({ emotion, energy, height = 22, variant = 'light' }: Props) {
  const palette = paletteFor(emotion);
  const barColor = palette.main;
  const opacity = variant === 'dark' ? 0.85 : 1;

  return (
    <div
      className="flex items-center gap-[2px] flex-1"
      style={{ height: `${height}px` }}
      aria-label={`${emotion} 음성 파형`}
    >
      {energy.map((e, i) => (
        <div
          key={i}
          className="flex-1 rounded-[1px] min-w-[1px]"
          style={{
            height: `${Math.max(e * 100, 12)}%`,
            background: barColor,
            opacity,
          }}
        />
      ))}
    </div>
  );
}
