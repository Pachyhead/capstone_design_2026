import type { EmotionLabel } from '@/types';
import { EMOTION_LABELS } from '@/types';
import { paletteFor } from '@/tokens/emotions';

interface Props {
  emotion: EmotionLabel;
  size?: 'sm' | 'md';
}

export function EmotionChip({ emotion, size = 'md' }: Props) {
  const palette = paletteFor(emotion.primary);
  const text = emotion.nuance
    ? `${EMOTION_LABELS[emotion.primary]} · ${emotion.nuance}`
    : EMOTION_LABELS[emotion.primary];

  const padding = size === 'sm' ? 'px-[6px] py-[2px] text-[10px]' : 'px-[7px] py-[2px] text-[11px]';

  return (
    <span
      className={`${padding} rounded-[7px] font-medium whitespace-nowrap`}
      style={{ background: palette.main, color: palette.deep }}
    >
      {text}
    </span>
  );
}
