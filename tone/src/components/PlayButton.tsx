import type { Emotion } from '@/types';
import { paletteFor } from '@/tokens/emotions';
import { api } from '@/lib/api';

interface Props {
  emotion: Emotion;
  size?: number;
  variant?: 'light' | 'dark';
  messageId?: string;
}

export function PlayButton({ emotion, size = 22, variant = 'light', messageId }: Props) {
  const palette = paletteFor(emotion);

  const bg = variant === 'dark' ? palette.main : palette.deep;
  const fg = variant === 'dark' ? palette.x : '#FFFFFF';

  const handleClick = () => {
    if (!messageId) return;
    api.playVoice(messageId).catch((err) => console.warn('[api] playVoice failed:', err));
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      className="rounded-full flex items-center justify-center flex-shrink-0"
      style={{ width: size, height: size, background: bg }}
      aria-label="재생"
    >
      <svg width={size * 0.45} height={size * 0.45} viewBox="0 0 10 10" fill="none">
        <path d="M2 1 L9 5 L2 9 Z" fill={fg} />
      </svg>
    </button>
  );
}
