import { useEffect, useRef, useState } from 'react';
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

  const [playing, setPlaying] = useState(false);
  const timeoutRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (timeoutRef.current !== null) window.clearTimeout(timeoutRef.current);
    };
  }, []);

  const handleClick = async () => {
    if (!messageId || playing) return;
    setPlaying(true);
    try {
      const result = await api.playVoice(messageId);
      const ms = Math.max(200, Math.round((result?.duration ?? 0) * 1000));
      if (timeoutRef.current !== null) window.clearTimeout(timeoutRef.current);
      timeoutRef.current = window.setTimeout(() => setPlaying(false), ms);
    } catch (err) {
      console.warn('[api] playVoice failed:', err);
      setPlaying(false);
    }
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={playing}
      className="rounded-full flex items-center justify-center flex-shrink-0 transition-transform"
      style={{
        width: size,
        height: size,
        background: bg,
        boxShadow: playing ? `0 0 0 3px ${bg}55` : 'none',
        animation: playing ? 'tone-play-pulse 1s ease-in-out infinite' : undefined,
      }}
      aria-label={playing ? '재생 중' : '재생'}
    >
      {playing ? (
        <svg width={size * 0.42} height={size * 0.42} viewBox="0 0 10 10" fill="none">
          <rect x="2" y="1.5" width="2.2" height="7" rx="0.6" fill={fg} />
          <rect x="5.8" y="1.5" width="2.2" height="7" rx="0.6" fill={fg} />
        </svg>
      ) : (
        <svg width={size * 0.45} height={size * 0.45} viewBox="0 0 10 10" fill="none">
          <path d="M2 1 L9 5 L2 9 Z" fill={fg} />
        </svg>
      )}
    </button>
  );
}
