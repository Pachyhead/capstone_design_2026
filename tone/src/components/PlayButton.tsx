import { useEffect, useRef, useState } from 'react';
import type { Emotion } from '@/types';
import { paletteFor } from '@/tokens/emotions';
import { api, describeError } from '@/lib/api';

interface Props {
  emotion: Emotion;
  size?: number;
  variant?: 'light' | 'dark';
  messageId?: string;
  onError?: (message: string) => void;
}

export function PlayButton({ emotion, size = 22, variant = 'light', messageId, onError }: Props) {
  const palette = paletteFor(emotion);

  const bg = variant === 'dark' ? palette.main : palette.deep;
  const fg = variant === 'dark' ? palette.x : '#FFFFFF';

  const [playing, setPlaying] = useState(false);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const handleClick = async () => {
    if (!messageId || playing) return;
    setPlaying(true);
    try {
      // 서버는 스피커 재생이 끝난 뒤에 응답하므로, 응답이 곧 재생 종료다.
      await api.playVoice(messageId);
    } catch (err) {
      console.warn('[api] playVoice failed:', err);
      onError?.(describeError('음성 재생', err));
    } finally {
      if (mountedRef.current) setPlaying(false);
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
