import { useEffect, useRef, useState } from 'react';
import type { Conversation, Emotion } from '@/types';
import { EMOTION_LABELS } from '@/types';
import { paletteFor } from '@/tokens/emotions';
import { speakerVoiceProfiles } from '@/data/mock';
import { useInbox } from '@/hooks/useInbox';
import { emotionFromBackend } from '@/lib/api';
import type { BackendId } from '@/hooks/useProfiles';

interface Props {
  conversation: Conversation;
  variant?: 'light' | 'dark';
  onClose: () => void;
}

const FALLBACK = {
  coverage: 6,
  pitch: '중간' as const,
  pace: '보통' as const,
  samples: ['happy', 'sad', 'neutral', 'fearful', 'surprised', 'angry'] as Emotion[],
};

export function VoiceProfilePopover({ conversation, variant = 'light', onClose }: Props) {
  const mockProfile = speakerVoiceProfiles[conversation.id] ?? FALLBACK;
  const { inbox } = useInbox();
  const bucket = inbox[conversation.backendId as BackendId] ?? [];

  // Derive coverage + samples from this peer's incoming messages.
  // Backend doesn't track pitch/pace, so those still come from mock.
  const realEmotions: Emotion[] = Array.from(
    new Set(bucket.map((m) => emotionFromBackend(m.emo_type))),
  );
  const coverage = bucket.length > 0 ? realEmotions.length : mockProfile.coverage;
  const samples: Emotion[] = bucket.length > 0 ? realEmotions : mockProfile.samples;

  const [playing, setPlaying] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  // close on outside click
  useEffect(() => {
    const onMouseDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('mousedown', onMouseDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [onClose]);

  const isDark = variant === 'dark';
  const cardBg = isDark ? '#48341F' : '#FFFFFF';
  const fg = isDark ? '#FFFFFF' : '#14130F';
  const muted = isDark ? 'rgba(255,255,255,0.6)' : '#6B5F4F';
  const hint = isDark ? 'rgba(255,255,255,0.4)' : '#9B8E7B';
  const divider = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(20,19,15,0.06)';

  const handlePlay = (emotion: string) => {
    setPlaying(emotion);
    // visual-only for the prototype — the real app would call TTS
    setTimeout(() => setPlaying((p) => (p === emotion ? null : p)), 1500);
  };

  return (
    <div
      ref={ref}
      className="absolute top-full left-0 mt-3 w-[340px] rounded-[14px] p-5 z-20"
      style={{
        background: cardBg,
        boxShadow: isDark
          ? '0 12px 32px rgba(0,0,0,0.5)'
          : '0 12px 32px rgba(20,19,15,0.12)',
        border: `0.5px solid ${divider}`,
      }}
    >
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="text-[15px] font-semibold leading-tight" style={{ color: fg }}>
            {conversation.name}의 목소리
          </div>
          <div className="text-[11px] mt-1" style={{ color: hint }}>
            {coverage}/9 감정 샘플 등록됨
          </div>
        </div>
        <button
          onClick={onClose}
          className="text-[18px] leading-none px-1"
          style={{ color: muted }}
          aria-label="닫기"
        >
          ×
        </button>
      </div>

      <div
        className="grid grid-cols-3 gap-3 py-3 mb-4 rounded-[10px]"
        style={{ background: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(20,19,15,0.03)' }}
      >
        <Stat label="감정 범위" value={`${coverage}/9`} fg={fg} hint={hint} />
        <Stat label="음높이" value={mockProfile.pitch} fg={fg} hint={hint} />
        <Stat label="말 속도" value={mockProfile.pace} fg={fg} hint={hint} />
      </div>

      <p
        className="text-[11px] tracking-wider font-medium mb-2"
        style={{ color: hint }}
      >
        톤 미리듣기
      </p>
      <div className="grid grid-cols-3 gap-2">
        {samples.map((emotion) => {
          const palette = paletteFor(emotion);
          const isPlaying = playing === emotion;
          return (
            <button
              key={emotion}
              type="button"
              onClick={() => handlePlay(emotion)}
              className="flex items-center justify-center gap-[5px] py-[9px] rounded-[10px] text-[12px] font-medium transition-all hover:scale-[1.02] active:scale-100"
              style={{
                background: palette.main,
                color: palette.x,
                opacity: isPlaying ? 0.85 : 1,
              }}
              aria-label={`${EMOTION_LABELS[emotion]} 톤 재생`}
            >
              {isPlaying ? (
                <span className="flex items-end gap-[2px] h-[10px] w-[8px]">
                  <span
                    className="flex-1 rounded-[1px] animate-pulse"
                    style={{ background: palette.x, height: '60%' }}
                  />
                  <span
                    className="flex-1 rounded-[1px] animate-pulse"
                    style={{ background: palette.x, height: '100%', animationDelay: '0.2s' }}
                  />
                  <span
                    className="flex-1 rounded-[1px] animate-pulse"
                    style={{ background: palette.x, height: '40%', animationDelay: '0.4s' }}
                  />
                </span>
              ) : (
                <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                  <path d="M2 1 L9 5 L2 9 Z" fill={palette.x} />
                </svg>
              )}
              <span>{EMOTION_LABELS[emotion]}</span>
            </button>
          );
        })}
      </div>

      <p className="text-[11px] mt-4 leading-relaxed" style={{ color: hint }}>
        받은 음성 메시지는 이 프로필을 사용해 {conversation.name}의 목소리로 재구성됩니다.
      </p>
    </div>
  );
}

function Stat({
  label,
  value,
  fg,
  hint,
}: {
  label: string;
  value: string;
  fg: string;
  hint: string;
}) {
  return (
    <div className="flex flex-col items-center text-center gap-[3px]">
      <span className="text-[10px] tracking-wider" style={{ color: hint }}>
        {label}
      </span>
      <span className="text-[13px] font-semibold" style={{ color: fg }}>
        {value}
      </span>
    </div>
  );
}
