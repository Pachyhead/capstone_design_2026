import { useEffect, useRef, type RefObject } from 'react';
import type { Conversation, Emotion } from '@/types';
import { EMOTION_LABELS } from '@/types';
import { paletteFor } from '@/tokens/emotions';
import { useInbox } from '@/hooks/useInbox';
import { emotionFromBackend } from '@/lib/api';
import type { BackendId } from '@/hooks/useProfiles';

interface Props {
  conversation: Conversation;
  variant?: 'light' | 'dark';
  onClose: () => void;
  // 팝업을 여닫는 버튼 영역. 이 안을 누른 것은 "바깥 클릭"으로 치지 않는다
  // (바깥 클릭으로 닫힌 직후 버튼 클릭이 다시 여는 것을 막는다).
  anchorRef?: RefObject<HTMLElement>;
}

export function VoiceProfilePopover({ conversation, variant = 'light', onClose, anchorRef }: Props) {
  const { inbox } = useInbox();
  const bucket = inbox[conversation.backendId as BackendId] ?? [];

  // 버킷에는 내가 보낸 메시지도 들어 있으므로, 상대가 보낸 것만 센다.
  const emotions: Emotion[] = Array.from(
    new Set(
      bucket
        .filter((m) => String(m.sender_id) === String(conversation.backendId))
        .map((m) => emotionFromBackend(m.emo_type)),
    ),
  );

  const ref = useRef<HTMLDivElement>(null);

  // close on outside click
  useEffect(() => {
    const onMouseDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (ref.current?.contains(target)) return;
      if (anchorRef?.current?.contains(target)) return;
      onClose();
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
  }, [onClose, anchorRef]);

  const isDark = variant === 'dark';
  const cardBg = isDark ? '#48341F' : '#FFFFFF';
  const fg = isDark ? '#FFFFFF' : '#14130F';
  const muted = isDark ? 'rgba(255,255,255,0.6)' : '#6B5F4F';
  const hint = isDark ? 'rgba(255,255,255,0.4)' : '#9B8E7B';
  const divider = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(20,19,15,0.06)';

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
            받은 메시지에서 감지된 감정 {emotions.length}/9
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

      {emotions.length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {emotions.map((emotion) => {
            const palette = paletteFor(emotion);
            return (
              <span
                key={emotion}
                className="px-[10px] py-[5px] rounded-[10px] text-[12px] font-medium"
                style={{ background: palette.main, color: palette.x }}
              >
                {EMOTION_LABELS[emotion]}
              </span>
            );
          })}
        </div>
      ) : (
        <p className="text-[12px]" style={{ color: muted }}>
          아직 {conversation.name}에게 받은 메시지가 없어요.
        </p>
      )}

      <p className="text-[11px] mt-4 leading-relaxed" style={{ color: hint }}>
        받은 음성 메시지는 {conversation.name}이(가) 등록한 목소리로 재구성됩니다.
      </p>
    </div>
  );
}
