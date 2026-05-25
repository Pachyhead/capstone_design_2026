import type { Message } from '@/types';
import { EMOTION_LABELS } from '@/types';
import { paletteFor } from '@/tokens/emotions';

interface Props {
  messages: Message[];
  variant?: 'light' | 'dark';
}

/**
 * Horizontal strip showing one bar per message in chronological order,
 * colored by the message's dominant emotion. Lets users see the emotional
 * arc of the conversation at a glance — something a text-based chat preview
 * can't show.
 */
export function EmotionArc({ messages, variant = 'light' }: Props) {
  const isDark = variant === 'dark';
  if (messages.length === 0) return null;

  const labelColor = isDark ? 'rgba(255,255,255,0.55)' : '#9B8E7B';
  const subColor = isDark ? 'rgba(255,255,255,0.35)' : '#C8BCAA';
  const borderColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(20,19,15,0.06)';
  const trackBg = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(20,19,15,0.04)';

  return (
    <div
      className="flex flex-col gap-[8px] px-10 py-[14px] flex-shrink-0"
      style={{ borderBottom: `0.5px solid ${borderColor}` }}
    >
      <div className="flex items-center gap-3">
        <span className="text-[11px] tracking-wider font-medium" style={{ color: labelColor }}>
          감정 흐름
        </span>
        <span className="text-[11px]" style={{ color: subColor }}>
          {messages.length}개 메시지 · {messages[0].sentAt} → {messages[messages.length - 1].sentAt}
        </span>
      </div>
      <div
        className="flex h-[8px] gap-[2px] rounded-full overflow-hidden"
        style={{ background: trackBg }}
      >
        {messages.map((m) => {
          const palette = paletteFor(m.emotion.primary);
          return (
            <div
              key={m.id}
              className="flex-1 transition-opacity hover:opacity-80"
              style={{ background: palette.main }}
              title={`${EMOTION_LABELS[m.emotion.primary]} · ${m.sentAt}`}
            />
          );
        })}
      </div>
    </div>
  );
}
