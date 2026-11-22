import { useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import { useConversations } from '@/hooks/useConversations';
import { paletteFor } from '@/tokens/emotions';
import { EMOTION_LABELS } from '@/types';
import { useProfiles, type BackendId } from '@/hooks/useProfiles';
import { useInbox } from '@/hooks/useInbox';
import { seenKey, useLastSeen } from '@/hooks/useLastSeen';
import { emotionFromBackend } from '@/lib/api';

function formatSentAt(raw: string | undefined): string {
  if (!raw) return '';
  const m = raw.match(/(\d{2}):(\d{2})/);
  return m ? `${m[1]}:${m[2]}` : raw.slice(0, 5);
}

export function ChatList() {
  const { activeProfile } = useProfiles();
  const { inbox, refresh } = useInbox();
  const lastSeen = useLastSeen();
  const conversations = useConversations();

  useEffect(() => {
    if (!activeProfile) return;
    refresh(activeProfile.backendId).catch((e) =>
      console.warn('[chatlist] inbox refresh failed', e),
    );
  }, [activeProfile?.backendId]);

  const visible = conversations.filter((c) => c.backendId !== activeProfile?.backendId);
  return (
    <aside
      className="flex flex-col h-full flex-shrink-0"
      style={{
        width: 360,
        background: '#FAF6EF',
        borderRight: '0.5px solid rgba(20,19,15,0.06)',
      }}
    >
      <header className="flex items-center justify-between px-5 pt-6 pb-4 flex-shrink-0">
        <h1 className="text-[20px] font-semibold text-ink leading-none">대화</h1>
      </header>

      <div className="flex-1 min-h-0 overflow-y-auto px-2 pb-3 flex flex-col gap-1">
        {visible.map((conv) => {
          const bucket = inbox[conv.backendId as BackendId] ?? [];
          const last = bucket.length ? bucket[bucket.length - 1] : null;
          const previewText = last?.message ?? '';
          const previewEmotion = last ? emotionFromBackend(last.emo_type) : 'neutral';
          const previewSentAt = formatSentAt(last?.send_time);
          const palette = paletteFor(previewEmotion);

          // 마지막으로 읽은 메시지 이후에 상대가 보낸 것만 안 읽은 수로 센다
          // (버킷에는 내가 보낸 메시지도 들어 있다).
          const seenId =
            activeProfile ? lastSeen[seenKey(activeProfile.backendId, conv.id)] : undefined;
          const seenIdx = seenId ? bucket.findIndex((m) => m.message_id === seenId) : -1;
          const unreadCount = bucket
            .slice(seenIdx + 1)
            .filter((m) => String(m.sender_id) === String(conv.backendId)).length;
          return (
            <NavLink
              key={conv.id}
              to={`/chat/${conv.id}`}
              className="block rounded-[12px] transition-colors"
            >
              {({ isActive }) => (
                <div
                  className="flex items-start gap-3 p-3 rounded-[12px]"
                  style={{
                    background: isActive ? palette.light : 'transparent',
                    boxShadow: isActive ? 'inset 0 0 0 0.5px rgba(20,19,15,0.06)' : 'none',
                  }}
                >
                  <div
                    className="w-10 h-10 rounded-full flex items-center justify-center text-[14px] font-medium flex-shrink-0"
                    style={{
                      background: isActive ? palette.main : '#2C2A26',
                      color: isActive ? palette.x : '#FFFFFF',
                    }}
                  >
                    {conv.initial}
                  </div>
                  <div className="flex-1 min-w-0 flex flex-col gap-[3px]">
                    <div className="flex items-center justify-between gap-2">
                      <span
                        className="text-[14px] font-medium leading-tight truncate"
                        style={{ color: isActive ? palette.x : '#14130F' }}
                      >
                        {conv.name}
                      </span>
                      <span
                        className="text-[11px] flex-shrink-0"
                        style={{ color: isActive ? palette.deep : '#9B8E7B', opacity: 0.8 }}
                      >
                        {previewSentAt}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span
                        className="text-[12px] flex-1 overflow-hidden text-ellipsis whitespace-nowrap leading-snug"
                        style={{
                          color: isActive ? palette.deep : '#6B5F4F',
                          opacity: previewText ? 0.9 : 0.5,
                          fontStyle: previewText ? 'normal' : 'italic',
                        }}
                      >
                        {previewText || '아직 메시지가 없습니다'}
                      </span>
                      {last && (
                        <span
                          className="text-[10px] px-[6px] py-[1px] rounded-[6px] font-medium flex-shrink-0"
                          style={{ background: palette.main, color: palette.deep }}
                        >
                          {EMOTION_LABELS[previewEmotion]}
                        </span>
                      )}
                      {unreadCount > 0 && (
                        <span className="bg-charcoal text-white text-[10px] px-[6px] py-[1px] rounded-[8px] min-w-[16px] text-center font-medium flex-shrink-0">
                          {unreadCount}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </NavLink>
          );
        })}
      </div>
    </aside>
  );
}
