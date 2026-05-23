import { NavLink } from 'react-router-dom';
import { conversations } from '@/data/mock';
import { paletteFor } from '@/tokens/emotions';
import { EMOTION_LABELS } from '@/types';
import { useProfiles } from '@/hooks/useProfiles';

export function ChatList() {
  const { activeProfile } = useProfiles();
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
        <button
          className="w-8 h-8 rounded-full bg-charcoal text-white flex items-center justify-center text-[18px] font-light hover:opacity-90 transition-opacity"
          aria-label="새 대화"
        >
          +
        </button>
      </header>

      <div
        className="px-4 pb-3 flex-shrink-0"
      >
        <div
          className="flex items-center gap-2 px-3 py-[7px] rounded-[10px]"
          style={{ background: 'rgba(20,19,15,0.04)' }}
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
            <circle cx="7" cy="7" r="5" stroke="#9B8E7B" strokeWidth="1.4" />
            <path d="M11 11L14 14" stroke="#9B8E7B" strokeWidth="1.4" strokeLinecap="round" />
          </svg>
          <span className="text-[12px] text-hint">검색</span>
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto px-2 pb-3 flex flex-col gap-1">
        {visible.map((conv) => {
          const palette = paletteFor(conv.lastMessage.emotion.primary);
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
                        {conv.lastMessage.sentAt}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span
                        className="text-[12px] flex-1 overflow-hidden text-ellipsis whitespace-nowrap leading-snug"
                        style={{
                          color: isActive ? palette.deep : '#6B5F4F',
                          opacity: 0.9,
                        }}
                      >
                        {conv.lastMessage.text}
                      </span>
                      <span
                        className="text-[10px] px-[6px] py-[1px] rounded-[6px] font-medium flex-shrink-0"
                        style={{ background: palette.main, color: palette.deep }}
                      >
                        {EMOTION_LABELS[conv.lastMessage.emotion.primary]}
                      </span>
                      {conv.unread > 0 && (
                        <span className="bg-charcoal text-white text-[10px] px-[6px] py-[1px] rounded-[8px] min-w-[16px] text-center font-medium flex-shrink-0">
                          {conv.unread}
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
