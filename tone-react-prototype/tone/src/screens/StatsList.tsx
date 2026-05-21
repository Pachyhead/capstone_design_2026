import { useMemo } from 'react';
import { NavLink } from 'react-router-dom';
import { conversations, messages } from '@/data/mock';
import { BubbleCluster } from '@/components/BubbleCluster';
import { computeDistribution } from '@/data/statsCompute';

export function StatsList() {
  const rows = useMemo(
    () =>
      conversations.map((conv) => {
        const convMessages = messages.filter((m) => m.conversationId === conv.id);
        return {
          conv,
          messageCount: convMessages.length,
          distribution: computeDistribution(convMessages),
        };
      }),
    [],
  );

  return (
    <aside
      className="flex flex-col h-full flex-shrink-0"
      style={{
        width: 360,
        background: '#FAF6EF',
        borderRight: '0.5px solid rgba(20,19,15,0.06)',
      }}
    >
      <header className="px-5 pt-6 pb-4 flex-shrink-0">
        <h1 className="text-[20px] font-semibold text-ink leading-none">통계</h1>
        <p className="text-[12px] text-hint mt-2">대화방별 감정 분포</p>
      </header>

      <div className="flex-1 min-h-0 overflow-y-auto px-2 pb-3 flex flex-col gap-1">
        <NavLink to="/stats" end className="block rounded-[12px]">
          {({ isActive }) => (
            <div
              className="flex items-center gap-3 p-3 rounded-[12px]"
              style={{
                background: isActive ? '#FFFFFF' : 'transparent',
                boxShadow: isActive ? 'inset 0 0 0 0.5px rgba(20,19,15,0.06)' : 'none',
              }}
            >
              <div
                className="w-9 h-9 rounded-full flex items-center justify-center text-[12px] font-medium flex-shrink-0"
                style={{ background: '#14130F', color: '#FAF6EF' }}
              >
                ∑
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-[13px] font-medium text-ink leading-tight truncate">
                  전체 대화
                </div>
                <div className="text-[11px] text-hint mt-[3px]">합산 통계</div>
              </div>
            </div>
          )}
        </NavLink>

        {rows.map(({ conv, messageCount, distribution }) => (
          <NavLink
            key={conv.id}
            to={`/stats/${conv.id}`}
            className="block rounded-[12px]"
          >
            {({ isActive }) => (
              <div
                className="flex items-center gap-3 p-3 rounded-[12px]"
                style={{
                  background: isActive ? '#FFFFFF' : 'transparent',
                  boxShadow: isActive ? 'inset 0 0 0 0.5px rgba(20,19,15,0.06)' : 'none',
                }}
              >
                <div className="w-9 h-9 rounded-full bg-charcoal text-white flex items-center justify-center text-[12px] font-medium flex-shrink-0">
                  {conv.initial}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-[13px] font-medium text-ink leading-tight truncate">
                    {conv.name}
                  </div>
                  <div className="text-[11px] text-hint mt-[3px]">{messageCount} 메시지</div>
                </div>
                <BubbleCluster data={distribution} size="small" showLabels={false} />
              </div>
            )}
          </NavLink>
        ))}
      </div>
    </aside>
  );
}
