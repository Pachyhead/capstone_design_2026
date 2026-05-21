import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { useMemo, useState } from 'react';
import { conversations, messages, previousPeriodDelta } from '@/data/mock';
import { paletteFor } from '@/tokens/emotions';
import { EMOTION_LABELS } from '@/types';
import { BubbleCluster } from '@/components/BubbleCluster';
import {
  computeDistribution,
  computeHourlyFlow,
  filterByAuthor,
  type AuthorFilter,
} from '@/data/statsCompute';

const PERIODS = ['하루', '주', '월', '년'] as const;
type Period = (typeof PERIODS)[number];

type Author = 'me' | 'them';

export function StatsDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const fromChat = location.pathname.startsWith('/chat/');
  const conv = conversations.find((c) => c.id === id) ?? conversations[0];
  const deltas = previousPeriodDelta[conv.id] ?? previousPeriodDelta.minsu;
  const [period, setPeriod] = useState<Period>('월');
  const [authors, setAuthors] = useState<AuthorFilter>(() => new Set<Author>(['me', 'them']));

  const filtered = useMemo(
    () => filterByAuthor(messages, conv.id, authors),
    [conv.id, authors],
  );
  const distribution = useMemo(() => computeDistribution(filtered), [filtered]);
  const flow = useMemo(() => computeHourlyFlow(filtered), [filtered]);

  const sortedEmotions = (Object.entries(distribution) as [keyof typeof EMOTION_LABELS, number][])
    .sort(([, a], [, b]) => b - a);
  const minor = sortedEmotions.slice(6);
  const isCombined = authors.has('me') && authors.has('them');

  const toggleAuthor = (a: Author) => {
    setAuthors((prev) => {
      const next = new Set(prev);
      if (next.has(a)) {
        if (next.size <= 1) return prev; // 최소 1개 보장
        next.delete(a);
      } else {
        next.add(a);
      }
      return next;
    });
  };

  return (
    <div className="flex flex-col h-full w-full bg-cream min-w-0">
      <header
        className="flex items-center gap-4 px-12 py-5 flex-shrink-0"
        style={{ borderBottom: '0.5px solid rgba(20,19,15,0.06)' }}
      >
        <div className="flex-1 min-w-0">
          <div className="text-[18px] font-semibold text-ink leading-tight">
            {conv.name}와의 대화
          </div>
          <div className="text-[12px] text-hint mt-[4px]">
            {filtered.length} 메시지 · 2026년 5월
          </div>
        </div>

        <AuthorToggle authors={authors} onToggle={toggleAuthor} />

        <div
          className="flex gap-1 p-[3px] rounded-[10px]"
          style={{ background: 'rgba(20,19,15,0.05)' }}
        >
          {PERIODS.map((p) => (
            <button
              key={p}
              onClick={() => setPeriod(p)}
              className={`px-[14px] py-[7px] text-[12px] rounded-[8px] transition-colors ${
                period === p ? 'bg-white text-ink font-medium' : 'text-muted hover:text-ink'
              }`}
            >
              {p}
            </button>
          ))}
        </div>

        <button
          type="button"
          onClick={() => navigate(fromChat ? `/chat/${conv.id}` : '/me')}
          className="w-9 h-9 rounded-full flex items-center justify-center text-muted hover:bg-white transition-colors flex-shrink-0"
          aria-label={fromChat ? '대화로 돌아가기' : '나로 돌아가기'}
          title={fromChat ? '대화로 돌아가기' : '나로 돌아가기'}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
            <path
              d="M6 6L18 18M18 6L6 18"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
            />
          </svg>
        </button>
      </header>

      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="px-12 py-8 grid grid-cols-2 gap-6">
          <section
            className="bg-white rounded-[16px] p-6 col-span-2"
            style={{ boxShadow: '0 1px 0 rgba(20,19,15,0.04)' }}
          >
            <h2 className="text-[12px] text-hint tracking-wider mb-4 font-medium">감정 분포</h2>
            {filtered.length === 0 ? (
              <div className="text-center text-[13px] text-muted py-12">
                선택한 화자의 메시지가 없어요
              </div>
            ) : (
              <>
                <div className="max-w-[760px] mx-auto">
                  <BubbleCluster data={distribution} size="large" />
                </div>
                {minor.length > 0 && (
                  <div
                    className="text-[12px] text-muted text-center mt-4 pt-4"
                    style={{ borderTop: '0.5px solid rgba(20,19,15,0.06)' }}
                  >
                    {minor
                      .map(([e, p]) => `${EMOTION_LABELS[e]} ${Math.round(p * 100)}%`)
                      .join(' · ')}
                  </div>
                )}
              </>
            )}
          </section>

          <section
            className="bg-white rounded-[16px] p-6 col-span-2"
            style={{ boxShadow: '0 1px 0 rgba(20,19,15,0.04)' }}
          >
            <div className="flex items-baseline gap-2 mb-4">
              <h2 className="text-[12px] text-hint tracking-wider font-medium">
                시간대별 감정 흐름
              </h2>
              <span className="text-[11px] text-hint opacity-70">0 ~ 24시</span>
            </div>
            <div className="flex items-end gap-[2px] h-[180px]">
              {flow.map((f) => {
                const palette = paletteFor(f.emotion);
                const heightPct = Math.max(f.intensity * 100, 2);
                return (
                  <div
                    key={f.hour}
                    className="flex-1 h-full flex items-end transition-opacity hover:opacity-80"
                    title={`${String(f.hour).padStart(2, '0')}시 · ${
                      EMOTION_LABELS[f.emotion]
                    } · ${Math.round(f.intensity * 100)}%`}
                  >
                    <div
                      className="w-full rounded-t-[3px]"
                      style={{
                        height: `${heightPct}%`,
                        background: palette.main,
                      }}
                    />
                  </div>
                );
              })}
            </div>
            <div className="relative mt-2 h-[14px] text-[11px] text-hint">
              {[0, 6, 12, 18, 23].map((h) => {
                const leftPct = (h / 23) * 100;
                return (
                  <span
                    key={h}
                    className="absolute"
                    style={{
                      left: `${leftPct}%`,
                      transform:
                        h === 0
                          ? 'translateX(0)'
                          : h === 23
                            ? 'translateX(-100%)'
                            : 'translateX(-50%)',
                    }}
                  >
                    {h === 23 ? '24시' : `${h}시`}
                  </span>
                );
              })}
            </div>
          </section>

          <section
            className="bg-white rounded-[16px] p-6 col-span-2"
            style={{ boxShadow: '0 1px 0 rgba(20,19,15,0.04)' }}
          >
            <div className="flex items-baseline justify-between gap-2 mb-4">
              <h2 className="text-[12px] text-hint tracking-wider font-medium">지난 달 대비</h2>
              {!isCombined && (
                <span className="text-[11px] text-hint">전체 기준</span>
              )}
            </div>
            <div className="flex flex-col">
              {deltas.map((d, i) => {
                const palette = paletteFor(d.emotion);
                const isUp = d.delta > 0;
                const arrowColor = isUp ? '#2D6852' : '#94402C';
                return (
                  <div
                    key={d.emotion}
                    className="flex items-center gap-3 py-[11px]"
                    style={{ borderTop: i > 0 ? '0.5px solid rgba(20,19,15,0.06)' : 'none' }}
                  >
                    <span
                      className="text-[14px] font-semibold w-4 text-center flex-shrink-0"
                      style={{ color: arrowColor }}
                    >
                      {isUp ? '↑' : '↓'}
                    </span>
                    <div
                      className="w-[10px] h-[10px] rounded-full flex-shrink-0"
                      style={{ background: palette.main }}
                    />
                    <span className="flex-1 text-[13px] text-ink">{EMOTION_LABELS[d.emotion]}</span>
                    <span className="text-[12px] font-mono" style={{ color: arrowColor }}>
                      {isUp ? '+' : '−'}
                      {Math.abs(Math.round(d.delta * 100))}%
                    </span>
                  </div>
                );
              })}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

function AuthorToggle({
  authors,
  onToggle,
}: {
  authors: AuthorFilter;
  onToggle: (a: Author) => void;
}) {
  const items: { key: Author; label: string }[] = [
    { key: 'me', label: '나' },
    { key: 'them', label: '상대' },
  ];
  return (
    <div
      className="flex items-center gap-1 p-[3px] rounded-[10px] flex-shrink-0"
      role="group"
      aria-label="화자 필터"
      style={{ background: 'rgba(20,19,15,0.05)' }}
      title="둘 다 선택 시 합산"
    >
      {items.map(({ key, label }) => {
        const active = authors.has(key);
        return (
          <button
            key={key}
            type="button"
            aria-pressed={active}
            onClick={() => onToggle(key)}
            className="px-[14px] py-[7px] text-[12px] rounded-[8px] transition-colors"
            style={{
              background: active ? '#14130F' : 'transparent',
              color: active ? '#FAF6EF' : '#6B5F4F',
              fontWeight: active ? 600 : 400,
            }}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}
