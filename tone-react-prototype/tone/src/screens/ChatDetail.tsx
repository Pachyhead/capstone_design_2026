import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams, useOutletContext } from 'react-router-dom';
import { conversations, messages } from '@/data/mock';
import { paletteFor } from '@/tokens/emotions';
import { EmotionWaveform } from '@/components/EmotionWaveform';
import { EmotionChip } from '@/components/EmotionChip';
import { PlayButton } from '@/components/PlayButton';
import { EmotionArc } from '@/components/EmotionArc';
import { VoiceProfilePopover } from '@/components/VoiceProfilePopover';
import { BubbleCluster } from '@/components/BubbleCluster';
import { computeDistribution } from '@/data/statsCompute';
import { EMOTION_LABELS } from '@/types';
import type { Conversation, Emotion, Message } from '@/types';
import type { ShellContext } from '@/App';
import { api, audioUrl } from '@/lib/api';

function nowHHMM(): string {
  const d = new Date();
  return `${d.getHours()}:${String(d.getMinutes()).padStart(2, '0')}`;
}

export function ChatDetail() {
  const { id } = useParams<{ id: string }>();
  const { mode, setMode } = useOutletContext<ShellContext>();
  const conversation = conversations.find((c) => c.id === id) ?? conversations[0];
  const baseThread = messages.filter((m) => m.conversationId === conversation.id);

  const [appended, setAppended] = useState<Message[]>([]);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [draft, setDraft] = useState<{ text: string; emotion: Emotion; durationSec: number; audioUrl?: string } | null>(null);
  const [sending, setSending] = useState(false);

  // reset appended messages and search when switching conversations
  useEffect(() => {
    setAppended([]);
    setSearchOpen(false);
    setSearchQuery('');
    const peerId = conversation.backendId;
    Promise.allSettled([api.setReceiverId(peerId), api.setSenderId(peerId)]).then((results) => {
      for (const r of results) if (r.status === 'rejected') console.warn('[api] set peer id:', r.reason);
    });
  }, [conversation.id]);

  const thread = [...baseThread, ...appended];
  const trimmedQuery = searchQuery.trim();
  const filteredThread = useMemo(() => {
    if (!searchOpen || !trimmedQuery) return thread;
    const q = trimmedQuery.toLowerCase();
    return thread.filter((m) => m.text.toLowerCase().includes(q));
  }, [thread, searchOpen, trimmedQuery]);
  const matchCount = searchOpen && trimmedQuery ? filteredThread.length : 0;

  const handleRecorded = async (durationSec: number) => {
    let result;
    try {
      result = await api.record(durationSec);
    } catch (err) {
      console.warn('[api] record failed:', err);
      return;
    }
    setDraft({
      text: result.text,
      emotion: result.emotion,
      durationSec: result.duration ?? durationSec,
      audioUrl: result.audio_url ? audioUrl(result.audio_url) : undefined,
    });
  };

  const handleConfirmSend = async () => {
    if (!draft) return;
    setSending(true);
    try {
      await api.send();
    } catch (err) {
      console.warn('[api] send failed:', err);
      setSending(false);
      return;
    }
    const newMessage: Message = {
      id: `local-${Date.now()}`,
      conversationId: conversation.id,
      authorId: 'me',
      text: draft.text.trim(),
      emotion: { primary: draft.emotion },
      durationSec: draft.durationSec,
      energy: Array.from({ length: 10 }, () => Math.random() * 0.5 + 0.4),
      sentAt: nowHHMM(),
    };
    setAppended((prev) => [...prev, newMessage]);
    setDraft(null);
    setSending(false);
  };

  const handleCancelDraft = () => setDraft(null);

  const handleDraftTextChange = (text: string) => {
    setDraft((prev) => (prev ? { ...prev, text } : prev));
  };

  const toggleSearch = () => {
    setSearchOpen((s) => {
      if (s) setSearchQuery('');
      return !s;
    });
  };

  const view =
    mode === 'voice' ? (
      <VoiceModeView
        conversation={conversation}
        thread={filteredThread}
        mode={mode}
        setMode={setMode}
        searchOpen={searchOpen}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        onSearchToggle={toggleSearch}
        matchCount={matchCount}
        highlight={trimmedQuery}
        onRecorded={handleRecorded}
        draft={draft}
        onDraftTextChange={handleDraftTextChange}
        onConfirmSend={handleConfirmSend}
        onCancelDraft={handleCancelDraft}
        sending={sending}
      />
    ) : (
      <TextModeView
        conversation={conversation}
        thread={filteredThread}
        mode={mode}
        setMode={setMode}
        searchOpen={searchOpen}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        onSearchToggle={toggleSearch}
        matchCount={matchCount}
        highlight={trimmedQuery}
        onRecorded={handleRecorded}
        draft={draft}
        onDraftTextChange={handleDraftTextChange}
        onConfirmSend={handleConfirmSend}
        onCancelDraft={handleCancelDraft}
        sending={sending}
      />
    );

  return view;
}

interface ModeViewProps {
  conversation: Conversation;
  thread: Message[];
  mode: 'text' | 'voice';
  setMode: (m: 'text' | 'voice') => void;
  searchOpen: boolean;
  searchQuery: string;
  onSearchChange: (q: string) => void;
  onSearchToggle: () => void;
  matchCount: number;
  highlight: string;
  onRecorded: (durationSec: number) => void;
  draft: { text: string; emotion: Emotion; durationSec: number; audioUrl?: string } | null;
  onDraftTextChange: (text: string) => void;
  onConfirmSend: () => void;
  onCancelDraft: () => void;
  sending: boolean;
}

function HighlightedText({
  text,
  query,
  baseColor,
  highlightBg,
  highlightColor,
}: {
  text: string;
  query: string;
  baseColor: string;
  highlightBg: string;
  highlightColor: string;
}) {
  if (!query) return <>{text}</>;
  const parts: Array<{ text: string; match: boolean }> = [];
  const lower = text.toLowerCase();
  const q = query.toLowerCase();
  let i = 0;
  while (i < text.length) {
    const idx = lower.indexOf(q, i);
    if (idx === -1) {
      parts.push({ text: text.slice(i), match: false });
      break;
    }
    if (idx > i) parts.push({ text: text.slice(i, idx), match: false });
    parts.push({ text: text.slice(idx, idx + q.length), match: true });
    i = idx + q.length;
  }
  return (
    <>
      {parts.map((p, k) =>
        p.match ? (
          <mark
            key={k}
            style={{
              background: highlightBg,
              color: highlightColor,
              padding: '0 2px',
              borderRadius: 3,
            }}
          >
            {p.text}
          </mark>
        ) : (
          <span key={k} style={{ color: baseColor }}>
            {p.text}
          </span>
        ),
      )}
    </>
  );
}

// -----------------------------------------------------------------------------
// Text mode — light cream theme
// -----------------------------------------------------------------------------
function TextModeView({
  conversation,
  thread,
  mode,
  setMode,
  searchOpen,
  searchQuery,
  onSearchChange,
  onSearchToggle,
  matchCount,
  highlight,
  onRecorded,
  draft,
  onDraftTextChange,
  onConfirmSend,
  onCancelDraft,
  sending,
}: ModeViewProps) {
  return (
    <div className="flex flex-col h-full w-full bg-cream min-w-0">
      <ChatHeader
        conversation={conversation}
        variant="light"
        mode={mode}
        setMode={setMode}
        onSearchToggle={onSearchToggle}
        searchOpen={searchOpen}
      />
      {searchOpen && (
        <SearchBar
          variant="light"
          query={searchQuery}
          onChange={onSearchChange}
          onClose={onSearchToggle}
          matchCount={matchCount}
          hasQuery={!!highlight}
        />
      )}
      <EmotionArc messages={thread} variant="light" />

      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="px-10 py-8 flex flex-col gap-3">
          {searchOpen && highlight && thread.length === 0 && (
            <div className="self-center text-[13px] text-muted py-4">
              "{highlight}"에 대한 결과가 없어요
            </div>
          )}
          {thread.map((m) => {
            const palette = paletteFor(m.emotion.primary);
            const isMine = m.authorId === 'me';
            return (
              <div
                key={m.id}
                className={`max-w-[640px] flex flex-col ${isMine ? 'self-end items-end' : 'self-start items-start'}`}
              >
                {!isMine && m.authorName && (
                  <span className="text-[11px] text-muted ml-3 mb-[3px]">{m.authorName}</span>
                )}
                <div
                  className={`px-4 py-3 ${
                    isMine ? 'rounded-[16px_4px_16px_16px]' : 'rounded-[4px_16px_16px_16px]'
                  }`}
                  style={{ background: palette.light }}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <PlayButton emotion={m.emotion.primary} size={26} variant="light" />
                    <EmotionWaveform emotion={m.emotion.primary} energy={m.energy} height={22} />
                    <span
                      className="text-[11px] font-mono flex-shrink-0"
                      style={{ color: palette.deep, opacity: 0.7 }}
                    >
                      {m.durationSec.toFixed(1)}s
                    </span>
                  </div>
                  <p
                    className="text-[14px] leading-snug font-medium m-0 mb-2"
                    style={{ color: palette.x }}
                  >
                    "
                    <HighlightedText
                      text={m.text}
                      query={highlight}
                      baseColor={palette.x}
                      highlightBg="#FFEFA8"
                      highlightColor="#14130F"
                    />
                    "
                  </p>
                  <div className="flex justify-between items-center gap-2">
                    <EmotionChip emotion={m.emotion} />
                    <span
                      className="text-[11px] flex-shrink-0"
                      style={{ color: palette.deep, opacity: 0.6 }}
                    >
                      {m.sentAt}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <ChatComposer
        variant="light"
        onRecorded={onRecorded}
        draft={draft}
        onDraftTextChange={onDraftTextChange}
        onConfirmSend={onConfirmSend}
        onCancelDraft={onCancelDraft}
        sending={sending}
      />
    </div>
  );
}

// -----------------------------------------------------------------------------
// Voice mode — dark theme
// -----------------------------------------------------------------------------
function VoiceModeView({
  conversation,
  thread,
  mode,
  setMode,
  searchOpen,
  searchQuery,
  onSearchChange,
  onSearchToggle,
  matchCount,
  highlight,
  onRecorded,
  draft,
  onDraftTextChange,
  onConfirmSend,
  onCancelDraft,
  sending,
}: ModeViewProps) {
  return (
    <div className="flex flex-col h-full w-full min-w-0" style={{ background: '#3A2A1A' }}>
      <ChatHeader
        conversation={conversation}
        variant="dark"
        mode={mode}
        setMode={setMode}
        onSearchToggle={onSearchToggle}
        searchOpen={searchOpen}
      />
      {searchOpen && (
        <SearchBar
          variant="dark"
          query={searchQuery}
          onChange={onSearchChange}
          onClose={onSearchToggle}
          matchCount={matchCount}
          hasQuery={!!highlight}
        />
      )}
      <EmotionArc messages={thread} variant="dark" />

      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="px-10 py-8 flex flex-col gap-3">
          {searchOpen && highlight && thread.length === 0 && (
            <div className="self-center text-[13px] py-4" style={{ color: 'rgba(255,255,255,0.55)' }}>
              "{highlight}"에 대한 결과가 없어요
            </div>
          )}
          {thread.map((m) => {
            const palette = paletteFor(m.emotion.primary);
            const isMine = m.authorId === 'me';
            return (
              <div
                key={m.id}
                className={`max-w-[640px] flex flex-col ${isMine ? 'self-end items-end' : 'self-start items-start'}`}
              >
                {!isMine && m.authorName && (
                  <span className="text-[11px] text-hint ml-3 mb-[3px]">{m.authorName}</span>
                )}
                <div
                  className={`px-4 py-3 ${
                    isMine ? 'rounded-[16px_4px_16px_16px]' : 'rounded-[4px_16px_16px_16px]'
                  }`}
                  style={{ background: isMine ? '#8E6E48' : '#785A38' }}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <PlayButton emotion={m.emotion.primary} size={28} variant="dark" />
                    <EmotionWaveform
                      emotion={m.emotion.primary}
                      energy={m.energy}
                      height={22}
                      variant="dark"
                    />
                    <span
                      className="text-[11px] font-mono flex-shrink-0 text-white"
                      style={{ opacity: 0.5 }}
                    >
                      {m.durationSec.toFixed(1)}s
                    </span>
                  </div>
                  <p
                    className="text-[12px] leading-snug m-0 mb-1"
                    style={{ color: 'rgba(255,255,255,0.65)' }}
                  >
                    "
                    <HighlightedText
                      text={m.text}
                      query={highlight}
                      baseColor="rgba(255,255,255,0.65)"
                      highlightBg="rgba(242,216,158,0.35)"
                      highlightColor="#F2D89E"
                    />
                    "
                  </p>
                  <p className="text-[12px] font-medium m-0" style={{ color: palette.main }}>
                    {EMOTION_LABELS[m.emotion.primary]}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <ChatComposer
        variant="dark"
        onRecorded={onRecorded}
        draft={draft}
        onDraftTextChange={onDraftTextChange}
        onConfirmSend={onConfirmSend}
        onCancelDraft={onCancelDraft}
        sending={sending}
      />
    </div>
  );
}

// -----------------------------------------------------------------------------
function ChatHeader({
  conversation,
  variant,
  mode,
  setMode,
  onSearchToggle,
  searchOpen,
}: {
  conversation: Conversation;
  variant: 'light' | 'dark';
  mode: 'text' | 'voice';
  setMode: (m: 'text' | 'voice') => void;
  onSearchToggle: () => void;
  searchOpen: boolean;
}) {
  const isDark = variant === 'dark';
  const [showProfile, setShowProfile] = useState(false);
  const navigate = useNavigate();
  const distribution = useMemo(
    () => computeDistribution(messages.filter((m) => m.conversationId === conversation.id)),
    [conversation.id],
  );

  return (
    <header
      className="flex items-center gap-3 px-10 py-4 flex-shrink-0 relative"
      style={{
        borderBottom: `0.5px solid ${isDark ? 'rgba(255,255,255,0.08)' : 'rgba(20,19,15,0.06)'}`,
      }}
    >
      <button
        type="button"
        onClick={() => setShowProfile((s) => !s)}
        className="w-10 h-10 rounded-full flex-shrink-0 flex items-center justify-center text-[14px] font-medium transition-transform hover:scale-105 active:scale-100"
        style={{
          background: isDark ? '#5A4126' : '#EDE7DB',
          color: isDark ? '#F2D89E' : '#14130F',
        }}
        aria-label={`${conversation.name} 음성 프로필 보기`}
      >
        {conversation.initial}
      </button>
      <button
        type="button"
        onClick={() => setShowProfile((s) => !s)}
        className="flex-1 min-w-0 text-left"
      >
        <div
          className="text-[15px] font-medium leading-tight"
          style={{ color: isDark ? '#FFFFFF' : '#14130F' }}
        >
          {conversation.name}
        </div>
        <div className="text-[12px] text-hint mt-[3px]">
          {conversation.language} · {conversation.lastSeen}
        </div>
      </button>

      {showProfile && (
        <VoiceProfilePopover
          conversation={conversation}
          variant={variant}
          onClose={() => setShowProfile(false)}
        />
      )}

      <ModeToggle isDark={isDark} mode={mode} setMode={setMode} />

      <button
        type="button"
        onClick={() => navigate(`/chat/${conversation.id}/stats`)}
        className="flex items-center justify-center flex-shrink-0 transition-colors rounded-[18px] px-3 h-9"
        style={{
          background: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(20,19,15,0.05)',
        }}
        aria-label="감정 통계 보기"
        title={`${conversation.name}와의 감정 통계`}
      >
        <BubbleCluster data={distribution} size="mini" showLabels={false} />
      </button>

      <button
        type="button"
        onClick={onSearchToggle}
        className="w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 transition-colors"
        style={{
          color: searchOpen
            ? isDark
              ? '#F2D89E'
              : '#14130F'
            : isDark
              ? '#C8BCAA'
              : '#6B5F4F',
          background: searchOpen
            ? isDark
              ? 'rgba(242,216,158,0.14)'
              : 'rgba(20,19,15,0.06)'
            : 'transparent',
        }}
        aria-label="대화 내용 검색"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <circle cx="11" cy="11" r="6.5" stroke="currentColor" strokeWidth="1.6" />
          <path d="M16 16L20 20" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
      </button>
    </header>
  );
}

function ModeToggle({
  isDark,
  mode,
  setMode,
}: {
  isDark: boolean;
  mode: 'text' | 'voice';
  setMode: (m: 'text' | 'voice') => void;
}) {
  const trackBg = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(20,19,15,0.05)';
  const activeBg = isDark ? '#F2D89E' : '#14130F';
  const activeFg = isDark ? '#14130F' : '#FAF6EF';
  const idleFg = isDark ? 'rgba(255,255,255,0.55)' : '#6B5F4F';

  return (
    <div
      className="flex items-center rounded-[10px] p-[3px] flex-shrink-0"
      role="tablist"
      aria-label="보기 모드"
      style={{ background: trackBg }}
    >
      {(['text', 'voice'] as const).map((m) => {
        const isActive = mode === m;
        return (
          <button
            key={m}
            type="button"
            role="tab"
            aria-selected={isActive}
            onClick={() => setMode(m)}
            className="text-[11px] font-medium tracking-wider px-[10px] py-[4px] rounded-[8px] transition-colors"
            style={{
              background: isActive ? activeBg : 'transparent',
              color: isActive ? activeFg : idleFg,
            }}
          >
            {m === 'text' ? '텍스트' : '음성'}
          </button>
        );
      })}
    </div>
  );
}

function SearchBar({
  variant,
  query,
  onChange,
  onClose,
  matchCount,
  hasQuery,
}: {
  variant: 'light' | 'dark';
  query: string;
  onChange: (q: string) => void;
  onClose: () => void;
  matchCount: number;
  hasQuery: boolean;
}) {
  const isDark = variant === 'dark';
  return (
    <div
      className="px-10 py-3 flex items-center gap-3 flex-shrink-0"
      style={{
        background: isDark ? '#42301E' : '#F5EFE3',
        borderBottom: `0.5px solid ${isDark ? 'rgba(255,255,255,0.08)' : 'rgba(20,19,15,0.06)'}`,
      }}
    >
      <div
        className="flex-1 flex items-center gap-2 px-4 py-[8px] rounded-[10px]"
        style={{
          background: isDark ? '#4D3823' : '#FFFFFF',
          border: `0.5px solid ${isDark ? 'rgba(255,255,255,0.08)' : 'rgba(20,19,15,0.08)'}`,
        }}
      >
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
          <circle cx="7" cy="7" r="5" stroke={isDark ? '#9B8E7B' : '#9B8E7B'} strokeWidth="1.4" />
          <path d="M11 11L14 14" stroke={isDark ? '#9B8E7B' : '#9B8E7B'} strokeWidth="1.4" strokeLinecap="round" />
        </svg>
        <input
          type="text"
          value={query}
          autoFocus
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Escape') onClose();
          }}
          placeholder="대화 내용에서 검색"
          className="flex-1 bg-transparent outline-none text-[13px]"
          style={{ color: isDark ? '#FAF6EF' : '#14130F' }}
        />
        {query && (
          <button
            type="button"
            onClick={() => onChange('')}
            className="w-5 h-5 rounded-full flex items-center justify-center text-[14px] leading-none"
            style={{ color: isDark ? 'rgba(255,255,255,0.55)' : '#9B8E7B' }}
            aria-label="검색어 지우기"
          >
            ×
          </button>
        )}
      </div>
      {hasQuery && (
        <span
          className="text-[12px] font-medium flex-shrink-0"
          style={{ color: isDark ? 'rgba(255,255,255,0.65)' : '#6B5F4F' }}
        >
          {matchCount}건
        </span>
      )}
      <button
        type="button"
        onClick={onClose}
        className="text-[12px] flex-shrink-0 hover:underline"
        style={{ color: isDark ? '#C8BCAA' : '#6B5F4F' }}
      >
        취소
      </button>
    </div>
  );
}

function ChatComposer({
  variant,
  onRecorded,
  draft,
  onDraftTextChange,
  onConfirmSend,
  onCancelDraft,
  sending,
}: {
  variant: 'light' | 'dark';
  onRecorded: (durationSec: number) => void;
  draft: { text: string; emotion: Emotion; durationSec: number; audioUrl?: string } | null;
  onDraftTextChange: (text: string) => void;
  onConfirmSend: () => void;
  onCancelDraft: () => void;
  sending: boolean;
}) {
  const isDark = variant === 'dark';
  const [isPressing, setIsPressing] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const startedAtRef = useRef<number | null>(null);
  const tickRef = useRef<number | null>(null);

  useEffect(() => {
    if (!isPressing) return;
    startedAtRef.current = Date.now();
    setElapsed(0);
    tickRef.current = window.setInterval(() => {
      if (startedAtRef.current !== null) {
        setElapsed((Date.now() - startedAtRef.current) / 1000);
      }
    }, 100);
    return () => {
      if (tickRef.current !== null) window.clearInterval(tickRef.current);
    };
  }, [isPressing]);

  const handlePressStart = () => {
    if (draft || sending) return;
    setIsPressing(true);
  };

  const handlePressEnd = () => {
    if (!isPressing) return;
    const durationSec = Math.max(1, Math.round(elapsed));
    setIsPressing(false);
    onRecorded(durationSec);
  };

  if (draft) {
    return (
      <DraftReview
        variant={variant}
        draft={draft}
        onTextChange={onDraftTextChange}
        onConfirm={onConfirmSend}
        onCancel={onCancelDraft}
        sending={sending}
      />
    );
  }

  return (
    <div
      className="px-10 py-4 flex-shrink-0 flex flex-col items-center gap-2"
      style={{
        borderTop: `0.5px solid ${isDark ? 'rgba(255,255,255,0.08)' : 'rgba(20,19,15,0.06)'}`,
      }}
    >
      {isPressing && (
        <div
          className="text-[12px] font-mono"
          style={{ color: isDark ? '#F2D89E' : '#94402C' }}
        >
          ● 녹음 중 {elapsed.toFixed(1)}s
        </div>
      )}
      <button
        type="button"
        onPointerDown={handlePressStart}
        onPointerUp={handlePressEnd}
        onPointerLeave={handlePressEnd}
        onPointerCancel={handlePressEnd}
        className="w-14 h-14 rounded-full flex items-center justify-center flex-shrink-0 transition-transform select-none"
        style={{
          background: isDark ? '#F2D89E' : '#14130F',
          color: isDark ? '#3A2A1A' : '#FFFFFF',
          transform: isPressing ? 'scale(1.15)' : 'scale(1)',
          boxShadow: isPressing
            ? `0 0 0 8px ${isDark ? 'rgba(242,216,158,0.18)' : 'rgba(20,19,15,0.10)'}`
            : 'none',
          touchAction: 'none',
        }}
        aria-label="누르고 있는 동안 녹음"
      >
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
          <rect x="9" y="3" width="6" height="12" rx="3" stroke="currentColor" strokeWidth="1.8" />
          <path d="M5 11C5 14.866 8.13401 18 12 18C15.866 18 19 14.866 19 11" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          <path d="M12 18V21" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        </svg>
      </button>
      <span
        className="text-[11px]"
        style={{ color: isDark ? 'rgba(255,255,255,0.5)' : '#9B8E7B' }}
      >
        {isPressing ? '손을 떼면 종료' : '눌러서 녹음'}
      </span>
    </div>
  );
}

function DraftReview({
  variant,
  draft,
  onTextChange,
  onConfirm,
  onCancel,
  sending,
}: {
  variant: 'light' | 'dark';
  draft: { text: string; emotion: Emotion; durationSec: number; audioUrl?: string };
  onTextChange: (text: string) => void;
  onConfirm: () => void;
  onCancel: () => void;
  sending: boolean;
}) {
  const isDark = variant === 'dark';
  const palette = paletteFor(draft.emotion);
  return (
    <div
      className="px-10 py-4 flex flex-col gap-3 flex-shrink-0"
      style={{
        borderTop: `0.5px solid ${isDark ? 'rgba(255,255,255,0.08)' : 'rgba(20,19,15,0.06)'}`,
        background: isDark ? 'rgba(255,255,255,0.03)' : 'rgba(20,19,15,0.02)',
      }}
    >
      <div className="flex items-center gap-2">
        <span
          className="px-3 py-[4px] rounded-[8px] text-[12px] font-medium"
          style={{ background: palette.main, color: palette.deep }}
        >
          {EMOTION_LABELS[draft.emotion]}
        </span>
        <span
          className="text-[11px] font-mono"
          style={{ color: isDark ? 'rgba(255,255,255,0.5)' : '#9B8E7B' }}
        >
          {draft.durationSec}s
        </span>
      </div>
      {draft.audioUrl && (
        <audio
          src={draft.audioUrl}
          controls
          preload="auto"
          className="w-full"
          style={{
            background: isDark ? '#4D3823' : '#FFFFFF',
            borderRadius: 10,
          }}
        />
      )}
      <textarea
        value={draft.text}
        onChange={(e) => onTextChange(e.target.value)}
        rows={2}
        className="w-full px-4 py-3 rounded-[12px] text-[14px] resize-none box-border focus:outline-none"
        style={{
          background: isDark ? '#4D3823' : '#FFFFFF',
          color: isDark ? '#FFFFFF' : '#14130F',
          border: `0.5px solid ${isDark ? 'rgba(255,255,255,0.10)' : 'rgba(20,19,15,0.10)'}`,
        }}
      />
      <div className="flex gap-2 justify-end">
        <button
          type="button"
          onClick={onCancel}
          disabled={sending}
          className="px-4 py-[8px] rounded-[10px] text-[13px] box-border disabled:opacity-50"
          style={{
            color: isDark ? 'rgba(255,255,255,0.65)' : '#6B5F4F',
            border: `0.5px solid ${isDark ? 'rgba(255,255,255,0.12)' : 'rgba(20,19,15,0.12)'}`,
          }}
        >
          취소
        </button>
        <button
          type="button"
          onClick={onConfirm}
          disabled={!draft.text.trim() || sending}
          className="px-5 py-[8px] rounded-[10px] text-[13px] font-medium disabled:opacity-50"
          style={{
            background: isDark ? '#F2D89E' : '#14130F',
            color: isDark ? '#3A2A1A' : '#FFFFFF',
          }}
        >
          {sending ? '전송 중…' : '전송'}
        </button>
      </div>
    </div>
  );
}
