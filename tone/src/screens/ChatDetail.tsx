import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { useConversations } from '@/hooks/useConversations';
import { useMessages } from '@/hooks/useMessages';
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
import { api, audioUrl, describeError } from '@/lib/api';
import { useProfiles } from '@/hooks/useProfiles';
import { useInbox } from '@/hooks/useInbox';
import { markSeen } from '@/hooks/useLastSeen';


export function ChatDetail() {
  const { id } = useParams<{ id: string }>();
  const conversations = useConversations();
  const conversation = conversations.find((c) => c.id === id) ?? conversations[0];
  const { thread } = useMessages(conversation.backendId, conversation.id);
  const { activeProfile } = useProfiles();
  const { inbox, refresh } = useInbox();
  const myBackendId = activeProfile?.backendId;

  // 채팅방 진입/전환 시 백엔드 수신자(receiver_id)를 현재 상대로 맞춘다.
  useEffect(() => {
    api.setReceiverId(conversation.backendId).catch((err) =>
      console.warn('[api] setReceiverId failed:', err),
    );
  }, [conversation.backendId]);

  useEffect(() => {
    const bucket = inbox[conversation.backendId as 0 | 1 | 2 | 3] ?? [];
    if (!bucket.length || myBackendId == null) return;
    markSeen(myBackendId, conversation.id, bucket[bucket.length - 1].message_id);
  }, [inbox, conversation.id, conversation.backendId, myBackendId]);

  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [draft, setDraft] = useState<{ text: string; emotion: Emotion; durationSec: number; audioUrl?: string } | null>(null);
  const [sending, setSending] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  // 손을 뗀 뒤 서버가 음성을 분석(STT·감정)하는 동안 true.
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 실패 안내는 잠시 보여 주고 스스로 사라진다.
  useEffect(() => {
    if (!error) return;
    const timer = window.setTimeout(() => setError(null), 5000);
    return () => window.clearTimeout(timer);
  }, [error]);

  // 녹음 시작 요청의 결과. 손을 뗄 때 이걸 기다려서 (1) 시작이 실패했으면 정지를
  // 보내지 않고 (2) 정지가 시작보다 먼저 서버에 닿지 않게 한다.
  const startResultRef = useRef<Promise<boolean> | null>(null);
  // 현재 보고 있는 방. 분석 결과가 돌아왔을 때 방이 바뀌어 있으면 초안을 버린다.
  const conversationIdRef = useRef(conversation.id);
  conversationIdRef.current = conversation.id;

  const handleRefreshInbox = async () => {
    if (refreshing || !activeProfile) return;
    setRefreshing(true);
    try {
      await refresh(activeProfile.backendId);
    } catch (err) {
      console.warn('[api] refresh inbox failed:', err);
      setError(describeError('새로고침', err));
    } finally {
      setRefreshing(false);
    }
  };

  // 방을 옮기면 검색과 초안을 비운다. 초안은 이전 상대에게 보내려고 녹음한 것이라
  // 남겨 두면 새 방의 상대에게 전송된다.
  useEffect(() => {
    setSearchOpen(false);
    setSearchQuery('');
    setDraft(null);
    setError(null);
  }, [conversation.id]);

  const trimmedQuery = searchQuery.trim();
  const filteredThread = useMemo(() => {
    if (!searchOpen || !trimmedQuery) return thread;
    const q = trimmedQuery.toLowerCase();
    return thread.filter((m) => m.text.toLowerCase().includes(q));
  }, [thread, searchOpen, trimmedQuery]);
  const matchCount = searchOpen && trimmedQuery ? filteredThread.length : 0;

  const handlePressStart = () => {
    setError(null);
    startResultRef.current = api
      .startRecording()
      .then(() => true)
      .catch((err) => {
        console.warn('[api] startRecording failed:', err);
        setError(describeError('녹음 시작', err));
        return false;
      });
  };

  const handlePressEnd = async () => {
    const startedIn = conversationIdRef.current;
    const started = await (startResultRef.current ?? Promise.resolve(false));
    startResultRef.current = null;
    if (!started) return;

    let result;
    setProcessing(true);
    try {
      result = await api.stopRecording();
    } catch (err) {
      console.warn('[api] stopRecording failed:', err);
      setError(describeError('음성 분석', err));
      return;
    } finally {
      setProcessing(false);
    }
    if (conversationIdRef.current !== startedIn) return;
    setDraft({
      text: result.text,
      emotion: result.emotion,
      durationSec: result.duration,
      audioUrl: result.audio_url ? `${audioUrl(result.audio_url)}?t=${Date.now()}` : undefined,
    });
  };

  const handleConfirmSend = async () => {
    if (!draft) return;
    setSending(true);
    try {
      await api.send(draft.text);
    } catch (err) {
      console.warn('[api] send failed:', err);
      setError(describeError('전송', err));
      setSending(false);
      return;
    }
    // 보낸 메시지는 서버 인박스에 포함돼 돌아온다. 로컬에 따로 붙이지 않고
    // 인박스를 다시 받아 서버 기준 순서/시간으로 표시한다.
    if (activeProfile) {
      try {
        await refresh(activeProfile.backendId);
      } catch (err) {
        console.warn('[api] refresh inbox after send failed:', err);
        setError('전송은 됐지만 대화 내역을 새로 받지 못했어요. 새로고침을 눌러 주세요.');
      }
    }
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

  return (
    <TextModeView
      conversation={conversation}
      thread={filteredThread}
      searchOpen={searchOpen}
      searchQuery={searchQuery}
      onSearchChange={setSearchQuery}
      onSearchToggle={toggleSearch}
      matchCount={matchCount}
      highlight={trimmedQuery}
      onPressStart={handlePressStart}
      onPressEnd={handlePressEnd}
      draft={draft}
      onDraftTextChange={handleDraftTextChange}
      onConfirmSend={handleConfirmSend}
      onCancelDraft={handleCancelDraft}
      sending={sending}
      processing={processing}
      onRefreshInbox={handleRefreshInbox}
      refreshing={refreshing}
      error={error}
      onError={setError}
      onDismissError={() => setError(null)}
    />
  );
}

interface ModeViewProps {
  conversation: Conversation;
  thread: Message[];
  searchOpen: boolean;
  searchQuery: string;
  onSearchChange: (q: string) => void;
  onSearchToggle: () => void;
  matchCount: number;
  highlight: string;
  onPressStart: () => void;
  onPressEnd: () => void;
  draft: { text: string; emotion: Emotion; durationSec: number; audioUrl?: string } | null;
  onDraftTextChange: (text: string) => void;
  onConfirmSend: () => void;
  onCancelDraft: () => void;
  sending: boolean;
  processing: boolean;
  onRefreshInbox: () => void;
  refreshing: boolean;
  error: string | null;
  onError: (message: string) => void;
  onDismissError: () => void;
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
  searchOpen,
  searchQuery,
  onSearchChange,
  onSearchToggle,
  matchCount,
  highlight,
  onPressStart,
  onPressEnd,
  draft,
  onDraftTextChange,
  onConfirmSend,
  onCancelDraft,
  sending,
  processing,
  onRefreshInbox,
  refreshing,
  error,
  onError,
  onDismissError,
}: ModeViewProps) {
  return (
    <div className="flex flex-col h-full w-full bg-cream min-w-0">
      <ChatHeader
        conversation={conversation}
        variant="light"
        onSearchToggle={onSearchToggle}
        searchOpen={searchOpen}
        onRefreshInbox={onRefreshInbox}
        refreshing={refreshing}
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
                    <PlayButton
                      emotion={m.emotion.primary}
                      size={26}
                      variant="light"
                      messageId={m.id}
                      onError={onError}
                    />
                    <EmotionWaveform emotion={m.emotion.primary} energy={m.energy} height={22} />
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

      {error && (
        <div
          role="alert"
          className="mx-10 mb-3 px-4 py-[10px] rounded-[12px] flex items-center gap-3 flex-shrink-0 text-[13px]"
          style={{ background: '#FBE3DB', color: '#7B341F' }}
        >
          <span className="flex-1">{error}</span>
          <button
            type="button"
            onClick={onDismissError}
            className="text-[16px] leading-none px-1"
            aria-label="안내 닫기"
          >
            ×
          </button>
        </div>
      )}

      <ChatComposer
        variant="light"
        onPressStart={onPressStart}
        onPressEnd={onPressEnd}
        draft={draft}
        onDraftTextChange={onDraftTextChange}
        onConfirmSend={onConfirmSend}
        onCancelDraft={onCancelDraft}
        sending={sending}
        processing={processing}
      />
    </div>
  );
}

// -----------------------------------------------------------------------------
function ChatHeader({
  conversation,
  variant,
  onSearchToggle,
  searchOpen,
  onRefreshInbox,
  refreshing,
}: {
  conversation: Conversation;
  variant: 'light' | 'dark';
  onSearchToggle: () => void;
  searchOpen: boolean;
  onRefreshInbox: () => void;
  refreshing: boolean;
}) {
  const isDark = variant === 'dark';
  const [showProfile, setShowProfile] = useState(false);
  const navigate = useNavigate();
  const { thread: headerThread } = useMessages(conversation.backendId, conversation.id);
  const distribution = useMemo(
    () => computeDistribution(headerThread),
    [headerThread],
  );
  const lastSentAt = headerThread.length ? headerThread[headerThread.length - 1].sentAt : '';
  // 팝업을 여닫는 버튼 묶음. display: contents 라서 배치에는 영향이 없다.
  const profileAnchorRef = useRef<HTMLDivElement>(null);

  return (
    <header
      className="flex items-center gap-3 px-10 py-4 flex-shrink-0 relative"
      style={{
        borderBottom: `0.5px solid ${isDark ? 'rgba(255,255,255,0.08)' : 'rgba(20,19,15,0.06)'}`,
      }}
    >
      <div ref={profileAnchorRef} className="contents">
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
          {lastSentAt ? `마지막 메시지 ${lastSentAt}` : '아직 메시지가 없어요'}
        </div>
      </button>
      </div>

      {showProfile && (
        <VoiceProfilePopover
          conversation={conversation}
          variant={variant}
          onClose={() => setShowProfile(false)}
          anchorRef={profileAnchorRef}
        />
      )}

      <button
        type="button"
        onClick={() => navigate(`/chat/${conversation.id}/stats`)}
        className="flex items-center justify-center flex-shrink-0 transition-colors rounded-[18px] px-3 h-9"
        style={{
          background: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(20,19,15,0.05)',
          color: isDark ? '#C8BCAA' : '#6B5F4F',
        }}
        aria-label="감정 통계 보기"
        title={`${conversation.name}와의 감정 통계`}
      >
        {Object.keys(distribution).length > 0 ? (
          <BubbleCluster data={distribution} size="mini" showLabels={false} />
        ) : (
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path
              d="M2 13V7M6 13V3M10 13V8M14 13V5"
              stroke="currentColor"
              strokeWidth="1.4"
              strokeLinecap="round"
            />
          </svg>
        )}
      </button>

      <button
        type="button"
        onClick={onRefreshInbox}
        disabled={refreshing}
        className="w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 transition-colors disabled:opacity-40"
        style={{
          color: isDark ? '#C8BCAA' : '#6B5F4F',
        }}
        aria-label="수신 메시지 새로고침"
        title="수신 메시지 새로고침"
      >
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          style={{
            transformOrigin: '50% 50%',
            animation: refreshing ? 'spin 0.8s linear infinite' : undefined,
          }}
        >
          <path
            d="M4 12a8 8 0 0 1 13.66-5.66L20 8.5M20 4v4.5h-4.5M20 12a8 8 0 0 1-13.66 5.66L4 15.5M4 20v-4.5h4.5"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
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
  onPressStart,
  onPressEnd,
  draft,
  onDraftTextChange,
  onConfirmSend,
  onCancelDraft,
  sending,
  processing,
}: {
  variant: 'light' | 'dark';
  onPressStart: () => void;
  onPressEnd: () => void;
  draft: { text: string; emotion: Emotion; durationSec: number; audioUrl?: string } | null;
  onDraftTextChange: (text: string) => void;
  onConfirmSend: () => void;
  onCancelDraft: () => void;
  sending: boolean;
  processing: boolean;
}) {
  const isDark = variant === 'dark';
  const [isPressing, setIsPressing] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const startedAtRef = useRef<number | null>(null);
  const releasingRef = useRef(false);

  const release = () => {
    if (releasingRef.current) return;
    releasingRef.current = true;
    setIsPressing(false);
    onPressEnd();
  };

  useEffect(() => {
    if (!isPressing) return;
    releasingRef.current = false;
    startedAtRef.current = Date.now();
    setElapsed(0);
    const id = window.setInterval(() => {
      if (startedAtRef.current === null) return;
      const sec = (Date.now() - startedAtRef.current) / 1000;
      setElapsed(sec);
      if (sec >= 10) release();
    }, 100);
    return () => window.clearInterval(id);
  }, [isPressing]);

  useEffect(() => {
    if (!isPressing) return;
    const onUp = () => release();
    document.addEventListener('pointerup', onUp);
    document.addEventListener('pointercancel', onUp);
    return () => {
      document.removeEventListener('pointerup', onUp);
      document.removeEventListener('pointercancel', onUp);
    };
  }, [isPressing]);

  const handlePointerDown = () => {
    if (draft || sending || processing || isPressing) return;
    setIsPressing(true);
    onPressStart();
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
          ● 녹음 중 {elapsed.toFixed(1)}/10.0s
        </div>
      )}
      <button
        type="button"
        onPointerDown={handlePointerDown}
        disabled={processing}
        className="w-14 h-14 rounded-full flex items-center justify-center flex-shrink-0 transition-transform select-none disabled:opacity-40"
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
        {processing ? '음성 분석 중…' : isPressing ? '손을 떼면 종료' : '눌러서 녹음'}
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
  const surfaceBg = isDark ? '#3F2E1C' : '#FFFFFF';
  const subtleBg = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(20,19,15,0.025)';
  const textColor = isDark ? '#FFFFFF' : '#14130F';
  const mutedText = isDark ? 'rgba(255,255,255,0.55)' : '#6B5F4F';
  const dividerColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(20,19,15,0.06)';
  return (
    <div
      className="px-10 py-5 flex flex-col gap-4 flex-shrink-0"
      style={{
        borderTop: `0.5px solid ${dividerColor}`,
        background: subtleBg,
      }}
    >
      <div
        className="rounded-[16px] flex flex-col gap-3 px-4 py-3"
        style={{
          background: surfaceBg,
          border: `0.5px solid ${dividerColor}`,
          boxShadow: isDark ? 'none' : '0 1px 4px rgba(20,19,15,0.04)',
        }}
      >
        <div className="flex items-center gap-2">
          <span
            className="px-[10px] py-[3px] rounded-full text-[11px] font-medium tracking-wide"
            style={{ background: palette.main, color: palette.deep }}
          >
            {EMOTION_LABELS[draft.emotion]}
          </span>
          <span className="text-[11px] font-mono" style={{ color: mutedText }}>
            {draft.durationSec.toFixed(1)}s
          </span>
        </div>
        <div className="flex items-center gap-3">
          <AudioPlayer src={draft.audioUrl} palette={palette} isDark={isDark} />
        </div>
      </div>

      <div className="flex flex-col gap-[6px]">
        <label className="text-[11px] tracking-wider font-medium" style={{ color: mutedText }}>
          내가 한 말
        </label>
        <textarea
          value={draft.text}
          onChange={(e) => onTextChange(e.target.value)}
          rows={2}
          className="w-full px-4 py-3 rounded-[14px] text-[14px] leading-relaxed resize-none box-border focus:outline-none transition-colors"
          style={{
            background: surfaceBg,
            color: textColor,
            border: `0.5px solid ${dividerColor}`,
          }}
        />
      </div>

      <div className="flex gap-2 justify-center">
        <button
          type="button"
          onClick={onCancel}
          disabled={sending}
          className="px-4 py-[9px] rounded-[12px] text-[13px] box-border transition-colors disabled:opacity-50"
          style={{
            color: mutedText,
            border: `0.5px solid ${isDark ? 'rgba(255,255,255,0.14)' : 'rgba(20,19,15,0.12)'}`,
          }}
        >
          취소
        </button>
        <button
          type="button"
          onClick={onConfirm}
          disabled={!draft.text.trim() || sending}
          className="px-5 py-[9px] rounded-[12px] text-[13px] font-medium flex items-center gap-2 transition-opacity hover:opacity-90 disabled:opacity-50"
          style={{
            background: isDark ? '#F2D89E' : '#14130F',
            color: isDark ? '#3A2A1A' : '#FAF6EF',
          }}
        >
          {sending ? '전송 중…' : '전송'}
          {!sending && <span aria-hidden>→</span>}
        </button>
      </div>
    </div>
  );
}

function AudioPlayer({
  src,
  palette,
  isDark,
}: {
  src?: string;
  palette: ReturnType<typeof paletteFor>;
  isDark: boolean;
}) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    setPlaying(false);
    setCurrentTime(0);
    setDuration(0);
  }, [src]);

  if (!src) {
    return (
      <div
        className="w-[44px] h-[44px] rounded-full flex items-center justify-center flex-shrink-0"
        style={{ background: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(20,19,15,0.06)' }}
      >
        <span className="text-[10px]" style={{ color: isDark ? 'rgba(255,255,255,0.45)' : '#9B8E7B' }}>
          —
        </span>
      </div>
    );
  }

  const toggle = async () => {
    const el = audioRef.current;
    if (!el) return;
    if (el.paused) {
      try {
        await el.play();
      } catch (err) {
        console.warn('[audio] play failed:', err);
      }
    } else {
      el.pause();
    }
  };

  const onSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    const el = audioRef.current;
    if (!el || !duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    el.currentTime = ratio * duration;
    setCurrentTime(el.currentTime);
  };

  const progress = duration > 0 ? Math.min(1, currentTime / duration) : 0;
  const trackBg = isDark ? 'rgba(255,255,255,0.10)' : 'rgba(20,19,15,0.08)';

  return (
    <>
      <button
        type="button"
        onClick={toggle}
        className="w-[44px] h-[44px] rounded-full flex items-center justify-center flex-shrink-0 transition-transform hover:scale-105 active:scale-100"
        style={{ background: palette.deep }}
        aria-label={playing ? '일시정지' : '재생'}
      >
        {playing ? (
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <rect x="2.5" y="2" width="3" height="10" rx="1" fill="#FFFFFF" />
            <rect x="8.5" y="2" width="3" height="10" rx="1" fill="#FFFFFF" />
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M3 2 L12 7 L3 12 Z" fill="#FFFFFF" />
          </svg>
        )}
      </button>
      <div className="flex-1 flex items-center gap-3 min-w-0">
        <div
          onClick={onSeek}
          className="flex-1 h-[6px] rounded-full cursor-pointer relative"
          style={{ background: trackBg, minWidth: 80 }}
          role="slider"
          aria-valuemin={0}
          aria-valuemax={duration}
          aria-valuenow={currentTime}
        >
          <div
            className="absolute inset-y-0 left-0 rounded-full"
            style={{
              width: `${progress * 100}%`,
              background: palette.deep,
              transition: playing ? 'width 0.1s linear' : 'none',
            }}
          />
        </div>
        <span
          className="text-[11px] font-mono flex-shrink-0"
          style={{ color: isDark ? 'rgba(255,255,255,0.6)' : '#6B5F4F' }}
        >
          {currentTime.toFixed(1)} / {duration.toFixed(1)}
        </span>
      </div>
      <audio
        ref={audioRef}
        key={src}
        src={src}
        preload="auto"
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration || 0)}
        onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        className="hidden"
      />
    </>
  );
}
