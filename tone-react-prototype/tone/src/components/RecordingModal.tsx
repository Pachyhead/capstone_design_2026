import { useEffect, useState } from 'react';
import { paletteFor } from '@/tokens/emotions';
import { EMOTION_LABELS } from '@/types';
import type { Conversation, Emotion } from '@/types';
import { pickScenario, type RecordingScenario } from '@/data/recordingScenarios';
import { api } from '@/lib/api';

export interface RecordingResult {
  text: string;
  emotion: Emotion;
  nuance?: string;
  durationSec: number;
  energy: number[]; // 10 normalized values
}

interface Props {
  conversation: Conversation;
  onClose: () => void;
  onSend: (r: RecordingResult) => void;
  variant?: 'light' | 'dark';
}

const LIVE_BARS = 48;
const TICK_MS = 100;

export function RecordingModal({ conversation, onClose, onSend, variant = 'light' }: Props) {
  const [phase, setPhase] = useState<'recording' | 'reviewing'>('recording');
  const [scenario, setScenario] = useState<RecordingScenario>(() => pickScenario());
  const [tick, setTick] = useState(0); // tenths of a second
  const [energies, setEnergies] = useState<number[]>(() =>
    Array.from({ length: LIVE_BARS }, () => 0.18),
  );
  const [editedText, setEditedText] = useState('');
  const [busy, setBusy] = useState<null | 'recording' | 'sending'>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [onClose]);

  useEffect(() => {
    if (phase !== 'recording') return;
    const interval = setInterval(() => {
      setTick((t) => t + 1);
      setEnergies((buf) => [...buf.slice(1), Math.random() * 0.65 + 0.2]);
    }, TICK_MS);
    return () => clearInterval(interval);
  }, [phase]);

  // emotion detection "kicks in" after 1.5s, before that show 감지 중
  const detecting = tick < 15;
  const liveEmotion: Emotion = detecting ? 'neutral' : scenario.emotion;

  const seconds = Math.floor(tick / 10);
  const tenths = tick % 10;
  const timerStr = `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}.${tenths}`;

  const handleStop = async () => {
    const durationSec = Math.max(1, Math.round(tick / 10));
    setBusy('recording');
    setError(null);
    try {
      await api.record(durationSec);
    } catch (err) {
      console.warn('[api] record failed:', err);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
    setEditedText(scenario.text);
    setPhase('reviewing');
  };

  const handleRedo = () => {
    setScenario((prev) => pickScenario(prev));
    setTick(0);
    setEnergies(Array.from({ length: LIVE_BARS }, () => 0.18));
    setEditedText('');
    setPhase('recording');
  };

  const handleSend = async () => {
    if (!editedText.trim()) return;
    setBusy('sending');
    setError(null);
    try {
      await api.send();
    } catch (err) {
      console.warn('[api] send failed:', err);
      setError(err instanceof Error ? err.message : String(err));
      setBusy(null);
      return;
    }
    setBusy(null);
    onSend({
      text: editedText.trim(),
      emotion: scenario.emotion,
      nuance: scenario.nuance,
      durationSec: scenario.durationSec,
      energy: makeStaticEnergy(),
    });
  };

  const isDark = variant === 'dark';
  const cardBg = isDark ? '#48341F' : '#FFFFFF';
  const fg = isDark ? '#FFFFFF' : '#14130F';
  const muted = isDark ? 'rgba(255,255,255,0.6)' : '#6B5F4F';
  const hint = isDark ? 'rgba(255,255,255,0.4)' : '#9B8E7B';
  const divider = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(20,19,15,0.06)';
  const subtleBg = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(20,19,15,0.03)';

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center px-6"
      style={{ background: 'rgba(0,0,0,0.45)', backdropFilter: 'blur(4px)' }}
      onMouseDown={(e) => {
        // click backdrop only (not card) closes
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="w-full max-w-[560px] rounded-[20px] overflow-hidden"
        style={{
          background: cardBg,
          boxShadow: '0 20px 60px rgba(0,0,0,0.35)',
          border: `0.5px solid ${divider}`,
        }}
      >
        <header
          className="flex items-center justify-between px-7 py-4"
          style={{ borderBottom: `0.5px solid ${divider}` }}
        >
          <div className="flex items-center gap-3">
            {phase === 'recording' ? (
              <>
                <span className="relative flex w-[10px] h-[10px]">
                  <span
                    className="absolute inset-0 rounded-full animate-ping"
                    style={{ background: '#94402C', opacity: 0.6 }}
                  />
                  <span className="relative w-full h-full rounded-full" style={{ background: '#94402C' }} />
                </span>
                <span className="text-[14px] font-semibold" style={{ color: fg }}>
                  녹음 중
                </span>
              </>
            ) : (
              <span className="text-[14px] font-semibold" style={{ color: fg }}>
                미리듣기
              </span>
            )}
            <span className="text-[12px]" style={{ color: hint }}>
              · {conversation.name}에게
            </span>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-[20px] leading-none px-2"
            style={{ color: muted }}
            aria-label="닫기"
          >
            ×
          </button>
        </header>

        {phase === 'recording' ? (
          <div className="px-7 py-7 flex flex-col items-center gap-6">
            <LiveWaveform energies={energies} emotion={liveEmotion} isDark={isDark} />

            <div className="font-mono text-[26px] tracking-wide" style={{ color: fg }}>
              {timerStr}
            </div>

            <div className="flex flex-col items-center gap-2">
              <span className="text-[11px] tracking-wider font-medium" style={{ color: hint }}>
                감지된 감정
              </span>
              {detecting ? (
                <div
                  className="flex items-center gap-2 px-4 py-[6px] rounded-[10px]"
                  style={{ background: subtleBg, color: muted }}
                >
                  <span className="w-[6px] h-[6px] rounded-full animate-pulse" style={{ background: muted }} />
                  <span className="text-[13px]">감지 중...</span>
                </div>
              ) : (
                <EmotionTag emotion={scenario.emotion} nuance={scenario.nuance} />
              )}
            </div>

            <div
              className="text-[11px] text-center px-4 py-[6px] rounded-[8px]"
              style={{
                background: subtleBg,
                color: hint,
              }}
            >
              당신의 등록된 목소리로 전송됩니다
            </div>
          </div>
        ) : (
          <div className="px-7 py-7 flex flex-col gap-5">
            <ReviewWaveform scenario={scenario} isDark={isDark} />

            <EmotionTagCenter emotion={scenario.emotion} nuance={scenario.nuance} />

            <div>
              <label
                className="text-[11px] tracking-wider font-medium mb-2 block"
                style={{ color: hint }}
              >
                자동 변환된 텍스트
              </label>
              <textarea
                value={editedText}
                onChange={(e) => setEditedText(e.target.value)}
                rows={2}
                className="w-full px-4 py-3 rounded-[12px] text-[14px] resize-none box-border focus:outline-none transition-colors"
                style={{
                  background: subtleBg,
                  color: fg,
                  border: `0.5px solid ${divider}`,
                }}
              />
            </div>
          </div>
        )}

        <footer
          className="flex gap-3 px-7 py-5"
          style={{ borderTop: `0.5px solid ${divider}` }}
        >
          {phase === 'recording' ? (
            <>
              <button
                type="button"
                onClick={onClose}
                className="flex-1 py-[12px] rounded-[12px] text-[14px] font-medium transition-colors box-border"
                style={{
                  color: muted,
                  border: `0.5px solid ${isDark ? 'rgba(255,255,255,0.12)' : 'rgba(20,19,15,0.12)'}`,
                }}
              >
                취소
              </button>
              <button
                type="button"
                onClick={handleStop}
                disabled={busy !== null}
                className="flex-1 py-[12px] rounded-[12px] text-[14px] font-medium flex items-center justify-center gap-2 transition-opacity hover:opacity-90 disabled:opacity-60"
                style={{
                  background: isDark ? '#F2D89E' : '#14130F',
                  color: isDark ? '#3A2A1A' : '#FFFFFF',
                }}
              >
                <span className="w-[10px] h-[10px] rounded-[2px]" style={{ background: 'currentColor' }} />
                {busy === 'recording' ? '처리 중…' : '정지'}
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                onClick={handleRedo}
                className="flex-1 py-[12px] rounded-[12px] text-[14px] font-medium transition-colors box-border"
                style={{
                  color: muted,
                  border: `0.5px solid ${isDark ? 'rgba(255,255,255,0.12)' : 'rgba(20,19,15,0.12)'}`,
                }}
              >
                다시 녹음
              </button>
              <button
                type="button"
                onClick={handleSend}
                disabled={!editedText.trim() || busy !== null}
                className="flex-1 py-[12px] rounded-[12px] text-[14px] font-medium flex items-center justify-center gap-2 transition-opacity hover:opacity-90 disabled:opacity-50"
                style={{
                  background: isDark ? '#F2D89E' : '#14130F',
                  color: isDark ? '#3A2A1A' : '#FFFFFF',
                }}
              >
                전송
                <span>→</span>
              </button>
            </>
          )}
        </footer>
      </div>
    </div>
  );
}

// -----------------------------------------------------------------------------
function LiveWaveform({
  energies,
  emotion,
  isDark,
}: {
  energies: number[];
  emotion: Emotion;
  isDark: boolean;
}) {
  const color = paletteFor(emotion).main;
  return (
    <div
      className="flex items-center gap-[2px] w-full h-[100px] rounded-[10px] px-3"
      style={{ background: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(20,19,15,0.03)' }}
      aria-label="녹음 파형"
    >
      {energies.map((e, i) => (
        <div
          key={i}
          className="flex-1 rounded-[1px] transition-all duration-100"
          style={{
            height: `${Math.max(e * 100, 8)}%`,
            background: color,
            opacity: isDark ? 0.85 : 1,
          }}
        />
      ))}
    </div>
  );
}

function ReviewWaveform({ scenario, isDark }: { scenario: RecordingScenario; isDark: boolean }) {
  const palette = paletteFor(scenario.emotion);
  // deterministic-ish energies based on scenario (different per scenario)
  const energies = STATIC_PATTERNS[scenario.emotion];

  return (
    <div className="flex items-center gap-3 px-4 py-3 rounded-[14px]" style={{ background: palette.light }}>
      <button
        type="button"
        className="w-12 h-12 rounded-full flex items-center justify-center flex-shrink-0 transition-transform hover:scale-105 active:scale-100"
        style={{ background: palette.deep }}
        aria-label="재생"
      >
        <svg width="14" height="14" viewBox="0 0 10 10" fill="none">
          <path d="M2 1 L9 5 L2 9 Z" fill="#FFFFFF" />
        </svg>
      </button>
      <div className="flex items-center gap-[2px] flex-1 h-[40px]">
        {energies.map((e, i) => (
          <div
            key={i}
            className="flex-1 rounded-[1px]"
            style={{
              height: `${Math.max(e * 100, 12)}%`,
              background: palette.main,
              opacity: isDark ? 0.85 : 1,
            }}
          />
        ))}
      </div>
      <span
        className="text-[12px] font-mono flex-shrink-0"
        style={{ color: palette.deep, opacity: 0.7 }}
      >
        {scenario.durationSec.toFixed(1)}s
      </span>
    </div>
  );
}

function EmotionTag({ emotion, nuance }: { emotion: Emotion; nuance?: string }) {
  const palette = paletteFor(emotion);
  const text = nuance ? `${EMOTION_LABELS[emotion]} · ${nuance}` : EMOTION_LABELS[emotion];
  return (
    <span
      className="px-4 py-[6px] rounded-[10px] text-[14px] font-medium"
      style={{ background: palette.main, color: palette.deep }}
    >
      {text}
    </span>
  );
}

function EmotionTagCenter({ emotion, nuance }: { emotion: Emotion; nuance?: string }) {
  return (
    <div className="flex justify-center">
      <EmotionTag emotion={emotion} nuance={nuance} />
    </div>
  );
}

// -----------------------------------------------------------------------------
const STATIC_PATTERNS: Record<Emotion, number[]> = {
  happy: [0.4, 0.7, 0.85, 0.6, 0.9, 0.75, 0.8, 0.55, 0.65, 0.5],
  sad: [0.25, 0.4, 0.35, 0.45, 0.3, 0.4, 0.35, 0.3, 0.35, 0.25],
  surprised: [0.3, 0.55, 0.9, 0.95, 0.75, 0.85, 0.6, 0.5, 0.4, 0.3],
  fearful: [0.35, 0.5, 0.3, 0.55, 0.4, 0.5, 0.35, 0.45, 0.3, 0.4],
  angry: [0.5, 0.75, 0.9, 0.7, 0.85, 0.95, 0.8, 0.7, 0.65, 0.55],
  neutral: [0.45, 0.55, 0.5, 0.6, 0.5, 0.55, 0.5, 0.45, 0.55, 0.45],
  disgusted: [0.4, 0.55, 0.45, 0.6, 0.5, 0.45, 0.55, 0.4, 0.5, 0.35],
  unk: [0.4, 0.55, 0.5, 0.65, 0.45, 0.55, 0.5, 0.45, 0.55, 0.4],
  other: [0.4, 0.5, 0.45, 0.55, 0.5, 0.45, 0.55, 0.45, 0.5, 0.4],
};

function makeStaticEnergy(): number[] {
  return Array.from({ length: 10 }, () => Math.random() * 0.5 + 0.4);
}
