import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { paletteFor } from '@/tokens/emotions';
import { EMOTION_LABELS } from '@/types';
import type { Emotion } from '@/types';
import { useVoiceProfile } from '@/hooks/useVoiceProfile';
import { api } from '@/lib/api';

const REFERENCE_PROMPT =
  '안녕하세요, 오늘 날씨가 참 좋아서 산책하기 딱 좋은 날인 것 같네요. 주말에는 보통 집에서 책을 읽거나 가까운 카페에 가서 시간을 보냅니다.';
const REFERENCE_EMOTION: Emotion = 'neutral';
const LIVE_BARS = 36;

export function Onboarding() {
  const [step, setStep] = useState<'welcome' | 'recording' | 'done'>('welcome');
  const navigate = useNavigate();
  const [profile, setProfile] = useVoiceProfile();

  const showClose = step !== 'done' && profile.registered;
  const handleClose = () => navigate('/');

  if (step === 'welcome')
    return (
      <Page onClose={showClose ? handleClose : undefined}>
        <Welcome onStart={() => setStep('recording')} />
      </Page>
    );

  if (step === 'recording') {
    const finish = (registered: boolean, durationSec: number) => {
      setProfile({
        registered,
        sentenceCount: registered ? 1 : 0,
        durationSec,
        emotionCoverage: registered ? 1 : 0,
        detectedEmotions: registered ? [REFERENCE_EMOTION] : [],
      });
      setStep('done');
    };

    return (
      <Page onClose={showClose ? handleClose : undefined}>
        <Recording
          onNext={(durationSec) => finish(true, durationSec)}
          onSkip={() => finish(false, 0)}
        />
      </Page>
    );
  }

  return (
    <Page>
      <Completion profile={profile} onFinish={() => navigate('/')} />
    </Page>
  );
}

// -----------------------------------------------------------------------------
function Page({
  children,
  onClose,
}: {
  children: React.ReactNode;
  onClose?: () => void;
}) {
  return (
    <div className="min-h-screen w-screen bg-cream flex items-center justify-center p-6">
      <div
        className="w-full max-w-[640px] bg-white rounded-[20px] p-12 relative"
        style={{ boxShadow: '0 4px 32px rgba(20,19,15,0.08)' }}
      >
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="absolute top-5 right-5 w-9 h-9 rounded-full flex items-center justify-center text-[20px] leading-none text-muted hover:bg-cream transition-colors"
            aria-label="닫기"
          >
            ×
          </button>
        )}
        {children}
      </div>
    </div>
  );
}

// -----------------------------------------------------------------------------
function Welcome({ onStart }: { onStart: () => void }) {
  return (
    <div className="flex flex-col items-center text-center">
      <div
        className="w-44 h-44 rounded-full flex items-center justify-center mb-8"
        style={{ background: '#FBF1D6' }}
      >
        <div className="flex items-center gap-[3px] h-20">
          {[35, 60, 85, 50, 95, 70, 80, 45, 65, 90, 55, 75].map((h, i) => (
            <div
              key={i}
              className="w-[5px] rounded-[2px]"
              style={{ height: `${h}%`, background: '#F2D89E' }}
            />
          ))}
        </div>
      </div>

      <h1 className="text-[28px] font-semibold text-ink leading-tight mb-3">
        당신의 목소리로 도착합니다
      </h1>
      <p className="text-[15px] text-muted leading-relaxed max-w-[440px] mb-3">
        한 문장을 녹음하면, 상대방이 받는 메시지가 당신의 목소리로 재구성됩니다.
      </p>
      <p className="text-[13px] text-hint leading-relaxed max-w-[440px] mb-10">
        음성 데이터는 당신의 기기에서만 사용되며, 서버로 전송되지 않습니다.
      </p>

      <button
        onClick={onStart}
        className="w-full max-w-[320px] py-[14px] rounded-[14px] text-[15px] font-medium bg-charcoal text-white hover:opacity-90 transition-opacity"
      >
        시작하기
      </button>
    </div>
  );
}

// -----------------------------------------------------------------------------
function Recording({
  onNext,
  onSkip,
}: {
  onNext: (durationSec: number) => void;
  onSkip: () => void;
}) {
  const [phase, setPhase] = useState<'idle' | 'recording' | 'reviewing'>('idle');
  const [tick, setTick] = useState(0); // tenths of a second
  const [energies, setEnergies] = useState<number[]>(() =>
    Array.from({ length: LIVE_BARS }, () => 0.18),
  );

  useEffect(() => {
    if (phase !== 'recording') return;
    const interval = setInterval(() => {
      setTick((t) => t + 1);
      setEnergies((buf) => [...buf.slice(1), Math.random() * 0.65 + 0.2]);
    }, 100);
    return () => clearInterval(interval);
  }, [phase]);

  const elapsedSec = tick / 10;
  const timerStr = `${elapsedSec.toFixed(1)}s`;
  const palette = paletteFor(REFERENCE_EMOTION);

  const handleStart = () => {
    setPhase('recording');
    api.startRecording().catch((err) => console.warn('[api] startRecording failed:', err));
  };

  const handleStop = () => {
    setPhase('reviewing');
    api.stopRecording().catch((err) => console.warn('[api] stopRecording failed:', err));
  };

  const handleRetake = () => {
    setPhase('idle');
    setTick(0);
    setEnergies(Array.from({ length: LIVE_BARS }, () => 0.18));
  };

  return (
    <div className="flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <p className="text-[12px] text-hint tracking-wider font-medium">읽어주세요</p>
        <button
          onClick={onSkip}
          className="text-[13px] text-hint hover:text-muted transition-colors"
        >
          건너뛰기
        </button>
      </div>

      <div className="bg-cream rounded-[14px] p-6 mb-6">
        <p className="text-[18px] text-ink leading-relaxed font-medium">"{REFERENCE_PROMPT}"</p>
      </div>

      {phase === 'idle' && (
        <div className="flex flex-col items-center justify-center min-h-[180px] gap-4 mb-6">
          <button
            onClick={handleStart}
            className="w-20 h-20 rounded-full flex items-center justify-center transition-transform hover:scale-105 active:scale-100"
            style={{ background: '#94402C' }}
            aria-label="녹음 시작"
          >
            <span className="w-7 h-7 rounded-full" style={{ background: '#FFFFFF' }} />
          </button>
          <p className="text-[12px] text-hint">탭해서 녹음 시작</p>
        </div>
      )}

      {phase === 'recording' && (
        <div className="flex flex-col items-center min-h-[180px] gap-4 mb-6">
          <div
            className="flex items-center gap-[2px] w-full h-[80px] rounded-[10px] px-3"
            style={{ background: 'rgba(20,19,15,0.03)' }}
            aria-label="녹음 파형"
          >
            {energies.map((e, i) => (
              <div
                key={i}
                className="flex-1 rounded-[1px] transition-all duration-100"
                style={{
                  height: `${Math.max(e * 100, 8)}%`,
                  background: palette.main,
                }}
              />
            ))}
          </div>
          <div className="font-mono text-[22px] text-ink">{timerStr}</div>
          <div className="flex items-center gap-2 text-[12px] text-muted">
            <span className="relative flex w-[8px] h-[8px]">
              <span
                className="absolute inset-0 rounded-full animate-ping"
                style={{ background: '#94402C', opacity: 0.6 }}
              />
              <span className="relative w-full h-full rounded-full" style={{ background: '#94402C' }} />
            </span>
            <span>녹음 중</span>
          </div>
        </div>
      )}

      {phase === 'reviewing' && (
        <div className="flex flex-col items-center min-h-[180px] gap-4 mb-6">
          <div
            className="flex items-center gap-[2px] w-full h-[80px] rounded-[10px] px-3"
            style={{ background: palette.light }}
          >
            {energies.map((e, i) => (
              <div
                key={i}
                className="flex-1 rounded-[1px]"
                style={{
                  height: `${Math.max(e * 100, 8)}%`,
                  background: palette.main,
                }}
              />
            ))}
          </div>
          <div className="flex items-center gap-2">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" fill="#DEF0E8" />
              <path d="M8 12 L11 15 L16 9" stroke="#2D6852" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span className="text-[13px] font-medium" style={{ color: palette.deep }}>
              {EMOTION_LABELS[REFERENCE_EMOTION]} 톤 등록됨
            </span>
          </div>
          <div className="font-mono text-[14px] text-hint">{timerStr}</div>
        </div>
      )}

      <div className="flex gap-3">
        {phase === 'idle' && (
          <button
            onClick={onSkip}
            className="flex-1 py-[13px] rounded-[14px] text-[14px] font-medium text-muted box-border hover:bg-cream transition-colors"
            style={{ border: '0.5px solid rgba(20,19,15,0.12)' }}
          >
            건너뛰기
          </button>
        )}

        {phase === 'recording' && (
          <button
            onClick={handleStop}
            className="flex-1 py-[13px] rounded-[14px] text-[14px] font-medium bg-charcoal text-white hover:opacity-90 transition-opacity flex items-center justify-center gap-2"
          >
            <span className="w-3 h-3 rounded-[2px] bg-white" />
            정지
          </button>
        )}

        {phase === 'reviewing' && (
          <>
            <button
              onClick={handleRetake}
              className="flex-1 py-[13px] rounded-[14px] text-[14px] font-medium text-muted box-border hover:bg-cream transition-colors"
              style={{ border: '0.5px solid rgba(20,19,15,0.12)' }}
            >
              다시 녹음
            </button>
            <button
              onClick={() => onNext(elapsedSec)}
              className="flex-1 py-[13px] rounded-[14px] text-[14px] font-medium bg-charcoal text-white hover:opacity-90 transition-opacity"
            >
              완료
            </button>
          </>
        )}
      </div>
    </div>
  );
}

// -----------------------------------------------------------------------------
function Completion({
  profile,
  onFinish,
}: {
  profile: import('@/types').VoiceProfile;
  onFinish: () => void;
}) {
  const detected = profile.detectedEmotions.length > 0
    ? profile.detectedEmotions
    : (['neutral'] as Emotion[]);

  const sampleLength = `${profile.durationSec.toFixed(1)}s`;
  const noneRecorded = profile.sentenceCount === 0;

  return (
    <div className="flex flex-col items-center text-center">
      <div
        className="w-20 h-20 rounded-full flex items-center justify-center mb-6"
        style={{ background: noneRecorded ? '#F2E5C9' : '#DEF0E8' }}
      >
        {noneRecorded ? (
          <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
            <circle cx="16" cy="16" r="13" stroke="#94402C" strokeWidth="2.5" />
            <path d="M16 9V18" stroke="#94402C" strokeWidth="2.5" strokeLinecap="round" />
            <circle cx="16" cy="22" r="1.4" fill="#94402C" />
          </svg>
        ) : (
          <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
            <path
              d="M8 16 L14 22 L24 10"
              stroke="#2D6852"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        )}
      </div>

      <h1 className="text-[24px] font-semibold text-ink leading-tight mb-3">
        {noneRecorded ? '등록된 음성이 없어요' : '목소리 등록 완료'}
      </h1>
      <p className="text-[14px] text-muted leading-relaxed max-w-[440px] mb-8">
        {noneRecorded
          ? '문장을 건너뛰셨어요. 메시지를 보내려면 목소리 등록이 필요합니다.'
          : '이제 당신이 보낸 메시지는 상대방에게 당신의 목소리로 도착합니다.'}
      </p>

      {!noneRecorded && (
        <div className="bg-cream rounded-[14px] p-6 w-full text-left mb-8">
          <p className="text-[12px] text-hint tracking-wider mb-4 font-medium">Voice ID</p>

          <InfoRow label="샘플 길이" value={sampleLength} />

          <div
            className="flex gap-3 items-center pt-4 mt-4"
            style={{ borderTop: '0.5px solid rgba(20,19,15,0.06)' }}
          >
            <span className="text-[12px] text-hint flex-shrink-0">등록된 톤</span>
            <div className="flex gap-[6px] flex-1 items-center justify-end flex-wrap">
              {detected.map((e, i) => {
                const size = Math.max(16 - i, 9);
                return (
                  <div
                    key={e}
                    className="rounded-full flex-shrink-0"
                    style={{
                      width: `${size}px`,
                      height: `${size}px`,
                      background: paletteFor(e).main,
                    }}
                    title={EMOTION_LABELS[e]}
                  />
                );
              })}
            </div>
          </div>
        </div>
      )}

      <button
        onClick={onFinish}
        className="w-full max-w-[320px] py-[14px] rounded-[14px] text-[15px] font-medium bg-charcoal text-white hover:opacity-90 transition-opacity"
      >
        시작하기
      </button>
    </div>
  );
}

function InfoRow({
  label,
  value,
  divider = false,
}: {
  label: string;
  value: string;
  divider?: boolean;
}) {
  return (
    <div
      className="flex justify-between text-[14px] text-ink py-[10px]"
      style={divider ? { borderTop: '0.5px solid rgba(20,19,15,0.06)' } : undefined}
    >
      <span className="text-muted">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  );
}
