import { useEffect, useRef, useState, type ReactNode } from 'react';
import { EMOTION_LABELS } from '@/types';
import { paletteFor } from '@/tokens/emotions';
import { useProfiles } from '@/hooks/useProfiles';
import { audioUrl } from './mockApi';
import { usePipeline, type NodeId, type PipelineState } from './pipeline';
import { audioPath, setForUser } from './scenario';

// 포스터의 System Architecture 그림을 그대로 옮긴 패널.
// 지금 진행 중인 단계가 켜지고, 지나간 단계는 옅게 남는다.

type Tone = 'plain' | 'text' | 'emotion';

function nodeClass(state: PipelineState, id: NodeId, tone: Tone, extra = ''): string {
  const status = state.active.includes(id) ? 'is-active' : state.done.includes(id) ? 'is-done' : '';
  return `pl-node pl-${tone} ${status} ${extra}`.trim();
}

function Node({
  state,
  id,
  label,
  sub,
  tone = 'plain',
  className,
  children,
}: {
  state: PipelineState;
  id: NodeId;
  label: string;
  sub?: string;
  tone?: Tone;
  className?: string;
  children?: ReactNode;
}) {
  return (
    <div className={nodeClass(state, id, tone, className)}>
      <span className="pl-label">{label}</span>
      {sub && <span className="pl-sub">{sub}</span>}
      {children}
    </div>
  );
}

function Arrow() {
  return <span className="pl-arrow" aria-hidden />;
}

/** 단말과 서버 사이의 전송 구간. 진행 중이면 점선이 흐른다. */
function Link({
  state,
  id,
  top,
  bottom,
}: {
  state: PipelineState;
  id: NodeId;
  top: string;
  bottom: string;
}) {
  const status = state.active.includes(id) ? 'is-active' : state.done.includes(id) ? 'is-done' : '';
  return (
    <div className={`pl-link ${status}`}>
      <span className="pl-link-top">{top}</span>
      <span className="pl-link-line" aria-hidden />
      <span className="pl-link-bottom">{bottom}</span>
    </div>
  );
}

/** 화자가 등록해 둔 ref 음성을 들려준다. */
function ReferenceButton({ setId }: { setId: string | null }) {
  const [playing, setPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const stop = () => {
    audioRef.current?.pause();
    audioRef.current = null;
    setPlaying(false);
  };

  // 화자가 바뀌거나 패널이 사라지면 재생을 멈춘다.
  useEffect(() => stop, [setId]);

  if (!setId) return null;

  const toggle = () => {
    if (playing) {
      stop();
      return;
    }
    const audio = new Audio(audioUrl(audioPath(setId, 'reference')));
    audioRef.current = audio;
    audio.addEventListener('ended', stop);
    audio.addEventListener('error', stop);
    setPlaying(true);
    audio.play().catch(stop);
  };

  return (
    <button type="button" className="pl-listen" onClick={toggle}>
      {playing ? '■ 정지' : '▶ 듣기'}
    </button>
  );
}

export function PipelinePanel({ onReset }: { onReset: () => void }) {
  const state = usePipeline();
  const { activeProfile } = useProfiles();
  const [open, setOpen] = useState(true);

  // P 키로 패널을 여닫는다 (글자를 입력하는 중에는 제외).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === 'p' || e.key === 'P') setOpen((o) => !o);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, []);

  if (!open) {
    return (
      <button type="button" className="pl-reopen" onClick={() => setOpen(true)}>
        파이프라인 보기 (P)
      </button>
    );
  }

  // 진행 중인 흐름의 화자가 없으면, 지금 로그인한 프로필의 ref 음성을 들려준다.
  const refSetId =
    state.speakerSetId ?? (activeProfile ? (setForUser(activeProfile.backendId)?.id ?? null) : null);
  const refOwner = state.speakerName ?? activeProfile?.name ?? null;
  const emotionPalette = state.emotion ? paletteFor(state.emotion) : null;

  return (
    <section className="pl" aria-label="시스템 파이프라인">
      <header className="pl-head">
        <span className="pl-title">SYSTEM ARCHITECTURE</span>
        <p className="pl-caption" aria-live="polite">
          {state.caption}
        </p>
        <button type="button" className="pl-btn" onClick={onReset}>
          처음부터
        </button>
        <button type="button" className="pl-btn" onClick={() => setOpen(false)}>
          숨기기 (P)
        </button>
      </header>

      <div className="pl-body">
        <div className="pl-box pl-sender">
          <h4>SENDER · 송신 단말</h4>
          <div className="pl-flow">
            <Node state={state} id="record" label="녹음" />
            <div className="pl-branches">
              <div className="pl-row">
                <Arrow />
                <Node state={state} id="stt" label="Whisper STT" />
                <Arrow />
                <Node state={state} id="text" label="텍스트 (가변)" tone="text" />
              </div>
              <div className="pl-row">
                <Arrow />
                <Node state={state} id="emotion2vec" label="emotion2vec" />
                <Arrow />
                <Node state={state} id="fsqEncode" label="FSQ 인코더" tone="emotion" />
                <Arrow />
                <Node state={state} id="bits" label="24bit" tone="emotion" />
              </div>
            </div>
          </div>
          <div className="pl-values">
            <span className="pl-value-text" title={state.text ?? undefined}>
              {state.text ? `“${state.text}”` : '1024d → 8d 격자 → 24bit'}
            </span>
            {state.emotion && emotionPalette && (
              <span
                className="pl-chip"
                style={{ background: emotionPalette.main, color: emotionPalette.x }}
              >
                {EMOTION_LABELS[state.emotion]}
              </span>
            )}
          </div>
        </div>

        <Link state={state} id="grpcUp" top="gRPC" bottom="텍스트+24bit" />

        <div className="pl-box pl-server">
          <h4>SERVER · gRPC + MySQL</h4>
          <div className="pl-server-top">
            <Node state={state} id="mysql" label="MySQL" sub="데이터 저장" />
            <Node
              state={state}
              id="fsqRestore"
              label="FSQ 복원"
              sub="24bit → 1024d"
              tone="emotion"
            />
            <Node
              state={state}
              id="ref"
              label="화자 ref 음성"
              sub={refOwner ? `${refOwner} · 사전 등록` : '사전 등록 · 보관'}
            >
              <ReferenceButton setId={refSetId} />
            </Node>
          </div>
          <div className="pl-gpu">
            <h5>API 호출 · 별도 GPU 도커</h5>
            <div className="pl-gpu-flow">
              <div className="pl-inputs">
                <Node state={state} id="ttsText" label="텍스트" />
                <Node state={state} id="ttsSpeaker" label="화자 임베딩 2048d" />
                <Node state={state} id="ttsEmotion" label="감정 1024d" />
              </div>
              <Arrow />
              <Node
                state={state}
                id="tts"
                label="Qwen3-TTS"
                sub="텍스트·화자·감정 → 음성 합성"
                tone="emotion"
                className="pl-tts"
              />
            </div>
          </div>
        </div>

        <Link state={state} id="grpcDown" top="합성 음성" bottom="gRPC 스트리밍" />

        <div className="pl-box pl-receiver">
          <h4>RECEIVER · 수신 단말</h4>
          <Node
            state={state}
            id="play"
            label="합성 음성 재생"
            sub="원 화자 목소리 + 전달된 감정으로"
            className="pl-speaker"
          >
            <svg width="34" height="34" viewBox="0 0 24 24" fill="none" aria-hidden>
              <path d="M4 9.5v5h3.5L12 18.5v-13L7.5 9.5H4Z" fill="currentColor" />
              <path
                d="M15 9a4 4 0 0 1 0 6M17.5 6.5a7.5 7.5 0 0 1 0 11"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
              />
            </svg>
          </Node>
        </div>
      </div>
    </section>
  );
}
