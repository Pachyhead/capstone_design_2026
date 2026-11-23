import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { EMOTION_LABELS } from '@/types';
import { paletteFor } from '@/tokens/emotions';
import { useProfiles } from '@/hooks/useProfiles';
import { audioUrl } from './mockApi';
import {
  setPanelOpen,
  usePanelOpen,
  usePipeline,
  type NodeId,
  type PipelineState,
} from './pipeline';
import { audioPath, setForUser } from './scenario';

// 포스터의 System Architecture 그림을 그대로 옮긴 패널.
// 지금 진행 중인 단계와 그 단계가 속한 구역이 강조되고, 나머지는 흐려진다.

type Tone = 'plain' | 'text' | 'emotion';

const SENDER_NODES: NodeId[] = ['record', 'stt', 'text', 'emotion2vec', 'fsqEncode', 'bits'];
const SERVER_NODES: NodeId[] = [
  'mysql',
  'fsqRestore',
  'ref',
  'ttsText',
  'ttsSpeaker',
  'ttsEmotion',
  'tts',
];
const RECEIVER_NODES: NodeId[] = ['play'];

function statusOf(state: PipelineState, id: NodeId): string {
  if (state.active.includes(id)) return 'is-active';
  if (state.done.includes(id)) return 'is-done';
  return '';
}

function Node({
  state,
  id,
  label,
  sub,
  tone = 'plain',
  className = '',
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
    <div className={`pl-node pl-${tone} ${statusOf(state, id)} ${className}`}>
      <span className="pl-label">{label}</span>
      {sub && <span className="pl-sub">{sub}</span>}
      {children}
    </div>
  );
}

function Arrow() {
  return <span className="pl-arrow" aria-hidden />;
}

/** 구역 하나(송신 단말 / 서버 / 수신 단말). 지금 단계가 이 안에 있으면 테두리가 켜진다. */
function Box({
  state,
  nodes,
  title,
  className,
  children,
}: {
  state: PipelineState;
  nodes: NodeId[];
  title: string;
  className: string;
  children: ReactNode;
}) {
  const current = state.active.some((n) => nodes.includes(n));
  return (
    <div className={`pl-box ${className} ${current ? 'is-current' : ''}`}>
      <h4>
        {title}
        {current && <span className="pl-now">지금</span>}
      </h4>
      {children}
    </div>
  );
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
  return (
    <div className={`pl-link ${statusOf(state, id)}`}>
      <span className="pl-link-top">{top}</span>
      <span className="pl-link-line" aria-hidden />
      <span className="pl-link-bottom">{bottom}</span>
    </div>
  );
}

/**
 * 감정 벡터의 크기 변화를 보여 주는 도식: 1024d(촘촘한 막대) → 8d(막대 8개) → 24bit(칸 24개).
 *
 * `code`(실제 FSQ 인덱스 8개, 각 0~7)가 있으면 8d 막대와 24bit 칸은 그 값을 그대로 그린다.
 * 없을 때, 그리고 1024d 는 항상, 크기만 맞춘 장식이다(실제 값이 아니다).
 */
function VectorViz({
  kind,
  seed,
  code,
}: {
  kind: '1024' | '8' | '24';
  seed: string;
  code: number[] | null;
}) {
  const real = kind !== '1024' && code !== null && code.length === 8;

  const values = useMemo(() => {
    if (real && kind === '8') return code.map((v) => (v + 1) / 8);
    if (real && kind === '24') {
      // 인덱스 하나가 3bit (상위 비트부터)
      return code.flatMap((v) => [(v >> 2) & 1, (v >> 1) & 1, v & 1]);
    }
    let h = 2166136261;
    for (let i = 0; i < seed.length; i++) h = Math.imul(h ^ seed.charCodeAt(i), 16777619);
    const count = kind === '1024' ? 56 : kind === '8' ? 8 : 24;
    return Array.from({ length: count }, () => {
      h = (h + 0x6d2b79f5) | 0;
      let t = Math.imul(h ^ (h >>> 15), 1 | h);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    });
  }, [kind, seed, code, real]);

  const label = kind === '24' ? '24bit' : `${kind}d`;
  const title = real
    ? `감정 코드 ${label}: ${code.join(' ')}`
    : `감정 벡터 ${label} (크기만 나타낸 도식)`;
  return (
    <span className={`pl-vec pl-vec-${kind}`} key={kind} title={title}>
      <span className="pl-vec-cells" aria-hidden>
        {values.map((v, i) =>
          kind === '24' ? (
            <i key={i} className={v > 0.5 ? 'on' : ''} />
          ) : (
            <i key={i} style={{ height: `${20 + v * 80}%` }} />
          ),
        )}
      </span>
      <span className="pl-vec-label">{real && kind === '8' ? code.join(' ') : label}</span>
    </span>
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

export function PipelinePanel() {
  const state = usePipeline();
  const { activeProfile } = useProfiles();
  const open = usePanelOpen();

  // 기본은 접혀 있다. 위쪽 제어 줄의 "구조 보기"나 P 키로 편다.
  if (!open) return null;

  // 진행 중인 흐름의 화자가 없으면, 지금 로그인한 프로필의 ref 음성을 들려준다.
  const refSetId =
    state.speakerSetId ?? (activeProfile ? (setForUser(activeProfile.backendId)?.id ?? null) : null);
  const refOwner = state.speakerName ?? activeProfile?.name ?? null;
  const emotionPalette = state.emotion ? paletteFor(state.emotion) : null;
  const seed = state.speakerSetId ?? 'tone';
  const busy = state.active.length > 0;

  const emotionChip = state.emotion && emotionPalette && (
    <span className="pl-chip" style={{ background: emotionPalette.main, color: emotionPalette.x }}>
      {EMOTION_LABELS[state.emotion]}
    </span>
  );

  return (
    <section className={`pl ${busy ? 'has-active' : ''}`} aria-label="시스템 파이프라인">
      <header className="pl-head">
        <span className="pl-title">SYSTEM ARCHITECTURE</span>
        <p className="pl-caption" aria-live="polite" key={state.caption}>
          {state.caption}
        </p>
        <button type="button" className="pl-btn" onClick={() => setPanelOpen(false)}>
          숨기기 (P)
        </button>
      </header>

      <div className="pl-body">
        <Box state={state} nodes={SENDER_NODES} title="SENDER · 송신 단말" className="pl-sender">
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
            {state.text ? (
              <span className="pl-value-text" title={state.text}>
                “{state.text}”
              </span>
            ) : (
              <span className="pl-value-text">1024d → 8d 격자 → 24bit</span>
            )}
            {emotionChip}
            {state.vector && <VectorViz kind={state.vector} seed={seed} code={state.code} />}
          </div>
        </Box>

        <Link state={state} id="grpcUp" top="gRPC" bottom="텍스트+24bit" />

        <Box state={state} nodes={SERVER_NODES} title="SERVER · gRPC + MySQL" className="pl-server">
          <div className="pl-server-top">
            <Node state={state} id="mysql" label="MySQL" sub="데이터 저장" />
            <Node
              state={state}
              id="fsqRestore"
              label="FSQ 복원"
              sub="24bit → 1024d"
              tone="emotion"
            >
              {state.restored && <VectorViz kind="1024" seed={seed} code={state.code} />}
            </Node>
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
        </Box>

        <Link state={state} id="grpcDown" top="합성 음성" bottom="gRPC 스트리밍" />

        <Box
          state={state}
          nodes={RECEIVER_NODES}
          title="RECEIVER · 수신 단말"
          className="pl-receiver"
        >
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
        </Box>
      </div>
    </section>
  );
}
