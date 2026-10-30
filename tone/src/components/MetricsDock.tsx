import { useState, type ReactNode } from 'react';
import { api, ApiError, type CodecName, type CodecSizes } from '@/lib/api';
import { setCodecState, useMetrics, type ApiCall } from '@/lib/metrics';
import { EmotionChip } from '@/components/EmotionChip';

interface Props {
  variant?: 'light' | 'dark';
}

const OPEN_KEY = 'tone:metricsDock:open';
const CODECS: CodecName[] = ['mp3', 'aac', 'opus'];

interface Theme {
  card: string;
  text: string;
  hint: string;
  border: string;
  track: string;
  bar: string;
  accent: string;
  toggleBg: string;
  toggleFg: string;
  warn: string;
}

const THEMES: Record<'light' | 'dark', Theme> = {
  light: {
    card: '#FFFFFF',
    text: '#14130F',
    hint: '#9B8E7B',
    border: 'rgba(20,19,15,0.06)',
    track: 'rgba(20,19,15,0.05)',
    bar: '#CFC5B5',
    accent: '#14130F',
    toggleBg: '#14130F',
    toggleFg: '#FAF6EF',
    warn: '#94402C',
  },
  dark: {
    card: '#2D1F13',
    text: '#FAF6EF',
    hint: 'rgba(250,246,239,0.55)',
    border: 'rgba(255,255,255,0.08)',
    track: 'rgba(250,246,239,0.08)',
    bar: '#F2D89E',
    accent: '#FAF6EF',
    toggleBg: '#F2D89E',
    toggleFg: '#3A2A1A',
    warn: '#F2B5A5',
  },
};

export function MetricsDock({ variant = 'light' }: Props) {
  const [open, setOpenState] = useState(() => window.localStorage.getItem(OPEN_KEY) === '1');
  const setOpen = (v: boolean) => {
    setOpenState(v);
    window.localStorage.setItem(OPEN_KEY, v ? '1' : '0');
  };
  const t = THEMES[variant];

  return (
    <div className="fixed right-5 bottom-5 z-30 flex flex-col items-end gap-3">
      {open && <MetricsCard theme={t} />}
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-label={open ? '시스템 지표 닫기' : '시스템 지표 열기'}
        title="시스템 지표"
        className="w-10 h-10 rounded-full flex items-center justify-center transition-transform hover:scale-105"
        style={{
          background: t.toggleBg,
          color: t.toggleFg,
          boxShadow: '0 4px 12px rgba(20,19,15,0.18)',
        }}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
          {open ? (
            <path d="M6 6L18 18M18 6L6 18" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          ) : (
            <path
              d="M3 12H7L10 5L14 19L17 12H21"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}
        </svg>
      </button>
    </div>
  );
}

function MetricsCard({ theme: t }: { theme: Theme }) {
  const { calls, lastRecording, codec, codecState, codecError } = useMetrics();

  const runMeasure = async () => {
    setCodecState({ codecState: 'running', codecError: null });
    try {
      const result = await api.measureCodecSizes();
      setCodecState({ codec: result, codecState: 'idle' });
    } catch (err) {
      setCodecState({ codecState: 'error', codecError: codecErrorMessage(err) });
    }
  };

  return (
    <div
      className="w-[340px] max-h-[60vh] overflow-y-auto rounded-[16px] p-5 flex flex-col gap-5"
      style={{
        background: t.card,
        color: t.text,
        border: `0.5px solid ${t.border}`,
        boxShadow: '0 8px 28px rgba(20,19,15,0.16)',
      }}
    >
      <div className="text-[14px] font-semibold">시스템 지표</div>

      <Section title="최근 녹음" theme={t}>
        {lastRecording ? (
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <EmotionChip emotion={{ primary: lastRecording.emotion }} size="sm" />
              <span className="text-[11px] font-mono" style={{ color: t.hint }}>
                {lastRecording.duration.toFixed(1)}s
                {lastRecording.payload_bytes !== undefined && ` · 전송 ${formatBytes(lastRecording.payload_bytes)}`}
              </span>
            </div>
            <p className="text-[12px] leading-snug line-clamp-2">
              {lastRecording.text || '(인식된 텍스트 없음)'}
            </p>
          </div>
        ) : (
          <Empty theme={t}>아직 녹음이 없어요</Empty>
        )}
      </Section>

      <Section
        title="코덱 비교 (ViSQOL ≥ 4.5)"
        theme={t}
        action={
          <button
            type="button"
            onClick={runMeasure}
            disabled={codecState === 'running'}
            className="px-[10px] py-[4px] rounded-[8px] text-[11px] font-medium flex items-center gap-[6px] disabled:opacity-60"
            style={{ background: t.track, color: t.text }}
          >
            {codecState === 'running' && (
              <span
                className="w-[10px] h-[10px] rounded-full border-[1.5px] border-current"
                style={{ borderTopColor: 'transparent', animation: 'spin 0.8s linear infinite' }}
              />
            )}
            {codecState === 'running' ? '측정 중…' : codec ? '다시 측정' : '측정'}
          </button>
        }
      >
        {codecState === 'error' && (
          <div className="text-[11px]" style={{ color: t.warn }}>
            {codecError}
          </div>
        )}
        {codec ? (
          <CodecBars codec={codec} payloadBytes={lastRecording?.payload_bytes} theme={t} />
        ) : (
          codecState !== 'error' && (
            <Empty theme={t}>최근 WAV 파일로 mp3·aac·opus 크기를 측정해요 (수십 초 소요)</Empty>
          )
        )}
      </Section>

      <Section title="API 응답 시간" theme={t}>
        {calls.length > 0 ? <ApiTimings calls={calls} theme={t} /> : <Empty theme={t}>아직 호출 기록이 없어요</Empty>}
      </Section>
    </div>
  );
}

interface SizeRow {
  name: string;
  size: number | null;
  note: string;
  accent?: boolean;
}

function CodecBars({
  codec,
  payloadBytes,
  theme: t,
}: {
  codec: CodecSizes;
  payloadBytes?: number;
  theme: Theme;
}) {
  const rows: SizeRow[] = [
    ...(codec.wav ? [{ name: 'wav', size: codec.wav.size_bytes, note: '원본' }] : []),
    ...CODECS.map((name) => {
      const c = codec[name];
      return {
        name,
        size: c?.size_bytes ?? null,
        note: c ? `${c.bitrate_kbps} kbps · ViSQOL ${c.quality.toFixed(2)}` : '목표 음질 미달',
      };
    }),
    ...(payloadBytes !== undefined
      ? [{ name: 'tone', size: payloadBytes, note: '텍스트 + 감정 벡터', accent: true }]
      : []),
  ];
  // sizes span B to hundreds of KB, so bar length is log10(size) / log10(max)
  const maxLog = Math.log10(Math.max(...rows.map((r) => r.size ?? 1)));

  return (
    <div className="flex flex-col gap-3">
      {rows.map((r) => (
        <div key={r.name} className="flex flex-col gap-1">
          <div className="flex items-center gap-2 text-[11px]">
            <span className="w-9 font-medium uppercase">{r.name}</span>
            <div className="flex-1 h-[8px] rounded-full overflow-hidden" style={{ background: t.track }}>
              {r.size !== null && (
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${(Math.log10(r.size) / maxLog) * 100}%`,
                    background: r.accent ? t.accent : t.bar,
                  }}
                />
              )}
            </div>
            <span className="w-[56px] text-right font-mono">{r.size !== null ? formatBytes(r.size) : '—'}</span>
          </div>
          <div className="pl-11 text-[10px]" style={{ color: t.hint }}>
            {r.note}
          </div>
        </div>
      ))}
      <div className="text-[10px]" style={{ color: t.hint }}>
        최근 WAV 파일 기준 · 막대 길이는 로그 스케일
      </div>
    </div>
  );
}

function ApiTimings({ calls, theme: t }: { calls: ApiCall[]; theme: Theme }) {
  const summaries = summarize(calls);
  const stopCalls = calls.filter((c) => c.path === '/stop_recording' && isOk(c.status)).slice(-10);
  const maxStop = Math.max(...stopCalls.map((c) => c.ms));

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-[6px]">
        {summaries.map((s) => (
          <div key={s.path} className="flex items-center gap-2 text-[11px]">
            <span className="flex-1 min-w-0 truncate font-mono">{s.path}</span>
            <span
              className="font-mono whitespace-nowrap flex-shrink-0"
              style={{ color: isOk(s.last.status) ? t.text : t.warn }}
            >
              {isOk(s.last.status) ? '' : `${s.last.status || 'ERR'} · `}
              {Math.round(s.last.ms)}ms
            </span>
            <span
              className="w-[100px] text-right font-mono whitespace-nowrap flex-shrink-0"
              style={{ color: t.hint }}
            >
              avg {Math.round(s.avgMs)} · {s.count}회
            </span>
          </div>
        ))}
      </div>

      {stopCalls.length > 0 && (
        <div>
          <div className="text-[10px] mb-[6px]" style={{ color: t.hint }}>
            /stop_recording 최근 {stopCalls.length}회 (≈ 인코딩 지연)
          </div>
          <div className="flex items-end gap-[3px] h-[36px]">
            {stopCalls.map((c, i) => (
              <div
                key={i}
                className="flex-1 rounded-t-[2px]"
                title={`${Math.round(c.ms)}ms`}
                style={{ height: `${Math.max((c.ms / maxStop) * 100, 4)}%`, background: t.bar }}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Section({
  title,
  theme,
  action,
  children,
}: {
  title: string;
  theme: Theme;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-[11px] tracking-wider font-medium" style={{ color: theme.hint }}>
          {title}
        </h3>
        {action}
      </div>
      {children}
    </section>
  );
}

function Empty({ theme, children }: { theme: Theme; children: ReactNode }) {
  return (
    <div className="text-[12px]" style={{ color: theme.hint }}>
      {children}
    </div>
  );
}

interface PathSummary {
  path: string;
  last: ApiCall;
  avgMs: number;
  count: number;
}

// most recently called path first
function summarize(calls: ApiCall[]): PathSummary[] {
  const byPath = new Map<string, ApiCall[]>();
  for (const c of calls) {
    const list = byPath.get(c.path);
    if (list) list.push(c);
    else byPath.set(c.path, [c]);
  }
  return [...byPath.entries()]
    .map(([path, list]) => ({
      path,
      last: list[list.length - 1],
      avgMs: list.reduce((sum, c) => sum + c.ms, 0) / list.length,
      count: list.length,
    }))
    .sort((a, b) => b.last.at - a.last.at);
}

function isOk(status: number): boolean {
  return status >= 200 && status < 300;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(1)} KB`;
}

function codecErrorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    return err.status === 404
      ? '저장된 WAV 파일이 없어요. 먼저 녹음해 주세요.'
      : `측정에 실패했어요 (HTTP ${err.status})`;
  }
  return '서버에 연결할 수 없어요';
}
