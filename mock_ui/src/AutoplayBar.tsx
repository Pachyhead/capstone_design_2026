import { useEffect, useState } from 'react';
import {
  AUTOPLAY_STEPS,
  startAutoplay,
  stopAutoplay,
  togglePause,
  useAutoplay,
} from './autoplay';
import { isPanelOpen, setPanelOpen, usePanelOpen } from './pipeline';

/** 화면 맨 위의 시연 제어 줄: 진행 표시와 시작·일시정지·정지 버튼. */
export function AutoplayBar({ onReset }: { onReset: () => void }) {
  const { status, mode, step, note } = useAutoplay();
  const panelOpen = usePanelOpen();
  const active = status === 'running' || status === 'paused';

  // Space 일시정지/계속, Esc 정지. 자동 시연 중에는 입력칸에 포커스가 있어도 가로챈다.
  useEffect(() => {
    if (!active) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === ' ') {
        e.preventDefault();
        e.stopPropagation();
        togglePause();
      } else if (e.key === 'Escape') {
        e.preventDefault();
        stopAutoplay();
      }
    };
    document.addEventListener('keydown', onKey, true);
    return () => document.removeEventListener('keydown', onKey, true);
  }, [active]);

  // P 키로 파이프라인 패널을 여닫는다 (글자를 입력하는 중에는 제외).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === 'p' || e.key === 'P') setPanelOpen(!isPanelOpen());
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, []);

  return (
    <header className="ap-bar">
      <span className="ap-brand">Tone 오프라인 데모</span>

      <ol className="ap-steps">
        {AUTOPLAY_STEPS[mode].map((label, i) => {
          const n = i + 1;
          const cls =
            status === 'done' || n < step ? 'is-done' : n === step && active ? 'is-active' : '';
          return (
            <li key={label} className={cls}>
              <span className="ap-num">{n}</span>
              {label}
            </li>
          );
        })}
      </ol>

      <span className="ap-note" aria-live="polite">
        {status === 'paused' ? '일시정지 — Space로 계속' : note}
      </span>

      <button
        type="button"
        className="ap-btn ap-quiet"
        onClick={() => setPanelOpen(!panelOpen)}
        title="시스템 구조 그림에서 지금 단계를 보여 줍니다 (P)"
      >
        {panelOpen ? '구조 숨기기' : '구조 보기'}
      </button>

      {active ? (
        <>
          <button type="button" className="ap-btn" onClick={togglePause}>
            {status === 'paused' ? '▶ 계속' : '❚❚ 일시정지'}
          </button>
          <button type="button" className="ap-btn" onClick={stopAutoplay}>
            ■ 정지
          </button>
        </>
      ) : (
        <>
          <button type="button" className="ap-btn ap-quiet" onClick={onReset}>
            초기화
          </button>
          <button type="button" className="ap-btn" onClick={() => startAutoplay('full')}>
            처음부터 시연
          </button>
          <button type="button" className="ap-btn ap-primary" onClick={() => startAutoplay('short')}>
            {status === 'done' ? '↻ 다시 보기' : '▶ 자동 시연'}
          </button>
        </>
      )}
    </header>
  );
}

/** 처음 열었을 때의 시작 화면. */
export function StartOverlay() {
  const [open, setOpen] = useState(true);
  if (!open) return null;

  const begin = (mode: 'short' | 'full') => {
    setOpen(false);
    startAutoplay(mode);
  };

  return (
    <div className="ap-overlay">
      <div className="ap-card">
        <div className="ap-logo">T</div>
        <h1>Tone 오프라인 데모</h1>
        <p>
          시작을 누르면 실제 화면이 저절로 진행됩니다.
          <br />
          녹음 → 전송 → 상대측 재생
        </p>
        <button type="button" className="ap-start" autoFocus onClick={() => begin('short')}>
          ▶ 자동 시연 시작
        </button>
        <button type="button" className="ap-second" onClick={() => begin('full')}>
          처음부터 보기
          <span>프로필 생성 · 목소리 등록 포함</span>
        </button>
        <button type="button" className="ap-manual" onClick={() => setOpen(false)}>
          직접 조작하기
        </button>
        <p className="ap-keys">Space 일시정지 · Esc 정지 · P 시스템 구조 보기</p>
      </div>
    </div>
  );
}

/** 자동 시연 중에는 실수로 화면을 눌러 흐름이 꼬이지 않게 Tone 화면 위를 덮는다. */
export function ClickShield() {
  const { status } = useAutoplay();
  if (status !== 'running' && status !== 'paused') return null;
  return <div className="ap-shield" aria-hidden />;
}
