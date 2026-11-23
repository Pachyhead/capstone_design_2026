import { useEffect, useState } from 'react';
import {
  AUTOPLAY_STEPS,
  startAutoplay,
  stopAutoplay,
  togglePause,
  useAutoplay,
} from './autoplay';

/** 화면 맨 위의 시연 제어 줄: 4단계 진행 표시와 시작·일시정지·정지 버튼. */
export function AutoplayBar() {
  const { status, step, note } = useAutoplay();
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

  return (
    <header className="ap-bar">
      <span className="ap-brand">Tone 오프라인 데모</span>

      <ol className="ap-steps">
        {AUTOPLAY_STEPS.map((label, i) => {
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
        <button type="button" className="ap-btn ap-primary" onClick={startAutoplay}>
          {status === 'done' ? '↻ 다시 보기' : '▶ 자동 시연'}
        </button>
      )}
    </header>
  );
}

/** 처음 열었을 때의 시작 화면. */
export function StartOverlay() {
  const [open, setOpen] = useState(true);
  if (!open) return null;

  return (
    <div className="ap-overlay">
      <div className="ap-card">
        <div className="ap-logo">T</div>
        <h1>Tone 오프라인 데모</h1>
        <p>
          시작을 누르면 실제 화면이 저절로 진행됩니다.
          <br />
          프로필 생성 → 목소리 등록 → 녹음 · 전송 → 상대측 재생
        </p>
        <button
          type="button"
          className="ap-start"
          autoFocus
          onClick={() => {
            setOpen(false);
            startAutoplay();
          }}
        >
          ▶ 자동 시연 시작
        </button>
        <button type="button" className="ap-manual" onClick={() => setOpen(false)}>
          직접 조작하기
        </button>
        <p className="ap-keys">Space 일시정지 · Esc 정지 · P 파이프라인 접기</p>
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
