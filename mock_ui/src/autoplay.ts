// 자동 시연. 시작을 누르면 실제 Tone 화면의 버튼을 순서대로 눌러 가며
// "프로필 생성 → 목소리 등록 → 녹음·전송 → 상대측 재생"을 영상처럼 보여 준다.
//
// 화면을 직접 그리지 않고 진짜 화면을 조작하므로, 보이는 것은 실제 앱과 같다.
// 소리는 구간마다 준비된 파일을 튼다: 목소리 등록 = reference.wav,
// 녹음 = source.wav, 상대측 재생 = output.wav (이건 mockApi 가 튼다).
import { useSyncExternalStore } from 'react';
import { EMOTION_LABELS } from '@/types';
import type { Emotion } from '@/types';
import { audioUrl, resetMockState } from './mockApi';
import { resetPipeline, setCaption } from './pipeline';
import { markSeen, seenKey } from '@/hooks/useLastSeen';
import { lastHistoryId } from './history';
import { AUTOPLAY, DEMO_PROFILES, audioPath, setForUser, type DemoProfile } from './scenario';

/**
 * short — 목소리가 등록된 상태에서 "녹음 → 전송 → 상대측 재생"만 보여 준다 (기본).
 * full  — 프로필 생성과 목소리 등록부터 보여 준다.
 */
export type AutoplayMode = 'short' | 'full';

export const AUTOPLAY_STEPS: Record<AutoplayMode, string[]> = {
  short: ['녹음', '전송', '상대측 재생'],
  full: ['프로필 생성', '목소리 등록', '녹음', '전송', '상대측 재생'],
};

export interface AutoplayState {
  status: 'idle' | 'running' | 'paused' | 'done';
  mode: AutoplayMode;
  /** 진행 중인 단계 (1부터). 시작 전은 0. */
  step: number;
  note: string;
}

let state: AutoplayState = { status: 'idle', mode: 'short', step: 0, note: '' };
const subscribers = new Set<() => void>();

function update(patch: Partial<AutoplayState>) {
  state = { ...state, ...patch };
  subscribers.forEach((cb) => cb());
}

export function useAutoplay(): AutoplayState {
  return useSyncExternalStore(
    (cb) => {
      subscribers.add(cb);
      return () => {
        subscribers.delete(cb);
      };
    },
    () => state,
  );
}

// ---------------------------------------------------------------------------
// 진행 제어: 정지하면 run 번호가 바뀌고, 진행 중이던 흐름은 다음 대기 지점에서 빠진다.
// ---------------------------------------------------------------------------
class Cancelled extends Error {}

let run = 0;
let audio: HTMLAudioElement | null = null;

const TICK = 80;

/** 일시정지 중이면 풀릴 때까지 기다리고, 정지됐으면 흐름을 끝낸다. */
async function checkpoint(mine: number) {
  for (;;) {
    if (mine !== run) throw new Cancelled();
    if (state.status !== 'paused') return;
    await new Promise((r) => window.setTimeout(r, TICK));
  }
}

async function wait(mine: number, ms: number) {
  let left = ms;
  while (left > 0) {
    await checkpoint(mine);
    const slice = Math.min(TICK, left);
    await new Promise((r) => window.setTimeout(r, slice));
    left -= slice;
  }
  await checkpoint(mine);
}

async function waitFor<T>(mine: number, find: () => T | null | undefined | false, what: string, timeoutMs = 8000): Promise<T> {
  const started = Date.now();
  for (;;) {
    await checkpoint(mine);
    const found = find();
    if (found) return found;
    if (Date.now() - started > timeoutMs) throw new Error(`화면에서 찾지 못했습니다: ${what}`);
    await new Promise((r) => window.setTimeout(r, TICK));
  }
}

// ---------------------------------------------------------------------------
// 화면 조작. Tone 화면(.mock-app) 안에서만 찾는다 — 바깥의 시연 제어 버튼과 섞이지 않게.
// ---------------------------------------------------------------------------
const appRoot = () => document.querySelector<HTMLElement>('.mock-app');

function bySelector(selector: string) {
  return () => appRoot()?.querySelector<HTMLElement>(selector) ?? null;
}

/** 글자가 정확히 일치하는 버튼. `startsWith` 면 그 글자로 시작하는 버튼. */
function byButtonText(text: string, startsWith = false) {
  return () =>
    [...(appRoot()?.querySelectorAll<HTMLElement>('button') ?? [])].find((b) => {
      const label = (b.textContent ?? '').trim();
      return !b.hasAttribute('disabled') && (startsWith ? label.startsWith(text) : label === text);
    }) ?? null;
}

let cursor: HTMLDivElement | null = null;

function ensureCursor(): HTMLDivElement {
  if (!cursor) {
    cursor = document.createElement('div');
    cursor.className = 'mock-cursor';
    cursor.style.left = `${window.innerWidth / 2}px`;
    cursor.style.top = `${window.innerHeight / 2}px`;
    document.body.appendChild(cursor);
  }
  return cursor;
}

/** 대화에 떠 있는 메시지들의 재생 버튼 (오래된 것부터). 대화 기록의 버튼도 포함된다. */
function messagePlayButtons(): HTMLElement[] {
  return [...(appRoot()?.querySelectorAll<HTMLElement>('button[aria-label="재생"]') ?? [])];
}

function removeCursor() {
  cursor?.remove();
  cursor = null;
}

async function moveTo(mine: number, el: HTMLElement) {
  el.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  const rect = el.getBoundingClientRect();
  const c = ensureCursor();
  c.style.left = `${rect.left + rect.width / 2}px`;
  c.style.top = `${rect.top + rect.height / 2}px`;
  await wait(mine, 650); // 커서 이동 애니메이션 길이 + 잠깐 멈춤
}

function pulse() {
  const c = ensureCursor();
  c.classList.remove('is-clicking');
  void c.offsetWidth; // 애니메이션을 다시 시작시킨다
  c.classList.add('is-clicking');
}

async function click(mine: number, find: () => HTMLElement | null, what: string, after = 600) {
  const el = await waitFor(mine, find, what);
  await moveTo(mine, el);
  pulse();
  el.click();
  await wait(mine, after);
}

/** 사람이 치는 것처럼 한 글자씩 입력한다 (React 가 값을 알아채도록 input 이벤트를 보낸다). */
async function type(mine: number, find: () => HTMLElement | null, text: string, what: string) {
  const input = (await waitFor(mine, find, what)) as HTMLInputElement;
  await moveTo(mine, input);
  pulse();
  input.focus();
  const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
  for (let i = 1; i <= text.length; i++) {
    setValue.call(input, text.slice(0, i));
    input.dispatchEvent(new Event('input', { bubbles: true }));
    await wait(mine, 220);
  }
  await wait(mine, 400);
}

/** 준비된 음성을 끝까지 튼다. */
async function playAudio(mine: number, path: string) {
  await checkpoint(mine);
  const el = new Audio(audioUrl(path));
  audio = el;
  let ended = false;
  let failed = false;
  el.addEventListener('ended', () => (ended = true));
  el.addEventListener('error', () => (failed = true));
  el.play().catch(() => (failed = true));
  try {
    for (;;) {
      if (mine !== run) throw new Cancelled();
      if (failed) throw new Error(`음성 파일을 재생할 수 없습니다: ${path}`);
      if (ended) return;
      // 일시정지는 소리에도 적용한다.
      if (state.status === 'paused' && !el.paused) el.pause();
      if (state.status === 'running' && el.paused && !ended) el.play().catch(() => (failed = true));
      await new Promise((r) => window.setTimeout(r, TICK));
    }
  } finally {
    el.pause();
    if (audio === el) audio = null;
  }
}

// ---------------------------------------------------------------------------
// 처음 상태 만들기
// ---------------------------------------------------------------------------
const AVATARS: Emotion[] = ['happy', 'surprised', 'neutral', 'sad'];

function profileRecord(p: DemoProfile) {
  return {
    id: p.id,
    name: p.name,
    avatar: { type: 'emotion', emotion: AVATARS[p.backendId % AVATARS.length] },
    backendId: p.backendId,
  };
}

/**
 * 브라우저에 남은 데모 상태를 지우고 주어진 프로필만 있는 상태로 만든다.
 * `loggedIn` 을 주면 그 프로필로 들어가 있고 목소리도 등록된 상태가 된다.
 *
 * 실제 앱의 저장소 훅들은 `storage` 이벤트를 듣고 다시 읽으므로, 새로고침 없이 반영된다.
 * (새로고침을 하면 브라우저가 "사용자가 누른 적 없음"으로 보고 소리 재생을 막는다.)
 */
export function seedProfiles(profiles: DemoProfile[], loggedIn?: DemoProfile) {
  const stale = Object.keys(window.localStorage).filter((k) => k.startsWith('tone:'));
  stale.forEach((k) => window.localStorage.removeItem(k));

  const entries: Record<string, string> = {
    'tone:profiles:v3': JSON.stringify(profiles.map(profileRecord)),
  };
  if (loggedIn) {
    entries['tone:activeProfileId:v3'] = loggedIn.id;
    entries['tone:userAvatar'] = JSON.stringify(profileRecord(loggedIn).avatar);
    entries[`tone:voiceProfile:v2:${loggedIn.backendId}`] = JSON.stringify({
      registered: true,
      sentenceCount: 1,
      durationSec: setForUser(loggedIn.backendId)?.referenceDuration ?? 0,
      emotionCoverage: 1,
      detectedEmotions: ['neutral'],
    });
  }
  // 미리 넣어 둔 대화 기록은 모두 읽은 것으로 둔다. 그래야 데모에서 새로 보낸
  // 메시지만 "안 읽음"으로 뜬다.
  const seen: Record<string, string> = {};
  for (const me of DEMO_PROFILES) {
    for (const peer of DEMO_PROFILES) {
      const last = me === peer ? undefined : lastHistoryId(me.backendId, peer.backendId);
      if (last) seen[seenKey(me.backendId, peer.id)] = last;
    }
  }
  entries['tone:lastSeen:v2'] = JSON.stringify(seen);

  for (const [key, value] of Object.entries(entries)) window.localStorage.setItem(key, value);

  // 읽음 기록 저장소는 storage 이벤트를 듣지 않고 메모리에 들고 있어서, 직접 알려 준다.
  for (const me of DEMO_PROFILES) {
    for (const peer of DEMO_PROFILES) {
      const last = seen[seenKey(me.backendId, peer.id)];
      if (last) markSeen(me.backendId, peer.id, last);
    }
  }

  const touched = new Set([
    ...stale,
    ...Object.keys(entries),
    'tone:activeProfileId:v3',
    'tone:userAvatar',
  ]);
  for (const key of touched) window.dispatchEvent(new StorageEvent('storage', { key }));
}

// ---------------------------------------------------------------------------
// 시나리오
// ---------------------------------------------------------------------------

/** 프로필을 새로 만들고 목소리를 등록한다 (full 에서만). 끝나면 대화 목록에 있다. */
async function createAndRegister(mine: number, sender: DemoProfile) {
  update({ step: 1, note: `${sender.name}의 프로필을 새로 만듭니다` });
  resetPipeline({ caption: '프로필 생성 — 처음 쓰는 사람이 자기 프로필을 만듭니다' });
  await click(mine, bySelector('button[aria-label="프로필 추가"]'), '프로필 추가 버튼');
  await type(mine, bySelector('input[placeholder="이름"]'), sender.name, '이름 입력칸');
  await click(
    mine,
    bySelector(`button[aria-label="${EMOTION_LABELS[AUTOPLAY.avatar]} 아바타"]`),
    '아바타 선택',
  );
  await click(mine, byButtonText('만들기'), '만들기 버튼', 900);
  await click(mine, bySelector(`button[aria-label="${sender.name} 프로필 선택"]`), '새 프로필', 900);

  update({ step: 2, note: '정해진 문장을 읽어 목소리를 한 번 등록합니다' });
  resetPipeline({ caption: '목소리 등록 — 처음 한 번, 정해진 문장을 읽어 녹음합니다' });
  await click(mine, bySelector('a[href="#/me"]'), '나 메뉴');
  await click(mine, byButtonText('등록 시작'), '등록 시작 버튼', 900);
  await click(mine, byButtonText('시작하기'), '시작하기 버튼', 900);
  await click(mine, bySelector('button[aria-label="녹음 시작"]'), '녹음 시작 버튼', 500);
  await waitFor(mine, byButtonText('정지'), '녹음 중 화면');
  await playAudio(mine, audioPath(String(sender.backendId), 'reference'));
  await wait(mine, 400);
  await click(mine, byButtonText('정지'), '정지 버튼', 400);
  await click(mine, byButtonText('완료'), '완료 버튼', 1200);
  await click(mine, byButtonText('시작하기'), '시작하기 버튼', 900);
}

/**
 * 대화 목록에서 시작해 "녹음 → 전송 → 상대측 재생"을 보여 준다.
 * `first` 는 이 구간의 첫 단계 번호다 (short 는 1, full 은 3).
 */
async function sendAndReceive(mine: number, sender: DemoProfile, receiver: DemoProfile, first: number) {
  // 녹음
  update({ step: first, note: `${receiver.name}에게 보낼 메시지를 말합니다` });
  resetPipeline({ caption: `녹음 — ${receiver.name}을(를) 골라 메시지를 말합니다` });
  await click(mine, bySelector(`a[href="#/chat/${receiver.id}"]`), '대화 상대', 900);
  const mic = await waitFor(mine, bySelector('button[aria-label="누르고 있는 동안 녹음"]'), '마이크 버튼');
  // 이 방에는 이미 대화 기록이 있다. 새 메시지가 붙었는지는 개수로 확인한다.
  const before = messagePlayButtons().length;
  await moveTo(mine, mic);
  pulse();
  ensureCursor().classList.add('is-holding');
  mic.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
  try {
    await playAudio(mine, audioPath(String(sender.backendId), 'source'));
    await wait(mine, 300);
  } finally {
    // 정지로 빠져나가더라도 눌린 채 남지 않게 손을 뗀다.
    cursor?.classList.remove('is-holding');
    document.dispatchEvent(new PointerEvent('pointerup', { bubbles: true }));
  }

  // 전송: 분석이 끝나 초안이 뜨면 문장과 감정을 확인하고 보낸다.
  await waitFor(mine, bySelector('textarea'), '초안 화면', 30000);
  update({ step: first + 1, note: '인식된 문장과 감정을 확인하고 전송합니다' });
  await wait(mine, 2200); // 초안을 읽을 시간
  await click(mine, byButtonText('전송', true), '전송 버튼', 400);
  // 전송이 끝나면 초안 화면이 닫히고 메시지가 대화에 나타난다.
  // (초안 화면에도 재생 버튼이 있어서, 초안이 닫힌 것부터 확인한다.)
  await waitFor(mine, () => !appRoot()?.querySelector('textarea'), '전송 완료', 30000);
  await waitFor(mine, () => messagePlayButtons().length > before, '보낸 메시지');
  await wait(mine, 1600);

  // 상대측 재생
  // 여기서는 패널을 지우지 않는다: 지나온 송신·서버 단계가 그대로 남아 있어야 흐름이 이어져 보인다.
  update({ step: first + 2, note: `${receiver.name}의 화면으로 바꿔 메시지를 재생합니다` });
  setCaption(`상대측 — ${receiver.name}이(가) 받은 메시지를 확인합니다`);
  await click(mine, bySelector('a[href="#/me"]'), '나 메뉴');
  await click(mine, byButtonText('프로필 전환'), '프로필 전환 버튼', 900);
  await click(mine, bySelector(`button[aria-label="${receiver.name} 프로필 선택"]`), '받는 사람 프로필', 900);
  await click(mine, bySelector(`a[href="#/chat/${sender.id}"]`), '받은 대화', 1200);
  // 방금 받은 메시지는 대화의 맨 아래에 있다 (위쪽은 음성이 없는 대화 기록이다).
  await click(mine, () => messagePlayButtons().pop() ?? null, '받은 메시지의 재생 버튼', 300);
  await waitFor(mine, bySelector('button[aria-label="재생 중"]'), '재생 시작');
  await waitFor(mine, () => !appRoot()?.querySelector('button[aria-label="재생 중"]'), '재생 끝', 90000);
  await wait(mine, 600);
}

async function play(mine: number, mode: AutoplayMode) {
  const sender = DEMO_PROFILES.find((p) => p.id === AUTOPLAY.senderId);
  const receiver = DEMO_PROFILES.find((p) => p.id === AUTOPLAY.receiverId);
  if (!sender || !receiver) throw new Error('scenario.ts 의 AUTOPLAY 프로필 id 가 잘못됐습니다');

  resetMockState();
  if (mode === 'full') {
    // 보내는 사람만 없는 상태에서 시작한다 (그 프로필을 새로 만드는 것부터 보여 준다).
    seedProfiles(DEMO_PROFILES.filter((p) => p.id !== sender.id));
    window.location.hash = '#/profiles';
  } else {
    // 보내는 사람으로 이미 들어가 있고 목소리도 등록된 상태에서 시작한다.
    seedProfiles(DEMO_PROFILES, sender);
    window.location.hash = '#/';
  }
  ensureCursor();
  await wait(mine, 900);

  if (mode === 'full') await createAndRegister(mine, sender);
  await sendAndReceive(mine, sender, receiver, mode === 'full' ? 3 : 1);
}

export function startAutoplay(mode: AutoplayMode = 'short') {
  stopAutoplay();
  const mine = run;
  update({ status: 'running', mode, step: 0, note: '' });
  play(mine, mode)
    .then(() => {
      if (mine !== run) return;
      removeCursor();
      update({ status: 'done', note: '시연이 끝났습니다. 이어서 직접 조작할 수 있습니다' });
    })
    .catch((err) => {
      if (err instanceof Cancelled || mine !== run) return;
      console.error('[autoplay]', err);
      removeCursor();
      update({ status: 'idle', step: 0, note: err instanceof Error ? err.message : String(err) });
    });
}

export function stopAutoplay() {
  run += 1;
  audio?.pause();
  audio = null;
  removeCursor();
  update({ status: 'idle', step: 0, note: '' });
}

export function togglePause() {
  if (state.status === 'running') update({ status: 'paused' });
  else if (state.status === 'paused') update({ status: 'running' });
}
