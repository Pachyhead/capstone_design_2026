import { useSyncExternalStore } from 'react';

// v2: 읽음 기록을 프로필(백엔드 id)별로 나눠 저장한다. 프로필마다 대화 내역이
// 다르므로 한 브라우저에서 프로필을 바꿔도 안 읽은 수가 섞이지 않는다.
const STORAGE_KEY = 'tone:lastSeen:v2';

type LastSeenMap = Record<string, string>;

function load(): LastSeenMap {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as LastSeenMap) : {};
  } catch {
    return {};
  }
}

let current: LastSeenMap = load();
const subscribers = new Set<() => void>();

function notify() {
  subscribers.forEach((cb) => cb());
}

function subscribe(cb: () => void) {
  subscribers.add(cb);
  return () => {
    subscribers.delete(cb);
  };
}

function read(): LastSeenMap {
  return current;
}

export function seenKey(myBackendId: number, conversationId: string): string {
  return `${myBackendId}:${conversationId}`;
}

export function markSeen(
  myBackendId: number,
  conversationId: string,
  messageId: string | undefined,
) {
  if (!messageId) return;
  const key = seenKey(myBackendId, conversationId);
  if (current[key] === messageId) return;
  current = { ...current, [key]: messageId };
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(current));
  } catch {
    /* ignore quota */
  }
  notify();
}

export function useLastSeen(): LastSeenMap {
  return useSyncExternalStore(subscribe, read, read);
}
