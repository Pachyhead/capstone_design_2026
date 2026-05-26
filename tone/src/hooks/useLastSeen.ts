import { useSyncExternalStore } from 'react';

const STORAGE_KEY = 'tone:lastSeen';

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

export function markSeen(conversationId: string, messageId: string | undefined) {
  if (!messageId) return;
  if (current[conversationId] === messageId) return;
  current = { ...current, [conversationId]: messageId };
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
