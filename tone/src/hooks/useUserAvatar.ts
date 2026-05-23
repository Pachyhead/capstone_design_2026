import { useSyncExternalStore } from 'react';
import type { Emotion } from '@/types';

export type UserAvatar =
  | { type: 'emotion'; emotion: Emotion }
  | { type: 'photo'; dataUrl: string };

const STORAGE_KEY = 'tone:userAvatar';
const DEFAULT: UserAvatar = { type: 'emotion', emotion: 'happy' };

const subscribers = new Set<() => void>();

// useSyncExternalStore requires getSnapshot to return a referentially stable
// value when the underlying data hasn't changed. JSON.parse always produces a
// fresh object, so we cache by the raw string and only re-parse on change.
let cachedRaw: string | null | undefined = undefined;
let cachedValue: UserAvatar = DEFAULT;

function read(): UserAvatar {
  if (typeof window === 'undefined') return DEFAULT;
  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (raw === cachedRaw) return cachedValue;
  cachedRaw = raw;
  if (!raw) {
    cachedValue = DEFAULT;
    return cachedValue;
  }
  try {
    const parsed = JSON.parse(raw) as UserAvatar;
    if (parsed && (parsed.type === 'emotion' || parsed.type === 'photo')) {
      cachedValue = parsed;
    } else {
      cachedValue = DEFAULT;
    }
  } catch {
    cachedValue = DEFAULT;
  }
  return cachedValue;
}

function subscribe(callback: () => void) {
  subscribers.add(callback);
  const onStorage = (e: StorageEvent) => {
    if (e.key === STORAGE_KEY) callback();
  };
  window.addEventListener('storage', onStorage);
  return () => {
    subscribers.delete(callback);
    window.removeEventListener('storage', onStorage);
  };
}

export function setUserAvatar(avatar: UserAvatar) {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(avatar));
  // invalidate cache so the next read() picks up the new value
  cachedRaw = undefined;
  subscribers.forEach((cb) => cb());
}

export function useUserAvatar(): [UserAvatar, (a: UserAvatar) => void] {
  const avatar = useSyncExternalStore(subscribe, read, () => DEFAULT);
  return [avatar, setUserAvatar];
}
