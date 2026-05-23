import { useSyncExternalStore } from 'react';
import type { UserAvatar } from '@/hooks/useUserAvatar';

export type BackendId = 0 | 1 | 2 | 3;

export interface Profile {
  id: string;
  name: string;
  avatar: UserAvatar;
  backendId: BackendId;
}

const PROFILES_KEY = 'tone:profiles:v2';
const ACTIVE_KEY = 'tone:activeProfileId:v2';

const DEFAULT_PROFILES: Profile[] = [
  { id: 'jongchan', name: '종찬', avatar: { type: 'emotion', emotion: 'neutral' }, backendId: 0 },
  { id: 'jaewoong', name: '재웅', avatar: { type: 'emotion', emotion: 'happy' }, backendId: 1 },
  { id: 'kyungtaek', name: '경택', avatar: { type: 'emotion', emotion: 'surprised' }, backendId: 2 },
  { id: 'taewon', name: '태원', avatar: { type: 'emotion', emotion: 'sad' }, backendId: 3 },
];

interface Store {
  profiles: Profile[];
  activeId: string | null;
}

const subscribers = new Set<() => void>();

let cachedKey = '__init__';
let cachedValue: Store = { profiles: DEFAULT_PROFILES, activeId: null };
const SERVER_SNAPSHOT: Store = { profiles: DEFAULT_PROFILES, activeId: null };

function read(): Store {
  if (typeof window === 'undefined') return SERVER_SNAPSHOT;
  const profilesRaw = window.localStorage.getItem(PROFILES_KEY);
  const activeRaw = window.localStorage.getItem(ACTIVE_KEY);
  const key = `${profilesRaw ?? ''}|${activeRaw ?? ''}`;
  if (key === cachedKey) return cachedValue;
  cachedKey = key;

  let profiles: Profile[] = DEFAULT_PROFILES;
  if (profilesRaw) {
    try {
      const parsed = JSON.parse(profilesRaw);
      if (Array.isArray(parsed)) profiles = parsed;
    } catch {
      profiles = DEFAULT_PROFILES;
    }
  }
  cachedValue = { profiles, activeId: activeRaw && activeRaw.length > 0 ? activeRaw : null };
  return cachedValue;
}

function notify() {
  subscribers.forEach((cb) => cb());
}

function subscribe(cb: () => void) {
  subscribers.add(cb);
  const onStorage = (e: StorageEvent) => {
    if (e.key === PROFILES_KEY || e.key === ACTIVE_KEY) cb();
  };
  window.addEventListener('storage', onStorage);
  return () => {
    subscribers.delete(cb);
    window.removeEventListener('storage', onStorage);
  };
}

function persistProfiles(profiles: Profile[]) {
  window.localStorage.setItem(PROFILES_KEY, JSON.stringify(profiles));
  cachedKey = '__init__';
  notify();
}

function persistActive(id: string | null) {
  if (id) window.localStorage.setItem(ACTIVE_KEY, id);
  else window.localStorage.removeItem(ACTIVE_KEY);
  cachedKey = '__init__';
  notify();
}

export function useProfiles() {
  const store = useSyncExternalStore(subscribe, read, () => SERVER_SNAPSHOT);
  const activeProfile = store.profiles.find((p) => p.id === store.activeId) ?? null;
  return {
    profiles: store.profiles,
    activeId: store.activeId,
    activeProfile,
    selectProfile: (id: string) => persistActive(id),
    clearActive: () => persistActive(null),
    addProfile: (profile: Profile) => persistProfiles([...store.profiles, profile]),
    removeProfile: (id: string) => {
      const next = store.profiles.filter((p) => p.id !== id);
      persistProfiles(next);
      if (store.activeId === id) persistActive(null);
    },
    renameProfile: (id: string, name: string) => {
      const next = store.profiles.map((p) => (p.id === id ? { ...p, name } : p));
      persistProfiles(next);
    },
    updateProfileAvatar: (id: string, avatar: UserAvatar) => {
      const next = store.profiles.map((p) => (p.id === id ? { ...p, avatar } : p));
      persistProfiles(next);
    },
  };
}
