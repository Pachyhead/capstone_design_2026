import { useSyncExternalStore } from 'react';
import type { VoiceProfile } from '@/types';
import { useProfiles } from '@/hooks/useProfiles';

// v2: 목소리 등록 상태를 프로필(백엔드 id)별로 저장한다. 서버의 레퍼런스 음성이
// 사용자 id별로 따로 있으므로, 브라우저의 등록 표시도 프로필마다 달라야 한다.
const STORAGE_PREFIX = 'tone:voiceProfile:v2:';

// Unregistered baseline. Onboarding writes real recorded data over this.
const DEFAULT: VoiceProfile = {
  registered: false,
  sentenceCount: 0,
  durationSec: 0,
  emotionCoverage: 0,
  detectedEmotions: [],
};

const subscribers = new Set<() => void>();

// snapshot caching per storage key — useSyncExternalStore needs a stable reference
// while the underlying string is unchanged.
const cache = new Map<string, { raw: string | null; value: VoiceProfile }>();

function read(key: string | null): VoiceProfile {
  if (key === null || typeof window === 'undefined') return DEFAULT;
  const raw = window.localStorage.getItem(key);
  const hit = cache.get(key);
  if (hit && hit.raw === raw) return hit.value;

  let value = DEFAULT;
  if (raw) {
    try {
      const parsed = JSON.parse(raw) as VoiceProfile;
      if (parsed && typeof parsed.registered === 'boolean' && Array.isArray(parsed.detectedEmotions)) {
        value = parsed;
      }
    } catch {
      value = DEFAULT;
    }
  }
  cache.set(key, { raw, value });
  return value;
}

function subscribe(callback: () => void) {
  subscribers.add(callback);
  const onStorage = (e: StorageEvent) => {
    if (e.key?.startsWith(STORAGE_PREFIX)) callback();
  };
  window.addEventListener('storage', onStorage);
  return () => {
    subscribers.delete(callback);
    window.removeEventListener('storage', onStorage);
  };
}

function write(key: string, profile: VoiceProfile) {
  window.localStorage.setItem(key, JSON.stringify(profile));
  subscribers.forEach((cb) => cb());
}

/** Voice-registration state of the active profile. Setter is a no-op without one. */
export function useVoiceProfile(): [VoiceProfile, (p: VoiceProfile) => void] {
  const { activeProfile } = useProfiles();
  const key = activeProfile ? `${STORAGE_PREFIX}${activeProfile.backendId}` : null;
  const profile = useSyncExternalStore(
    subscribe,
    () => read(key),
    () => DEFAULT,
  );
  const setProfile = (p: VoiceProfile) => {
    if (key !== null) write(key, p);
  };
  return [profile, setProfile];
}
