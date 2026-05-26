import { useSyncExternalStore } from 'react';
import type { Emotion, VoiceProfile } from '@/types';

const STORAGE_KEY = 'tone:voiceProfile';

// Unregistered baseline. Onboarding writes real recorded data over this.
const DEFAULT: VoiceProfile = {
  registered: false,
  sentenceCount: 0,
  durationSec: 0,
  emotionCoverage: 0,
  detectedEmotions: [],
};

const subscribers = new Set<() => void>();

// snapshot caching — same trick as useUserAvatar to satisfy useSyncExternalStore
let cachedRaw: string | null | undefined = undefined;
let cachedValue: VoiceProfile = DEFAULT;

function read(): VoiceProfile {
  if (typeof window === 'undefined') return DEFAULT;
  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (raw === cachedRaw) return cachedValue;
  cachedRaw = raw;
  if (!raw) {
    cachedValue = DEFAULT;
    return cachedValue;
  }
  try {
    const parsed = JSON.parse(raw) as VoiceProfile;
    if (parsed && typeof parsed.registered === 'boolean' && Array.isArray(parsed.detectedEmotions)) {
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

export function setVoiceProfile(profile: VoiceProfile) {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(profile));
  cachedRaw = undefined;
  subscribers.forEach((cb) => cb());
}

/** Build a VoiceProfile from the emotions the user recorded during onboarding. */
export function buildProfileFromRecorded(recordedEmotions: Emotion[]): VoiceProfile {
  // de-dupe while preserving order (first occurrence wins)
  const seen = new Set<Emotion>();
  const ordered: Emotion[] = [];
  for (const e of recordedEmotions) {
    if (!seen.has(e)) {
      seen.add(e);
      ordered.push(e);
    }
  }
  const count = recordedEmotions.length;
  return {
    registered: count > 0,
    sentenceCount: count,
    durationSec: count * 15, // rough estimate, ~15s per sentence
    emotionCoverage: ordered.length,
    detectedEmotions: ordered,
  };
}

export function useVoiceProfile(): [VoiceProfile, (p: VoiceProfile) => void] {
  const profile = useSyncExternalStore(subscribe, read, () => DEFAULT);
  return [profile, setVoiceProfile];
}
