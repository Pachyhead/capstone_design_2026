import type { Emotion } from '@/types';

const BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');

export class ApiError extends Error {
  constructor(public status: number, public path: string, message: string) {
    super(`[${status}] ${path}: ${message}`);
  }
}

async function post<T = unknown>(path: string, params?: Record<string, string | number>): Promise<T> {
  const url = new URL(`${BASE}${path}`, window.location.origin);
  if (params) {
    for (const [k, v] of Object.entries(params)) url.searchParams.set(k, String(v));
  }
  const label = `[api] POST ${path}${params ? ` ${JSON.stringify(params)}` : ''}`;
  let res: Response;
  try {
    res = await fetch(url.toString(), { method: 'POST' });
  } catch (err) {
    console.error(`${label} — network error:`, err);
    throw err;
  }
  const text = await res.text();
  let parsed: unknown = text;
  try {
    parsed = JSON.parse(text);
  } catch {
    /* keep as text */
  }
  if (!res.ok) {
    console.warn(`${label} → ${res.status}`, parsed);
    throw new ApiError(res.status, path, text || res.statusText);
  }
  console.log(`${label} → ${res.status}`, parsed);
  return parsed as T;
}

export interface RecordResult {
  text: string;
  emotion: Emotion;
  duration: number;
  audio_url?: string;
}

export interface StartRecordingResult {
  status: string;
}

export interface EmotionLabelResult {
  emotion_label: Emotion;
  emotion_score: number;
}

export interface PlayVoiceResult {
  duration: number;
}

export interface ReceivedMessage {
  message_id: string;
  sender_id: number;
  message: string;
  emo_type: number | string;
  send_time: string;
}

// /set_my_id response: 4-element array bucketed by sender backend id.
// Bucket at index === my own backend id is always empty (self).
// Inner lists are time-ordered (oldest → newest).
export type InboxBuckets = [
  ReceivedMessage[],
  ReceivedMessage[],
  ReceivedMessage[],
  ReceivedMessage[],
];

// backend EmotionLabel IntEnum → frontend Emotion
const EMOTION_INDEX_TO_NAME: Record<number, Emotion> = {
  0: 'angry',
  1: 'disgusted',
  2: 'fearful',
  3: 'happy',
  4: 'neutral',
  5: 'other',
  6: 'sad',
  7: 'surprised',
  8: 'unk',
};

export function emotionFromBackend(value: number | string): Emotion {
  if (typeof value === 'number') return EMOTION_INDEX_TO_NAME[value] ?? 'unk';
  const lower = value.toLowerCase();
  if ((Object.values(EMOTION_INDEX_TO_NAME) as string[]).includes(lower)) return lower as Emotion;
  const asNum = Number(value);
  if (!Number.isNaN(asNum)) return EMOTION_INDEX_TO_NAME[asNum] ?? 'unk';
  return 'unk';
}

export function audioUrl(path: string): string {
  if (/^https?:/i.test(path)) return path;
  return `${BASE}${path.startsWith('/') ? '' : '/'}${path}`;
}

export const api = {
  setMyId: (id: number) => post<InboxBuckets>('/set_my_id', { value: id, my_id: id }),
  startRecording: () => post<StartRecordingResult>('/start_recording'),
  stopRecording: () => post<RecordResult>('/stop_recording'),
  send: (message: string) => post('/send', { message }),
  sendRef: () => post('/send_ref'),
  getEmotionLabel: () => post<EmotionLabelResult>('/get_emotion_label'),
  playVoice: (messageId: string) =>
    post<PlayVoiceResult>('/play_voice', { message_id: messageId }),
};
