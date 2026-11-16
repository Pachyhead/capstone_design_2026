import type { Emotion } from '@/types';
import { recordApiCall, setLastRecording } from '@/lib/metrics';

const BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');

export class ApiError extends Error {
  constructor(public status: number, public path: string, message: string) {
    super(`[${status}] ${path}: ${message}`);
  }
}

async function post<T = unknown>(
  path: string,
  params?: Record<string, string | number>,
  opts?: { silentStatuses?: number[] },
): Promise<T> {
  const url = new URL(`${BASE}${path}`, window.location.origin);
  if (params) {
    for (const [k, v] of Object.entries(params)) url.searchParams.set(k, String(v));
  }
  const label = `[api] POST ${path}${params ? ` ${JSON.stringify(params)}` : ''}`;
  const started = performance.now();
  let res: Response;
  try {
    res = await fetch(url.toString(), { method: 'POST' });
  } catch (err) {
    recordApiCall({ path, status: 0, ms: performance.now() - started, at: Date.now() });
    console.error(`${label} — network error:`, err);
    throw err;
  }
  const text = await res.text();
  recordApiCall({ path, status: res.status, ms: performance.now() - started, at: Date.now() });
  let parsed: unknown = text;
  try {
    parsed = JSON.parse(text);
  } catch {
    /* keep as text */
  }
  if (!res.ok) {
    if (!opts?.silentStatuses?.includes(res.status)) {
      console.warn(`${label} → ${res.status}`, parsed);
    }
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
  payload_bytes?: number;
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

// /measure_audio_codec_sizes: each codec's size at its service bitrate (bitrate × duration).
export interface CodecSize {
  bitrate_kbps: number;
  size_bytes: number;
}

export type CodecName = 'opus' | 'volte' | 'codec2';

// wav: size of the measured source file.
export type CodecSizes = Record<CodecName, CodecSize> & {
  wav: { size_bytes: number };
};

export interface ReceivedMessage {
  message_id: string;
  sender_id: number | string;
  receiver_id: number | string;
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

// set_receiver_id는 sender 서버 전용. receiver 서버(receiver/main.py)엔 없어 405가 난다.
// 첫 405를 감지해 이후 호출을 생략한다(같은 dist를 두 서버가 공유 서빙하므로 origin만으론 구분 불가).
let setReceiverSupported = true;

export const api = {
  setMyId: (id: number) => post<InboxBuckets>('/set_my_id', { value: id, my_id: id }),
  setReceiverId: async (id: number) => {
    if (!setReceiverSupported) return;
    try {
      return await post('/set_receiver_id', { value: id }, { silentStatuses: [405] });
    } catch (err) {
      if (err instanceof ApiError && err.status === 405) {
        setReceiverSupported = false;
        console.info('[api] /set_receiver_id 미지원 서버(receiver) 감지 — 이후 호출 생략');
        return;
      }
      throw err;
    }
  },
  startRecording: () => post<StartRecordingResult>('/start_recording'),
  stopRecording: async () => {
    const result = await post<RecordResult>('/stop_recording');
    setLastRecording(result);
    return result;
  },
  send: (message: string) => post('/send', { message }),
  sendRef: () => post('/send_ref'),
  getEmotionLabel: () => post<EmotionLabelResult>('/get_emotion_label'),
  playVoice: (messageId: string) =>
    post<PlayVoiceResult>('/play_voice', { message_id: messageId }),
  measureCodecSizes: () => post<CodecSizes>('/measure_audio_codec_sizes'),
};
