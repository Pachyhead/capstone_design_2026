import type { Emotion } from '@/types';
import { recordApiCall, setLastRecording } from '@/lib/metrics';

const BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');

export class ApiError extends Error {
  constructor(public status: number, public path: string, message: string) {
    super(`[${status}] ${path}: ${message}`);
  }
}

/** 화면에 보여 줄 짧은 실패 사유. `action`은 "전송", "녹음" 같은 동작 이름. */
export function describeError(action: string, err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 502 || err.status === 503) {
      return `${action}에 실패했어요. 중계 서버에 연결할 수 없습니다.`;
    }
    if (err.status === 409) {
      return '엔드포인트가 다시 시작돼 사용자 정보를 맞추는 중이에요. 잠시 후 다시 시도해 주세요.';
    }
    return `${action}에 실패했어요. (오류 ${err.status})`;
  }
  return `${action}에 실패했어요. 엔드포인트 서버에 연결할 수 없습니다.`;
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
    // 409: 엔드포인트가 사용자/수신자 id를 모름 (재시작 직후). 다시 알려 준다.
    if (res.status === 409) endpointResetHandler?.();
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
// 마지막으로 보낸 수신자 id. 엔드포인트가 재시작되면 다시 보낸다 (resendReceiverId).
let lastReceiverId: number | null = null;

// 엔드포인트가 id를 잃었을 때(409) 부를 함수. useInboxEvents가 등록한다.
let endpointResetHandler: (() => void) | null = null;
export function setEndpointResetHandler(fn: (() => void) | null): void {
  endpointResetHandler = fn;
}

async function setReceiverId(id: number) {
  lastReceiverId = id;
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
}

export const api = {
  setMyId: (id: number) => post<InboxBuckets>('/set_my_id', { value: id, my_id: id }),
  setReceiverId,
  /** 마지막으로 고른 수신자를 다시 알린다. 아직 고른 적이 없으면 아무것도 하지 않는다. */
  resendReceiverId: async () => {
    if (lastReceiverId == null) return;
    return setReceiverId(lastReceiverId);
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
  // 엔드포인트 스피커로 재생한다. 서버는 재생이 끝난 뒤 true로 응답한다.
  playVoice: (messageId: string) => post<boolean>('/play_voice', { message_id: messageId }),
  measureCodecSizes: () => post<CodecSizes>('/measure_audio_codec_sizes'),
};
