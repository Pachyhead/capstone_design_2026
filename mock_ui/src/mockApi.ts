// `@/lib/api` 자리에 들어가는 모듈. 엔드포인트 서버 없이 시나리오대로 응답하고,
// 각 호출이 실제 시스템의 어느 단계에 해당하는지 파이프라인 패널에 알린다.
//
// 실제 시스템과 같은 점: 메시지는 "텍스트 + 감정"으로 저장되고, 음성은 받는 쪽이
// 재생을 누를 때 합성된다. 다른 점: 마이크와 스피커가 엔드포인트가 아니라 이
// 브라우저이고, 녹음 결과와 합성 음성은 미리 준비한 파일이다.
import {
  ApiError,
  describeError,
  emotionFromBackend,
  type CodecName,
  type CodecSize,
  type CodecSizes,
  type EmotionLabelResult,
  type InboxBuckets,
  type ReceivedMessage,
  type RecordResult,
  type StartRecordingResult,
} from '../../tone/src/lib/api';
import { recordApiCall, setLastRecording } from '@/lib/metrics';
import { clearNodes, resetPipeline, stage, type NodeId } from './pipeline';
import { DEMO_PROFILES, audioPath, setById, setForUser, type VoiceSet } from './scenario';

export { ApiError, describeError, emotionFromBackend };
export type {
  CodecName,
  CodecSize,
  CodecSizes,
  EmotionLabelResult,
  InboxBuckets,
  ReceivedMessage,
  RecordResult,
  StartRecordingResult,
};

/** 파일로 직접 열어도 찾을 수 있게 index.html 기준 상대 경로로 돌려준다. */
export function audioUrl(path: string): string {
  if (/^https?:/i.test(path)) return path;
  return `./${path.replace(/^\/+/, '')}`;
}

// 각 단계를 화면에 머무르게 하는 시간(ms). 실제 처리 시간이 아니라, 청중이 그 단계의
// 설명을 읽고 그림에서 위치를 확인할 여유다.
const MS = {
  split: 2200, // 음성을 Whisper STT · emotion2vec 에 넣는다
  splitResult: 2400, // 텍스트 + 1024d 감정 벡터로 나뉜다
  compress: 2400, // FSQ 인코더: 1024d → 8d
  upload: 2200, // 텍스트 + 24bit 를 한 번에 전송
  restore: 2400, // 서버: 24bit → 1024d 로 되돌려 MySQL 에 저장
  gather: 2200, // TTS 입력 모으기
  synth: 2600, // Qwen3-TTS
  stream: 1800, // gRPC 스트리밍
  refUpload: 1500,
};

// endpoint/sender_src/utils.py 의 SERVICE_BITRATES_KBPS 와 같은 값
const SERVICE_BITRATES_KBPS: Record<CodecName, number> = { opus: 16, volte: 12.65, codec2: 0.7 };

const STORAGE_KEY = 'tone-mock:messages';

interface StoredMessage extends ReceivedMessage {
  set_id: string;
}

function loadMessages(): StoredMessage[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as StoredMessage[]) : [];
  } catch {
    return [];
  }
}

let messages: StoredMessage[] = loadMessages();
let myId = 0;
let peerId = 1;
/** 마지막 녹음 결과. 전송할 때와 코덱 크기 비교에 쓴다. */
let recorded: VoiceSet | null = null;

// 화면 표시용 흐름 번호. 새 흐름이 시작되면 이전 흐름은 더 이상 패널을 건드리지 않는다.
let flow = 0;
let stopPlayback: (() => void) | null = null;

function persist() {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
  } catch {
    /* 저장 실패는 데모 진행에 지장이 없다 */
  }
}

const sleep = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms));

function nameOf(backendId: number): string {
  return DEMO_PROFILES.find((p) => p.backendId === backendId)?.name ?? `사용자 ${backendId}`;
}

function nowStamp(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

/** 실제 api.ts 처럼 호출 시간을 지표 패널에 남긴다. */
async function call<T>(path: string, run: () => Promise<T>): Promise<T> {
  const started = performance.now();
  try {
    const result = await run();
    recordApiCall({ path, status: 200, ms: performance.now() - started, at: Date.now() });
    return result;
  } catch (err) {
    const status = err instanceof ApiError ? err.status : 0;
    recordApiCall({ path, status, ms: performance.now() - started, at: Date.now() });
    throw err;
  }
}

/** 서버의 get_chats_by_user_id 와 같은 모양: 상대 id 순서로 4칸, 내 자리는 빈 칸. */
function bucketsFor(userId: number): InboxBuckets {
  const buckets: InboxBuckets = [[], [], [], []];
  for (const m of messages) {
    const sender = Number(m.sender_id);
    const receiver = Number(m.receiver_id);
    if (sender === userId && receiver !== userId) buckets[receiver as 0 | 1 | 2 | 3]?.push(m);
    else if (receiver === userId && sender !== userId) buckets[sender as 0 | 1 | 2 | 3]?.push(m);
  }
  return buckets;
}

/** 준비된 파일을 끝까지 재생한다. 다른 재생이 시작되면 중간에 끝난다. */
function playFile(path: string): Promise<void> {
  stopPlayback?.();
  return new Promise((resolve, reject) => {
    const audio = new Audio(audioUrl(path));
    const done = () => {
      if (stopPlayback === stop) stopPlayback = null;
      resolve();
    };
    const stop = () => {
      audio.pause();
      done();
    };
    stopPlayback = stop;
    audio.addEventListener('ended', done);
    audio.addEventListener('error', () => {
      if (stopPlayback === stop) stopPlayback = null;
      reject(new ApiError(404, '/play_voice', `audio file not found: ${path}`));
    });
    audio.play().catch(() => {
      if (stopPlayback === stop) stopPlayback = null;
      reject(new ApiError(500, '/play_voice', `cannot play: ${path}`));
    });
  });
}

/** 데모를 처음 상태로 되돌린다 (메시지 삭제). */
export function resetMockState() {
  flow += 1;
  stopPlayback?.();
  messages = [];
  recorded = null;
  window.localStorage.removeItem(STORAGE_KEY);
  resetPipeline();
}

export const api = {
  setMyId: (id: number) =>
    call('/set_my_id', async () => {
      myId = id;
      return bucketsFor(id);
    }),

  setReceiverId: (id: number) =>
    call('/set_receiver_id', async () => {
      peerId = id;
      return { message: 'receiver_id updated', receiver_id: id };
    }),

  startRecording: () =>
    call<StartRecordingResult>('/start_recording', async () => {
      const set = setForUser(myId);
      if (!set) throw new ApiError(500, '/start_recording', 'no demo voice set for this profile');
      flow += 1;
      stopPlayback?.();
      resetPipeline({ speakerSetId: set.id, speakerName: nameOf(myId) });
      // 같은 녹음 호출을 목소리 등록 화면도 쓴다. 그때는 메시지 녹음이 아니다.
      if (window.location.hash.includes('/onboarding')) {
        stage(['ref'], `목소리 등록 — ${nameOf(myId)}이(가) 정해진 문장을 읽습니다`);
      } else {
        stage(['record'], `녹음 — ${nameOf(myId)}이(가) 메시지를 말합니다`);
      }
      return { status: 'recording start' };
    }),

  stopRecording: () =>
    call<RecordResult>('/stop_recording', async () => {
      const set = setForUser(myId);
      if (!set) throw new ApiError(500, '/stop_recording', 'no demo voice set for this profile');
      const mine = flow;
      const show = (nodes: NodeId[], caption: string, patch = {}) => {
        if (mine === flow) stage(nodes, caption, patch);
      };

      show(
        ['stt', 'emotion2vec'],
        '분리 — 녹음한 음성을 Whisper STT와 emotion2vec에 함께 넣습니다',
      );
      await sleep(MS.split);
      show(['text'], '분리 결과 — 음성이 텍스트와 1024차원 감정 벡터로 나뉩니다', {
        text: set.text,
        emotion: set.emotion,
        vector: '1024',
        code: set.emotionCode ?? null,
      });
      await sleep(MS.splitResult);
      show(['fsqEncode'], '압축 — FSQ 인코더가 감정 벡터를 1024차원에서 8차원으로 줄입니다', {
        vector: '8',
      });
      await sleep(MS.compress);
      show(['bits'], '양자화 — 8차원 격자를 24bit 코드로 만듭니다. 이제 보낼 준비가 끝났습니다', {
        vector: '24',
      });

      recorded = set;
      const result: RecordResult = {
        text: set.text,
        emotion: set.emotion,
        duration: set.sourceDuration,
        audio_url: audioPath(set.id, 'source'),
      };
      setLastRecording(result);
      return result;
    }),

  send: (message: string) =>
    call('/send', async () => {
      const set = recorded ?? setForUser(myId);
      if (!set) throw new ApiError(400, '/send', 'no encoded message available');
      const mine = flow;
      const show = (nodes: NodeId[], caption: string, patch = {}) => {
        if (mine === flow) stage(nodes, caption, patch);
      };

      show(
        ['grpcUp'],
        '전송 — 텍스트와 24bit를 한 번에 서버로 보냅니다. 음성 파일은 보내지 않습니다',
        { text: message, vector: '24' },
      );
      await sleep(MS.upload);
      show(
        ['fsqRestore', 'mysql'],
        '복원 · 저장 — 서버가 24bit(8차원)를 다시 1024차원 감정 벡터로 늘려 저장합니다',
        { restored: true },
      );
      await sleep(MS.restore);

      messages = [
        ...messages,
        {
          message_id: `mock-${Date.now()}-${messages.length}`,
          sender_id: String(myId),
          receiver_id: String(peerId),
          message,
          emo_type: set.emoType,
          send_time: nowStamp(),
          set_id: set.id,
        },
      ];
      persist();
      show([], `전송 완료 — ${nameOf(peerId)}이(가) 재생을 누르면 합성이 시작됩니다`);
      return { message: 'sent', text: message };
    }),

  sendRef: () =>
    call('/send_ref', async () => {
      const set = setForUser(myId);
      flow += 1;
      resetPipeline({ speakerSetId: set?.id ?? null, speakerName: nameOf(myId) });
      stage(['ref'], `목소리 등록 — ${nameOf(myId)}의 ref 음성을 서버에 보관합니다`);
      await sleep(MS.refUpload);
      stage([], '목소리 등록 완료');
      return { message: 'reference sent' };
    }),

  getEmotionLabel: () =>
    call<EmotionLabelResult>('/get_emotion_label', async () => {
      if (!recorded) throw new ApiError(400, '/get_emotion_label', 'no encoded result found');
      return { emotion_label: recorded.emotion, emotion_score: 1 };
    }),

  // 실제 시스템처럼 재생을 누르는 순간 합성 과정을 거쳐 소리가 난다.
  playVoice: (messageId: string) =>
    call<boolean>('/play_voice', async () => {
      const message = messages.find((m) => m.message_id === messageId);
      const set = message ? setById(message.set_id) : undefined;
      if (!message || !set) throw new ApiError(404, '/play_voice', 'Such Message ID Not Found');

      flow += 1;
      const mine = flow;
      const current = () => mine === flow;
      const speaker = nameOf(Number(message.sender_id));

      stopPlayback?.();
      clearNodes(['ref', 'ttsText', 'ttsSpeaker', 'ttsEmotion', 'tts', 'grpcDown', 'play']);

      stage(
        ['ref', 'ttsText', 'ttsSpeaker', 'ttsEmotion'],
        `합성 준비 — 텍스트, ${speaker}의 등록된 목소리, 감정 벡터를 TTS에 넣습니다`,
        {
          text: message.message,
          emotion: set.emotion,
          vector: '24',
          code: set.emotionCode ?? null,
          restored: true,
          speakerSetId: set.id,
          speakerName: speaker,
        },
      );
      await sleep(MS.gather);
      if (!current()) return true;
      stage(['tts'], `합성 — Qwen3-TTS가 ${speaker}의 목소리와 감정으로 음성을 만듭니다`);
      await sleep(MS.synth);
      if (!current()) return true;
      stage(['grpcDown'], '전달 — 합성된 음성을 gRPC 스트리밍으로 받는 쪽에 보냅니다');
      await sleep(MS.stream);
      if (!current()) return true;
      stage(['play'], `재생 — ${speaker}의 목소리와 감정으로 들립니다`);
      await playFile(audioPath(set.id, 'output'));
      if (current()) stage([], '재생 완료');
      return true;
    }),

  // 마지막 녹음(source.wav)의 실제 길이로 계산한다. 서버 구현과 같은 식: 비트레이트 × 길이 / 8
  measureCodecSizes: () =>
    call<CodecSizes>('/measure_audio_codec_sizes', async () => {
      if (!recorded) throw new ApiError(404, '/measure_audio_codec_sizes', 'No WAV file found in storage');
      const size = (kbps: number): CodecSize => ({
        bitrate_kbps: kbps,
        size_bytes: Math.round((kbps * 1000 * recorded!.sourceDuration) / 8),
      });
      return {
        wav: { size_bytes: recorded.sourceBytes },
        opus: size(SERVICE_BITRATES_KBPS.opus),
        volte: size(SERVICE_BITRATES_KBPS.volte),
        codec2: size(SERVICE_BITRATES_KBPS.codec2),
      };
    }),
};
