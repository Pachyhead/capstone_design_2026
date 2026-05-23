import type {
  ChatroomStats,
  Conversation,
  Emotion,
  Message,
  VoiceProfile,
} from '@/types';

const wave = (...values: number[]): number[] => values.map((v) => v / 100);

export const conversations: Conversation[] = [
  {
    id: 'jongchan',
    name: '종찬',
    initial: '종',
    language: '한국어',
    lastSeen: '5분 전',
    unread: 1,
    backendId: 0,
    lastMessage: {
      text: '내일 만나자, 진짜 기대돼!',
      emotion: { primary: 'happy', nuance: '들뜸' },
      sentAt: '3:42',
    },
  },
  {
    id: 'jaewoong',
    name: '재웅',
    initial: '재',
    language: '한국어',
    lastSeen: '1시간 전',
    unread: 0,
    backendId: 1,
    lastMessage: {
      text: '알겠어 확인해볼게',
      emotion: { primary: 'neutral' },
      sentAt: '2:30',
    },
  },
  {
    id: 'kyungtaek',
    name: '경택',
    initial: '경',
    language: '한국어',
    lastSeen: '방금',
    unread: 3,
    backendId: 2,
    lastMessage: {
      text: '헐 진짜?? 대박!',
      emotion: { primary: 'surprised' },
      sentAt: '1:30',
    },
  },
  {
    id: 'taewon',
    name: '태원',
    initial: '태',
    language: '한국어',
    lastSeen: '오늘',
    unread: 0,
    backendId: 3,
    lastMessage: {
      text: '회의 시간 변경 가능할까요',
      emotion: { primary: 'fearful', nuance: '망설임' },
      sentAt: '12:45',
    },
  },
];

export const messages: Message[] = [
  // ---- 종찬 (joy 들뜸) ----
  {
    id: 'jc-1',
    conversationId: 'jongchan',
    authorId: 'jongchan',
    text: '내일 만나자, 진짜 기대돼!',
    emotion: { primary: 'happy', nuance: '들뜸' },
    durationSec: 1.6,
    energy: wave(40, 60, 75, 80, 70, 75, 65, 55, 60, 50),
    sentAt: '3:42',
  },
  {
    id: 'jc-2',
    conversationId: 'jongchan',
    authorId: 'me',
    text: '어 ㅋㅋ 몇 시에 볼까',
    emotion: { primary: 'happy' },
    durationSec: 1.3,
    energy: wave(40, 55, 65, 70, 60, 65, 55, 50, 55, 45),
    sentAt: '3:43',
  },

  // ---- 재웅 (neutral) ----
  {
    id: 'jw-1',
    conversationId: 'jaewoong',
    authorId: 'me',
    text: '지난번 그 자료 좀 보내줄 수 있어?',
    emotion: { primary: 'neutral' },
    durationSec: 2.8,
    energy: wave(40, 50, 55, 45, 55, 50, 55, 45, 50, 40),
    sentAt: '2:00',
  },
  {
    id: 'jw-2',
    conversationId: 'jaewoong',
    authorId: 'jaewoong',
    text: '알겠어 확인해볼게',
    emotion: { primary: 'neutral' },
    durationSec: 2.0,
    energy: wave(40, 50, 45, 55, 50, 45, 50, 40, 45, 40),
    sentAt: '2:30',
  },

  // ---- 경택 (surprise) ----
  {
    id: 'kt-1',
    conversationId: 'kyungtaek',
    authorId: 'kyungtaek',
    text: '헐 진짜?? 대박!',
    emotion: { primary: 'surprised' },
    durationSec: 1.4,
    energy: wave(30, 60, 90, 95, 75, 85, 60, 50, 40, 30),
    sentAt: '1:30',
  },
  {
    id: 'kt-2',
    conversationId: 'kyungtaek',
    authorId: 'me',
    text: '응 진짜야 ㅋㅋ',
    emotion: { primary: 'happy' },
    durationSec: 1.1,
    energy: wave(40, 60, 65, 70, 60, 55, 50, 45, 50, 40),
    sentAt: '1:31',
  },

  // ---- 태원 (fear 망설임) ----
  {
    id: 'tw-1',
    conversationId: 'taewon',
    authorId: 'taewon',
    text: '회의 시간 변경 가능할까요',
    emotion: { primary: 'fearful', nuance: '망설임' },
    durationSec: 2.4,
    energy: wave(35, 45, 40, 50, 45, 40, 45, 35, 40, 35),
    sentAt: '12:45',
  },
  {
    id: 'tw-2',
    conversationId: 'taewon',
    authorId: 'me',
    text: '네 알겠습니다',
    emotion: { primary: 'neutral' },
    durationSec: 1.5,
    energy: wave(40, 50, 45, 55, 50, 45, 50, 40, 45, 40),
    sentAt: '12:46',
  },
];

export const chatroomStats: ChatroomStats[] = [
  {
    conversationId: 'jongchan',
    messageCount: 247,
    distribution: { happy: 0.45, sad: 0.15, fearful: 0.12, surprised: 0.08, neutral: 0.08, angry: 0.07, disgusted: 0.03, unk: 0.02 },
  },
  {
    conversationId: 'jaewoong',
    messageCount: 156,
    distribution: { neutral: 0.45, happy: 0.25, sad: 0.12, fearful: 0.10, surprised: 0.08 },
  },
  {
    conversationId: 'kyungtaek',
    messageCount: 189,
    distribution: { surprised: 0.40, happy: 0.30, angry: 0.12, neutral: 0.10, fearful: 0.05, sad: 0.03 },
  },
  {
    conversationId: 'taewon',
    messageCount: 98,
    distribution: { neutral: 0.55, fearful: 0.20, sad: 0.15, happy: 0.10 },
  },
];

export interface HourlyEmotion {
  hour: number;
  emotion: Emotion;
  intensity: number;
}

function buildFlow(...blocks: Array<[number, number, Emotion, number]>): HourlyEmotion[] {
  const out: HourlyEmotion[] = Array.from({ length: 24 }, (_, h) => ({
    hour: h,
    emotion: 'neutral' as Emotion,
    intensity: 0,
  }));
  for (const [start, end, emotion, intensity] of blocks) {
    for (let h = start; h <= end; h++) {
      out[h] = { hour: h, emotion, intensity };
    }
  }
  return out;
}

export const hourlyEmotionFlow: Record<string, HourlyEmotion[]> = {
  jongchan: buildFlow(
    [0, 5, 'neutral', 0.05],
    [6, 8, 'neutral', 0.30],
    [9, 11, 'happy', 0.55],
    [12, 14, 'happy', 0.70],
    [15, 17, 'happy', 0.85],
    [18, 20, 'fearful', 0.55],
    [21, 23, 'happy', 0.40],
  ),
  jaewoong: buildFlow(
    [0, 7, 'neutral', 0.05],
    [8, 11, 'neutral', 0.45],
    [12, 13, 'happy', 0.40],
    [14, 17, 'neutral', 0.55],
    [18, 20, 'neutral', 0.40],
    [21, 23, 'happy', 0.30],
  ),
  kyungtaek: buildFlow(
    [0, 1, 'happy', 0.30],
    [2, 7, 'neutral', 0.05],
    [8, 10, 'neutral', 0.20],
    [11, 14, 'happy', 0.55],
    [15, 18, 'surprised', 0.70],
    [19, 22, 'happy', 0.85],
    [23, 23, 'happy', 0.50],
  ),
  taewon: buildFlow(
    [0, 8, 'neutral', 0.05],
    [9, 11, 'neutral', 0.40],
    [12, 13, 'neutral', 0.20],
    [14, 16, 'fearful', 0.55],
    [17, 18, 'fearful', 0.70],
    [19, 23, 'neutral', 0.05],
  ),
};

export const previousPeriodDelta: Record<string, { emotion: Emotion; delta: number }[]> = {
  jongchan: [
    { emotion: 'happy', delta: 0.12 },
    { emotion: 'sad', delta: -0.05 },
    { emotion: 'fearful', delta: 0.03 },
  ],
};

export interface SpeakerVoiceProfile {
  coverage: number;
  pitch: '낮음' | '중간' | '높음';
  pace: '느림' | '보통' | '빠름';
  samples: Emotion[];
}

export const speakerVoiceProfiles: Record<string, SpeakerVoiceProfile> = {
  jongchan: {
    coverage: 8,
    pitch: '중간',
    pace: '빠름',
    samples: ['happy', 'surprised', 'sad', 'fearful', 'neutral', 'angry'],
  },
  jaewoong: {
    coverage: 6,
    pitch: '중간',
    pace: '보통',
    samples: ['neutral', 'happy', 'sad', 'fearful', 'surprised'],
  },
  kyungtaek: {
    coverage: 7,
    pitch: '높음',
    pace: '빠름',
    samples: ['surprised', 'happy', 'angry', 'neutral', 'sad', 'disgusted'],
  },
  taewon: {
    coverage: 5,
    pitch: '중간',
    pace: '보통',
    samples: ['neutral', 'fearful', 'angry', 'happy'],
  },
};

export const myProfile = {
  name: '이재웅',
  handle: '@jaewoong',
  initial: '재',
  friendCount: 12,
};

export const voiceProfile: VoiceProfile = {
  registered: true,
  sentenceCount: 6,
  durationSec: 90,
  emotionCoverage: 8,
  detectedEmotions: ['happy', 'sad', 'surprised', 'fearful', 'neutral', 'angry', 'disgusted', 'unk'],
};
