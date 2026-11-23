import type { Emotion } from '@/types';

// 데모 시나리오. 문장·이름을 바꾸려면 이 파일을 고치고 `npm run build` 를 다시 한다.
//
// 프로필은 백엔드 id 로 구분한다: 0 종찬, 1 재웅, 2 경택, 3 태원.
// 음성은 그 사람의 id 폴더에 둔다 (src/audio/<id>/):
//   reference.wav — 목소리 등록 때 정해진 문장을 읽는 소리
//   source.wav    — 메시지를 보낼 때 말하는 소리 (녹음 구간에서 들린다)
//   output.wav    — 받는 쪽에서 합성된 소리 (상대측 재생에서 들린다). `text` 를 말한다
//
// 처음부터 들어 있는 대화 기록(음성 없음)은 history.ts 에 있다.

export type BackendId = 0 | 1 | 2 | 3;

/** 한 사람이 데모에서 보내는 메시지 하나와 거기에 딸린 음성. */
export interface VoiceSet {
  /** 음성 폴더 이름. 그 사람의 백엔드 id 와 같다. */
  id: string;
  emotion: Emotion;
  /** 백엔드 EmotionLabel 번호 (angry 0, fearful 2, happy 3, surprised 7 …) */
  emoType: number;
  /** 감정 분류의 확신도 (0~1). 실제 인코딩 결과의 emotion_score. */
  emotionScore: number;
  /** output.wav 가 말하는 문장. 채팅 메시지로 보인다. */
  text: string;
  /** reference.wav 의 길이(초). 목소리가 이미 등록된 상태로 시작할 때 표시에 쓴다. */
  referenceDuration: number;
  /** source.wav 의 길이(초)와 크기(바이트). 녹음 결과와 전송 크기 비교에 쓴다. */
  sourceDuration: number;
  sourceBytes: number;
  /**
   * DB 에 저장된 실제 감정 코드: FSQ 인덱스 8개, 각각 0~7 (3bit × 8 = 24bit).
   * 넣어 두면 구조 그림의 8d 막대와 24bit 칸이 이 값을 그대로 보여 준다.
   * 없으면 그림은 크기만 맞춘 장식으로 그려진다.
   */
  emotionCode?: number[];
}

export interface DemoProfile {
  id: string;
  name: string;
  backendId: BackendId;
}

export const DEMO_PROFILES: DemoProfile[] = [
  { id: 'jongchan', name: '종찬', backendId: 0 },
  { id: 'jaewoong', name: '재웅', backendId: 1 },
  { id: 'kyungtaek', name: '경택', backendId: 2 },
  { id: 'taewon', name: '태원', backendId: 3 },
];

// 준비된 음성. 지금은 종찬(0)만 있다. 나머지 프로필로는 대화 기록과 통계만 볼 수 있고,
// 새 메시지를 녹음하면 "준비된 음성이 없다"는 안내가 뜬다.
//
// 종찬의 값은 실제 시스템이 이 source.wav 를 인코딩한 결과(EncodeResult)를 그대로 옮긴 것이다:
//   text='진짜? 재웅이가 그런 걸 했다고?', emotion_indices=(0, 7, 7, 3, 7, 2, 0, 4),
//   emotion_label=SURPRISED(7), emotion_score=0.9943621158599854
export const VOICE_SETS: VoiceSet[] = [
  {
    id: '0',
    emotion: 'surprised',
    emoType: 7,
    emotionScore: 0.9943621158599854,
    emotionCode: [0, 7, 7, 3, 7, 2, 0, 4],
    text: '진짜? 재웅이가 그런 걸 했다고?',
    referenceDuration: 8.7,
    sourceDuration: 2.2,
    // 녹음기가 저장한 원본(32비트 float WAV)의 크기. src/audio 의 파일은 16비트로 변환한 사본이다.
    sourceBytes: 141370,
  },
];

// 자동 시연에서 누가 누구에게 보내는지. "처음부터 보기"는 보내는 사람의 프로필을
// 새로 만드는 것부터 보여 준다.
export const AUTOPLAY: { senderId: string; receiverId: string; avatar: Emotion } = {
  senderId: 'jongchan',
  receiverId: 'jaewoong',
  /** 프로필을 만들 때 고르는 아바타 표정 */
  avatar: 'happy',
};

export function setById(id: string): VoiceSet | undefined {
  return VOICE_SETS.find((s) => s.id === id);
}

/** 이 사용자(백엔드 id)의 음성 세트. 준비된 음성이 없으면 undefined. */
export function setForUser(backendId: number): VoiceSet | undefined {
  return setById(String(backendId));
}

export function audioPath(setId: string, role: 'reference' | 'source' | 'output'): string {
  return `audio/${setId}/${role}.wav`;
}
