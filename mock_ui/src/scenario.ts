import type { Emotion } from '@/types';

// 데모 시나리오. 문장·이름을 바꾸려면 이 파일을 고치고 `npm run build` 를 다시 한다.
//
// 프로필은 백엔드 id 로 구분한다: 0 종찬, 1 재웅, 2 경택, 3 태원.
// 음성은 그 사람의 id 폴더에 둔다 (public/audio/<id>/):
//   reference.wav — 목소리 등록 때 정해진 문장을 읽는 소리
//   source.wav    — 메시지를 보낼 때 말하는 소리 (녹음 구간에서 들린다)
//   output.wav    — 받는 쪽에서 합성된 소리 (상대측 재생에서 들린다). `text` 를 말한다
//
// 지금 들어 있는 파일은 임시다. 팀원 목소리로 녹음한 파일을 받으면 같은 이름으로
// 바꿔 넣는다. 문장은 음성을 Whisper 로 받아 적은 것이라 틀린 곳이 있을 수 있다.
//
// 처음부터 들어 있는 대화 기록(음성 없음)은 history.ts 에 있다.

export type BackendId = 0 | 1 | 2 | 3;

/** 한 사람이 데모에서 보내는 메시지 하나와 거기에 딸린 음성. */
export interface VoiceSet {
  /** 음성 폴더 이름. 그 사람의 백엔드 id 와 같다. */
  id: string;
  emotion: Emotion;
  /** 백엔드 EmotionLabel 번호 (angry 0, fearful 2, happy 3 …) */
  emoType: number;
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

// 준비된 음성. 태원(3)은 음성이 없어서, 태원으로는 대화 기록과 통계만 볼 수 있고
// 새 메시지를 녹음하면 "준비된 음성이 없다"는 안내가 뜬다.
export const VOICE_SETS: VoiceSet[] = [
  {
    id: '0',
    emotion: 'happy',
    emoType: 3,
    text: '혹시 신비한 동물 사전은 다 봤나요?',
    referenceDuration: 8.7,
    sourceDuration: 3.3,
    sourceBytes: 105772,
  },
  {
    id: '1',
    emotion: 'angry',
    emoType: 0,
    text: '니가 원하는 거 해. 아까 그 얘기 마저 하자. 그 얘기 아까 더 할 거 남아있어.',
    referenceDuration: 5.1,
    sourceDuration: 7.3,
    sourceBytes: 232748,
  },
  {
    id: '2',
    emotion: 'fearful',
    emoType: 2,
    text: '그래서 내가 얼마만큼 이해를 해줘야 이해를 잘 해줬다고 소문이 날까. 얼마만큼 이해를 해줘야…',
    referenceDuration: 8.2,
    sourceDuration: 3.1,
    sourceBytes: 98604,
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
