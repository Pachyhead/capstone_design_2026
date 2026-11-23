import type { Emotion } from '@/types';

// 데모 시나리오. 문장·이름을 바꾸려면 이 파일을 고치고 `npm run build` 를 다시 한다.
//
// 음성 세트 하나의 구성 (public/audio/<id>/):
//   reference.wav — 화자가 미리 등록한 목소리 샘플
//   source.wav    — 감정이 담긴 발화. 여기서 감정만 뽑는다
//   output.wav    — 합성 결과. reference 의 목소리로 `text` 를 source 의 감정으로 말한다
//
// 문장은 음성을 Whisper 로 받아 적은 것이라 틀린 곳이 있을 수 있다.

export interface VoiceSet {
  id: string;
  emotion: Emotion;
  /** 백엔드 EmotionLabel 번호 (angry 0, fearful 2, happy 3 …) */
  emoType: number;
  /** output.wav 가 말하는 문장. 채팅 메시지로 보인다. */
  text: string;
  /** source.wav 의 길이(초)와 크기(바이트). 녹음 결과와 전송 크기 비교에 쓴다. */
  sourceDuration: number;
  sourceBytes: number;
}

export interface DemoProfile {
  id: string;
  name: string;
  backendId: 0 | 1 | 2;
  /** 이 프로필이 메시지를 보낼 때 쓰는 음성 세트 */
  setId: string;
}

export const VOICE_SETS: VoiceSet[] = [
  {
    id: 'happy',
    emotion: 'happy',
    emoType: 3,
    text: '혹시 신비한 동물 사전은 다 봤나요?',
    sourceDuration: 3.3,
    sourceBytes: 105772,
  },
  {
    id: 'angry',
    emotion: 'angry',
    emoType: 0,
    text: '니가 원하는 거 해. 아까 그 얘기 마저 하자. 그 얘기 아까 더 할 거 남아있어.',
    sourceDuration: 7.3,
    sourceBytes: 232748,
  },
  {
    id: 'fearful',
    emotion: 'fearful',
    emoType: 2,
    text: '그래서 내가 얼마만큼 이해를 해줘야 이해를 잘 해줬다고 소문이 날까. 얼마만큼 이해를 해줘야…',
    sourceDuration: 3.1,
    sourceBytes: 98604,
  },
];

export const DEMO_PROFILES: DemoProfile[] = [
  { id: 'jongchan', name: '종찬', backendId: 0, setId: 'happy' },
  { id: 'jaewoong', name: '재웅', backendId: 1, setId: 'angry' },
  { id: 'kyungtaek', name: '경택', backendId: 2, setId: 'fearful' },
];

export function setById(id: string): VoiceSet | undefined {
  return VOICE_SETS.find((s) => s.id === id);
}

/** 이 사용자(백엔드 id)가 보낼 때 쓰는 음성 세트. 데모 프로필이 아니면 undefined. */
export function setForUser(backendId: number): VoiceSet | undefined {
  const profile = DEMO_PROFILES.find((p) => p.backendId === backendId);
  return profile ? setById(profile.setId) : undefined;
}

export function audioPath(setId: string, role: 'reference' | 'source' | 'output'): string {
  return `audio/${setId}/${role}.wav`;
}
