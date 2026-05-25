import type { Emotion } from '@/types';

export interface RecordingScenario {
  text: string;
  emotion: Emotion;
  nuance?: string;
  durationSec: number;
}

// pre-defined "outcomes" the prototype cycles through. picked randomly when
// the user starts recording — simulates STT + emotion2vec returning different
// results based on what the user actually said. real backend would replace this.
export const RECORDING_SCENARIOS: RecordingScenario[] = [
  { text: '내일 약속 잘 부탁해!', emotion: 'happy', nuance: '들뜸', durationSec: 2.4 },
  { text: '잠깐만, 다시 생각해볼게', emotion: 'fearful', nuance: '망설임', durationSec: 3.1 },
  { text: '와 진짜 멋지다 그거', emotion: 'surprised', durationSec: 1.8 },
  { text: '오늘 진짜 힘들었어 ㅠ', emotion: 'sad', durationSec: 4.2 },
  { text: '응 알겠어, 그렇게 하자', emotion: 'neutral', durationSec: 2.0 },
];

export function pickScenario(exclude?: RecordingScenario): RecordingScenario {
  if (!exclude) {
    return RECORDING_SCENARIOS[Math.floor(Math.random() * RECORDING_SCENARIOS.length)];
  }
  const others = RECORDING_SCENARIOS.filter((s) => s !== exclude);
  return others[Math.floor(Math.random() * others.length)];
}
