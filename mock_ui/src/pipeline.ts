import { useSyncExternalStore } from 'react';
import type { Emotion } from '@/types';

// 파이프라인 패널이 보여 주는 상태. mockApi 가 쓰고 PipelinePanel 이 읽는다.
// 노드 이름은 포스터의 System Architecture 그림을 따른다.
export type NodeId =
  | 'record' // 녹음
  | 'stt' // Whisper STT
  | 'text' // 텍스트 (가변)
  | 'emotion2vec'
  | 'fsqEncode' // FSQ 인코더
  | 'bits' // 24bit
  | 'grpcUp' // 송신 단말 → 서버 (텍스트 + 24bit)
  | 'mysql'
  | 'fsqRestore' // FSQ 복원 24bit → 1024d
  | 'ref' // 화자 ref 음성
  | 'ttsText'
  | 'ttsSpeaker' // 화자 임베딩 2048d
  | 'ttsEmotion' // 감정 1024d
  | 'tts' // Qwen3-TTS
  | 'grpcDown' // 서버 → 수신 단말 (합성 음성 스트리밍)
  | 'play'; // 합성 음성 재생

export interface PipelineState {
  active: NodeId[];
  done: NodeId[];
  caption: string;
  /** 지금 흐름에서 드러난 값들. 해당 단계가 끝나야 채워진다. */
  text: string | null;
  emotion: Emotion | null;
  /** 송신 단말에서 감정 벡터가 지금 어떤 모양인지: 1024d → 8d → 24bit */
  vector: '1024' | '8' | '24' | null;
  /** 이 메시지의 실제 감정 코드(FSQ 인덱스 8개). 없으면 도식은 장식으로 그린다. */
  code: number[] | null;
  /** 서버가 24bit 를 1024d 로 되돌렸는지 */
  restored: boolean;
  /** 지금 흐름의 화자가 쓰는 음성 세트 (ref 음성 미리듣기에 쓴다) */
  speakerSetId: string | null;
  speakerName: string | null;
}

export const IDLE_CAPTION = '마이크 버튼을 누르고 있으면 녹음이 시작됩니다';

const EMPTY: PipelineState = {
  active: [],
  done: [],
  caption: IDLE_CAPTION,
  text: null,
  emotion: null,
  vector: null,
  code: null,
  restored: false,
  speakerSetId: null,
  speakerName: null,
};

let state: PipelineState = EMPTY;
const subscribers = new Set<() => void>();

function update(patch: Partial<PipelineState>) {
  state = { ...state, ...patch };
  subscribers.forEach((cb) => cb());
}

/** 새 흐름을 시작한다: 지난 표시를 모두 지운다. */
export function resetPipeline(patch: Partial<PipelineState> = {}) {
  state = { ...EMPTY, ...patch };
  subscribers.forEach((cb) => cb());
}

/** 다음 단계로 넘어간다: 지금까지 켜져 있던 노드는 완료로 바뀐다. */
export function stage(active: NodeId[], caption: string, patch: Partial<PipelineState> = {}) {
  const done = [...new Set([...state.done, ...state.active])].filter((n) => !active.includes(n));
  update({ ...patch, active, done, caption });
}

/** 설명 문구만 바꾼다 (켜진 노드와 값은 그대로 둔다). */
export function setCaption(caption: string) {
  update({ caption });
}

/** 지정한 노드의 완료 표시를 지운다 (같은 메시지를 다시 재생할 때). */
export function clearNodes(nodes: NodeId[]) {
  update({
    active: state.active.filter((n) => !nodes.includes(n)),
    done: state.done.filter((n) => !nodes.includes(n)),
  });
}

// 패널이 펼쳐져 있는지. 기본은 접힘: 이 데모의 목적은 실제 시스템처럼 보이는 것이고,
// 구조 설명은 포스터가 맡는다. 펼치면 mockApi 가 단계마다 설명을 읽을 시간을 준다.
let panelOpen = false;
const panelSubscribers = new Set<() => void>();

export function isPanelOpen(): boolean {
  return panelOpen;
}

export function setPanelOpen(open: boolean) {
  panelOpen = open;
  panelSubscribers.forEach((cb) => cb());
}

export function usePanelOpen(): boolean {
  return useSyncExternalStore(
    (cb) => {
      panelSubscribers.add(cb);
      return () => {
        panelSubscribers.delete(cb);
      };
    },
    () => panelOpen,
  );
}

export function usePipeline(): PipelineState {
  return useSyncExternalStore(
    (cb) => {
      subscribers.add(cb);
      return () => {
        subscribers.delete(cb);
      };
    },
    () => state,
  );
}
