import { useSyncExternalStore } from 'react';
import type { CodecSizes, RecordResult } from '@/lib/api';

// in-memory store for the floating metrics dock. api.ts writes, MetricsDock reads.
// state is replaced as a whole on every update so useSyncExternalStore sees a new snapshot.

export interface ApiCall {
  path: string;
  status: number; // 0 = network error
  ms: number;
  at: number; // epoch ms
}

export type CodecState = 'idle' | 'running' | 'error';

export interface MetricsState {
  calls: ApiCall[];
  lastRecording: RecordResult | null;
  codec: CodecSizes | null;
  codecState: CodecState;
  codecError: string | null;
}

const MAX_CALLS = 30;

let state: MetricsState = {
  calls: [],
  lastRecording: null,
  codec: null,
  codecState: 'idle',
  codecError: null,
};

const subscribers = new Set<() => void>();

function update(patch: Partial<MetricsState>) {
  state = { ...state, ...patch };
  subscribers.forEach((cb) => cb());
}

function subscribe(cb: () => void) {
  subscribers.add(cb);
  return () => {
    subscribers.delete(cb);
  };
}

export function recordApiCall(call: ApiCall) {
  update({ calls: [...state.calls, call].slice(-MAX_CALLS) });
}

export function setLastRecording(result: RecordResult) {
  update({ lastRecording: result });
}

export function setCodecState(patch: Pick<Partial<MetricsState>, 'codec' | 'codecState' | 'codecError'>) {
  update(patch);
}

export function useMetrics(): MetricsState {
  return useSyncExternalStore(subscribe, () => state);
}
