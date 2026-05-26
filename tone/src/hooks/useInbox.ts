import { useSyncExternalStore } from 'react';
import { api, type InboxBuckets, type ReceivedMessage } from '@/lib/api';

// In-memory only: each /set_my_id call replaces the buckets.
// Bucket at index === my own backend id stays empty (self bucket).
const EMPTY: InboxBuckets = [[], [], [], []];

let current: InboxBuckets = EMPTY;
const subscribers = new Set<() => void>();

function notify() {
  subscribers.forEach((cb) => cb());
}

function subscribe(cb: () => void) {
  subscribers.add(cb);
  return () => {
    subscribers.delete(cb);
  };
}

function read(): InboxBuckets {
  return current;
}

// Backend may return the buckets directly OR wrap them as
// { message, user_id, get_pending_messages: [[],[],[],[]] }.
// Accept both shapes here so callers don't have to care.
function normalize(resp: unknown): InboxBuckets {
  if (Array.isArray(resp)) return resp as InboxBuckets;
  if (resp && typeof resp === 'object' && 'get_pending_messages' in resp) {
    const inner = (resp as { get_pending_messages: unknown }).get_pending_messages;
    if (Array.isArray(inner)) return inner as InboxBuckets;
  }
  return EMPTY;
}

async function refresh(myBackendId: number): Promise<void> {
  const resp = await api.setMyId(myBackendId);
  current = normalize(resp);
  notify();
}

function setBuckets(buckets: InboxBuckets) {
  current = buckets;
  notify();
}

function getBucket(peerBackendId: 0 | 1 | 2 | 3): ReceivedMessage[] {
  return current[peerBackendId] ?? [];
}

export function useInbox() {
  const inbox = useSyncExternalStore(subscribe, read, () => EMPTY);
  return {
    inbox,
    refresh,
    setBuckets,
    getBucket,
  };
}
