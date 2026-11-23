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

// 지금 버킷이 누구의 인박스인지. 실시간 메시지가 어느 버킷에 들어갈지 정할 때 쓴다.
let currentUserId: number | null = null;
// refresh 응답이 오는 동안 실시간으로 받은 메시지. 응답이 그보다 먼저 조회된 목록이면
// 덮어쓸 때 빠질 수 있으므로, 덮어쓴 뒤 다시 합친다.
let inFlight = 0;
let liveDuringRefresh: ReceivedMessage[] = [];

async function refresh(myBackendId: number): Promise<void> {
  inFlight += 1;
  try {
    const resp = await api.setMyId(myBackendId);
    current = normalize(resp);
    currentUserId = myBackendId;
    for (const rec of liveDuringRefresh) insert(rec);
    notify();
  } finally {
    inFlight -= 1;
    if (inFlight === 0) liveDuringRefresh = [];
  }
}

// 메시지 하나를 상대 버킷에 시간순으로 넣는다. 이미 있으면(같은 message_id) 넣지 않는다.
// 바뀌었으면 true.
function insert(rec: ReceivedMessage): boolean {
  if (currentUserId == null) return false;
  const me = String(currentUserId);
  const sender = String(rec.sender_id);
  const receiver = String(rec.receiver_id);
  if (sender !== me && receiver !== me) return false; // 다른 사용자의 메시지 (프로필 전환 직후 등)
  const peer = Number(sender === me ? receiver : sender);
  if (!(peer >= 0 && peer <= 3) || peer === currentUserId) return false;

  const bucket = current[peer] ?? [];
  if (bucket.some((m) => m.message_id === rec.message_id)) return false;
  const next = [...bucket, rec].sort((a, b) => String(a.send_time).localeCompare(String(b.send_time)));
  const buckets = [...current] as InboxBuckets;
  buckets[peer] = next;
  current = buckets;
  return true;
}

/** 실시간(SSE)으로 받은 메시지 하나를 인박스에 더한다. */
function addMessage(rec: ReceivedMessage): void {
  if (inFlight > 0) liveDuringRefresh.push(rec);
  if (insert(rec)) notify();
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
    addMessage,
    setBuckets,
    getBucket,
  };
}
