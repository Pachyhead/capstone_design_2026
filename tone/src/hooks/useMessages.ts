import type { Message } from '@/types';
import { useInbox } from '@/hooks/useInbox';
import { useProfiles, type BackendId } from '@/hooks/useProfiles';
import { emotionFromBackend, type ReceivedMessage } from '@/lib/api';

export interface MessagesHook {
  thread: Message[];
  reload: () => void;
}

// Backend ships "YYYY-MM-DD HH:MM:SS.ffffff" (or ISO). UI wants "HH:mm".
// Pulls the first HH:MM pair out so either format works.
function formatSentAt(raw: string | undefined): string {
  if (!raw) return '';
  const m = raw.match(/(\d{2}):(\d{2})/);
  return m ? `${m[1]}:${m[2]}` : raw.slice(0, 5);
}

// 백엔드가 메시지별 에너지를 주지 않아 파형은 장식용이다. 메시지 id에서 만든
// 고정 시드를 써서 같은 메시지는 다시 그려도 항상 같은 모양이 되게 한다.
function placeholderEnergy(messageId: string): number[] {
  let seed = 2166136261;
  for (let i = 0; i < messageId.length; i++) {
    seed = Math.imul(seed ^ messageId.charCodeAt(i), 16777619);
  }
  return Array.from({ length: 10 }, () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    const unit = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    return unit * 0.5 + 0.4;
  });
}

export function backendToMessage(
  rec: ReceivedMessage,
  conversationId: string,
  myBackendId?: BackendId,
  peerName?: string,
): Message {
  const isMine = myBackendId != null && String(rec.sender_id) === String(myBackendId);
  return {
    id: rec.message_id,
    conversationId,
    authorId: isMine ? 'me' : String(rec.sender_id),
    authorName: isMine ? undefined : peerName,
    text: rec.message,
    emotion: { primary: emotionFromBackend(rec.emo_type) },
    durationSec: 0,
    energy: placeholderEnergy(String(rec.message_id)),
    sentAt: formatSentAt(rec.send_time),
  };
}

// `peerBackendId` keys into the inbox returned by /set_my_id (both directions:
// messages from the peer and messages I sent to the peer, time-ordered).
// `conversationId` is the front-end-side string id (used for filtering only).
export function useMessages(peerBackendId: BackendId, conversationId: string): MessagesHook {
  const { inbox } = useInbox();
  const { profiles, activeProfile } = useProfiles();
  const peerName = profiles.find((p) => p.backendId === peerBackendId)?.name;
  const myBackendId = activeProfile?.backendId;

  const base: Message[] = (inbox[peerBackendId] ?? []).map((rec) =>
    backendToMessage(rec, conversationId, myBackendId, peerName),
  );

  return {
    // 서버 인박스가 유일한 원본이다. 보낸 메시지도 서버가 함께 돌려주므로
    // 로컬 에코를 따로 들고 있지 않는다(들고 있으면 새로고침 때 중복된다).
    thread: base,
    // refresh button on the chat header drives the inbox; reload is a no-op now.
    reload: () => {},
  };
}
