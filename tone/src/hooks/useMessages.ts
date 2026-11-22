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
    // 10-bar placeholder energy — backend doesn't ship per-message energy yet
    energy: Array.from({ length: 10 }, () => Math.random() * 0.5 + 0.4),
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
