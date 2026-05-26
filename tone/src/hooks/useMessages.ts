import { useEffect, useState } from 'react';
import type { Message } from '@/types';
import { useInbox } from '@/hooks/useInbox';
import { useProfiles, type BackendId } from '@/hooks/useProfiles';
import { emotionFromBackend, type ReceivedMessage } from '@/lib/api';

export interface MessagesHook {
  thread: Message[];
  append: (m: Message) => void;
  reload: () => void;
}

// Backend ships "YYYY-MM-DD HH:MM:SS.ffffff" (or ISO). UI wants "HH:mm".
// Pulls the first HH:MM pair out so either format works.
function formatSentAt(raw: string | undefined): string {
  if (!raw) return '';
  const m = raw.match(/(\d{2}):(\d{2})/);
  return m ? `${m[1]}:${m[2]}` : raw.slice(0, 5);
}

function backendToMessage(
  rec: ReceivedMessage,
  conversationId: string,
  peerName?: string,
): Message {
  return {
    id: rec.message_id,
    conversationId,
    authorId: String(rec.sender_id),
    authorName: peerName,
    text: rec.message,
    emotion: { primary: emotionFromBackend(rec.emo_type) },
    durationSec: 0,
    // 10-bar placeholder energy — backend doesn't ship per-message energy yet
    energy: Array.from({ length: 10 }, () => Math.random() * 0.5 + 0.4),
    sentAt: formatSentAt(rec.send_time),
  };
}

// `peerBackendId` keys into the inbox returned by /set_my_id.
// `conversationId` is the front-end-side string id (used for filtering only).
export function useMessages(peerBackendId: BackendId, conversationId: string): MessagesHook {
  const { inbox } = useInbox();
  const { profiles } = useProfiles();
  const peerName = profiles.find((p) => p.backendId === peerBackendId)?.name;

  const base: Message[] = (inbox[peerBackendId] ?? []).map((rec) =>
    backendToMessage(rec, conversationId, peerName),
  );

  // Local echo for messages the user just sent — backend's inbox is incoming-only,
  // so outgoing bubbles live in component-local state until next mount.
  const [appended, setAppended] = useState<Message[]>([]);
  useEffect(() => {
    setAppended([]);
  }, [conversationId]);

  return {
    thread: [...base, ...appended],
    append: (m) => setAppended((prev) => [...prev, m]),
    // refresh button on the chat header drives the inbox; reload is a no-op now.
    reload: () => {},
  };
}
