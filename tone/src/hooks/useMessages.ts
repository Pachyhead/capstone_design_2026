import { useEffect, useState } from 'react';
import { messages as mockMessages } from '@/data/mock';
import type { Message } from '@/types';

export interface MessagesHook {
  thread: Message[];
  append: (m: Message) => void;
  reload: () => void;
}

// Single swap point for moving from mock to real backend fetch.
// When backend exposes `GET /messages?conversation_id=X`, replace the body
// of `fetchThread` with `api.getMessages(conversationId)` and add polling.
async function fetchThread(conversationId: string): Promise<Message[]> {
  return mockMessages.filter((m) => m.conversationId === conversationId);
}

export function useMessages(conversationId: string): MessagesHook {
  const [base, setBase] = useState<Message[]>([]);
  const [appended, setAppended] = useState<Message[]>([]);

  useEffect(() => {
    setAppended([]);
    let active = true;
    fetchThread(conversationId).then((next) => {
      if (active) setBase(next);
    });
    return () => {
      active = false;
    };
  }, [conversationId]);

  return {
    thread: [...base, ...appended],
    append: (m) => setAppended((prev) => [...prev, m]),
    reload: () => {
      fetchThread(conversationId).then(setBase);
    },
  };
}
