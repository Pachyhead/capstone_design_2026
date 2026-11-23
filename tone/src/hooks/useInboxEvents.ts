import { useEffect, useRef } from 'react';
import { audioUrl, type ReceivedMessage } from '@/lib/api';
import { useInbox } from '@/hooks/useInbox';

// 엔드포인트의 /events(SSE)를 구독해 새 메시지를 새로고침 없이 인박스에 반영한다.
//   message: 새 메시지 1건 → 상대 버킷에 추가 (이미 있으면 무시)
//   resync:  엔드포인트가 서버 구독을 (다시) 연결함 → 끊겨 있던 동안의 메시지를 위해 목록을 다시 받음
// 연결이 끊기면 브라우저(EventSource)가 알아서 다시 접속한다.
export function useInboxEvents(myBackendId: number | undefined): void {
  const { refresh, addMessage } = useInbox();
  const myIdRef = useRef(myBackendId);
  myIdRef.current = myBackendId;

  useEffect(() => {
    if (typeof EventSource === 'undefined') return;
    const source = new EventSource(audioUrl('/events'));

    const onMessage = (e: MessageEvent<string>) => {
      try {
        addMessage(JSON.parse(e.data) as ReceivedMessage);
      } catch (err) {
        console.warn('[events] bad message event', err);
      }
    };
    const onResync = () => {
      const id = myIdRef.current;
      if (id == null) return;
      refresh(id).catch((err) => console.warn('[events] resync failed', err));
    };

    source.addEventListener('message', onMessage);
    source.addEventListener('resync', onResync);
    return () => {
      source.removeEventListener('message', onMessage);
      source.removeEventListener('resync', onResync);
      source.close();
    };
  }, []);
}
