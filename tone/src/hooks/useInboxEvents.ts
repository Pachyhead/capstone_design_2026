import { useEffect, useRef } from 'react';
import { api, audioUrl, setEndpointResetHandler, type ReceivedMessage } from '@/lib/api';
import { useInbox } from '@/hooks/useInbox';

// 엔드포인트의 /events(SSE)를 구독해 새 메시지를 새로고침 없이 인박스에 반영한다.
//   message: 새 메시지 1건 → 상대 버킷에 추가 (이미 있으면 무시)
//   resync:  엔드포인트가 서버 구독을 (다시) 연결함 → 끊겨 있던 동안의 메시지를 위해 목록을 다시 받음
// 연결이 끊기면 브라우저(EventSource)가 알아서 다시 접속한다.
//
// 엔드포인트가 재시작되면 사용자/수신자 id가 기본값으로 돌아간다. 그러면 SSE도 반드시
// 끊겼다가 다시 연결되므로, 끊긴 뒤 다시 연결될 때 id를 다시 알린다 (syncEndpoint).
// 엔드포인트가 409(id 모름)로 거부할 때도 같은 처리를 한다.
// 페이지를 처음 열 때의 첫 연결은 App.tsx(set_my_id)와 ChatDetail.tsx(set_receiver_id)가
// 이미 알리므로 건너뛴다.
export function useInboxEvents(myBackendId: number | undefined): void {
  const { refresh, addMessage } = useInbox();
  const myIdRef = useRef(myBackendId);
  myIdRef.current = myBackendId;

  useEffect(() => {
    if (typeof EventSource === 'undefined') return;

    let syncing: Promise<void> | null = null;
    // 사용자 id를 먼저 보낸다 (/set_receiver_id는 내 id와 같은 값을 거부하므로).
    const syncEndpoint = () => {
      const id = myIdRef.current;
      if (id == null || syncing) return;
      syncing = refresh(id)
        .then(() => api.resendReceiverId())
        .then(() => console.info('[events] endpoint ids re-sent'))
        .catch((err) => console.warn('[events] endpoint id sync failed', err))
        .finally(() => {
          syncing = null;
        });
    };

    const source = new EventSource(audioUrl('/events'));
    let dropped = false; // 이 연결이 한 번이라도 끊겼는지 (처음 연결이 실패한 경우 포함)

    const onOpen = () => {
      if (!dropped) return;
      dropped = false;
      syncEndpoint();
    };
    const onError = () => {
      dropped = true;
    };
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

    source.addEventListener('open', onOpen);
    source.addEventListener('error', onError);
    source.addEventListener('message', onMessage);
    source.addEventListener('resync', onResync);
    setEndpointResetHandler(syncEndpoint);
    return () => {
      setEndpointResetHandler(null);
      source.removeEventListener('open', onOpen);
      source.removeEventListener('error', onError);
      source.removeEventListener('message', onMessage);
      source.removeEventListener('resync', onResync);
      source.close();
    };
  }, []);
}
