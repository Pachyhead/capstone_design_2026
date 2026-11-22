import { useEffect, useMemo } from 'react';
import type { Message } from '@/types';
import { conversations } from '@/data/mock';
import { useInbox } from '@/hooks/useInbox';
import { useProfiles, type BackendId } from '@/hooks/useProfiles';
import { backendToMessage } from '@/hooks/useMessages';

// 통계용 메시지: 백엔드 /set_my_id 가 돌려주는 대화 내역(보낸 것 + 받은 것)을
// 대화방별 Message 로 변환한다. 내 프로필 버킷은 항상 비어 있으므로 제외된다.
export function useStatsMessages(): Message[] {
  const { activeProfile } = useProfiles();
  const { inbox, refresh } = useInbox();
  const myBackendId = activeProfile?.backendId;

  // 통계 화면으로 바로 들어와도 최신 내역을 받도록 진입 시 한 번 갱신한다.
  useEffect(() => {
    if (myBackendId == null) return;
    refresh(myBackendId).catch((e) => console.warn('[stats] inbox refresh failed', e));
  }, [myBackendId]);

  return useMemo(() => {
    const out: Message[] = [];
    for (const conv of conversations) {
      if (conv.backendId === myBackendId) continue;
      const bucket = inbox[conv.backendId as BackendId] ?? [];
      for (const rec of bucket) {
        out.push({ ...backendToMessage(rec, conv.id, myBackendId, conv.name), energy: [] });
      }
    }
    return out;
  }, [inbox, myBackendId]);
}
