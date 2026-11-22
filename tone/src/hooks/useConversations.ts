import { useMemo } from 'react';
import type { Conversation } from '@/types';
import { conversations } from '@/data/mock';
import { useProfiles } from '@/hooks/useProfiles';

// 대화 상대 목록. 이름과 이니셜은 프로필 화면에 보이는 이름과 같아야 하므로
// 프로필(백엔드 id 기준)에서 가져오고, 해당 프로필이 없을 때만 기본값을 쓴다.
export function useConversations(): Conversation[] {
  const { profiles } = useProfiles();
  return useMemo(
    () =>
      conversations.map((conv) => {
        const profile = profiles.find((p) => p.backendId === conv.backendId);
        if (!profile) return conv;
        return { ...conv, name: profile.name, initial: profile.name.slice(0, 1) };
      }),
    [profiles],
  );
}
