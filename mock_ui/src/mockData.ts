// `@/data/mock` 자리에 들어가는 모듈. 대화 상대를 데모 프로필 3명으로 맞춘다.
import type { Conversation } from '@/types';
import { DEMO_PROFILES } from './scenario';

export const conversations: Conversation[] = DEMO_PROFILES.map((p) => ({
  id: p.id,
  name: p.name,
  initial: p.name.slice(0, 1),
  language: '한국어',
  lastSeen: '',
  unread: 0,
  backendId: p.backendId,
  lastMessage: { text: '', emotion: { primary: 'neutral' }, sentAt: '' },
}));
