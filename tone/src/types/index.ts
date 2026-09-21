// 9 emotions — labels mirror the backend emotion2vec categorical output verbatim
export type Emotion =
  | 'happy'
  | 'sad'
  | 'angry'
  | 'surprised'
  | 'fearful'
  | 'disgusted'
  | 'unk'
  | 'neutral'
  | 'other';

export const EMOTION_LABELS: Record<Emotion, string> = {
  happy: '기쁨',
  sad: '슬픔',
  angry: '분노',
  surprised: '놀람',
  fearful: '두려움',
  disgusted: '혐오',
  unk: '미상',
  neutral: '중립',
  other: '기타',
};

// 'unk'(미상)은 백엔드 오류/미상 폴백값이므로 사용자 선택 목록에서는 제외
export const AVATAR_EMOTIONS: Emotion[] = [
  'happy',
  'sad',
  'surprised',
  'angry',
  'fearful',
  'disgusted',
  'neutral',
  'other',
];

// optional secondary descriptor — purely UI sugar for now
export interface EmotionLabel {
  primary: Emotion;
  nuance?: string; // 들뜸, 망설임, 걱정 etc.
}

export interface Message {
  id: string;
  conversationId: string;
  authorId: string; // 'me' | friendId
  authorName?: string; // shown above received bubbles in group chats
  text: string;
  emotion: EmotionLabel;
  durationSec: number;
  // 10 normalized energy values [0..1] — UI display only
  // backend would derive this from audio RMS energy, not from FSQ
  energy: number[];
  sentAt: string; // HH:mm
}

export interface Conversation {
  id: string;
  name: string;
  initial: string;
  language: string;
  lastSeen: string;
  unread: number;
  backendId: 0 | 1 | 2 | 3;
  // last message preview — enough to render the chat list row
  lastMessage: Pick<Message, 'text' | 'emotion' | 'sentAt'>;
}

export interface ViewMode {
  kind: 'text' | 'voice';
}

export interface VoiceProfile {
  registered: boolean;
  sentenceCount: number;
  durationSec: number;
  emotionCoverage: number; // 0..9
  detectedEmotions: Emotion[]; // ordered by prominence
}

export interface ChatroomStats {
  conversationId: string;
  messageCount: number;
  // distribution of emotions in this conversation, sums to 1
  distribution: Partial<Record<Emotion, number>>;
}
