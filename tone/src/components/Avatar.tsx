import { EmotionAvatar } from '@/components/EmotionAvatar';
import type { UserAvatar } from '@/hooks/useUserAvatar';

interface Props {
  avatar: UserAvatar;
  size?: number;
}

export function Avatar({ avatar, size = 48 }: Props) {
  if (avatar.type === 'photo') {
    return (
      <img
        src={avatar.dataUrl}
        alt=""
        width={size}
        height={size}
        className="rounded-full object-cover flex-shrink-0"
        style={{
          width: size,
          height: size,
          boxShadow: 'inset 0 0 0 0.5px rgba(20,19,15,0.08)',
        }}
      />
    );
  }
  return <EmotionAvatar emotion={avatar.emotion} size={size} />;
}
