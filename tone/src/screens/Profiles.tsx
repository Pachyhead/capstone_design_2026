import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Avatar } from '@/components/Avatar';
import { EmotionAvatar } from '@/components/EmotionAvatar';
import { useProfiles, type Profile } from '@/hooks/useProfiles';
import { useUserAvatar } from '@/hooks/useUserAvatar';
import { EMOTION_LABELS, AVATAR_EMOTIONS } from '@/types';
import type { Emotion } from '@/types';
import { api } from '@/lib/api';
import { useInbox } from '@/hooks/useInbox';

const MAX_PROFILES = 4;

export function Profiles() {
  const { profiles, addProfile, removeProfile, selectProfile } = useProfiles();
  const [, setUserAvatar] = useUserAvatar();
  const { refresh } = useInbox();
  const [creating, setCreating] = useState(false);
  const [editing, setEditing] = useState(false);
  const navigate = useNavigate();

  const handlePick = (p: Profile) => {
    setUserAvatar(p.avatar);
    selectProfile(p.id);
    refresh(p.backendId).catch((err) => console.warn('[api] setMyId failed:', err));
    navigate('/');
  };

  const handleCreate = (name: string, emotion: Emotion) => {
    const used = new Set(profiles.map((p) => p.backendId));
    const next = ([0, 1, 2, 3] as const).find((id) => !used.has(id));
    if (next === undefined) {
      setCreating(false);
      return;
    }
    const profile: Profile = {
      id: `profile-${Date.now()}`,
      name,
      avatar: { type: 'emotion', emotion },
      backendId: next,
    };
    addProfile(profile);
    setCreating(false);
  };

  const canAdd = profiles.length < MAX_PROFILES;

  return (
    <div className="min-h-screen w-screen bg-cream flex flex-col items-center justify-center p-6">
      <div className="flex flex-col items-center mb-12">
        <div
          className="w-9 h-9 rounded-[10px] flex items-center justify-center text-[14px] font-semibold mb-5"
          style={{ background: '#14130F', color: '#FAF6EF' }}
        >
          T
        </div>
        <h1 className="text-[32px] font-semibold text-ink mb-2 leading-tight">
          누가 사용 중인가요?
        </h1>
        <p className="text-[14px] text-muted">프로필을 선택하세요</p>
      </div>

      <div className="flex items-start gap-8 flex-wrap justify-center max-w-[640px]">
        {profiles.map((p) => (
          <ProfileCircle
            key={p.id}
            profile={p}
            editing={editing}
            onPick={() => handlePick(p)}
            onRemove={() => removeProfile(p.id)}
          />
        ))}
        {canAdd && !creating && !editing && (
          <AddCircle onClick={() => setCreating(true)} />
        )}
        {creating && (
          <CreateCard
            onSubmit={handleCreate}
            onCancel={() => setCreating(false)}
          />
        )}
      </div>

      {profiles.length > 0 && !creating && (
        <button
          type="button"
          onClick={() => setEditing((e) => !e)}
          className="mt-12 px-5 py-[10px] rounded-[10px] text-[13px] text-muted box-border hover:bg-white transition-colors"
          style={{ border: '0.5px solid rgba(20,19,15,0.18)' }}
        >
          {editing ? '완료' : '프로필 관리'}
        </button>
      )}
    </div>
  );
}

function ProfileCircle({
  profile,
  editing,
  onPick,
  onRemove,
}: {
  profile: Profile;
  editing: boolean;
  onPick: () => void;
  onRemove: () => void;
}) {
  return (
    <div className="flex flex-col items-center gap-3 relative">
      <div className="relative">
        <button
          type="button"
          onClick={editing ? undefined : onPick}
          className="rounded-full flex items-center justify-center transition-transform hover:scale-105 active:scale-100"
          style={{
            width: 120,
            height: 120,
            opacity: editing ? 0.6 : 1,
            cursor: editing ? 'default' : 'pointer',
          }}
          aria-label={`${profile.name} 프로필 선택`}
        >
          <Avatar avatar={profile.avatar} size={120} />
        </button>
        {editing && (
          <button
            type="button"
            onClick={onRemove}
            className="absolute -top-1 -right-1 w-8 h-8 rounded-full bg-white text-[18px] flex items-center justify-center hover:bg-cream transition-colors"
            style={{ boxShadow: '0 2px 12px rgba(20,19,15,0.18)', color: '#94402C' }}
            aria-label={`${profile.name} 프로필 삭제`}
          >
            ×
          </button>
        )}
      </div>
      <span className="text-[14px] text-ink font-medium">{profile.name}</span>
    </div>
  );
}

function AddCircle({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex flex-col items-center gap-3 transition-transform hover:scale-105 active:scale-100"
      aria-label="프로필 추가"
    >
      <div
        className="w-[120px] h-[120px] rounded-full flex items-center justify-center text-[40px] text-muted"
        style={{
          background: '#FFFFFF',
          border: '1.5px dashed rgba(20,19,15,0.18)',
        }}
      >
        +
      </div>
      <span className="text-[14px] text-muted font-medium">프로필 추가</span>
    </button>
  );
}

function CreateCard({
  onSubmit,
  onCancel,
}: {
  onSubmit: (name: string, emotion: Emotion) => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState('');
  const [emotion, setEmotion] = useState<Emotion>('happy');
  const trimmed = name.trim();

  return (
    <div
      className="bg-white rounded-[18px] p-5 flex flex-col items-center gap-4 w-[260px]"
      style={{ boxShadow: '0 4px 24px rgba(20,19,15,0.10)' }}
    >
      <EmotionAvatar emotion={emotion} size={88} />
      <input
        type="text"
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="이름"
        autoFocus
        className="w-full px-3 py-[8px] text-[14px] text-center text-ink bg-cream rounded-[10px] outline-none"
        maxLength={20}
      />
      <div className="flex flex-wrap gap-[6px] justify-center">
        {AVATAR_EMOTIONS.map((e) => {
          const selected = emotion === e;
          return (
            <button
              key={e}
              type="button"
              onClick={() => setEmotion(e)}
              className="rounded-full transition-transform hover:scale-105 active:scale-100"
              style={{
                boxShadow: selected
                  ? '0 0 0 2px #FFFFFF, 0 0 0 4px #14130F'
                  : 'inset 0 0 0 0.5px rgba(20,19,15,0.08)',
                borderRadius: '50%',
              }}
              aria-label={`${EMOTION_LABELS[e]} 아바타`}
            >
              <EmotionAvatar emotion={e} size={32} />
            </button>
          );
        })}
      </div>
      <div className="flex gap-2 w-full">
        <button
          type="button"
          onClick={onCancel}
          className="flex-1 py-[9px] rounded-[10px] text-[13px] text-muted box-border hover:bg-cream transition-colors"
          style={{ border: '0.5px solid rgba(20,19,15,0.12)' }}
        >
          취소
        </button>
        <button
          type="button"
          onClick={() => trimmed && onSubmit(trimmed, emotion)}
          disabled={!trimmed}
          className="flex-1 py-[9px] rounded-[10px] text-[13px] font-medium bg-charcoal text-white hover:opacity-90 transition-opacity disabled:opacity-40"
        >
          만들기
        </button>
      </div>
    </div>
  );
}
