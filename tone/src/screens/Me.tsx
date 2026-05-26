import { useNavigate, useOutletContext } from 'react-router-dom';
import { paletteFor } from '@/tokens/emotions';
import { EmotionAvatar } from '@/components/EmotionAvatar';
import { Avatar } from '@/components/Avatar';
import { EMOTION_LABELS, AVATAR_EMOTIONS } from '@/types';
import { useUserAvatar, type UserAvatar } from '@/hooks/useUserAvatar';
import { useVoiceProfile } from '@/hooks/useVoiceProfile';
import { useProfiles } from '@/hooks/useProfiles';
import type { ShellContext } from '@/App';

export function Me() {
  const { mode, setMode } = useOutletContext<ShellContext>();
  const [avatar, setAvatar] = useUserAvatar();
  const { activeProfile, profiles, clearActive, updateProfileAvatar } = useProfiles();
  const navigate = useNavigate();

  const displayName = activeProfile?.name ?? '프로필 미선택';
  const handle = activeProfile ? `@${activeProfile.id}` : '';
  const friendCount = Math.max(profiles.length - 1, 0);

  const handleAvatarChange = (next: UserAvatar) => {
    setAvatar(next);
    if (activeProfile) updateProfileAvatar(activeProfile.id, next);
  };

  const handleSwitchProfile = () => {
    clearActive();
    navigate('/profiles');
  };

  return (
    <div className="flex flex-col h-full bg-cream w-full min-w-0">
      <header
        className="px-12 py-5 flex-shrink-0 flex items-center justify-between"
        style={{ borderBottom: '0.5px solid rgba(20,19,15,0.06)' }}
      >
        <h1 className="text-[18px] font-semibold text-ink">나</h1>
        <button
          type="button"
          onClick={handleSwitchProfile}
          className="px-3 py-[6px] rounded-[10px] text-[12px] font-medium text-muted box-border hover:bg-white transition-colors"
          style={{ border: '0.5px solid rgba(20,19,15,0.12)' }}
        >
          프로필 전환
        </button>
      </header>

      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="px-12 py-8 flex flex-col gap-6">
          <section className="flex items-center gap-5">
            <Avatar avatar={avatar} size={80} />
            <div className="flex-1 min-w-0">
              <div className="text-[20px] font-semibold text-ink leading-tight">
                {displayName}
              </div>
              {handle && (
                <div className="text-[13px] text-muted font-mono mt-1">{handle}</div>
              )}
            </div>
          </section>

          <section>
            <p className="text-[12px] text-hint tracking-wider mb-2 font-medium">프로필 이미지</p>
            <AvatarPicker selected={avatar} onSelect={handleAvatarChange} />
          </section>

          <section>
            <p className="text-[12px] text-hint tracking-wider mb-2 font-medium">화자 프로필</p>
            <VoiceProfileCard />
          </section>

          <section>
            <p className="text-[12px] text-hint tracking-wider mb-2 font-medium">보기 모드</p>
            <div
              className="bg-white rounded-[14px] grid grid-cols-2"
              style={{ boxShadow: '0 1px 0 rgba(20,19,15,0.04)' }}
            >
              <ModeCard
                label="텍스트 모드"
                desc="텍스트 우선. 라이트 테마, 감정 색·강도 시각화."
                selected={mode === 'text'}
                onSelect={() => setMode('text')}
              />
              <ModeCard
                label="음성 모드"
                desc="음성 우선. 다크 테마, 자동재생."
                selected={mode === 'voice'}
                onSelect={() => setMode('voice')}
                divider
              />
            </div>
          </section>

          <section>
            <p className="text-[12px] text-hint tracking-wider mb-2 font-medium">설정</p>
            <div
              className="bg-white rounded-[14px] overflow-hidden"
              style={{ boxShadow: '0 1px 0 rgba(20,19,15,0.04)' }}
            >
              <MenuRow label="친구" meta={`${friendCount}명`} />
              <MenuRow label="통계" divider onClick={() => navigate('/stats')} />
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

function AvatarPicker({
  selected,
  onSelect,
}: {
  selected: UserAvatar;
  onSelect: (a: UserAvatar) => void;
}) {
  return (
    <div
      className="bg-white rounded-[14px] p-5"
      style={{ boxShadow: '0 1px 0 rgba(20,19,15,0.04)' }}
    >
      <div className="grid grid-cols-9 gap-2">
        {AVATAR_EMOTIONS.map((emotion) => {
          const isSelected =
            selected.type === 'emotion' && selected.emotion === emotion;
          return (
            <button
              key={emotion}
              type="button"
              onClick={() => onSelect({ type: 'emotion', emotion })}
              className="flex flex-col items-center gap-[6px] py-1 transition-transform hover:scale-105 active:scale-100"
              aria-label={`${EMOTION_LABELS[emotion]} 표정 선택`}
            >
              <span
                className="rounded-full inline-flex items-center justify-center"
                style={{
                  boxShadow: isSelected
                    ? `0 0 0 3px #FFFFFF, 0 0 0 5px #14130F`
                    : 'inset 0 0 0 0.5px rgba(20,19,15,0.08)',
                  borderRadius: '50%',
                }}
              >
                <EmotionAvatar emotion={emotion} size={56} />
              </span>
              <span
                className="text-[11px] tracking-wide"
                style={{
                  color: isSelected ? '#14130F' : '#9B8E7B',
                  fontWeight: isSelected ? 600 : 400,
                }}
              >
                {EMOTION_LABELS[emotion]}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function VoiceProfileCard() {
  const [voiceProfile] = useVoiceProfile();
  const detected = voiceProfile.detectedEmotions.slice(0, 8);
  const navigate = useNavigate();
  const isRegistered = voiceProfile.registered;

  return (
    <div
      className="bg-white rounded-[14px] p-6"
      style={{ boxShadow: '0 1px 0 rgba(20,19,15,0.04)' }}
    >
      <div className="flex items-center gap-4 mb-5">
        <div
          className="w-20 h-12 flex-shrink-0 flex items-center gap-[1.5px] p-[6px] rounded-[8px]"
          style={{ background: '#FBF1D6' }}
        >
          {[30, 60, 80, 50, 90, 65, 75, 45, 70, 55].map((h, i) => (
            <div
              key={i}
              className="flex-1 rounded-[1px] min-w-[1px]"
              style={{ height: `${h}%`, background: '#F2D89E' }}
            />
          ))}
        </div>
        <div className="flex-1 min-w-0">
          {isRegistered ? (
            <span
              className="inline-flex items-center gap-[6px] text-[12px] px-[10px] py-[3px] rounded-[8px] font-medium"
              style={{ background: '#DEF0E8', color: '#2D6852' }}
            >
              <span className="w-[6px] h-[6px] rounded-full" style={{ background: '#2D6852' }} />
              등록됨
            </span>
          ) : (
            <span
              className="inline-flex items-center gap-[6px] text-[12px] px-[10px] py-[3px] rounded-[8px] font-medium"
              style={{ background: '#FBE3DB', color: '#94402C' }}
            >
              <span className="w-[6px] h-[6px] rounded-full" style={{ background: '#94402C' }} />
              미등록
            </span>
          )}
          <div className="text-[12px] text-muted mt-2 leading-snug">
            {isRegistered
              ? `${voiceProfile.sentenceCount}개 문장 · ${Math.round(voiceProfile.durationSec / 60)}분 ${voiceProfile.durationSec % 60}초 · 감정 범위 ${voiceProfile.emotionCoverage}/9`
              : '목소리를 등록해야 음성 메시지를 전송할 수 있어요.'}
          </div>
        </div>
      </div>

      <div
        className="flex gap-3 items-center pt-4"
        style={{ borderTop: '0.5px solid rgba(20,19,15,0.06)' }}
      >
        <span className="text-[12px] text-hint flex-shrink-0">감지된 감정</span>
        <div className="flex gap-[6px] flex-1 items-center">
          {detected.map((e, i) => {
            const size = Math.max(16 - i * 0.9, 8);
            return (
              <div
                key={e}
                className="rounded-full flex-shrink-0"
                style={{
                  width: `${size}px`,
                  height: `${size}px`,
                  background: paletteFor(e).main,
                }}
              />
            );
          })}
        </div>
        <div className="flex gap-2 flex-shrink-0">
          <button
            type="button"
            onClick={() => navigate('/onboarding')}
            className="px-4 py-[8px] rounded-[10px] text-[12px] font-medium bg-charcoal text-white hover:opacity-90 transition-opacity"
          >
            {isRegistered ? '재등록' : '등록 시작'}
          </button>
          <button
            className="px-4 py-[8px] rounded-[10px] text-[12px] font-medium text-muted box-border hover:bg-cream transition-colors"
            style={{ border: '0.5px solid rgba(20,19,15,0.12)' }}
          >
            상세 보기
          </button>
        </div>
      </div>
    </div>
  );
}

function ModeCard({
  label,
  desc,
  selected,
  onSelect,
  divider = false,
}: {
  label: string;
  desc: string;
  selected: boolean;
  onSelect: () => void;
  divider?: boolean;
}) {
  return (
    <button
      onClick={onSelect}
      className="flex items-start gap-3 p-5 text-left transition-colors hover:bg-cream"
      style={
        divider
          ? { borderLeft: '0.5px solid rgba(20,19,15,0.06)' }
          : undefined
      }
    >
      <div
        className="w-[18px] h-[18px] rounded-full border-[1.5px] flex items-center justify-center flex-shrink-0 box-border mt-[2px]"
        style={{ borderColor: selected ? '#14130F' : '#C8BCAA' }}
      >
        {selected && <div className="w-2 h-2 rounded-full bg-ink" />}
      </div>
      <div className="flex-1">
        <div className="text-[14px] text-ink font-medium leading-tight">{label}</div>
        <div className="text-[12px] text-muted mt-1 leading-snug">{desc}</div>
      </div>
    </button>
  );
}

function MenuRow({
  label,
  meta,
  divider = false,
  onClick,
}: {
  label: string;
  meta?: string;
  divider?: boolean;
  onClick?: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex items-center px-5 py-4 gap-3 w-full hover:bg-cream transition-colors"
      style={divider ? { borderTop: '0.5px solid rgba(20,19,15,0.06)' } : undefined}
    >
      <span className="flex-1 text-[14px] text-ink text-left">{label}</span>
      {meta && <span className="text-[12px] text-hint">{meta}</span>}
      <span className="text-hint text-[16px]">›</span>
    </button>
  );
}
