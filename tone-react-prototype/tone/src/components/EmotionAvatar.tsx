import type { Emotion } from '@/types';
import { paletteFor } from '@/tokens/emotions';

interface Props {
  emotion: Emotion;
  size?: number;
  selected?: boolean;
  onClick?: () => void;
  ariaLabel?: string;
}

export function EmotionAvatar({ emotion, size = 72, selected, onClick, ariaLabel }: Props) {
  const palette = paletteFor(emotion);
  const stroke = palette.x;
  const isButton = typeof onClick === 'function';

  const inner = (
    <svg viewBox="0 0 100 100" width={size} height={size} aria-hidden="true">
      <circle cx="50" cy="50" r="50" fill={palette.main} />
      <Features emotion={emotion} stroke={stroke} />
    </svg>
  );

  if (isButton) {
    return (
      <button
        type="button"
        onClick={onClick}
        aria-label={ariaLabel}
        className="rounded-full flex items-center justify-center transition-transform hover:scale-105 active:scale-100 focus:outline-none"
        style={{
          width: size,
          height: size,
          boxShadow: selected
            ? `0 0 0 3px #FAF6EF, 0 0 0 5px ${palette.deep}`
            : 'inset 0 0 0 0.5px rgba(20,19,15,0.08)',
          borderRadius: '50%',
        }}
      >
        {inner}
      </button>
    );
  }

  return (
    <div
      className="rounded-full flex items-center justify-center"
      style={{ width: size, height: size }}
    >
      {inner}
    </div>
  );
}

function Features({ emotion, stroke }: { emotion: Emotion; stroke: string }) {
  const sw = 3;
  const props = { stroke, strokeWidth: sw, fill: 'none', strokeLinecap: 'round' as const };

  switch (emotion) {
    case 'happy':
      return (
        <>
          <path d="M28 44 Q34 36 40 44" {...props} />
          <path d="M60 44 Q66 36 72 44" {...props} />
          <path d="M30 60 Q50 80 70 60" {...props} />
        </>
      );

    case 'sad':
      return (
        <>
          <circle cx="34" cy="46" r="3" fill={stroke} />
          <circle cx="66" cy="46" r="3" fill={stroke} />
          <path d="M30 74 Q50 62 70 74" {...props} />
        </>
      );

    case 'angry':
      return (
        <>
          <path d="M24 32 L42 40" {...props} />
          <path d="M76 32 L58 40" {...props} />
          <circle cx="34" cy="50" r="2.6" fill={stroke} />
          <circle cx="66" cy="50" r="2.6" fill={stroke} />
          <path d="M34 70 Q50 64 66 70" {...props} />
        </>
      );

    case 'surprised':
      return (
        <>
          <circle cx="34" cy="44" r="5" {...props} />
          <circle cx="66" cy="44" r="5" {...props} />
          <ellipse cx="50" cy="70" rx="6" ry="8" {...props} />
        </>
      );

    case 'fearful':
      return (
        <>
          <ellipse cx="34" cy="44" rx="4.5" ry="5.5" {...props} />
          <ellipse cx="66" cy="44" rx="4.5" ry="5.5" {...props} />
          <path d="M34 72 Q42 66 50 70 Q58 74 66 70" {...props} />
        </>
      );

    case 'disgusted':
      return (
        <>
          <path d="M26 44 L42 44" {...props} />
          <path d="M58 44 L74 44" {...props} />
          <path d="M32 70 L42 64 L52 70 L62 64 L70 70" {...props} />
        </>
      );

    case 'unk':
      return (
        <>
          <circle cx="34" cy="46" r="2.6" fill={stroke} />
          <path d="M58 46 L74 46" {...props} />
          <path d="M34 70 Q44 64 60 74" {...props} />
        </>
      );

    case 'neutral':
      return (
        <>
          <circle cx="34" cy="46" r="2.6" fill={stroke} />
          <circle cx="66" cy="46" r="2.6" fill={stroke} />
          <path d="M34 70 L66 70" {...props} />
        </>
      );

    case 'other':
      return (
        <>
          <circle cx="34" cy="46" r="2.6" fill={stroke} />
          <circle cx="66" cy="46" r="2.6" fill={stroke} />
          <text
            x="50"
            y="78"
            textAnchor="middle"
            fontSize="20"
            fontWeight="600"
            fill={stroke}
          >
            ?
          </text>
        </>
      );
  }
}
