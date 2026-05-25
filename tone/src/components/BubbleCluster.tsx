import type { Emotion } from '@/types';
import { EMOTION_LABELS } from '@/types';
import { paletteFor } from '@/tokens/emotions';

interface Bubble {
  emotion: Emotion;
  percentage: number; // 0..1
}

interface Props {
  data: Partial<Record<Emotion, number>>;
  size?: 'mini' | 'small' | 'large';
  showLabels?: boolean;
}

// pre-computed positions for "small" layout (used in stats list rows): horizontal sorted by size
// the cluster layout for "large" uses an organic packing — we hand-place 6 stops since exact
// circle packing isn't worth the JS for a static screen
const LARGE_SLOTS = [
  { cx: 50, cy: 50, scale: 1.0 }, // dominant — center
  { cx: 18, cy: 26, scale: 0.55 }, // top-left
  { cx: 82, cy: 26, scale: 0.50 }, // top-right
  { cx: 14, cy: 78, scale: 0.42 }, // bottom-left
  { cx: 88, cy: 78, scale: 0.42 }, // bottom-right
  { cx: 50, cy: 90, scale: 0.38 }, // bottom-center
];

export function BubbleCluster({ data, size = 'large', showLabels = true }: Props) {
  // sort emotions by percentage desc and keep top 6
  const sorted: Bubble[] = (Object.entries(data) as [Emotion, number][])
    .sort(([, a], [, b]) => b - a)
    .slice(0, 6)
    .map(([emotion, percentage]) => ({ emotion, percentage }));

  if (size === 'mini') {
    return (
      <div className="flex items-center gap-[3px] flex-shrink-0">
        {sorted.slice(0, 5).map(({ emotion, percentage }, i) => {
          const palette = paletteFor(emotion);
          const radius = Math.max(Math.sqrt(percentage) * 18, 4);
          return (
            <div
              key={`${emotion}-${i}`}
              className="rounded-full flex-shrink-0"
              style={{
                width: `${radius}px`,
                height: `${radius}px`,
                background: palette.main,
              }}
            />
          );
        })}
      </div>
    );
  }

  if (size === 'small') {
    return (
      <div className="flex items-center gap-1 flex-shrink-0">
        {sorted.map(({ emotion, percentage }) => {
          const palette = paletteFor(emotion);
          const radius = Math.max(Math.sqrt(percentage) * 28, 6);
          return (
            <div
              key={emotion}
              className="rounded-full flex-shrink-0"
              style={{
                width: `${radius}px`,
                height: `${radius}px`,
                background: palette.main,
              }}
              title={`${EMOTION_LABELS[emotion]} ${Math.round(percentage * 100)}%`}
            />
          );
        })}
      </div>
    );
  }

  // large: SVG positioned cluster
  return (
    <svg
      viewBox="0 0 320 180"
      xmlns="http://www.w3.org/2000/svg"
      className="w-full h-auto"
      role="img"
      aria-label="감정 분포 클러스터"
    >
      {sorted.map((bubble, i) => {
        const slot = LARGE_SLOTS[i];
        if (!slot) return null;
        const palette = paletteFor(bubble.emotion);
        // bubble radius: dominant ~46, others scaled
        const r = Math.sqrt(bubble.percentage) * 70 * slot.scale + 14;
        const cx = (slot.cx / 100) * 320;
        const cy = (slot.cy / 100) * 180;
        return (
          <g key={bubble.emotion}>
            <circle cx={cx} cy={cy} r={r} fill={palette.main} />
            {showLabels && r > 18 && (
              <>
                <text
                  x={cx}
                  y={cy - 2}
                  textAnchor="middle"
                  fontSize={r > 38 ? 13 : 11}
                  fontWeight={500}
                  fill={palette.deep}
                >
                  {EMOTION_LABELS[bubble.emotion]}
                </text>
                <text
                  x={cx}
                  y={cy + (r > 38 ? 12 : 11)}
                  textAnchor="middle"
                  fontSize={11}
                  fill={palette.deep}
                  opacity={0.6}
                >
                  {Math.round(bubble.percentage * 100)}%
                </text>
              </>
            )}
          </g>
        );
      })}
    </svg>
  );
}
