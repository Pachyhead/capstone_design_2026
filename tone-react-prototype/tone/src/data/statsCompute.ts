import type { Emotion, Message } from '@/types';

export type AuthorFilter = Set<'me' | 'them'>;

export function filterByAuthor(
  all: Message[],
  conversationId: string,
  authors: AuthorFilter,
): Message[] {
  return all.filter((m) => {
    if (m.conversationId !== conversationId) return false;
    const isMe = m.authorId === 'me';
    return (isMe && authors.has('me')) || (!isMe && authors.has('them'));
  });
}

export function computeDistribution(
  messages: Message[],
): Partial<Record<Emotion, number>> {
  if (messages.length === 0) return {};
  const counts: Partial<Record<Emotion, number>> = {};
  for (const m of messages) {
    counts[m.emotion.primary] = (counts[m.emotion.primary] ?? 0) + 1;
  }
  const total = messages.length;
  const result: Partial<Record<Emotion, number>> = {};
  for (const k in counts) {
    const e = k as Emotion;
    result[e] = (counts[e] ?? 0) / total;
  }
  return result;
}

export interface HourFlow {
  hour: number;
  emotion: Emotion;
  intensity: number;
}

function parseHour(sentAt: string): number | null {
  const m = /^(\d{1,2}):(\d{2})$/.exec(sentAt);
  if (!m) return null;
  const h = Number(m[1]);
  if (!Number.isFinite(h) || h < 0 || h > 23) return null;
  return h;
}

function meanEnergy(energy: number[]): number {
  if (!energy.length) return 0;
  let sum = 0;
  for (const v of energy) sum += v;
  return sum / energy.length;
}

export function computeHourlyFlow(messages: Message[]): HourFlow[] {
  const hourBuckets: Map<number, Message[]> = new Map();
  for (const m of messages) {
    const h = parseHour(m.sentAt);
    if (h === null) continue;
    const bucket = hourBuckets.get(h);
    if (bucket) bucket.push(m);
    else hourBuckets.set(h, [m]);
  }

  const flow: HourFlow[] = [];
  for (let h = 0; h < 24; h++) {
    const bucket = hourBuckets.get(h);
    if (!bucket || bucket.length === 0) {
      flow.push({ hour: h, emotion: 'neutral', intensity: 0 });
      continue;
    }
    const counts: Partial<Record<Emotion, number>> = {};
    let dominant: Emotion = bucket[0].emotion.primary;
    let dominantCount = 0;
    for (const m of bucket) {
      const e = m.emotion.primary;
      const c = (counts[e] ?? 0) + 1;
      counts[e] = c;
      if (c > dominantCount) {
        dominant = e;
        dominantCount = c;
      }
    }
    let intensitySum = 0;
    for (const m of bucket) intensitySum += meanEnergy(m.energy);
    const intensity = intensitySum / bucket.length;
    flow.push({ hour: h, emotion: dominant, intensity });
  }
  return flow;
}
