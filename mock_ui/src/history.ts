// 데모를 열었을 때 이미 오간 대화. 감정 통계와 대화 목록이 비어 보이지 않게 넣어 둔 가짜 기록이다.
// 이 메시지들에는 음성 파일이 없어서 재생을 누르면 "음성이 없는 기록"이라고 안내한다.
// 데모 중에 새로 보낸 메시지만 준비된 음성으로 들을 수 있다.
//
// 백엔드 id: 0 종찬, 1 재웅, 2 경택, 3 태원
// 감정 번호(emo): 0 분노, 1 혐오, 2 두려움, 3 기쁨, 4 중립, 5 기타, 6 슬픔, 7 놀람

interface Line {
  from: 0 | 1 | 2 | 3;
  to: 0 | 1 | 2 | 3;
  /** 2026-11-22 의 시각 "HH:MM" */
  at: string;
  emo: number;
  text: string;
}

const DAY = '2026-11-22';

const LINES: Line[] = [
  // 종찬 ↔ 재웅
  { from: 0, to: 1, at: '09:12', emo: 4, text: '오늘 발표 리허설 몇 시에 해?' },
  { from: 1, to: 0, at: '09:15', emo: 4, text: '두 시쯤 하자. 그 전에 슬라이드 한 번 더 볼게' },
  { from: 0, to: 1, at: '09:16', emo: 3, text: '좋아 그럼 점심 먹고 보자' },
  { from: 1, to: 0, at: '13:40', emo: 7, text: '헐 서버 갑자기 안 켜지는데?' },
  { from: 0, to: 1, at: '13:42', emo: 2, text: '진짜? 리허설 전에 고칠 수 있을까' },
  { from: 1, to: 0, at: '13:58', emo: 3, text: '됐다! 포트 설정 문제였어' },
  { from: 0, to: 1, at: '13:59', emo: 3, text: '다행이다 진짜 고생했어' },
  { from: 1, to: 0, at: '18:20', emo: 6, text: '리허설 생각보다 길어져서 좀 지친다' },
  { from: 0, to: 1, at: '18:24', emo: 4, text: '오늘은 일찍 쉬고 내일 마무리하자' },

  // 종찬 ↔ 경택
  { from: 2, to: 0, at: '10:05', emo: 2, text: '포스터 인쇄 내일까지 맞출 수 있을까요?' },
  { from: 0, to: 2, at: '10:08', emo: 4, text: '오늘 저녁에 파일 넘기면 될 것 같아' },
  { from: 2, to: 0, at: '10:09', emo: 3, text: '오 다행이다. 그럼 표만 다시 정리할게요' },
  { from: 0, to: 2, at: '15:30', emo: 7, text: '그래프 색 바꾼 거 완전 잘 보인다' },
  { from: 2, to: 0, at: '15:33', emo: 3, text: '감사합니다 ㅎㅎ 감정별 색 맞춰 봤어요' },
  { from: 2, to: 0, at: '21:10', emo: 0, text: '인쇄소가 파일 형식 틀렸다고 다시 보내래요' },
  { from: 0, to: 2, at: '21:12', emo: 4, text: 'PDF로 다시 뽑아서 보내 줄게' },

  // 종찬 ↔ 태원
  { from: 3, to: 0, at: '11:20', emo: 4, text: '모델 체크포인트 공유 드라이브에 올렸어' },
  { from: 0, to: 3, at: '11:25', emo: 3, text: '고마워! 바로 받아 볼게' },
  { from: 3, to: 0, at: '16:45', emo: 6, text: '감정 유사도가 생각보다 낮게 나왔네' },
  { from: 0, to: 3, at: '16:50', emo: 4, text: '라벨이 애매한 감정이 섞여서 그런 것 같아' },
  { from: 3, to: 0, at: '16:52', emo: 3, text: '그래도 놀람이랑 슬픔은 꽤 잘 따라가더라' },

  // 재웅 ↔ 경택
  { from: 1, to: 2, at: '09:40', emo: 4, text: '시연 영상 녹화는 누가 하기로 했지?' },
  { from: 2, to: 1, at: '09:44', emo: 2, text: '제가 하기로 했는데 마이크가 좀 불안해요' },
  { from: 1, to: 2, at: '09:45', emo: 4, text: '내 마이크 빌려 줄게 걱정 마' },
  { from: 2, to: 1, at: '14:20', emo: 3, text: '녹화 한 번에 끝났어요!' },
  { from: 1, to: 2, at: '14:22', emo: 7, text: '와 벌써? 대박이다' },
  { from: 2, to: 1, at: '19:05', emo: 1, text: '근데 편집 프로그램이 계속 튕겨요…' },
  { from: 1, to: 2, at: '19:08', emo: 4, text: '내가 내일 아침에 같이 봐 줄게' },

  // 재웅 ↔ 태원
  { from: 3, to: 1, at: '12:10', emo: 7, text: '24bit로 보내는데 감정이 이 정도로 살아?' },
  { from: 1, to: 3, at: '12:13', emo: 3, text: '응 FSQ로 줄여도 꽤 남더라' },
  { from: 3, to: 1, at: '17:30', emo: 0, text: 'TTS 서버 GPU 메모리 또 터졌어' },
  { from: 1, to: 3, at: '17:34', emo: 2, text: '내일 시연 때 터지면 어떡하지' },
  { from: 3, to: 1, at: '17:40', emo: 4, text: '배치 줄여서 다시 띄웠어 이제 괜찮을 거야' },

  // 경택 ↔ 태원
  { from: 2, to: 3, at: '13:05', emo: 4, text: '설문 결과 정리 끝났어요' },
  { from: 3, to: 2, at: '13:10', emo: 3, text: '오 빠르다 고마워' },
  { from: 2, to: 3, at: '20:15', emo: 6, text: '혐오는 맞힌 사람이 한 명도 없어서 아쉬워요' },
  { from: 3, to: 2, at: '20:20', emo: 4, text: '그건 한계로 정리해서 포스터에 넣자' },
];

export interface HistoryMessage {
  message_id: string;
  sender_id: string;
  receiver_id: string;
  message: string;
  emo_type: number;
  send_time: string;
  /** 음성 세트 id. 기록 메시지는 음성이 없어서 null. */
  set_id: string | null;
}

/** 시간 순으로 정렬된 대화 기록 (서버 응답처럼 각 대화방 안에서 오래된 것부터). */
export const HISTORY: HistoryMessage[] = [...LINES]
  .sort((a, b) => a.at.localeCompare(b.at))
  .map((line, i) => ({
    message_id: `history-${String(i).padStart(3, '0')}`,
    sender_id: String(line.from),
    receiver_id: String(line.to),
    message: line.text,
    emo_type: line.emo,
    send_time: `${DAY}T${line.at}:00`,
    set_id: null,
  }));

/** 두 사람 사이의 마지막 기록 메시지 id. 처음 열었을 때 기록을 "읽음"으로 두는 데 쓴다. */
export function lastHistoryId(a: number, b: number): string | undefined {
  const pair = HISTORY.filter(
    (m) =>
      (m.sender_id === String(a) && m.receiver_id === String(b)) ||
      (m.sender_id === String(b) && m.receiver_id === String(a)),
  );
  return pair[pair.length - 1]?.message_id;
}
