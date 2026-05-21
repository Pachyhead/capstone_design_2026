# Tone — UI Prototype

저트래픽 voice messenger UI 프로토타입. 음성을 텍스트(STT)와 24비트 감정 코드(emotion2vec → FSQ-AE)로 분리해 전송하고, 수신측에서 화자 프로필 기반 TTS로 재구성하는 시스템의 클라이언트 화면을 React로 구현한 것.

## 실행

```bash
npm install
npm run dev
```

브라우저에서 `http://localhost:5173` 열기. 음성 모드를 보려면 `?mode=voice` 쿼리 추가 (예: `http://localhost:5173/chat/minsu?mode=voice`).

## 라우트

| 경로 | 화면 |
|---|---|
| `/` | 대화 list |
| `/chat/:id` | 대화 detail (텍스트/음성 모드) |
| `/stats` | 통계 list (per-chatroom) |
| `/stats/:id` | 통계 detail (큰 cluster + 시간 흐름 + 비교) |
| `/me` | 나 (프로필, 화자 등록, 보기 모드, 친구·설정) |
| `/onboarding` | 화자 등록 3단계 |

## 디자인 토큰

`src/tokens/emotions.ts` 가 single source of truth. 9 emotions × 4 stops (light / main / deep / x).

```
joy:      #FBF1D6  #F2D89E  #5C400D  #3D2A0E
sad:      #E5EDF4  #BBCFE5  #283F66  #1B2945
angry:    #FBE3DB  #F2B5A5  #94402C  #7B341F
surprise: #DEF0E8  #B0DECD  #2D6852  #1F4838
fear:     #ECE7F2  #CDC4DE  #3B2E5E  #2D1F47
disgust:  #ECEFD9  #CDD5AE  #545E33  #3E4424
contempt: #F7E7EE  #E8C5D2  #5C2C40  #7E445A
neutral:  #ECE7DE  #CFC5B5  #4A4232  #5D5241
other:    #FAEAD9  #F0CCB5  #8A4828  #6E371F
```

Chrome (Tailwind 클래스):
- `bg-cream` `#FAF6EF` — 라이트 테마 화면 배경
- `bg-sand` `#EDE7DB` — 폰 외곽 / 세컨더리 표면
- `bg-charcoal` `#2C2A26` — 라이트 모드 강조 (FAB, 액센트)
- `text-ink` `#14130F` — 본문
- `text-muted` `#6B5F4F` — 보조 본문
- `text-hint` `#9B8E7B` — 힌트
- `bg-dk-bg` `#14130F` — 음성 모드 배경
- `bg-dk-card-recv` `#1F1D1A` / `bg-dk-card-sent` `#262320` — 음성 모드 말풍선

## 디렉토리 구조

```
src/
├── main.tsx                   # 진입점
├── App.tsx                    # 라우팅 + view mode 상태
├── index.css                  # Tailwind base + 글로벌
├── types/index.ts             # 도메인 타입 + 라벨
├── tokens/emotions.ts         # 감정 팔레트
├── data/mock.ts               # 시드 데이터
├── components/
│   ├── PhoneFrame.tsx         # 폰 셸 (light/dark)
│   ├── BottomNav.tsx          # 3-tab 하단 nav
│   ├── EmotionWaveform.tsx    # 단일 감정 색 waveform
│   ├── EmotionChip.tsx        # 감정 라벨 칩
│   ├── PlayButton.tsx         # 감정 색 재생 버튼
│   └── BubbleCluster.tsx      # 면적 비례 분포 시각화
└── screens/
    ├── ChatList.tsx
    ├── ChatDetail.tsx         # 텍스트 + 음성 모드 분기
    ├── StatsList.tsx
    ├── StatsDetail.tsx
    ├── Me.tsx
    └── Onboarding.tsx
```

## 핵심 디자인 결정 (이번 프로토타입에서 구현됨)

- **Waveform = 발화 단위 단일 감정 색** — 한 메시지 안의 막대는 모두 같은 색, 높이만 audio energy를 따라 변함. emotion2vec_plus_large가 utterance-level 임베딩을 출력하므로, 한 메시지 내부에서 감정 변화를 가짜로 만들지 않음.
- **3-tab bottom nav** — 대화 / 통계 / 나. FAB 제거, "+ 새 대화"는 대화 list 헤더로 이동.
- **두 보기 모드** — 텍스트 모드 (라이트 크림) / 음성 모드 (다크 차콜 + 옐로우 액센트). `?mode=voice` 또는 Me 화면 라디오로 전환. 같은 데이터, 다른 렌더링.
- **Bubble cluster** — 도넛 차트 폐기. r ∝ √percentage 면적 비례 배치.
- **Emotion-tinted chat list rows** — 각 row가 latest 메시지 감정에 맞춰 light pastel bg + emotion-deep text.

## 다음 작업

- `mode` 상태를 Zustand 또는 Context로 끌어올려 Me 화면에서 라디오 클릭 시 즉시 반영
- 친구 / 설정 / 정보 sub-screens
- 새 대화 시작 (친구 picker) flow
- 녹음 modal (chat detail 안에서 inline)
- 통계 detail 내 segmented period 동작 (현재는 UI만)
- Mock 데이터를 backend API 스키마에 맞춰 정리
- API 연동 — STT, emotion2vec, FSQ-AE encoder, Qwen3-TTS reconstruction
